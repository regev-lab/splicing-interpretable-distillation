from collections import defaultdict
import contextlib
import numpy as np
import os
import pandas as pd
import random
from ray import tune
import torch
from typing import Tuple

from datasets.sequence import SequenceDataset
from metrics import losses
from metrics import metrics
from models.nbm import Conv1DNBM


# torch.backends.cudnn.benchmark = True
def set_experiment_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class Experiment(tune.Trainable):
    """
    An experiment object.
    """

    def load_datasets(
        self, config: dict
    ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        if not hasattr(self, "trainval_pd"):
            # In case we are resetting, do not waste time reloading the data.
            self.trainval_pd = pd.read_csv(config["train_csv"])

        # Sample the training and validation sets
        # Use the experiment seed to ensure that the same samples are used for each trial
        print("Sampling train and validation splits")
        train_pd = self.trainval_pd.sample(
            frac=config["train_fraction"], random_state=config["experiment_seed"]
        )
        val_pd = self.trainval_pd.drop(train_pd.index)

        print("Initializing datasets")
        trainset = SequenceDataset(
            data_path=None,
            data_df=train_pd,
            sequence_column=config["sequence_column"],
            target_column=config["target_column"],
            upstream_sequence=config["upstream_sequence"],
            downstream_sequence=config["downstream_sequence"],
        )
        validset = SequenceDataset(
            data_path=None,
            data_df=val_pd,
            sequence_column=config["sequence_column"],
            target_column=config["target_column"],
            upstream_sequence=config["upstream_sequence"],
            downstream_sequence=config["downstream_sequence"],
        )
        
        return trainset, validset

    def parse_config(self, config: dict) -> Tuple[torch.nn.Module, dict]:
        """
        Parse the configuration and create the model. Returns the model and updated config.
        """
        if config["model_architecture"] == "conv1d_nbm":
            num_sequence_kernel_types = config["sequence_kernels"]["num_types"]
            config["sequence_kernels"] = [
                (
                    int(2 ** config["sequence_kernels"][f"kernel{i}_num"]),
                    int(config["sequence_kernels"][f"kernel{i}_size"]),
                )
                for i in range(num_sequence_kernel_types)
            ]
            
            config["feature_hidden_size"] = int(config["feature_hidden_size"])
            config["tuner_hidden_size"] = int(config["tuner_hidden_size"])

            config["batch_size"] = int(2 ** config["batch_size"])

            return Conv1DNBM(**config), config
        else:
            raise ValueError(
                f"Unknown model architecture: {config['model_architecture']}"
            )

    def setup(self, config: dict):
        """
        Setup the experiment.
        """
        # Parse control parameters from the config
        skip_dataset = "skip_dataset" in config and config["skip_dataset"]
        skip_dataloader = "skip_dataloader" in config and config["skip_dataloader"]
        skip_compiling = "skip_compiling" in config and config["skip_compiling"]

        if "trial_seed" in config:
            set_experiment_seed(config["trial_seed"])

        print("Loading model")
        self.model, self.config = self.parse_config(config)
        if "device" in config:
            self.device = config["device"]
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not skip_dataset:
            print("Loading datasets")
            self.trainset, self.validset = self.load_datasets(config)

        # Use gradient accumulation over batch sizes of batch_size_iter
        # to achieve an effective batch size of batch_size.
        # This is useful for large batch sizes that do not fit in memory.
        self.batch_size = int(config["batch_size"])
        self.batch_size_iter = min(
            int(config["batch_size_iter"]), int(config["batch_size"])
        )

        if self.batch_size % self.batch_size_iter != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by batch_size_iter ({self.batch_size_iter})"
            )

        self.iters_to_accumulate = int(self.batch_size / self.batch_size_iter)

        if not skip_dataloader:
            print("Loading dataloaders")
            self.trainloader = torch.utils.data.DataLoader(
                self.trainset,
                batch_size=self.batch_size_iter,
                shuffle=True,
                num_workers=2,
                prefetch_factor=5,
                pin_memory=True,
                persistent_workers=True,
                drop_last=True,
            )
            self.validloader = torch.utils.data.DataLoader(
                self.validset,
                batch_size=self.batch_size_iter,
                shuffle=False,
                num_workers=2,
                prefetch_factor=5,
                pin_memory=True,
                persistent_workers=True,
                drop_last=True,
            )

        print("Sending model to device")
        self.model = self.model.to(self.device)

        if config["loss"] in losses.LOSSES:
            self.loss = losses.LOSSES[config["loss"]]
        else:
            raise ValueError(f"Unknown loss function: {config['loss']}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["init_lr"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=config["lr_decay_factor"],
            patience=config["lr_decay_patience"],
            min_lr=1e-8,
        )
        self.scaler = torch.amp.GradScaler(device=self.device, enabled=config["amp"])

        self.eval_fns = dict()
        for metric in config["metrics"]:
            self.eval_fns[metric] = metrics.METRICS[metric]

        if not skip_compiling:
            print("Compiling model")
            self.model = torch.compile(self.model, mode="default")

    def reset_config(self, new_config: dict):
        """
        Reset the configuration.
        """
        self.setup(new_config | {"skip_dataset": True})
        return True

    def _calculate_loss(
        self,
        seq,
        target,
        weight_rebal: torch.Tensor = None,
        normalization_factor: float = 1.0,
    ):
        """
        Helper function to calculate the loss.
        """
        pred = self.model(seq).view(-1)
        return (
            (
                self.loss(pred, target, weight_rebal) / normalization_factor
                if weight_rebal is not None
                else self.loss(pred, target) / normalization_factor
            ),
            pred,
        )

    def _step(
        self,
        metrics_prefix: str,
        loader: torch.utils.data.DataLoader,
        train: bool,
    ) -> dict:
        """
        Run one step of training or evaluation.

        Args:
            metrics_prefix (str): Prefix for metrics.
            loader (torch.utils.data.DataLoader): Data loader.
            train (bool): Whether to train the model.
        """

        losses = defaultdict(float)
        track_metrics = not train

        # Track metrics when not training
        if track_metrics:
            all_targets = []
            all_preds = []

        for i, batch in enumerate(loader):
            seq, target = batch

            seq, target = (
                seq.to(self.device, non_blocking=train),
                target.to(self.device, non_blocking=train),
            )

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.config["amp"],
            ):
                loss, pred = self._calculate_loss(
                    seq, target, None, len(loader)
                )
                losses[
                    f'{metrics_prefix}-{self.config["loss"]}-loss'
                ] += loss.item()
                total_loss = loss

                if train:
                    reg_penalty = self.model.get_regularization_penalty() / len(
                        loader
                    )
                    total_loss += reg_penalty
                    losses[f"{metrics_prefix}-reg-penalty"] += reg_penalty.item()

                losses[f"{metrics_prefix}-total-loss"] += total_loss.item()

            if train:
                self.scaler.scale(total_loss).backward()

                if (i + 1) % self.iters_to_accumulate == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

            if track_metrics:
                all_preds.extend(pred.tolist())
                all_targets.extend(target.view(-1).tolist())

        # Report metrics
        step_metrics = {k: v for k, v in losses.items()}

        if track_metrics:
            all_preds = torch.tensor(all_preds)
            all_targets = torch.tensor(all_targets)
            step_metrics.update(
                {
                    f"{metrics_prefix}-{metric}": fn(all_preds, all_targets)
                    for metric, fn in self.eval_fns.items()
                }
            )

        return step_metrics

    def step(self):
        """
        Run one step of training and evaluation.
        """
        self.model.train()
        train_metrics = self._step(
            metrics_prefix="train",
            loader=self.trainloader,
            train=True,
        )
        self.model.eval()
        with torch.no_grad():
            valid_metrics = self._step(
                metrics_prefix="valid",
                loader=self.validloader,
                train=False,
            )

        return {**train_metrics, **valid_metrics}

    def save_checkpoint(self, checkpoint_path: str) -> str:
        """
        Save the model checkpoint.
        """
        path = os.path.join(checkpoint_path, "checkpoint.pt")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
            },
            path,
        )

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load the model checkpoint.
        """
        path = os.path.join(checkpoint_path, "checkpoint.pt")

        state = torch.load(path)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scaler.load_state_dict(state["scaler_state_dict"])
