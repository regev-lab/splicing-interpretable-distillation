import torch

LOSSES = {
    "mse": lambda preds, targets: torch.nn.MSELoss()(preds, targets),
    "kl_divergence": lambda preds, targets: lambda preds, targets: torch.nn.functional.binary_cross_entropy(
        preds, targets
    )
    - torch.nn.functional.binary_cross_entropy(targets, targets),
    "kl_divergence_logits": lambda preds, targets: torch.nn.functional.binary_cross_entropy_with_logits(
        preds, targets
    )
    - torch.nn.functional.binary_cross_entropy(targets, targets),
}
