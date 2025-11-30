import scipy
import torch

METRICS = {
    "mse": lambda preds, labels: torch.mean(torch.square(preds - labels)).item(),
    "mse-logits": lambda preds, labels: torch.mean(
        torch.square(torch.special.expit(preds) - labels)
    ).item(),
    "pearson-r": lambda preds, labels: scipy.stats.pearsonr(preds, labels).statistic,
    "pearson-r-pval": lambda preds, labels: scipy.stats.pearsonr(preds, labels).pvalue,
    "pearson-logits-r": lambda preds, labels: scipy.stats.pearsonr(
        torch.special.expit(preds), labels
    ).statistic,
    "pearson-logits-r-pval": lambda preds, labels: scipy.stats.pearsonr(
        torch.special.expit(preds), labels
    ).pvalue,
}
