"""
evaluation_metrics.py
======================

Standalone evaluation utilities for AV pretraining / finetuning.

Implements:
- Accuracy (top-1)
- Top-k accuracy
- Precision / Recall / F1 (macro + per-class)
- Confusion matrix
- Intersection over Union (IoU) from a confusion matrix
- ROC AUC (binary)
- Grad-CAM heatmaps
"""

from typing import Dict, Tuple, Optional, Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_num_classes(logits: torch.Tensor) -> int:
    if logits.ndim == 1:
        return 2
    if logits.size(-1) == 1:
        return 2
    return logits.size(-1)


def _to_pred_labels(logits: torch.Tensor) -> torch.Tensor:
    """
    Turn logits into discrete class predictions.

    Binary case:
        - shape [..., 1] or [...]
        - threshold at 0 to produce {0,1}
    Multi-class case:
        - shape [..., C] with C > 1
        - argmax over last dim
    """
    if logits.ndim == 1:
        # treat as binary logits
        return (logits >= 0).long().view(-1)
    if logits.size(-1) == 1:
        return (logits.squeeze(-1) >= 0).long().view(-1)
    # multi-class
    return torch.argmax(logits, dim=-1).view(-1)


def _flatten_targets(targets: torch.Tensor) -> torch.Tensor:
    """
    Flatten targets to 1D long tensor.
    """
    if targets.ndim > 1:
        targets = targets.view(-1)
    return targets.long()


# ---------------------------------------------------------------------------
# Core classification metrics
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Top-1 accuracy.

    Args:
        logits: (N, C) or (N,) or (N, 1)
        targets: (N,) class indices or {0,1} for binary.

    Returns:
        Scalar tensor with accuracy in [0,1].
    """
    preds = _to_pred_labels(logits)
    t = _flatten_targets(targets).to(preds.device)
    assert preds.numel() == t.numel(), "logits and targets must have same number of elements"

    correct = (preds == t).float().sum()
    return correct / t.numel()


def top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """
    Top-k accuracy for multi-class classification.

    If logits are binary (C==1 or 1D), this falls back to normal accuracy.

    Args:
        logits: (N, C)
        targets: (N,)
        k: top-k value

    Returns:
        Scalar tensor with top-k accuracy in [0,1].
    """
    num_classes = _infer_num_classes(logits)
    if num_classes <= 2:
        return accuracy(logits, targets)

    t = _flatten_targets(targets)
    if logits.ndim != 2:
        raise ValueError("top_k_accuracy expects logits with shape (N, C) for multi-class problems.")

    _, pred_topk = logits.topk(k, dim=-1)  # (N, k)
    correct_topk = pred_topk.eq(t.view(-1, 1).to(pred_topk.device)).any(dim=1)
    return correct_topk.float().mean()


def confusion_matrix(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute confusion matrix.

    Args:
        logits: (N, C) or (N,) or (N, 1)
        targets: (N,)
        num_classes: optional number of classes. If None, inferred from logits & targets.

    Returns:
        Tensor of shape (num_classes, num_classes):
            [i, j] = count of examples with true class i and predicted class j
    """
    preds = _to_pred_labels(logits)
    t = _flatten_targets(targets).to(preds.device)

    if num_classes is None:
        num_classes = max(_infer_num_classes(logits), int(t.max().item()) + 1)

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=preds.device)

    for true_c, pred_c in zip(t, preds):
        if 0 <= true_c < num_classes and 0 <= pred_c < num_classes:
            cm[true_c, pred_c] += 1

    return cm


def precision_recall_f1_from_confusion(
    cm: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-class and macro-averaged precision, recall and F1 from confusion matrix.

    Args:
        cm: (C, C) tensor, where rows are true labels and columns are predicted labels.
        eps: numerical stability term.

    Returns:
        Dict with keys:
            - "precision_per_class" : (C,)
            - "recall_per_class"    : (C,)
            - "f1_per_class"        : (C,)
            - "precision_macro"     : (C,)
            - "recall_macro"        : (C,)
            - "f1_macro"            : scalar
    """
    # True positives: diagonal
    tp = cm.diag().float()
    # For each predicted class j, sum over true labels i -> column sum
    pred_sum = cm.sum(dim=0).float()
    # For each true class i, sum over predicted labels j -> row sum
    true_sum = cm.sum(dim=1).float()

    precision_per_class = tp / (pred_sum + eps)
    recall_per_class = tp / (true_sum + eps)
    f1_per_class = 2 * precision_per_class * recall_per_class / (
        precision_per_class + recall_per_class + eps
    )

    precision_macro = precision_per_class.mean()
    recall_macro = recall_per_class.mean()
    f1_macro = f1_per_class.mean()

    return {
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


# ---------------------------------------------------------------------------
# Intersection over Union (IoU)
# ---------------------------------------------------------------------------

def iou_from_confusion(cm: torch.Tensor, eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    """
    Compute per-class IoU and mean IoU from a confusion matrix.

    IoU_c = TP_c / (TP_c + FP_c + FN_c)

    Args:
        cm: (C, C) confusion matrix.

    Returns:
        Dict with:
            - "iou_per_class" : (C,)
            - "iou_mean"      : scalar
    """
    tp = cm.diag().float()
    pred_sum = cm.sum(dim=0).float()   # TP + FP
    true_sum = cm.sum(dim=1).float()   # TP + FN

    fp = pred_sum - tp
    fn = true_sum - tp

    iou_per_class = tp / (tp + fp + fn + eps)
    iou_mean = iou_per_class.mean()

    return {
        "iou_per_class": iou_per_class,
        "iou_mean": iou_mean,
    }


# ---------------------------------------------------------------------------
# ROC AUC (binary)
# ---------------------------------------------------------------------------

def binary_auc_roc(
    logits_or_probs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute ROC AUC for a binary classifier.

    Args:
        logits_or_probs: (N,) or (N, 1) or (N, 2)
            - If shape (N, 2): assumes scores for [neg_class, pos_class] and uses col 1.
            - Else: applies sigmoid and treats as score for positive class.
        targets: (N,) with values in {0,1}.

    Returns:
        Scalar tensor with AUC in [0,1].
    """
    t = _flatten_targets(targets).float()

    scores = logits_or_probs
    if scores.ndim == 2 and scores.size(-1) == 2:
        scores = scores[:, 1]
    scores = scores.view(-1)

    # Convert logits to probabilities in (0,1) if needed
    if scores.min() < 0 or scores.max() > 1:
        scores = torch.sigmoid(scores)

    # Sort by predicted score (descending)
    sorted_scores, indices = torch.sort(scores, descending=True)
    sorted_targets = t[indices]

    # True positives and false positives as we sweep threshold
    cum_pos = torch.cumsum(sorted_targets, dim=0)
    cum_neg = torch.cumsum(1.0 - sorted_targets, dim=0)

    total_pos = cum_pos[-1].clamp(min=1.0)
    total_neg = cum_neg[-1].clamp(min=1.0)

    tpr = cum_pos / total_pos      # true positive rate
    fpr = cum_neg / total_neg      # false positive rate

    # Add (0,0) at the start for a proper ROC curve
    tpr = torch.cat([torch.zeros(1, device=tpr.device), tpr])
    fpr = torch.cat([torch.zeros(1, device=fpr.device), fpr])

    # Numerical integration (trapezoidal rule)
    auc = torch.trapz(tpr, fpr)
    return auc


# ---------------------------------------------------------------------------
# High-level classification report (now includes AUC for binary)
# ---------------------------------------------------------------------------

def classification_report(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    topk: Sequence[int] = (1, 5),
) -> Dict[str, torch.Tensor]:
    """
    Convenience wrapper to compute a full set of classification metrics.

    Computes:
        - accuracy (top-1)
        - top-k accuracies
        - confusion matrix
        - precision / recall / F1 (macro + per-class)
        - IoU (per-class and mean)
        - AUC (for binary classification)
    """
    if num_classes is None:
        num_classes = _infer_num_classes(logits)

    acc_top1 = accuracy(logits, targets)

    # Top-k accuracies
    topk_results: Dict[str, torch.Tensor] = {}
    for k in topk:
        topk_results[f"top_{k}_acc"] = top_k_accuracy(logits, targets, k=k)

    cm = confusion_matrix(logits, targets, num_classes=num_classes)
    prf = precision_recall_f1_from_confusion(cm)
    iou = iou_from_confusion(cm)

    out: Dict[str, torch.Tensor] = {
        "acc": acc_top1,
        "confusion_matrix": cm,
        "iou_per_class": iou["iou_per_class"],
        "iou_mean": iou["iou_mean"],
        **prf,
        **topk_results,
    }

    # Only meaningful for binary problems
    if num_classes <= 2:
        out["auc_roc"] = binary_auc_roc(logits, targets)

    return out


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Minimal Grad-CAM implementation for visualizing model decisions.

    Typical usage:

        gradcam = GradCAM(model, target_layer_name="layer4")
        heatmap = gradcam(input_tensor, target_class=None)

    - `model` should be a nn.Module.
    - `target_layer_name` should point to a convolutional feature layer
       (e.g., "backbone.layer4" or similar).
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        self.model = model
        self.target_layer = self._get_submodule(target_layer_name)

        self._activations = None
        self._gradients = None

        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _get_submodule(self, target_layer_name: str) -> torch.nn.Module:
        submodule = self.model
        for attr in target_layer_name.split("."):
            submodule = getattr(submodule, attr)
        return submodule

    def _forward_hook(self, module, input, output):
        # Save activations from the target layer
        self._activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple; take gradients w.r.t. output feature maps
        self._gradients = grad_output[0]

    def __call__(
        self,
        inputs: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute Grad-CAM heatmap.

        Args:
            inputs: input tensor (e.g., image or batch of images).
            target_class: optional integer specifying which class logit to
                backpropagate. If None, uses the argmax over logits.

        Returns:
            Heatmap tensor with the same spatial size as the target layer's
            feature maps (upsampling to input size can be done outside).
        """
        self.model.zero_grad()

        outputs = self.model(inputs)  # assumes model returns logits
        if isinstance(outputs, dict):
            # try to pull "logits" if this is a dict-style output
            if "logits" not in outputs:
                raise ValueError(
                    "GradCAM expects model(inputs) to return logits or a dict containing 'logits'."
                )
            logits = outputs["logits"]
        else:
            logits = outputs

        if target_class is None:
            target_class = torch.argmax(logits, dim=-1)

        # Handle batch-wise target selection
        indices = target_class.view(-1, 1)
        selected_logits = torch.gather(logits, 1, indices).squeeze()

        selected_logits.backward(torch.ones_like(selected_logits))

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients. Check target_layer_name.")

        grads = self._gradients
        activations = self._activations

        # Assumes activations: (N, C, H, W) or (N, C, T, H, W)
        if activations.ndim == 4:
            weights = grads.mean(dim=(2, 3), keepdim=True)  # (N, C, 1, 1)
            cam = (weights * activations).sum(dim=1, keepdim=True)  # (N, 1, H, W)
        elif activations.ndim == 5:
            # 3D case (e.g., video): average over T, H, W
            weights = grads.mean(dim=(2, 3, 4), keepdim=True)  # (N, C, 1, 1, 1)
            cam = (weights * activations).sum(dim=1, keepdim=True)  # (N, 1, T, H, W)
        else:
            raise ValueError(
                f"Unsupported activation ndim for Grad-CAM: {activations.ndim}. "
                "Expected 4D or 5D feature maps."
            )

        cam = F.relu(cam)
        # Normalize each example to [0, 1]
        cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(
            -1, 1, 1, *([1] * (cam.ndim - 3))
        )
        cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(
            -1, 1, 1, *([1] * (cam.ndim - 3))
        )
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam
