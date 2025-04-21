import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve, mean_absolute_error,mean_squared_error, r2_score
)

def accuracy(y_true, y_pred):
    """Compute accuracy for classification models."""
    y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_true_labels = y_true.cpu().numpy()
    return accuracy_score(y_true_labels, y_pred_labels)

def auc_score(y_true, y_scores):
    """Compute Area Under the Curve (AUC-ROC)."""
    return roc_auc_score(y_true.cpu().numpy(), y_scores.cpu().numpy())

def plot_roc_curve(y_true, y_scores):
    """Plot ROC Curve."""
    fpr, tpr, _ = roc_curve(y_true.cpu().numpy(), y_scores.cpu().numpy())
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(auc_score(y_true, y_scores)))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

def plot_precision_recall_curve(y_true, y_scores):
    """Plot Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true.cpu().numpy(), y_scores.cpu().numpy())
    plt.figure()
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

def top_k_accuracy(y_true, y_pred, k=5):
    """Compute Top-K Accuracy."""
    top_k_preds = torch.topk(y_pred, k=k, dim=1)[1]
    correct = torch.eq(top_k_preds, y_true.view(-1, 1)).sum().item()
    return correct / y_true.shape[0]


def mse(y_true, y_pred):
    """Mean Squared Error."""
    return mean_squared_error(y_true.cpu().numpy(), y_pred.cpu().numpy())

def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return mean_absolute_error(y_true.cpu().numpy(), y_pred.cpu().numpy())

def r2(y_true, y_pred):
    """R-Squared Score."""
    return r2_score(y_true.cpu().numpy(), y_pred.cpu().numpy())


def misclassification_severity(y_true, y_pred):
    """
    Misclassification Severity Score (MSS)
    - Measures how far off the model's prediction is from the true class.
    - Generates a bar plot showing the 10 worst misclassified instances.

    Returns:
        - Mean severity of misclassifications
    """
    y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_true_labels = y_true.cpu().numpy()

    # Severity is measured by the absolute distance from the correct class
    severity = np.abs(y_pred_labels - y_true_labels)

    # Identify top 10 worst cases
    worst_indices = np.argsort(severity)[-10:]
    worst_severity = severity[worst_indices]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(range(10), worst_severity, tick_label=worst_indices, color='red')
    plt.xlabel("Input Index")
    plt.ylabel("Misclassification Severity")
    plt.title("Top 10 Worst Misclassified Inputs")
    plt.show()

    return np.mean(severity)  # Average severity across all instances


def confidence_deviation(y_true, y_pred):
    """
    Confidence Deviation Score (CDS)
    - Identifies cases where the model is highly confident but wrong.
    - Generates a bar plot showing the 10 worst confidence deviations.

    Returns:
        - Mean confidence of incorrect predictions
    """
    y_pred_probs = torch.softmax(y_pred, dim=1).cpu().numpy()
    y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_true_labels = y_true.cpu().numpy()

    wrong_predictions = y_pred_labels != y_true_labels
    confidence_scores = np.max(y_pred_probs, axis=1)

    # Extract confidence scores of wrong predictions
    wrong_confidence_scores = confidence_scores[wrong_predictions]
    wrong_indices = np.where(wrong_predictions)[0]

    if len(wrong_confidence_scores) == 0:
        print("No incorrect high-confidence predictions found.")
        return 0.0  # No incorrect high-confidence predictions

    # Identify top 10 worst cases
    worst_indices = np.argsort(wrong_confidence_scores)[-10:]
    worst_confidence = wrong_confidence_scores[worst_indices]
    worst_input_indices = wrong_indices[worst_indices]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(range(10), worst_confidence, tick_label=worst_input_indices, color='blue')
    plt.xlabel("Input Index")
    plt.ylabel("Confidence Score")
    plt.title("Top 10 High-Confidence Wrong Predictions")
    plt.show()

    return np.mean(worst_confidence)  # Mean confidence of incorrect predictions