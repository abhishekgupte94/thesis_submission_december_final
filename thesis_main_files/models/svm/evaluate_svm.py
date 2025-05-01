import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

class SVMEvaluator:
    @staticmethod
    def evaluate(model, X_test, y_test, plot_curves=True, plot_dir=None):
        """
        Evaluates an SVM model on test data and optionally saves ROC + PR curves.
        """
        y_pred = model.predict(X_test)
        y_scores = model.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_scores)
        ap = average_precision_score(y_test, y_scores)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        if plot_curves and plot_dir:
            os.makedirs(plot_dir, exist_ok=True)

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.2f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title("ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(plot_dir, "roc_curve.png"))
            plt.close()

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_scores)
            plt.figure()
            plt.plot(recall, precision, label=f"AP = {ap:.2f}")
            plt.title("Precision-Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(plot_dir, "precision_recall_curve.png"))
            plt.close()

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "average_precision": ap,
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }
