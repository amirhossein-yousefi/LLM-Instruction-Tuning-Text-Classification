from __future__ import annotations
from typing import List, Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def evaluate_predictions(y_true_series, y_pred: List[str], labels: List[str]) -> Dict[str, float]:
    mapping = {lab: i for i, lab in enumerate(labels)}
    y_true_idx = np.array([mapping.get(y, -1) for y in y_true_series.tolist()])
    y_pred_idx = np.array([mapping.get(y, -1) for y in y_pred])

    mask = y_true_idx != -1
    y_true_idx = y_true_idx[mask]
    y_pred_idx = y_pred_idx[mask]

    acc = accuracy_score(y_true_idx, y_pred_idx)
    f1_micro = f1_score(y_true_idx, y_pred_idx, average="micro")
    f1_macro = f1_score(y_true_idx, y_pred_idx, average="macro")

    print(f"\nAccuracy:   {acc:.3f}")
    print(f"F1 (micro): {f1_micro:.3f}")
    print(f"F1 (macro): {f1_macro:.3f}")

    print("\nPer-class report:")
    print(classification_report(
        y_true_idx,
        y_pred_idx,
        target_names=labels,
        labels=list(range(len(labels))),
        digits=3,
        zero_division=0,
    ))

    print("\nConfusion matrix (rows = true, cols = pred):")
    print(confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(labels)))))

    return {"accuracy": acc, "f1_micro": f1_micro, "f1_macro": f1_macro}
