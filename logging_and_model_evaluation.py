import evaluate
import numpy as np
import wandb
from transformers import EvalPrediction

vivit_eval_metrics = evaluate.combine(["accuracy", "precision", "recall"])


def vivit_compute_metrics(p: EvalPrediction) -> dict:
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    result = vivit_eval_metrics.compute(predictions=preds, references=labels)

    log_confusion_matrix(preds, labels)

    return result


convnext_eval_metrics = evaluate.combine(["accuracy", "precision", "recall"])


def convnext_compute_metrics(p: EvalPrediction) -> dict:
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids

    log_confusion_matrix(preds, labels)
    return convnext_eval_metrics.compute(predictions=preds, references=labels)


def log_confusion_matrix(preds: np.ndarray, labels: list) -> None:
    wandb.log(
        {
            "confusion_matrix": wandb.plot.confusion_matrix(
                preds=preds,
                y_true=labels,
                class_names=[
                    "Negative",
                    "Positive",
                ],
            ),
        },
    )
