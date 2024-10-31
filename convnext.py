import evaluate
import numpy as np
from transformers import ConvNextForImageClassification, ConvNextConfig
import torch
from datasets import Dataset


eval_metrics = evaluate.combine(["accuracy", "precision", "recall"])


def initialize_convnext(
    shuffled_dataset: Dataset,
    device: str,
    model: str,
) -> ConvNextForImageClassification:
    """initialize model"""
    labels = shuffled_dataset["train"].features["labels"].names
    config = ConvNextConfig.from_pretrained(model)
    config.id2label = {str(i): c for i, c in enumerate(labels)}
    config.label2id = {c: str(i) for i, c in enumerate(labels)}
    config.num_labels = len(labels)
    return ConvNextForImageClassification.from_pretrained(
        model,
        config=config,
        ignore_mismatched_sizes=True,
    ).to(device)


def compute_metrics(p) -> dict:
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return eval_metrics.compute(predictions=preds, references=labels)

