import evaluate
import numpy as np
from transformers import ConvNextForImageClassification, ConvNextConfig
import torch
from datasets import Dataset


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


