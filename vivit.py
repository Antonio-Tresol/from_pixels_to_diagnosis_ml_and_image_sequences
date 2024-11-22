from datasets import Dataset
from transformers import VivitConfig, VivitForVideoClassification
import torch


def collate_fn(batch: torch.Tensor) -> dict:
    return {
        "pixel_values": torch.stack(
            [(torch.tensor(x["pixel_values"])) for x in batch],
        ),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def initialize_vivit(
    shuffled_dataset: Dataset,
    device: str,
    model: str,
) -> VivitForVideoClassification:
    """initialize model"""
    labels = shuffled_dataset["train"].features["labels"].names
    config = VivitConfig.from_pretrained(model)
    config.num_classes = len(labels)
    config.id2label = {str(i): c for i, c in enumerate(labels)}
    config.label2id = {c: str(i) for i, c in enumerate(labels)}
    config.num_frames = 18
    config.video_size = [18, 224, 224]

    return VivitForVideoClassification.from_pretrained(
        model,
        ignore_mismatched_sizes=True,
        config=config,
    ).to(device)
