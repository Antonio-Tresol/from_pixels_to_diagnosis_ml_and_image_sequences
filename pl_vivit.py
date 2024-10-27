from torchvision import transforms
from torch import nn
import torch
from transformers import VivitConfig, VivitForVideoClassification
import config


def get_vivit_config(
    num_classes: int,
    labels: list[str],
    model: str,
    num_frames: int,
) -> VivitConfig:
    """initialize model"""
    config = VivitConfig.from_pretrained(model)
    config.num_classes = num_classes
    config.id2label = {str(i): c for i, c in enumerate(labels)}
    config.label2id = {c: str(i) for i, c in enumerate(labels)}
    config.num_frames = num_frames
    config.video_size = [18, 224, 224]
    return config


class ViViT(nn.Module):
    def __init__(self, num_classes: int, device: str) -> None:
        super().__init__()
        self.vivit = VivitForVideoClassification(
            config=get_vivit_config(
                num_classes=num_classes,
                labels=["Negative", "Positive"],
                model=config.MODEL_NAME,
                num_frames=config.NUM_FRAMES,
            ),
        ).to(device)

    def forward(self, x, labels=None) -> torch.Tensor:
        if labels is None:
            return self.vivit.forward(pixel_values=x)
        return self.vivit.forward(pixel_values=x, labels=labels)


def get_vivit_transformations() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
    )


def testing() -> None:
    from pl_vivit import get_vivit_transformations
    from datasets import ImageSequenceClassificationDataset
    from data_module import ImageSequenceDataModule
    from data_tools.sampling import Sampling

    train_transform = test_transform = get_vivit_transformations()

    train_dataset = ImageSequenceClassificationDataset(
        root_dir="dataset",
        transform=train_transform,
        device=config.DEVICE,
    )
    test_dataset = ImageSequenceClassificationDataset(
        root_dir="dataset",
        transform=test_transform,
        device=config.DEVICE,
    )

    vivit_dm = ImageSequenceDataModule(
        dataset="testing",
        root_dir="dataset",
        batch_size=4,
        train_base_dataset=train_dataset,
        test_base_dataset=test_dataset,
        train_size=0.80,
        test_size=0.20,
        use_index=True,
        indices_dir="indices",
        sampling=Sampling.NONE,
    )
    vivit = ViViT(2, config.DEVICE)
    x = train_dataset[0][0].unsqueeze(0)
    y = train_dataset[0][1]
    y = (
        torch.Tensor([1, 0])
        if y == torch.Tensor(1).to(config.DEVICE)
        else torch.Tensor([0, 1])
    )
    y.to(config.DEVICE)
    print("x:", x.shape, "\n y ", y.shape)
    ouput = vivit(x)
    from torch.nn import CrossEntropyLoss
    loss_fct = CrossEntropyLoss()
    y_hat = ouput.logits
    print("y_hat:", y_hat.shape, "\n y ", y.shape)
    loss = loss_fct(y_hat, y.unsqueeze(0).to(config.DEVICE))
    print(loss)


if __name__ == "__main__":
    testing()
