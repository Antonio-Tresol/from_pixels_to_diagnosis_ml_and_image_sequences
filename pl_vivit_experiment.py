import pandas as pd
import torch
from pytorch_lightning.loggers import WandbLogger

from pl_module import ClassificationLightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from datasets import ImageSequenceClassificationDataset
from data_module import ImageSequenceDataModule
from data_tools.sampling import Sampling
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
)
from torchmetrics import MetricCollection
from pl_vivit import ViViT, get_vivit_transformations
from torch import nn
import wandb
import config
import gc


def main() -> None:
    torch.set_float32_matmul_precision("high")
    class_count = 2

    metrics = MetricCollection(
        {
            "Accuracy": BinaryAccuracy(),
            "Precision": BinaryPrecision(),
            "Recall": BinaryRecall(),
        },
    )
    train_transform = test_transform = get_vivit_transformations()

    train_dataset = ImageSequenceClassificationDataset(
        root_dir=config.DATASET_DIR,
        transform=train_transform,
        device=config.DEVICE,
    )
    test_dataset = ImageSequenceClassificationDataset(
        root_dir=config.DATASET_DIR,
        transform=test_transform,
        device=config.DEVICE,
    )

    patient_dm = ImageSequenceDataModule(
        dataset=config.DATASET_DIR,
        root_dir=config.DATASET_DIR,
        batch_size=config.BATCH_SIZE,
        train_base_dataset=train_dataset,
        test_base_dataset=test_dataset,
        train_size=config.TRAIN_SIZE,
        test_size=config.TEST_SIZE,
        use_index=config.USE_INDEX,
        indices_dir=config.INDICES_DIR,
        sampling=Sampling.NONE,
    )

    vivit = ViViT(num_classes=class_count, device=config.DEVICE)
    model = ClassificationLightningModule(
        model=vivit,
        model_name=config.MODEL_NAME,
        loss_fn=nn.CrossEntropyLoss(),
        metrics=metrics,
        lr=config.LR,
        scheduler_max_it=config.SCHEDULER_MAX_IT,
        class_names=config.CLASS_NAMES,
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=config.PATIENCE,
        strict=False,
        verbose=False,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=config.VIVIT_DIR,
        filename=config.VIVIT_FILENAME,
        save_top_k=config.TOP_K_SAVES,
        mode="min",
    )

    wandb_id = config.VIVIT_FILENAME + wandb.util.generate_id()
    wandb_logger = WandbLogger(
        project=config.WANDB_PROJECT,
        id=wandb_id,
        resume="allow",
    )

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=config.EPOCHS,
        log_every_n_steps=1,
    )
    torch.cuda.empty_cache()
    gc.collect()
    trainer.fit(model, datamodule=patient_dm)
    trainer.test(model, datamodule=patient_dm)

    wandb.finish()


if __name__ == "__main__":
    main()
