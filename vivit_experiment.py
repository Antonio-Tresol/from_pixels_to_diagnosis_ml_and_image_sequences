from pathlib import Path
import torch
import data_handling as dh
from vivit import initialize_vivit
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    VivitForVideoClassification,
)
import wandb
import config
from log_and_eval import vivit_compute_metrics
from key import WANDB_KEY
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from datasets import Dataset
import pandas as pd
import numpy as np


def create_optimizer(vivit: VivitForVideoClassification) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        vivit.parameters(),
        lr=config.LEARNING_RATE,
        betas=config.BETAS,
        eps=config.EPSILON,
    )


def create_training_arguments(run_num: int, seed: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=f"{config.TRAINING_DIR}_run_{run_num}",
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH,
        per_device_eval_batch_size=config.EVAL_BATCH,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir=f"{config.LOGGING_DIR}_run_{run_num}",
        logging_steps=config.LOGGING_STEPS,
        seed=seed,
        eval_strategy=config.EVAL_STRATEGY,
        eval_steps=config.EVAL_STEPS,
        warmup_steps=config.WARMUP_STEPS,
        optim=config.OPTIMIZATION_ALGORITHM,
        lr_scheduler_type=config.SCHEDULER,
        fp16=config.SMALL_FLOATING_POINT,
        report_to=config.LOGGER,
        run_name=f"{config.RUN_NAME}_run_{run_num}",
        # so the trainer takes the best checkpoint at the end of training
        load_best_model_at_end=True,
        # how many checkpoints to save
        save_total_limit=1,
        # the metric to save the model checkpoint
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_strategy=config.EVAL_STRATEGY,
    )


def create_trainer(
    train_dataset: dict[str, Dataset],
    vivit: VivitForVideoClassification,
    training_args: TrainingArguments,
    optimizer: torch.optim.AdamW,
) -> Trainer:
    return Trainer(
        model=vivit,
        args=training_args,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["test"],
        optimizers=(optimizer, None),
        compute_metrics=vivit_compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING),
        ],
    )


def save_metrics_and_confusion_matrix_locally(
    run_num: int,
    train_dataset: Dataset,
    trainer: Trainer,
) -> None:
    # Compute predictions and metrics
    predictions, labels, _ = trainer.predict(train_dataset["test"])
    pred_labels = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels, average="macro")
    recall = recall_score(labels, pred_labels, average="macro")
    conf_matrix = confusion_matrix(labels, pred_labels)

    # Create metrics DataFrame with run number
    metrics = pd.DataFrame(
        {
            "Run": [run_num] * 3,
            "Metric": ["Accuracy", "Precision", "Recall"],
            "Value": [accuracy, precision, recall],
        },
    )

    # Save metrics - append if file exists
    vivit_metrics_dir = Path(config.VIVIT_LOCAL_METRICS_DIR)
    vivit_metrics_dir.mkdir(exist_ok=True)
    metrics_file = vivit_metrics_dir / Path("vivit_validation_metrics_all_runs.csv")
    if metrics_file.exists():
        metrics.to_csv(metrics_file, mode="a", header=False, index=False)
    else:
        metrics.to_csv(metrics_file, index=False)

    # Save confusion matrix for this run
    conf_matrix_df = pd.DataFrame(conf_matrix)
    confusion_matrix_file: Path = vivit_metrics_dir / Path(
        f"confusion_matrix_run_{run_num}.csv",
    )
    conf_matrix_df.to_csv(confusion_matrix_file, index=False)

    # Log to wandb
    wandb.log(
        {
            "validation_metrics": wandb.Table(dataframe=metrics),
            "confusion_matrix": wandb.Table(dataframe=conf_matrix_df),
        },
    )


def train_and_evaluate_model(
    run_num: int,
    train_dataset: dict[str, Dataset],
    trainer: Trainer,
) -> None:
    train_results = trainer.train()
    checkpoint_dir = Path(config.VIVIT_CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_name: Path = checkpoint_dir / Path(f"vivit_model_run_{run_num}")
    trainer.save_model(checkpoint_name)
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # Evaluate the model
    eval_results = trainer.evaluate()
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)

    # Get predictions on the validation set
    save_metrics_and_confusion_matrix_locally(
        run_num,
        train_dataset,
        trainer,
    )


def run_experiment() -> None:
    wandb_key = WANDB_KEY
    wandb.login(key=wandb_key)

    for run_num in range(config.REPLICATES):
        seed = config.SEED + run_num
        torch.manual_seed(seed)
        np.random.default_rng(seed)

        train_dataset = dh.create_vivit_dataset(
            directory=config.DATASET_DIR,
            test_size=config.TEST_SIZE,
            seed=seed,
            save_dataset=config.VIVIT_SHALL_SAVE_DATASET,
            dataset_name=config.VIVIT_SAVE_DATASET_DIR,
        )

        device = config.DEVICE
        vivit = initialize_vivit(train_dataset, device, config.VIVIT_MODEL_NAME)

        training_args = create_training_arguments(run_num, seed)

        optimizer = create_optimizer(vivit)

        trainer = create_trainer(train_dataset, vivit, training_args, optimizer)

        with wandb.init(
            project=config.PROJECT,
            job_type="train",
            tags=[config.RUN_NAME],
            name=f"{config.RUN_NAME}_run_{run_num}",
            reinit=True,
        ):
            # Train the model
            train_and_evaluate_model(run_num, train_dataset, trainer)


if __name__ == "__main__":
    run_experiment()
