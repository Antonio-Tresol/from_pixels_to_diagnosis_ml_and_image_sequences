from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
)
import torch
import wandb
import numpy as np

from datasets import Dataset, concatenate_datasets
from convnext import initialize_convnext
from transformers import ConvNextForImageClassification
from logging_and_model_evaluation import convnext_compute_metrics
from transformers import TrainingArguments, Trainer
from key import WANDB_KEY
import convnext_data_handling as cdh
import config


def create_optimizer(convnext: ConvNextForImageClassification) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        convnext.parameters(),
        lr=config.LEARNING_RATE,
        betas=config.BETAS,
        eps=config.EPSILON,
    )


def create_trainer(
    dataset: dict,
    convnext: ConvNextForImageClassification,
    training_args: TrainingArguments,
    optimizer: torch.optim.AdamW,
) -> Trainer:
    return Trainer(
        model=convnext,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        optimizers=(optimizer, None),
        compute_metrics=convnext_compute_metrics,
    )


def create_training_arguments(
    run_num: int,
    seed: int,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=f"{config.TRAINING_DIR}_convnext_run_{run_num}",
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH,
        per_device_eval_batch_size=config.EVAL_BATCH,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir=f"{config.LOGGING_DIR}_convnext_run_{run_num}",
        logging_steps=config.LOGGING_STEPS,
        seed=seed,
        eval_strategy=config.EVAL_STRATEGY,
        eval_steps=config.EVAL_STEPS,
        warmup_steps=config.WARMUP_STEPS,
        optim=config.OPTIMIZATION_ALGORITHM,
        lr_scheduler_type=config.SCHEDULER,
        fp16=config.SMALL_FLOATING_POINT,
        report_to=config.LOGGER,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        run_name=f"{config.RUN_NAME_CONVNEXT}_run_{run_num}",
        save_strategy=config.EVAL_STRATEGY,
    )


def post_evaluate_and_save_metrics(
    run_num: int,
    convnext: ConvNextForImageClassification,
    split: dict,
    complete_dataset: Dataset,
) -> None:
    """Evaluate ConvNext model at patient level using existing dataset"""
    test_patients, test_labels = split["test"]
    device = convnext.device
    convnext.eval()

    patient_predictions = []
    for patient_dir in test_patients:
        patient_images = complete_dataset.filter(
            lambda x, patient_dir=patient_dir: x["id"] == patient_dir,
        )

        patient_image_predictions = []
        with torch.inference_mode():
            for image_data in patient_images:
                inputs = {
                    "pixel_values": torch.tensor(image_data["pixel_values"])
                    .unsqueeze(0)
                    .to(device),
                }
                outputs = convnext(**inputs)
                prediction = outputs.logits.cpu().argmax(dim=1).item()
                patient_image_predictions.append(prediction)

        patient_prediction = 1 if 1 in patient_image_predictions else 0
        patient_predictions.append(patient_prediction)

    test_labels = np.array(test_labels)
    patient_predictions = np.array(patient_predictions)
    accuracy = accuracy_score(test_labels, patient_predictions)
    precision = precision_score(test_labels, patient_predictions)
    recall = recall_score(test_labels, patient_predictions)
    conf_matrix = confusion_matrix(test_labels, patient_predictions)

    metrics = pd.DataFrame(
        {
            "Run": [run_num] * 3,
            "Metric": ["Accuracy", "Precision", "Recall"],
            "Value": [accuracy, precision, recall],
        },
    )

    metrics_dir = Path(config.CONVNEXT_LOCAL_METRICS_DIR)
    metrics_dir.mkdir(exist_ok=True)

    metrics_file = metrics_dir / "convnext_validation_metrics_all_runs.csv"
    if metrics_file.exists():
        metrics.to_csv(metrics_file, mode="a", header=False, index=False)
    else:
        metrics.to_csv(metrics_file, index=False)

    conf_matrix_df = pd.DataFrame(conf_matrix)
    confusion_matrix_file: Path = metrics_dir / Path(
        f"confusion_matrix_run_{run_num}.csv",
    )
    conf_matrix_df.to_csv(confusion_matrix_file, index=False)

    wandb.log(
        {
            "validation_metrics": wandb.Table(dataframe=metrics),
            "confusion_matrix": wandb.Table(dataframe=conf_matrix_df),
        },
    )


def train_and_evaluate_model(trainer: Trainer, run_num: int, split: dict) -> None:
    train_results = trainer.train()

    checkpoint_dir = Path(config.CONVNEXT_CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_name: Path = checkpoint_dir / Path(f"convnext_model_run_{run_num}")
    trainer.save_model(checkpoint_name)

    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    # evaluate the model on individual images
    eval_results = trainer.evaluate()
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)
    complete_dataset = concatenate_datasets(
        [
            trainer.train_dataset,
            trainer.eval_dataset,
        ],
    )
    # evaluate the model on patient level
    post_evaluate_and_save_metrics(
        run_num,
        trainer.model,
        split,
        complete_dataset,
    )


def run_convnext_experiment() -> None:
    device = config.DEVICE
    wandb.login(key=WANDB_KEY)
    for run_num in range(config.REPLICATES):
        seed = config.SEED + run_num
        dataset, split = cdh.create_convnext_dataset(
            directory=config.DATASET_DIR,
            test_size=config.TEST_SIZE,
            seed=seed,
            dataset_name=config.CONVNEXT_SAVE_DATASET_DIR,
        )

        convnext = initialize_convnext(dataset, device, config.CONVNEXT_MODEL_NAME)

        training_args = create_training_arguments(
            run_num=run_num,
            seed=seed,
        )

        optimizer = create_optimizer(convnext)

        trainer = create_trainer(dataset, convnext, training_args, optimizer)

        with wandb.init(
            project=config.PROJECT,
            job_type="train",
            tags=[config.RUN_NAME_CONVNEXT],
            name=f"{config.RUN_NAME_CONVNEXT}_run_{run_num}",
            reinit=True,
        ):
            train_and_evaluate_model(trainer, run_num, split)


if __name__ == "__main__":
    run_convnext_experiment()
