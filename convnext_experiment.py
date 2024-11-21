import torch
import wandb
import json

from datasets import Dataset, DatasetDict, ClassLabel, Features, Sequence, Value
from convnext import initialize_convnext
from log_and_eval import convnext_compute_patient_metrics
from transformers import TrainingArguments, Trainer
from key import WANDB_KEY
import data_handling as dh
import config


def main() -> None:
    device = config.DEVICE
    dataset = dh.create_convnext_dataset(
        directory=config.DATASET_DIR,
        test_size=config.TEST_SIZE,
        seed=config.SEED,
        save_dataset=config.CONVNEXT_SHALL_SAVE_DATASET,
        dataset_name=config.CONVNEXT_SAVE_DATASET_DIR,
    )

    # train using the individual images for each patient
    training_dataset = create_training_datasetdict(dataset)

    dataset["train"] = training_dataset

    convnext = initialize_convnext(dataset, device, config.CONVNEXT_MODEL_NAME)


    training_args = TrainingArguments(
        output_dir=config.TRAINING_DIR,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.TRAIN_BATCH,
        per_device_eval_batch_size=config.EVAL_BATCH,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir=config.LOGGING_DIR,
        logging_steps=config.LOGGING_STEPS,
        seed=config.SEED,
        eval_strategy=config.EVAL_STRATEGY,
        eval_steps=config.EVAL_STEPS,
        warmup_steps=config.WARMUP_STEPS,
        optim=config.OPTIMIZATION_ALGORITHM,
        lr_scheduler_type=config.SCHEDULER,
        fp16=config.SMALL_FLOATING_POINT,
        report_to=config.LOGGER,
        run_name=config.RUN_NAME,
    )

    wandb_key = WANDB_KEY
    wandb.login(key=wandb_key)

    optimizer = torch.optim.AdamW(
        convnext.parameters(),
        lr=config.LEARNING_RATE,
        betas=config.BETAS,
        eps=config.EPSILON,
    )

    trainer = Trainer(
        model=convnext,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=dataset["test"],
        optimizers=(optimizer, None),
        compute_metrics=convnext_compute_patient_metrics,
    )
    with wandb.init(
        project=config.PROJECT,
        job_type="train",
        tags=[config.RUN_NAME_CONVNEXT],
        id=config.RUN_NAME_CONVNEXT,
    ):
        train_results = trainer.train()
        trainer.save_model("model")
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()
        eval_results = trainer.evaluate()
        trainer.log_metrics("eval", eval_results)
        trainer.save_metrics("eval", eval_results)


def create_training_datasetdict(dataset: DatasetDict, label_multiplier: int = 18) -> Dataset:
    class_names = dataset['train'].features['labels'].names
    # Create new lists for labels and pixel_values
    new_labels = [label for label in dataset['train']['labels'] for _ in range(label_multiplier)]
    new_pixel_values = [image for patient in dataset['train']['pixel_values'] for image in patient]

    # Create a new dataset with the modified labels and pixel_values
    new_dataset = {
        'labels': new_labels,
        'pixel_values': new_pixel_values
    }
    
    new_features = {
        'labels': ClassLabel(names=['Negative', 'Positive'], id=None),
        'pixel_values': Sequence(feature=Sequence(feature=Sequence(feature=Value(dtype='float32', id=None))))
    }

    new_features = Features(new_features)


    new_dataset = Dataset.from_dict(new_dataset, features=new_features)

    print(len(new_dataset['pixel_values']))

    return new_dataset

    


if __name__ == "__main__":
    main()
