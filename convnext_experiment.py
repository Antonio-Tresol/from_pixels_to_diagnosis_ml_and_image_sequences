import torch
from convnext import initialize_convnext
from transformers import TrainingArguments, Trainer
import data_handling as dh
import config

def main() -> None:
    device = config.DEVICE
    dataset = dh.create_convnext_dataset(
        directory=config.DATASET_DIR,
        test_size=config.TEST_SIZE,
        seed=config.SEED,
        save_dataset=False,
        dataset_name="dataset",
    )

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

    optimizer = torch.optim.AdamW(
        convnext.parameters(),
        lr=config.LEARNING_RATE,
        betas=config.BETAS,
        eps=config.EPSILON,
    )

    # trainer = Trainer(
    #     model=convnext,
    #     args=training_args,
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset["test"],
    #     optimizers=(optimizer, None),
    # )

    # trainer.train()



if __name__ == "__main__":
    main()