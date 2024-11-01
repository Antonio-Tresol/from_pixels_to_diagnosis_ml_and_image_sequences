import torch
import data_handling as dh
from vivit import initialize_vivit, compute_metrics
from transformers import Trainer, TrainingArguments
import wandb
import config
from key import WANDB_KEY
import random


def main() -> None:
    device = config.DEVICE
    train_dataset = dh.create_vivit_dataset(
        directory=config.DATASET_DIR,
        test_size=config.TEST_SIZE,
        seed=config.SEED,
        save_dataset=False,
        dataset_name="saved_dataset",
    )
    vivit = initialize_vivit(train_dataset, device, config.MODEL_NAME)

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
        vivit.parameters(),
        lr=config.LEARNING_RATE,
        betas=config.BETAS,
        eps=config.EPSILON,
    )

    trainer = Trainer(
        model=vivit,
        args=training_args,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["test"],
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,
    )
    with wandb.init(
        project=config.PROJECT,
        job_type="train",
        tags=[config.RUN_NAME],
        id=config.RUN_NAME,
    ):
        train_results = trainer.train()
        trainer.save_model("model")
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()
        eval_results = trainer.evaluate()
        trainer.log_metrics("eval", eval_results)
        trainer.save_metrics("eval", eval_results)


if __name__ == "__main__":
    main()
