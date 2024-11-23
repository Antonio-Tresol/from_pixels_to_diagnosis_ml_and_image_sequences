import pandas as pd
import torch
import random

# Project constants
PROJECT = "cs_research"
VIVIT_MODEL_NAME = "google/vivit-b-16x2-kinetics400"
CONVNEXT_MODEL_NAME = "facebook/convnext-tiny-224"
RUN_NAME_CONVNEXT = f"convnext_{random.randint(0, 9999)}"
VIVIT_RUN_NAME = f"vivit_{random.randint(0, 9999)}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# General constants
DATASET_DIR = "dataset_mini"
DATASET_INFO = pd.read_csv("dataset/hemorrhage_diagnosis_per_slice.csv")

VIVIT_SAVE_DATASET_DIR = f"vivit_saved_{DATASET_DIR}"
VIVIT_LOCAL_METRICS_DIR = "vivit_metrics"
VIVIT_CHECKPOINT_DIR = "vivit_model_checkpoints"
VIVIT_METRICS = "vivit_new_val_metrics_all_runs.csv"
VIVIT_CM = "vivit_cm_run_"

CONVNEXT_SAVE_DATASET_DIR = f"convnext_saved_{DATASET_DIR}"
CONVNEXT_LOCAL_METRICS_DIR = "convnext_metrics"
CONVNEXT_CHECKPOINT_DIR = "convnext_model_checkpoints"
CONVNEXT_METRICS = "convnext_validation_metrics_all_runs.csv"
CONVNEXT_CM = "convnext_cm_run_"

TEST_SIZE = 0.3
SEED = 42
EPOCHS = 5
TRAIN_BATCH = 10
EVAL_BATCH = 10
WEIGHT_DECAY = 0.01
TRAINING_DIR = "/tmp/results"

# Optimizarion algorithm constants
OPTIMIZATION_ALGORITHM = "adamw_torch"
LEARNING_RATE = 0.000005
BETAS = (0.9, 0.999)
EPSILON = 1e-08
EARLY_STOPPING = 5

# Logging constants
LOGGER = "wandb"
LOGGING_DIR = "./logs"
LOGGING_STEPS = 10

# Other constants
EVAL_STRATEGY = "epoch"
EVAL_STEPS = 10
WARMUP_STEPS = int(0.1 * 20)
SCHEDULER = "linear"
SMALL_FLOATING_POINT = True
REPLICATES = 2
