import torch
import random

# Project constants
PROJECT = "cs_research"
MODEL_NAME = "google/vivit-b-16x2-kinetics400"
RUN_NAME = f"vivit_dummy_run_{random.randint(0, 9999)}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# General constants
DATASET_DIR = "dataset_mini"
TEST_SIZE = 0.3
TRAIN_SIZE = 1 - TEST_SIZE
SEED = 42
EPOCHS = 20
TRAIN_BATCH = 2
EVAL_BATCH = 2
WEIGHT_DECAY = 0.01
TRAINING_DIR = "/tmp/results"

NUM_FRAMES = 18
# Optimizarion algorithm constants
OPTIMIZATION_ALGORITHM = "adamw_torch"
LEARNING_RATE = 0.000005
BETAS = (0.9, 0.999)
EPSILON = 1e-08

# Logging constants
LOGGER = "wandb"
LOGGING_DIR = "./logs"
LOGGING_STEPS = 10

# Other constants
EVAL_STRATEGY = "steps"
EVAL_STEPS = 10
WARMUP_STEPS = int(0.1 * 20)
SCHEDULER = "linear"
SMALL_FLOATING_POINT = True


LR = 0.0001
SCHEDULER_MAX_IT = 30
WEIGH_DECAY = 1e-4
EPSILON = 1e-4

# train loop
BATCH_SIZE = 2
USE_INDEX = False
# callback
PATIENCE = 15
TOP_K_SAVES = 1
# training loop
NUM_TRIALS = 1

INDICES_DIR = "indices/"
CHECKPOINTS_DIR = "checkpoints/"
METRICS_DIR = "metrics/"
WANDB_PROJECT = PROJECT

VIVIT_DIR = CHECKPOINTS_DIR + "vivit/"
VIVIT_FILENAME = "vivit_"

CLASS_NAMES = ["Negative", "Positive"]
