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
