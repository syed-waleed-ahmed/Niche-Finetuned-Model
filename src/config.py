import os

# Base model (small open-source chat model)
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "fastapi_qa_train.jsonl")
EVAL_PATH = os.path.join(DATA_DIR, "fastapi_qa_eval.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "fastapi_tinyllama_lora")

# Training hyperparameters (you can tweak)
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]