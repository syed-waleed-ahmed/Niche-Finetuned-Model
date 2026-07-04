import os


def _env_int(name: str, default: int) -> int:
	value = os.getenv(name)
	if value is None:
		return default
	try:
		return int(value)
	except ValueError as exc:
		raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_float(name: str, default: float) -> float:
	value = os.getenv(name)
	if value is None:
		return default
	try:
		return float(value)
	except ValueError as exc:
		raise ValueError(f"{name} must be a number, got {value!r}") from exc

# Base model (small open-source chat model)
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "fastapi_qa_train.jsonl")
EVAL_PATH = os.path.join(DATA_DIR, "fastapi_qa_eval.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "fastapi_tinyllama_lora")

# Service settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = _env_int("API_PORT", 8000)
MODEL_MAX_NEW_TOKENS = _env_int("MODEL_MAX_NEW_TOKENS", 256)
MODEL_TEMPERATURE = _env_float("MODEL_TEMPERATURE", 0.4)
MODEL_TOP_P = _env_float("MODEL_TOP_P", 0.9)
MODEL_REPETITION_PENALTY = _env_float("MODEL_REPETITION_PENALTY", 1.05)

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