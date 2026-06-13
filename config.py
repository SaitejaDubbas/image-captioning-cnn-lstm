import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Dataset
KAGGLE_DATASET = "adityajn105/flickr8k"
IMAGE_DIR = RAW_DIR / "Images"
CAPTIONS_FILE = RAW_DIR / "captions.txt"

# Preprocessing
IMAGE_SIZE = (299, 299)  # InceptionV3 input
MAX_CAPTION_LENGTH = 40
VOCAB_SIZE = 8000
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1

# Model
EMBEDDING_DIM = 256
UNITS = 512         # LSTM + attention units
FEATURES_SHAPE = 2048
ATTENTION_FEATURES_SHAPE = 64

# Training
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-4
FINE_TUNE_AT = 249   # InceptionV3 layer to start fine-tuning from

# Paths for saved artifacts
TOKENIZER_PATH = PROCESSED_DIR / "tokenizer.pkl"
FEATURES_PATH = PROCESSED_DIR / "features.pkl"
CHECKPOINT_PATH = MODELS_DIR / "checkpoints"
FINAL_MODEL_PATH = MODELS_DIR / "caption_model.keras"

# Flask API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 5000))
MAX_IMAGE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
