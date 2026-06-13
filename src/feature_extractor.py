"""InceptionV3-based image feature extraction with optional fine-tuning."""

import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import IMAGE_SIZE, FEATURES_PATH, IMAGE_DIR, FINE_TUNE_AT


def build_encoder(fine_tune: bool = False, fine_tune_at: int = FINE_TUNE_AT) -> Model:
    """Build InceptionV3 encoder.

    Returns spatial features (batch, 8*8, 2048) for attention, or
    pooled features (batch, 2048) for standard LSTM.
    """
    base = InceptionV3(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3))

    if fine_tune:
        # Freeze all layers up to fine_tune_at
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base.layers[fine_tune_at:]:
            layer.trainable = True
    else:
        base.trainable = False

    # Spatial features for attention: (batch, 64, 2048)
    x = base.output
    x = tf.keras.layers.Reshape((-1, x.shape[-1]))(x)  # (batch, H*W, C)
    model = Model(inputs=base.input, outputs=x, name="inception_encoder")
    return model


def load_image(image_path: str) -> tf.Tensor:
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = preprocess_input(img)
    return img


def extract_features(image_dir: Path = IMAGE_DIR,
                     output_path: Path = FEATURES_PATH,
                     batch_size: int = 32) -> dict:
    """Extract and cache InceptionV3 features for all images."""
    encoder = build_encoder(fine_tune=False)
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    print(f"Extracting features from {len(image_paths)} images...")
    features = {}

    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i: i + batch_size]
        imgs = tf.stack([load_image(str(p)) for p in batch_paths])
        feats = encoder(imgs, training=False)  # (B, 64, 2048)
        for path, feat in zip(batch_paths, feats):
            features[path.name] = feat.numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(features, f)

    print(f"Features saved to {output_path}")
    return features


def load_features(path: Path = FEATURES_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def preprocess_single_image(image_path: str) -> np.ndarray:
    """Load and preprocess one image for inference."""
    img = load_image(image_path)
    return tf.expand_dims(img, 0).numpy()
