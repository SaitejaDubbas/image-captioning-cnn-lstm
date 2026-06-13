"""Flask REST API for image captioning."""

import io
import os
import time
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ALLOWED_EXTENSIONS, MAX_IMAGE_SIZE_MB, TOKENIZER_PATH,
    IMAGE_SIZE, FINAL_MODEL_PATH
)
from src.data_loader import Tokenizer
from src.model import build_model
from src.feature_extractor import build_encoder
from src.predict import generate_caption_from_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE_MB * 1024 * 1024

# ── Lazy-loaded globals ───────────────────────────────────────────────────────
_tokenizer = None
_model = None
_encoder = None


def get_tokenizer() -> Tokenizer:
    global _tokenizer
    if _tokenizer is None:
        logger.info("Loading tokenizer...")
        _tokenizer = Tokenizer.load(TOKENIZER_PATH)
    return _tokenizer


def get_model():
    global _model
    if _model is None:
        logger.info("Loading captioning model...")
        tok = get_tokenizer()
        _model = build_model(vocab_size=len(tok.word2idx))
        # Warm-up forward pass to build the model before loading weights
        dummy_feat = tf.zeros((1, 64, 2048))
        dummy_cap = tf.zeros((1, 40), dtype=tf.int32)
        _ = _model((dummy_feat, dummy_cap))
        weights_path = str(FINAL_MODEL_PATH).replace(".keras", "_best.weights.h5")
        if Path(weights_path).exists():
            _model.load_weights(weights_path)
            logger.info(f"Loaded weights from {weights_path}")
        else:
            logger.warning("No trained weights found — model will produce random output.")
    return _model


def get_encoder():
    global _encoder
    if _encoder is None:
        logger.info("Loading InceptionV3 encoder...")
        _encoder = build_encoder(fine_tune=False)
    return _encoder


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_pil(pil_image: Image.Image) -> np.ndarray:
    """Resize + normalize a PIL image for InceptionV3."""
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    img = pil_image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, 0)


def extract_features_from_array(img_array: np.ndarray) -> np.ndarray:
    encoder = get_encoder()
    tensor = tf.constant(img_array)
    features = encoder(tensor, training=False)  # (1, 64, 2048)
    return features[0].numpy()                  # (64, 2048)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "timestamp": time.time()})


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts multipart/form-data with field 'image'.
    Optional query param: strategy=greedy|beam, beam_width=3
    Returns: {"caption": "...", "latency_ms": ...}
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send as multipart 'image' field."}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    strategy = request.args.get("strategy", "greedy")
    beam_width = int(request.args.get("beam_width", 3))

    try:
        t0 = time.time()
        pil_img = Image.open(io.BytesIO(file.read()))
        img_array = preprocess_pil(pil_img)
        features = extract_features_from_array(img_array)

        model = get_model()
        tokenizer = get_tokenizer()
        caption = generate_caption_from_features(
            features, model, tokenizer,
            strategy=strategy, beam_width=beam_width
        )
        latency = round((time.time() - t0) * 1000, 2)

        return jsonify({
            "caption": caption,
            "strategy": strategy,
            "latency_ms": latency,
        })
    except Exception as exc:
        logger.exception("Prediction error")
        return jsonify({"error": str(exc)}), 500


@app.route("/predict/url", methods=["POST"])
def predict_url():
    """
    POST /predict/url
    Body JSON: {"url": "https://...", "strategy": "greedy"}
    """
    data = request.get_json(force=True)
    if not data or "url" not in data:
        return jsonify({"error": "Provide JSON body with 'url' key."}), 400

    import urllib.request
    url = data["url"]
    strategy = data.get("strategy", "greedy")

    try:
        t0 = time.time()
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            raw = resp.read()
        pil_img = Image.open(io.BytesIO(raw))
        img_array = preprocess_pil(pil_img)
        features = extract_features_from_array(img_array)

        model = get_model()
        tokenizer = get_tokenizer()
        caption = generate_caption_from_features(features, model, tokenizer, strategy=strategy)
        latency = round((time.time() - t0) * 1000, 2)

        return jsonify({"caption": caption, "latency_ms": latency})
    except Exception as exc:
        logger.exception("URL prediction error")
        return jsonify({"error": str(exc)}), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    POST /batch_predict
    Accepts up to 10 images as multipart fields: image_0, image_1, ...
    """
    results = []
    model = get_model()
    tokenizer = get_tokenizer()
    strategy = request.args.get("strategy", "greedy")

    for i in range(10):
        key = f"image_{i}"
        if key not in request.files:
            break
        file = request.files[key]
        if not allowed_file(file.filename):
            results.append({"index": i, "error": "Invalid file type"})
            continue
        try:
            pil_img = Image.open(io.BytesIO(file.read()))
            img_array = preprocess_pil(pil_img)
            features = extract_features_from_array(img_array)
            caption = generate_caption_from_features(features, model, tokenizer, strategy=strategy)
            results.append({"index": i, "filename": file.filename, "caption": caption})
        except Exception as exc:
            results.append({"index": i, "error": str(exc)})

    return jsonify({"results": results, "count": len(results)})


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Image Captioning API",
        "version": "1.0.0",
        "endpoints": {
            "GET  /health": "Health check",
            "POST /predict": "Caption an uploaded image",
            "POST /predict/url": "Caption an image from URL",
            "POST /batch_predict": "Caption multiple images",
        }
    })


if __name__ == "__main__":
    from config import API_HOST, API_PORT
    app.run(host=API_HOST, port=API_PORT, debug=False)
