"""BLEU score evaluation for the image captioning model."""

from pathlib import Path
from typing import List

import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MAX_CAPTION_LENGTH, TOKENIZER_PATH, FEATURES_PATH, PROCESSED_DIR
from src.data_loader import load_captions, Tokenizer, split_dataset, load_processed
from src.feature_extractor import load_features
from src.predict import generate_caption_from_features


def download_nltk():
    for pkg in ["punkt", "punkt_tab"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


def compute_bleu(references: List[List[List[str]]],
                 hypotheses: List[List[str]]) -> dict:
    """Compute BLEU-1 through BLEU-4."""
    smooth = SmoothingFunction().method1
    return {
        "bleu1": corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smooth),
        "bleu2": corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
        "bleu3": corpus_bleu(references, hypotheses, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth),
        "bleu4": corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth),
    }


def evaluate(model, tokenizer: Tokenizer, split: str = "test",
             max_samples: int = None) -> dict:
    download_nltk()

    captions = load_captions()
    features = load_features()
    train_imgs, val_imgs, test_imgs = split_dataset(captions)
    split_map = {"train": train_imgs, "val": val_imgs, "test": test_imgs}
    image_names = split_map[split]

    if max_samples:
        image_names = image_names[:max_samples]

    references = []
    hypotheses = []

    for img in image_names:
        if img not in features:
            continue
        feat = features[img]

        # Generate prediction
        pred_caption = generate_caption_from_features(feat, model, tokenizer)
        hyp = pred_caption.split()

        # All ground-truth captions as references
        refs = []
        for cap in captions[img]:
            tokens = [w for w in cap.split() if w not in {"<start>", "<end>", "<pad>"}]
            refs.append(tokens)

        references.append(refs)
        hypotheses.append(hyp)

    scores = compute_bleu(references, hypotheses)
    print(f"\n=== BLEU Scores ({split} set, {len(hypotheses)} samples) ===")
    for k, v in scores.items():
        print(f"  {k.upper()}: {v:.4f}")
    return scores


if __name__ == "__main__":
    from src.model import build_model

    tokenizer = Tokenizer.load()
    model = build_model(vocab_size=len(tokenizer.word2idx))

    weights_path = Path(PROCESSED_DIR).parent / "models" / "caption_model_best.weights.h5"
    if weights_path.exists():
        # Warm up the model with a dummy forward pass before loading weights
        import tensorflow as tf
        dummy_feat = tf.zeros((1, 64, 2048))
        dummy_cap = tf.zeros((1, 40), dtype=tf.int32)
        _ = model((dummy_feat, dummy_cap))
        model.load_weights(str(weights_path))

    evaluate(model, tokenizer, split="test")
