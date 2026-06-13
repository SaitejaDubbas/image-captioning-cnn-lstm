"""Greedy and beam-search inference for image captioning."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MAX_CAPTION_LENGTH, TOKENIZER_PATH
from src.data_loader import Tokenizer


def generate_caption_from_features(
    image_features: np.ndarray,
    model,
    tokenizer: Tokenizer,
    max_len: int = MAX_CAPTION_LENGTH,
    strategy: str = "greedy",
    beam_width: int = 3,
) -> str:
    """Generate a caption given pre-extracted image features."""
    if strategy == "beam":
        return _beam_search(image_features, model, tokenizer, max_len, beam_width)
    return _greedy(image_features, model, tokenizer, max_len)


def _greedy(image_features: np.ndarray, model, tokenizer: Tokenizer,
            max_len: int) -> str:
    feat = tf.expand_dims(image_features, 0)          # (1, 64, 2048)
    enc_feat = model.encoder(feat, training=False)    # (1, 64, emb)

    hidden, cell = model.decoder.reset_state(1)
    start_idx = tokenizer.word2idx.get("<start>", 1)
    end_idx = tokenizer.word2idx.get("<end>", 2)

    token = tf.constant([[start_idx]])
    result = []

    for _ in range(max_len):
        logits, hidden, cell, _ = model.decoder(token, enc_feat, hidden, cell, training=False)
        pred_idx = int(tf.argmax(logits, axis=-1).numpy()[0])
        if pred_idx == end_idx:
            break
        word = tokenizer.idx2word.get(pred_idx, "")
        if word and word not in {"<pad>", "<start>", "<end>"}:
            result.append(word)
        token = tf.constant([[pred_idx]])

    return " ".join(result)


def _beam_search(image_features: np.ndarray, model, tokenizer: Tokenizer,
                 max_len: int, beam_width: int) -> str:
    feat = tf.expand_dims(image_features, 0)
    enc_feat = model.encoder(feat, training=False)
    hidden, cell = model.decoder.reset_state(1)

    start_idx = tokenizer.word2idx.get("<start>", 1)
    end_idx = tokenizer.word2idx.get("<end>", 2)

    # Each beam: (log_prob, token_ids, hidden, cell)
    beams = [(0.0, [start_idx], hidden, cell)]
    completed = []

    for _ in range(max_len):
        candidates = []
        for log_prob, seq, h, c in beams:
            token = tf.constant([[seq[-1]]])
            logits, new_h, new_c, _ = model.decoder(token, enc_feat, h, c, training=False)
            log_probs = tf.nn.log_softmax(logits[0]).numpy()
            top_k = np.argsort(log_probs)[-beam_width:]
            for idx in top_k:
                candidates.append((log_prob + log_probs[idx], seq + [int(idx)], new_h, new_c))

        # Keep top beam_width beams
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = []
        for cand in candidates[:beam_width]:
            if cand[1][-1] == end_idx:
                completed.append(cand)
            else:
                beams.append(cand)
        if not beams:
            break

    if not completed:
        completed = beams
    best = max(completed, key=lambda x: x[0] / len(x[1]))
    words = [tokenizer.idx2word.get(i, "") for i in best[1][1:]
             if i != end_idx and tokenizer.idx2word.get(i, "") not in {"<pad>", "<start>", "<end>", ""}]
    return " ".join(words)


def caption_image_file(image_path: str, model, tokenizer: Tokenizer,
                       strategy: str = "greedy") -> Tuple[str, np.ndarray]:
    """Load an image file, extract features, and return a caption."""
    from src.feature_extractor import build_encoder, load_image
    encoder = build_encoder(fine_tune=False)
    img = tf.expand_dims(load_image(image_path), 0)
    features = encoder(img, training=False)[0].numpy()  # (64, 2048)
    caption = generate_caption_from_features(features, model, tokenizer, strategy=strategy)
    return caption, features
