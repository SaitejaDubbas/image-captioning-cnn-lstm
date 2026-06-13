"""Flickr8k dataset loading and caption preprocessing."""

import os
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CAPTIONS_FILE, IMAGE_DIR, MAX_CAPTION_LENGTH, VOCAB_SIZE,
    TOKENIZER_PATH, TEST_SPLIT, VAL_SPLIT, PROCESSED_DIR
)


def download_dataset():
    """Download Flickr8k from Kaggle."""
    try:
        import kaggle
    except ImportError:
        raise ImportError("Install kaggle: pip install kaggle")

    print("Downloading Flickr8k from Kaggle...")
    import subprocess
    subprocess.run([
        "kaggle", "datasets", "download", "-d", "adityajn105/flickr8k",
        "-p", str(RAW_DIR := Path(CAPTIONS_FILE).parent),
        "--unzip"
    ], check=True)
    print("Dataset downloaded.")


def clean_caption(caption: str) -> str:
    caption = caption.lower().strip()
    caption = re.sub(r"[^a-z\s]", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return f"<start> {caption} <end>"


def load_captions(captions_file: Path = CAPTIONS_FILE) -> dict:
    """Load and clean captions. Returns {image_name: [caption, ...]}."""
    df = pd.read_csv(captions_file)
    # Flickr8k captions.txt has columns: image, caption
    if "image" not in df.columns:
        df.columns = ["image", "caption"]

    captions = {}
    for _, row in df.iterrows():
        img = row["image"].split("#")[0] if "#" in str(row["image"]) else row["image"]
        cap = clean_caption(str(row["caption"]))
        captions.setdefault(img, []).append(cap)
    return captions


def build_vocabulary(captions: dict, vocab_size: int = VOCAB_SIZE):
    """Build word-to-index mapping from captions."""
    counter = Counter()
    for caps in captions.values():
        for cap in caps:
            counter.update(cap.split())

    # Reserve 0 for padding, keep most frequent words
    vocab = ["<pad>"] + [w for w, _ in counter.most_common(vocab_size - 1)]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word, vocab


class Tokenizer:
    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.word2idx: dict = {}
        self.idx2word: dict = {}

    def fit(self, captions: dict):
        self.word2idx, self.idx2word, self.vocab = build_vocabulary(captions, self.vocab_size)
        return self

    def encode(self, caption: str) -> list:
        return [self.word2idx.get(w, self.word2idx.get("<pad>", 0)) for w in caption.split()]

    def decode(self, indices: list) -> str:
        words = [self.idx2word.get(i, "") for i in indices]
        words = [w for w in words if w not in {"<pad>", "<start>", "<end>", ""}]
        return " ".join(words)

    def save(self, path: Path = TOKENIZER_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path = TOKENIZER_PATH) -> "Tokenizer":
        with open(path, "rb") as f:
            return pickle.load(f)


def split_dataset(captions: dict, test_split=TEST_SPLIT, val_split=VAL_SPLIT):
    """Split image names into train/val/test sets."""
    image_names = list(captions.keys())
    train_val, test = train_test_split(image_names, test_size=test_split, random_state=42)
    val_size = val_split / (1 - test_split)
    train, val = train_test_split(train_val, test_size=val_size, random_state=42)
    return train, val, test


def build_dataset(image_names: list, captions: dict, tokenizer: Tokenizer,
                  features: dict, max_len: int = MAX_CAPTION_LENGTH):
    """Build (image_feature, input_seq, target_seq) for training."""
    img_features, input_seqs, target_seqs = [], [], []

    for img in image_names:
        if img not in features:
            continue
        feat = features[img]
        for cap in captions[img]:
            seq = tokenizer.encode(cap)
            seq = seq[:max_len]
            for i in range(1, len(seq)):
                in_seq = seq[:i]
                out_seq = seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_len, padding="post")[0]
                img_features.append(feat)
                input_seqs.append(in_seq)
                target_seqs.append(out_seq)

    return (
        np.array(img_features, dtype=np.float32),
        np.array(input_seqs, dtype=np.int32),
        np.array(target_seqs, dtype=np.int32),
    )


def save_processed(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_processed(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)
