"""Training script for the image captioning model."""

import pickle
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, FINE_TUNE_LEARNING_RATE,
    CHECKPOINT_PATH, FINAL_MODEL_PATH, FEATURES_PATH, TOKENIZER_PATH,
    VOCAB_SIZE, EMBEDDING_DIM, UNITS, PROCESSED_DIR
)
from src.data_loader import (
    load_captions, Tokenizer, split_dataset, build_dataset, save_processed
)
from src.feature_extractor import extract_features, load_features
from src.model import build_model


def masked_sparse_categorical_crossentropy(real, pred):
    """Cross-entropy loss ignoring padding (index 0)."""
    loss = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)
    mask = tf.cast(real != 0, dtype=loss.dtype)
    return tf.reduce_mean(loss * mask)


def train(epochs: int = EPOCHS, resume: bool = False):
    # ── Data preparation ──────────────────────────────────────────────────
    print("Loading captions...")
    captions = load_captions()

    print("Building tokenizer...")
    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.fit(captions)
    tokenizer.save()

    print("Loading/extracting image features...")
    if FEATURES_PATH.exists():
        features = load_features()
    else:
        features = extract_features()

    train_imgs, val_imgs, _ = split_dataset(captions)

    print("Building dataset tensors...")
    X_img, X_cap, y = build_dataset(train_imgs, captions, tokenizer, features)
    Xv_img, Xv_cap, yv = build_dataset(val_imgs, captions, tokenizer, features)

    train_ds = (
        tf.data.Dataset.from_tensor_slices(((X_img, X_cap), y))
        .shuffle(10000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices(((Xv_img, Xv_cap), yv))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ── Model & optimizer ─────────────────────────────────────────────────
    model = build_model(vocab_size=len(tokenizer.word2idx))
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = masked_sparse_categorical_crossentropy

    checkpoint_path = Path(CHECKPOINT_PATH)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, str(checkpoint_path), max_to_keep=3)

    if resume and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(f"Restored checkpoint: {ckpt_manager.latest_checkpoint}")

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = 0.0
        steps = 0

        for (img_feats, cap_in), cap_target in train_ds:
            # Manual teacher-forcing step
            with tf.GradientTape() as tape:
                enc_features = model.encoder(img_feats, training=True)
                batch_sz = tf.shape(img_feats)[0]
                hidden, cell = model.decoder.reset_state(batch_sz)
                seq_loss = 0.0
                seq_len = tf.shape(cap_in)[1]

                for t in range(seq_len - 1):
                    token = tf.expand_dims(cap_in[:, t], 1)
                    logits, hidden, cell, _ = model.decoder(
                        token, enc_features, hidden, cell, training=True
                    )
                    seq_loss += loss_fn(cap_target[:, t + 1] if cap_target.shape[1] > 1
                                        else cap_target, logits)

            variables = (model.encoder.trainable_variables +
                         model.decoder.trainable_variables)
            grads = tape.gradient(seq_loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
            train_loss += seq_loss / tf.cast(seq_len - 1, tf.float32)
            steps += 1

        train_loss /= steps

        # Validation
        val_loss = 0.0
        val_steps = 0
        for (img_feats, cap_in), cap_target in val_ds:
            enc_features = model.encoder(img_feats, training=False)
            batch_sz = tf.shape(img_feats)[0]
            hidden, cell = model.decoder.reset_state(batch_sz)
            seq_len = tf.shape(cap_in)[1]
            step_loss = 0.0
            for t in range(seq_len - 1):
                token = tf.expand_dims(cap_in[:, t], 1)
                logits, hidden, cell, _ = model.decoder(
                    token, enc_features, hidden, cell, training=False
                )
                step_loss += loss_fn(
                    cap_target[:, t + 1] if cap_target.shape[1] > 1 else cap_target, logits
                )
            val_loss += step_loss / tf.cast(seq_len - 1, tf.float32)
            val_steps += 1
        val_loss /= val_steps

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        elapsed = time.time() - start
        print(f"Epoch {epoch}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"time={elapsed:.1f}s")

        ckpt_manager.save()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(str(FINAL_MODEL_PATH).replace(".keras", "_best.weights.h5"))
            print(f"  → Best model saved (val_loss={best_val_loss:.4f})")

    # Save final model weights
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(FINAL_MODEL_PATH).replace(".keras", ".weights.h5"))
    save_processed(history, PROCESSED_DIR / "training_history.pkl")
    print("Training complete.")
    return model, tokenizer, history


if __name__ == "__main__":
    train()
