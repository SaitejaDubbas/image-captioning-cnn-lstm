"""CNN-LSTM Image Captioning model with Bahdanau Attention."""

from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    EMBEDDING_DIM, UNITS, VOCAB_SIZE, ATTENTION_FEATURES_SHAPE,
    FEATURES_SHAPE, MAX_CAPTION_LENGTH
)


class BahdanauAttention(layers.Layer):
    """Additive (Bahdanau) attention over spatial image features."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.W1 = layers.Dense(units)   # applied to features
        self.W2 = layers.Dense(units)   # applied to hidden state
        self.V = layers.Dense(1)

    def call(self, features, hidden):
        # features: (batch, 64, 2048)
        # hidden:   (batch, units)
        hidden_expanded = tf.expand_dims(hidden, 1)          # (batch, 1, units)
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_expanded)))
        # score: (batch, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)     # (batch, 64, 1)
        context = attention_weights * features               # (batch, 64, 2048)
        context = tf.reduce_sum(context, axis=1)             # (batch, 2048)
        return context, attention_weights


class CNNEncoder(layers.Layer):
    """Project InceptionV3 spatial features to embedding dimension."""

    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.fc = layers.Dense(embedding_dim, activation="relu")

    def call(self, x):
        return self.fc(x)   # (batch, 64, embedding_dim)


class RNNDecoder(layers.Layer):
    """LSTM decoder with Bahdanau attention."""

    def __init__(self, embedding_dim: int, units: int, vocab_size: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTMCell(units)
        self.attention = BahdanauAttention(units)
        self.fc1 = layers.Dense(units, activation="relu")
        self.fc2 = layers.Dense(vocab_size)
        self.dropout = layers.Dropout(0.3)

    def call(self, x, features, hidden, cell, training=False):
        """
        x:        (batch, 1)          current token
        features: (batch, 64, emb)    encoded image features
        hidden:   (batch, units)
        cell:     (batch, units)
        """
        context, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)              # (batch, 1, emb)
        x = tf.squeeze(x, axis=1)         # (batch, emb)
        x = tf.concat([x, context], axis=-1)
        output, [new_hidden, new_cell] = self.lstm(x, states=[hidden, cell])
        output = self.dropout(output, training=training)
        output = self.fc1(output)
        logits = self.fc2(output)          # (batch, vocab_size)
        return logits, new_hidden, new_cell, attention_weights

    def reset_state(self, batch_size: int):
        return (
            tf.zeros((batch_size, self.units)),
            tf.zeros((batch_size, self.units)),
        )


class ImageCaptioningModel(tf.keras.Model):
    """End-to-end image captioning model."""

    def __init__(self,
                 embedding_dim: int = EMBEDDING_DIM,
                 units: int = UNITS,
                 vocab_size: int = VOCAB_SIZE,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = CNNEncoder(embedding_dim)
        self.decoder = RNNDecoder(embedding_dim, units, vocab_size)

    def call(self, inputs, training=False):
        """
        inputs: (image_features, input_seq)
          image_features: (batch, 64, 2048)
          input_seq:      (batch, max_len)
        Returns logits over one caption step (teacher forcing handled in train_step).
        """
        image_features, input_seq = inputs
        features = self.encoder(image_features)
        batch_size = tf.shape(image_features)[0]
        hidden, cell = self.decoder.reset_state(batch_size)

        # Feed one token at a time (teacher forcing uses all steps)
        # Here we do a single step for the call interface; training loop handles full seq
        token = tf.expand_dims(input_seq[:, 0], 1)
        logits, _, _, _ = self.decoder(token, features, hidden, cell, training=training)
        return logits

    def train_step_manual(self, image_features, input_seq, target_seq,
                          optimizer, loss_fn, training=True):
        """Full teacher-forcing train step over a sequence."""
        with tf.GradientTape() as tape:
            features = self.encoder(image_features, training=training)
            batch_size = tf.shape(image_features)[0]
            hidden, cell = self.decoder.reset_state(batch_size)
            total_loss = 0.0
            seq_len = tf.shape(input_seq)[1]

            for t in range(seq_len - 1):
                token = tf.expand_dims(input_seq[:, t], 1)
                logits, hidden, cell, _ = self.decoder(
                    token, features, hidden, cell, training=training
                )
                total_loss += loss_fn(target_seq[:, t + 1], logits)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(total_loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        return total_loss / tf.cast(seq_len - 1, tf.float32)


def build_model(embedding_dim=EMBEDDING_DIM, units=UNITS,
                vocab_size=VOCAB_SIZE) -> ImageCaptioningModel:
    return ImageCaptioningModel(embedding_dim=embedding_dim,
                                units=units, vocab_size=vocab_size)
