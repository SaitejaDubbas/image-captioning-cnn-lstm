"""API integration tests — run without a trained model (smoke tests)."""

import io
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def client():
    """Create a Flask test client with mocked ML dependencies."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "a": 3, "dog": 4}
    mock_tokenizer.idx2word = {v: k for k, v in mock_tokenizer.word2idx.items()}

    mock_model = MagicMock()
    mock_encoder = MagicMock()
    mock_encoder.return_value = np.zeros((1, 64, 2048), dtype=np.float32)

    with (
        patch("app.api.get_tokenizer", return_value=mock_tokenizer),
        patch("app.api.get_model", return_value=mock_model),
        patch("app.api.get_encoder", return_value=mock_encoder),
        patch("app.api.generate_caption_from_features", return_value="a dog running in a field"),
        patch("app.api.extract_features_from_array", return_value=np.zeros((64, 2048))),
    ):
        from app.api import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


def make_image_bytes() -> bytes:
    """Create a minimal valid JPEG in-memory."""
    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(100, 149, 237))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_root_lists_endpoints(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "endpoints" in data


class TestPredict:
    def test_predict_with_valid_image(self, client):
        img_bytes = make_image_bytes()
        resp = client.post(
            "/predict",
            data={"image": (io.BytesIO(img_bytes), "test.jpg")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "caption" in data
        assert isinstance(data["caption"], str)
        assert "latency_ms" in data

    def test_predict_no_file_returns_400(self, client):
        resp = client.post("/predict", data={}, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_predict_invalid_extension_returns_400(self, client):
        resp = client.post(
            "/predict",
            data={"image": (io.BytesIO(b"data"), "file.txt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_predict_greedy_strategy(self, client):
        img_bytes = make_image_bytes()
        resp = client.post(
            "/predict?strategy=greedy",
            data={"image": (io.BytesIO(img_bytes), "img.jpg")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.get_json()["strategy"] == "greedy"


class TestDataLoader:
    def test_clean_caption(self):
        from src.data_loader import clean_caption
        result = clean_caption("A Dog Running!! 123")
        assert result.startswith("<start>")
        assert result.endswith("<end>")
        assert "123" not in result
        assert "!!" not in result

    def test_tokenizer_encode_decode(self):
        from src.data_loader import Tokenizer
        tok = Tokenizer(vocab_size=100)
        captions = {"img1.jpg": ["<start> a dog runs <end>", "<start> dog is fast <end>"]}
        tok.fit(captions)
        encoded = tok.encode("<start> a dog <end>")
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        decoded = tok.decode(encoded)
        assert isinstance(decoded, str)


class TestModel:
    def test_model_instantiation(self):
        from src.model import build_model
        model = build_model(embedding_dim=64, units=128, vocab_size=100)
        assert model is not None

    def test_bahdanau_attention_output_shape(self):
        import tensorflow as tf
        from src.model import BahdanauAttention
        attn = BahdanauAttention(units=64)
        features = tf.zeros((2, 64, 2048))
        hidden = tf.zeros((2, 64))
        context, weights = attn(features, hidden)
        assert context.shape == (2, 2048)
        assert weights.shape == (2, 64, 1)
