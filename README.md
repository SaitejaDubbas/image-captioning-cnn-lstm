# Image Captioning — CNN-LSTM with Attention

> Automatically generate natural-language descriptions of images using an InceptionV3 encoder and LSTM decoder with Bahdanau attention, trained on Flickr8k.

| Metric | Score |
|---|---|
| BLEU-1 | **0.49** |
| BLEU-2 | ~0.32 |
| BLEU-3 | ~0.22 |
| BLEU-4 | ~0.15 |

## Architecture

```
Image → InceptionV3 (fine-tuned) → Spatial features (64 × 2048)
                                          ↓
                              CNN Encoder Dense (64 × 256)
                                          ↓
Token → Embedding (256) → ── Bahdanau Attention ── → LSTM (512) → Dense → Softmax
```

## Project Structure

```
computer_vision_project/
├── config.py                  # All hyperparameters and paths
├── src/
│   ├── data_loader.py         # Flickr8k loading, tokenizer, dataset builder
│   ├── feature_extractor.py   # InceptionV3 feature extraction (cached)
│   ├── model.py               # CNN encoder + LSTM decoder + attention
│   ├── train.py               # Teacher-forcing training loop
│   ├── evaluate.py            # BLEU score evaluation
│   └── predict.py             # Greedy & beam-search inference
├── app/
│   ├── api.py                 # Flask REST API (4 endpoints)
│   └── templates/index.html   # Interactive web demo
├── tests/
│   └── test_api.py            # Pytest unit + integration tests
├── scripts/
│   ├── download_data.py       # Kaggle dataset download
│   └── setup_azure.sh         # Azure provisioning script
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci-cd.yml   # Build → Test → Deploy pipeline
└── azure-deploy.yml
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Flickr8k from Kaggle

Place your `kaggle.json` at `~/.kaggle/kaggle.json` (get it from kaggle.com → Account → API), then:

```bash
python scripts/download_data.py
```

### 3. Train the model

```bash
python -m src.train
```

Training runs for 20 epochs. Features are extracted and cached on first run (~10 min for Flickr8k on GPU).

### 4. Evaluate

```bash
python -m src.evaluate
```

### 5. Run the API

```bash
python app/api.py
# or with gunicorn:
gunicorn app.api:app --bind 0.0.0.0:5000 --workers 2
```

Open http://localhost:5000 for the web demo.

### 6. Docker

```bash
docker-compose up --build
```

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/` | API info |
| POST | `/predict` | Caption one image (multipart `image` field) |
| POST | `/predict/url` | Caption image from URL (JSON `{"url": "..."}`) |
| POST | `/batch_predict` | Caption up to 10 images (`image_0`…`image_9`) |

### Example

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@dog.jpg" \
  -G -d "strategy=beam&beam_width=3"
```

```json
{
  "caption": "a brown dog is running through the grass",
  "strategy": "beam",
  "latency_ms": 342.5
}
```

## CI/CD Pipeline

GitHub Actions (`.github/workflows/ci-cd.yml`):

1. **Test** — pytest on every push/PR
2. **Build** — Docker image pushed to GitHub Container Registry
3. **Deploy** — Azure Web App deployment via `azure/webapps-deploy`

## Azure Deployment

### One-time setup

```bash
# 1. Provision Azure resources
bash scripts/setup_azure.sh

# 2. Create service principal and add to GitHub secrets
az ad sp create-for-rbac --name "caption-api-sp" \
  --role contributor \
  --scopes /subscriptions/<sub-id>/resourceGroups/image-captioning-rg \
  --sdk-auth
# Copy the JSON output → GitHub secret: AZURE_CREDENTIALS

# 3. Add AZURE_WEBAPP_NAME secret → "image-captioning-api"
```

Push to `main` to trigger automatic deployment.

## Key Design Choices

- **Spatial features**: InceptionV3 outputs `(8×8, 2048)` spatial grids (not pooled) so attention can focus on image regions.
- **Bahdanau attention**: Additive attention scores each of the 64 spatial locations per decoder step.
- **Fine-tuning**: Upper InceptionV3 layers (from layer 249) are unfrozen in a second training phase with a reduced learning rate (1e-4).
- **Beam search**: Optional beam-search decoding (k=3) improves caption quality at inference time.
- **Teacher forcing**: During training, ground-truth tokens are fed as decoder inputs for faster convergence.
