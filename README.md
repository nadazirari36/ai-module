# AI Recommendation Module

Ensemble recommendation engine combining **AutoEncoder** (collaborative filtering), **Neural Collaborative Filtering (NCF)**, and **LSTM** (sequential prediction) models trained on the UCI Online Retail Dataset.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Ensemble Inference                       │
│                                                             │
│   ┌───────────────┐  ┌──────────┐  ┌────────────────────┐   │
│   │  AutoEncoder  │  │   NCF    │  │   LSTM Sequential  │   │
│   │  (4070→128→   │  │ User+Item│  │  Sequence→Next     │   │
│   │   4070)       │  │ Embed→MLP│  │  Item Prediction   │   │
│   └──────┬────────┘  └────┬─────┘  └─────────┬──────────┘   │
│          │                │                  │              │
│          └────────────────┘──────────────────┘              │
│                           │                                 │
│                     Average & Re-rank                       │
│                   Filter Purchased Items                    │
│                           │                                 │
│                  Top-N Recommendations                      │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
ai_module/
├── config.py              # All paths, hyperparameters, constants
├── preprocessing.py       # Data cleaning + matrix building + sequences
├── apriori_rules.py       # Association rules + heatmap visualization
├── model_autoencoder.py   # AutoEncoder definition + training
├── model_ncf.py           # Neural Collaborative Filtering
├── model_lstm.py          # LSTM sequential model
├── evaluate.py            # Metrics: Precision@K, Recall@K, NDCG@K
├── inference.py           # Ensemble fusion logic
├── api.py                 # FastAPI endpoints
├── train_pipeline.py      # Full training orchestration
├── requirements.txt       # Python dependencies
├── data/
│   └── Online_Retail.xlsx # UCI Online Retail Dataset
├── saved_models/          # Exported .keras models + artifacts
│   ├── autoencoder.keras
│   ├── ncf.keras
│   ├── lstm.keras
│   ├── mappings.pkl
│   ├── popular_items.pkl
│   └── association_rules.csv
└── notebooks/
    └── full_pipeline.ipynb
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Training

### Full Pipeline (recommended)

```bash
cd ai_module
python train_pipeline.py
```

This runs all steps: preprocessing → Apriori → AutoEncoder → NCF → LSTM → Evaluation.

### Individual Models

```bash
python preprocessing.py      # Data cleaning only
python apriori_rules.py      # Association rules only
python model_autoencoder.py  # AutoEncoder only
python model_ncf.py          # NCF only
python model_lstm.py         # LSTM only
```

## Launching the API

```bash
cd ai_module
uvicorn api:app --host 0.0.0.0 --port 8000
```

Or:
```bash
python api.py
```

## API Endpoints

### POST /recommend
Generate personalized recommendations.

**Request:**
```json
{
  "user_id": "17850",
  "purchased_items": ["85123A", "71053", "84406B"]
}
```

**Response:**
```json
{
  "recommendations": [
    {"product_id": "22423", "description": "REGENCY CAKESTAND 3 TIER", "score": 0.94},
    {"product_id": "47566", "description": "PARTY BUNTING", "score": 0.87}
  ],
  "strategy": "ensemble_autoencoder_ncf_lstm"
}
```

### GET /health
```json
{
  "status": "healthy",
  "models_loaded": {
    "autoencoder": true,
    "ncf": true,
    "lstm": true
  }
}
```

### GET /popular
Top 10 most purchased items (cold-start fallback).

## Integration with Java Backend

The Java backend calls `POST /recommend` with a user ID and their purchase history. If `user_id` is unknown (new user), fall back to `GET /popular`.

```java
// Example Java integration
HttpClient client = HttpClient.newHttpClient();
String json = "{\"user_id\":\"17850\",\"purchased_items\":[\"85123A\"]}";
HttpRequest request = HttpRequest.newBuilder()
    .uri(URI.create("http://localhost:8000/recommend"))
    .header("Content-Type", "application/json")
    .POST(HttpRequest.BodyPublishers.ofString(json))
    .build();
HttpResponse<String> response = client.send(request, BodyHandlers.ofString());
```

## Model Details

| Model | Input | Architecture | Loss |
|-------------|-------|-------------|------|
| AutoEncoder | Binary user-item vector (4070 dims) | 4070→512→256→128→256→512→4070 | Binary Cross-Entropy |
| NCF | User ID + Item ID embeddings (64-dim) | Concat→128→64→1 | Binary Cross-Entropy |
| LSTM | Sequence of item embeddings | Embed→LSTM(128)→Dense(64)→Dense(4070) | Categorical Cross-Entropy |

All models use Adam optimizer with ReduceLROnPlateau, Dropout(0.3), and seed=42.

## Evaluation Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended  
- **NDCG@K**: Normalized Discounted Cumulative Gain (position-aware)

Computed at K=5 and K=10. Baseline: pure Apriori association rules.
