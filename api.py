"""
FastAPI REST API for Recommendation Engine.
Endpoints: POST /recommend, GET /health, GET /popular
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time

from inference import RecommendationEngine
from config import API_HOST, API_PORT, MAX_RECOMMENDATIONS

app = FastAPI(
    title="AI Recommendation Module",
    description="Ensemble recommendation engine (AutoEncoder + NCF + LSTM)",
    version="1.0.0"
)

# Global engine instance
engine = RecommendationEngine()


class RecommendRequest(BaseModel):
    user_id: str
    purchased_items: Optional[List[str]] = []


class RecommendItem(BaseModel):
    product_id: str
    description: str
    score: float


class RecommendResponse(BaseModel):
    recommendations: List[RecommendItem]
    strategy: str


@app.on_event("startup")
async def startup():
    """Load models on startup."""
    engine.load()


@app.get("/health")
async def health():
    """API health check."""
    return {
        "status": "healthy",
        "models_loaded": {
            "autoencoder": engine.ae_model is not None,
            "ncf": engine.ncf_model is not None,
            "lstm": engine.lstm_model is not None,
        }
    }


@app.get("/popular")
async def popular():
    """Top 10 most purchased items (cold-start fallback)."""
    if engine.popular_items is None:
        raise HTTPException(status_code=503, detail="Popular items not loaded")
    return {"items": engine.popular_items[:10]}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """Generate personalized recommendations."""
    start = time.time()

    try:
        # Convert items to strings to match stored format
        purchased_items = [str(item) for item in request.purchased_items] if request.purchased_items else []
        
        recommendations, strategy = engine.recommend(
            user_id=str(request.user_id),
            purchased_items=purchased_items,
            top_n=MAX_RECOMMENDATIONS
        )

        elapsed = time.time() - start
        if elapsed > 0.5:
            print(f"WARNING: Inference took {elapsed:.2f}s (>500ms target)")

        return RecommendResponse(
            recommendations=[RecommendItem(**r) for r in recommendations],
            strategy=strategy
        )
    except Exception as e:
        print(f"ERROR in /recommend: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=False)
