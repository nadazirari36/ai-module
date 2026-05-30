"""
api.py — API REST FastAPI pour le moteur de recommandation.
Expose trois endpoints :
  - GET  /health    : vérification de l'état de l'API et des modèles chargés
  - GET  /popular   : top 10 des articles les plus populaires (cold-start)
  - POST /recommend : recommandations personnalisées pour un utilisateur

Lancement :
    uvicorn api:app --host 0.0.0.0 --port 8000
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Supprime les logs verbeux de TensorFlow

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time

from inference import RecommendationEngine
from config import API_HOST, API_PORT, MAX_RECOMMENDATIONS

# ── Initialisation de l'application FastAPI ───────────────────────────────────
app = FastAPI(
    title="Module de Recommandation IA",
    description="Moteur de recommandation par ensemble (AutoEncodeur + NCF + LSTM)",
    version="1.0.0"
)

# Instance globale du moteur (chargée une seule fois au démarrage)
engine = RecommendationEngine()


# ── Modèles de données Pydantic (validation automatique des requêtes/réponses) ──

class RecommendRequest(BaseModel):
    """Corps de la requête POST /recommend."""
    user_id: str                          # Identifiant de l'utilisateur (CustomerID)
    purchased_items: Optional[List[str]] = []  # Liste des articles déjà achetés (codes StockCode)


class RecommendItem(BaseModel):
    """Un article recommandé avec son score de pertinence."""
    product_id:  str    # Code article (StockCode)
    description: str    # Description textuelle du produit
    score:       float  # Score de recommandation normalisé (0 à 1)


class RecommendResponse(BaseModel):
    """Réponse de l'endpoint POST /recommend."""
    recommendations: List[RecommendItem]  # Liste ordonnée des articles recommandés
    strategy:        str                   # Stratégie utilisée (ex. 'ensemble_autoencoder_ncf_lstm')


# ── Événement de démarrage ────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """
    Charge tous les modèles et mappings au démarrage de l'API.
    Cette opération est lente (~30 sec) mais n'est exécutée qu'une seule fois.
    Toutes les requêtes suivantes utilisent les modèles déjà en mémoire.
    """
    engine.load()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Vérifie que l'API est opérationnelle et que les modèles sont chargés.
    Utile pour les health checks en production (Docker, Kubernetes, etc.).
    Retourne un dict avec le statut et l'état de chaque modèle.
    """
    return {
        "status": "opérationnel",
        "modèles_chargés": {
            "autoencoder": engine.ae_model   is not None,
            "ncf":         engine.ncf_model  is not None,
            "lstm":        engine.lstm_model is not None,
        }
    }


@app.get("/popular")
async def popular():
    """
    Retourne les 10 articles les plus achetés.
    Utilisé comme fallback pour les nouveaux utilisateurs sans historique (cold-start total).
    Lève une erreur 503 si les articles populaires ne sont pas chargés.
    """
    if engine.popular_items is None:
        raise HTTPException(status_code=503, detail="Articles populaires non chargés")
    return {"articles": engine.popular_items[:10]}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Génère des recommandations personnalisées pour un utilisateur.

    Corps de la requête :
      - user_id         : identifiant de l'utilisateur
      - purchased_items : liste des articles déjà achetés (peut être vide)

    Comportement :
      - Utilisateur connu + historique → ensemble AE + NCF + LSTM
      - Utilisateur connu sans historique → NCF + articles populaires
      - Utilisateur inconnu avec historique → LSTM + AE
      - Utilisateur inconnu sans historique → articles populaires (cold-start)

    Lève une erreur 500 en cas d'exception interne.
    Avertit si le traitement dépasse 500 ms (objectif de latence).
    """
    start = time.time()

    try:
        # Convertit les items en chaînes pour correspondre au format stocké
        purchased_items = [str(item) for item in request.purchased_items] if request.purchased_items else []

        recommendations, strategy = engine.recommend(
            user_id=str(request.user_id),
            purchased_items=purchased_items,
            top_n=MAX_RECOMMENDATIONS
        )

        # Avertissement si la latence dépasse l'objectif de 500 ms
        elapsed = time.time() - start
        if elapsed > 0.5:
            print(f"ATTENTION : inférence en {elapsed:.2f}s (objectif < 500ms)")

        return RecommendResponse(
            recommendations=[RecommendItem(**r) for r in recommendations],
            strategy=strategy
        )

    except Exception as e:
        # Journalise l'erreur complète (stack trace) pour le débogage
        print(f"ERREUR dans /recommend : {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── Point d'entrée direct ───────────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn
    # Lance le serveur uvicorn directement (sans rechargement automatique en production)
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=False)
