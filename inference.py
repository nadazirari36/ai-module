"""
Inference Module.
Loads all trained models and fuses predictions for recommendations.
"""
import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from config import (SAVED_MODELS_DIR, MAPPINGS_PATH, POPULAR_PATH, MAX_RECOMMENDATIONS,
                    SEED, LSTM_SEQ_LEN, ENSEMBLE_WEIGHT_AE, ENSEMBLE_WEIGHT_NCF, ENSEMBLE_WEIGHT_LSTM)

np.random.seed(SEED)


class RecommendationEngine:
    """Ensemble recommendation engine combining AutoEncoder + NCF + LSTM."""

    def __init__(self):
        self.ae_model = None
        self.ncf_model = None
        self.lstm_model = None
        self.mappings = None
        self.popular_items = None
        self.customer_vectors = None

    def load(self, customer_item_matrix=None):
        """Load all models and mappings."""
        print("Loading recommendation engine...")

        # Load mappings
        with open(MAPPINGS_PATH, 'rb') as f:
            self.mappings = pickle.load(f)

        # Load popular items
        with open(POPULAR_PATH, 'rb') as f:
            self.popular_items = pickle.load(f)

        # Load models — compile=False skips loss/optimizer restore (inference only)
        ae_path = os.path.join(SAVED_MODELS_DIR, 'autoencoder.keras')
        if os.path.exists(ae_path):
            self.ae_model = keras.models.load_model(ae_path, compile=False)
            print("  AutoEncoder loaded.")

        ncf_path = os.path.join(SAVED_MODELS_DIR, 'ncf.keras')
        if os.path.exists(ncf_path):
            self.ncf_model = keras.models.load_model(ncf_path, compile=False)
            print("  NCF loaded.")

        lstm_path = os.path.join(SAVED_MODELS_DIR, 'lstm.keras')
        if os.path.exists(lstm_path):
            self.lstm_model = keras.models.load_model(lstm_path, compile=False)
            print("  LSTM loaded.")

        if customer_item_matrix is not None:
            self.customer_vectors = customer_item_matrix

        print("Engine ready.")

    def _get_user_vector(self, user_id, purchased_items=None):
        """Build a binary user vector from customer matrix or purchased items."""
        n_items = self.mappings['n_items']
        item2idx = self.mappings['item2idx']

        if self.customer_vectors is not None and user_id in self.customer_vectors.index:
            return self.customer_vectors.loc[user_id].values.astype(np.float32)

        # Build from purchased items list
        vec = np.zeros(n_items, dtype=np.float32)
        if purchased_items:
            for item in purchased_items:
                if item in item2idx:
                    vec[item2idx[item]] = 1.0
        return vec

    def _ae_scores(self, user_vector):
        """Get AutoEncoder reconstruction scores."""
        if self.ae_model is None:
            return np.zeros(self.mappings['n_items'])
        inp = user_vector.reshape(1, -1)
        return self.ae_model.predict(inp, verbose=0)[0]

    def _ncf_scores(self, user_id):
        """Get NCF scores for all items."""
        if self.ncf_model is None:
            return np.zeros(self.mappings['n_items'])
        user2idx = self.mappings['user2idx']
        n_items = self.mappings['n_items']
        if user_id not in user2idx:
            return np.zeros(n_items)
        uidx = user2idx[user_id]
        user_arr = np.full(n_items, uidx)
        item_arr = np.arange(n_items)
        return self.ncf_model.predict([user_arr, item_arr], verbose=0, batch_size=1024).flatten()

    def _lstm_scores(self, purchased_items):
        """Get LSTM next-item scores."""
        if self.lstm_model is None or not purchased_items:
            return np.zeros(self.mappings['n_items'])
        item2idx = self.mappings['item2idx']
        indices = [item2idx[it] for it in purchased_items if it in item2idx]
        if not indices:
            return np.zeros(self.mappings['n_items'])
        indices = indices[-LSTM_SEQ_LEN:]
        if len(indices) < LSTM_SEQ_LEN:
            indices = [0] * (LSTM_SEQ_LEN - len(indices)) + indices
        X = np.array(indices).reshape(1, -1)
        return self.lstm_model.predict(X, verbose=0)[0]

    def recommend(self, user_id, purchased_items=None, top_n=MAX_RECOMMENDATIONS):
        """
        Generate ensemble recommendations.
        Averages scores from AE + NCF + LSTM and filters already-purchased items.
        """
        idx2item = self.mappings['idx2item']
        item2idx = self.mappings['item2idx']
        desc_map = self.mappings['desc_map']
        n_items = self.mappings['n_items']

        # Check cold start
        is_known = user_id in self.mappings['user2idx']
        has_items = purchased_items and len(purchased_items) > 0

        if not is_known and not has_items:
            return self.popular_items[:top_n], 'popular_fallback'

        # Get user vector
        user_vector = self._get_user_vector(user_id, purchased_items)
        purchased_set = set()
        if purchased_items:
            purchased_set = set(purchased_items)
        # Also add items from user vector
        for i in range(n_items):
            if user_vector[i] > 0:
                purchased_set.add(idx2item[i])

        def _normalize(sc):
            mn, mx = sc.min(), sc.max()
            return (sc - mn) / (mx - mn) if mx - mn > 1e-8 else sc

        # Weighted ensemble — LSTM dominates, NCF supplements, AE minor
        scores = np.zeros(n_items)
        w_sum = 0.0

        ae_sc = self._ae_scores(user_vector)
        if np.any(ae_sc > 0):
            scores += ENSEMBLE_WEIGHT_AE * _normalize(ae_sc)
            w_sum += ENSEMBLE_WEIGHT_AE

        ncf_sc = self._ncf_scores(user_id)
        if np.any(ncf_sc > 0):
            scores += ENSEMBLE_WEIGHT_NCF * _normalize(ncf_sc)
            w_sum += ENSEMBLE_WEIGHT_NCF

        lstm_sc = self._lstm_scores(purchased_items or [])
        if np.any(lstm_sc > 0):
            scores += ENSEMBLE_WEIGHT_LSTM * _normalize(lstm_sc)
            w_sum += ENSEMBLE_WEIGHT_LSTM

        if w_sum > 0:
            scores /= w_sum

        # Rank and filter
        ranked_indices = np.argsort(-scores)
        recommendations = []
        for idx in ranked_indices:
            item_code = idx2item[idx]
            if item_code in purchased_set:
                continue
            recommendations.append({
                'product_id': str(item_code),  # Ensure string format
                'description': str(desc_map.get(item_code, item_code)),  # Ensure string
                'score': round(float(scores[idx]), 4)
            })
            if len(recommendations) >= top_n:
                break

        # Cold-start blend: if few purchases, mix with popular
        if has_items and len(purchased_items) < 3:
            pop_ids = {p['product_id'] for p in recommendations}
            for pop_item in self.popular_items:
                if pop_item['product_id'] not in pop_ids and pop_item['product_id'] not in purchased_set:
                    # Ensure string format for popular items too
                    pop_item_fixed = {
                        'product_id': str(pop_item['product_id']),
                        'description': str(pop_item.get('description', pop_item['product_id'])),
                        'score': float(pop_item.get('score', 1.0))
                    }
                    recommendations.append(pop_item_fixed)
                    if len(recommendations) >= top_n:
                        break

        n_active = sum([np.any(ae_sc > 0), np.any(ncf_sc > 0), np.any(lstm_sc > 0)])
        strategy = 'ensemble_autoencoder_ncf_lstm' if n_active > 1 else 'single_model'
        return recommendations[:top_n], strategy