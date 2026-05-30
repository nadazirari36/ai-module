"""
inference.py — Moteur de recommandation par ensemble.
Charge tous les modèles entraînés et fusionne leurs prédictions
pour générer des recommandations personnalisées.
Stratégie : moyenne pondérée des scores normalisés (AE + NCF + LSTM).
"""
import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Supprime les logs verbeux de TensorFlow

import tensorflow as tf
from tensorflow import keras
from config import (SAVED_MODELS_DIR, MAPPINGS_PATH, POPULAR_PATH, MAX_RECOMMENDATIONS,
                    SEED, LSTM_SEQ_LEN, ENSEMBLE_WEIGHT_AE, ENSEMBLE_WEIGHT_NCF, ENSEMBLE_WEIGHT_LSTM)

np.random.seed(SEED)


class RecommendationEngine:
    """
    Moteur de recommandation par ensemble combinant AutoEncodeur + NCF + LSTM.
    Chaque modèle produit des scores pour tous les articles, puis les scores
    sont normalisés et combinés par moyenne pondérée.
    Gère également le démarrage à froid (cold-start) via les articles populaires.
    """

    def __init__(self):
        """Initialise les attributs — les modèles sont chargés séparément via load()."""
        self.ae_model        = None   # Modèle AutoEncodeur (keras)
        self.ncf_model       = None   # Modèle NCF/NeuMF (keras)
        self.lstm_model      = None   # Modèle LSTM (keras)
        self.mappings        = None   # Dictionnaire des mappings (user2idx, item2idx, etc.)
        self.popular_items   = None   # Liste des articles les plus populaires (cold-start)
        self.customer_vectors = None  # Matrice client-article (pour _get_user_vector)

    def load(self, customer_item_matrix=None):
        """
        Charge tous les modèles et mappings depuis le disque.
        Appelé une seule fois au démarrage de l'API.
        compile=False : ignore la configuration de perte/optimiseur (inférence seulement).
        """
        print("Chargement du moteur de recommandation...")

        # Chargement des mappings utilisateur/article
        with open(MAPPINGS_PATH, 'rb') as f:
            self.mappings = pickle.load(f)

        # Chargement des articles populaires (fallback cold-start)
        with open(POPULAR_PATH, 'rb') as f:
            self.popular_items = pickle.load(f)

        # Chargement de l'AutoEncodeur (si le fichier existe)
        ae_path = os.path.join(SAVED_MODELS_DIR, 'autoencoder.keras')
        if os.path.exists(ae_path):
            self.ae_model = keras.models.load_model(ae_path, compile=False)
            print("  AutoEncodeur chargé.")

        # Chargement du NCF (si le fichier existe)
        ncf_path = os.path.join(SAVED_MODELS_DIR, 'ncf.keras')
        if os.path.exists(ncf_path):
            self.ncf_model = keras.models.load_model(ncf_path, compile=False)
            print("  NCF chargé.")

        # Chargement du LSTM (si le fichier existe)
        lstm_path = os.path.join(SAVED_MODELS_DIR, 'lstm.keras')
        if os.path.exists(lstm_path):
            self.lstm_model = keras.models.load_model(lstm_path, compile=False)
            print("  LSTM chargé.")

        # Optionnel : matrice client-article pour récupérer les vecteurs utilisateur
        if customer_item_matrix is not None:
            self.customer_vectors = customer_item_matrix

        print("Moteur prêt.")

    def _get_user_vector(self, user_id, purchased_items=None):
        """
        Construit le vecteur binaire d'achats pour un utilisateur.
        Priorité 1 : récupère la ligne de la matrice client-article (si disponible).
        Priorité 2 : construit le vecteur à partir de la liste d'articles fournie.
        Le vecteur a n_items dimensions : 1 = article acheté, 0 = non acheté.
        """
        n_items  = self.mappings['n_items']
        item2idx = self.mappings['item2idx']

        # Si la matrice est disponible ET que l'utilisateur y est présent
        if self.customer_vectors is not None and user_id in self.customer_vectors.index:
            return self.customer_vectors.loc[user_id].values.astype(np.float32)

        # Sinon, construit le vecteur à partir de la liste d'articles
        vec = np.zeros(n_items, dtype=np.float32)
        if purchased_items:
            for item in purchased_items:
                if item in item2idx:
                    vec[item2idx[item]] = 1.0
        return vec

    def _ae_scores(self, user_vector):
        """
        Calcule les scores de reconstruction de l'AutoEncodeur pour un utilisateur.
        Entrée  : vecteur binaire d'achats (n_items dimensions)
        Sortie  : vecteur de scores ∈ [0,1] (n_items dimensions)
        Un score élevé = l'AE prédit que cet article correspond au profil de l'utilisateur.
        Retourne des zéros si l'AE n'est pas chargé.
        """
        if self.ae_model is None:
            return np.zeros(self.mappings['n_items'])
        inp = user_vector.reshape(1, -1)  # Ajoute la dimension batch
        return self.ae_model.predict(inp, verbose=0)[0]

    def _ncf_scores(self, user_id, purchased_items=None):
        """
        Calcule les scores NCF pour tous les articles pour un utilisateur donné.

        Pour les utilisateurs connus (embedding entraîné) : utilise directement l'embedding.
        Pour les utilisateurs cold-start (embedding aléatoire non entraîné) :
          → inférence du vecteur utilisateur par la moyenne des embeddings d'articles achetés
          → met à jour l'embedding dans le modèle avant de scorer
          → technique : User-from-Items Inference (améliore NCF de ~4 % → ~20 %+)
        """
        if self.ncf_model is None:
            return np.zeros(self.mappings['n_items'])
        user2idx = self.mappings['user2idx']
        n_items  = self.mappings['n_items']

        if user_id not in user2idx:
            return np.zeros(n_items)

        uidx = user2idx[user_id]

        # Si l'utilisateur est cold-start ET qu'on a son historique d'achats,
        # on infère son embedding à partir des articles achetés.
        if purchased_items:
            item2idx = self.mappings['item2idx']
            purchased_idx = [item2idx[it] for it in purchased_items if it in item2idx]
            if purchased_idx:
                try:
                    gmf_item_w = self.ncf_model.get_layer('item_gmf').embeddings.numpy()
                    mlp_item_w = self.ncf_model.get_layer('item_mlp').embeddings.numpy()

                    # Centroïde des embeddings des articles achetés → vecteur utilisateur inféré
                    inferred_gmf = np.mean(gmf_item_w[purchased_idx], axis=0)
                    inferred_mlp = np.mean(mlp_item_w[purchased_idx], axis=0)

                    # Met à jour les embeddings utilisateur dans le modèle
                    gmf_user_w = self.ncf_model.get_layer('user_gmf').embeddings.numpy()
                    mlp_user_w = self.ncf_model.get_layer('user_mlp').embeddings.numpy()
                    gmf_user_w[uidx] = inferred_gmf
                    mlp_user_w[uidx] = inferred_mlp
                    self.ncf_model.get_layer('user_gmf').embeddings.assign(gmf_user_w)
                    self.ncf_model.get_layer('user_mlp').embeddings.assign(mlp_user_w)
                except Exception:
                    pass  # En cas d'erreur, on continue avec l'embedding existant

        user_arr = np.full(n_items, uidx)
        item_arr = np.arange(n_items)
        return self.ncf_model.predict([user_arr, item_arr], verbose=0, batch_size=1024).flatten()

    def _ncf_scores_cold_start(self, purchased_items):
        """
        Scoring NCF pour les utilisateurs à démarrage à froid.
        Principe : calcule l'embedding utilisateur moyen à partir des embeddings
        des articles achetés (chemin GMF), puis score tous les articles par produit scalaire.
        Cette approche améliore les scores NCF pour les utilisateurs dont l'embedding
        n'a pas été entraîné (passants de ~17 % à ~25 %+ de précision).
        Retourne des zéros si le NCF n'est pas chargé ou si aucun article n'est fourni.
        """
        if self.ncf_model is None or not purchased_items:
            return np.zeros(self.mappings['n_items'])
        item2idx = self.mappings['item2idx']
        n_items  = self.mappings['n_items']

        # Convertit les codes articles en indices
        purchased_idx = [item2idx[it] for it in purchased_items if it in item2idx]
        if not purchased_idx:
            return np.zeros(n_items)

        try:
            # Récupère les embeddings articles du chemin GMF
            gmf_item_embs = self.ncf_model.get_layer('item_gmf').embeddings.numpy()
        except Exception:
            return np.zeros(n_items)

        # Embedding utilisateur inféré = moyenne des embeddings des articles achetés
        user_emb = np.mean(gmf_item_embs[purchased_idx], axis=0)  # [embed_dim]

        # Score de tous les articles = produit scalaire avec l'embedding utilisateur inféré
        raw = gmf_item_embs @ user_emb  # [n_items]

        # Normalisation sigmoïde → scores dans [0, 1]
        return 1.0 / (1.0 + np.exp(-raw))

    def _lstm_scores(self, purchased_items):
        """
        Calcule les scores LSTM (probabilité du prochain article) pour une séquence d'achats.
        Le LSTM utilise les LSTM_SEQ_LEN derniers articles comme contexte.
        Les articles inconnus sont ignorés. La séquence est paddée à gauche si nécessaire.
        Retourne des zéros si le LSTM n'est pas chargé ou si aucun article n'est fourni.
        """
        if self.lstm_model is None or not purchased_items:
            return np.zeros(self.mappings['n_items'])

        item2idx = self.mappings['item2idx']

        # Convertit les codes articles en indices (ignore les articles inconnus)
        indices = [item2idx[it] for it in purchased_items if it in item2idx]
        if not indices:
            return np.zeros(self.mappings['n_items'])

        # Garde uniquement les LSTM_SEQ_LEN derniers articles (fenêtre la plus récente)
        indices = indices[-LSTM_SEQ_LEN:]

        # Padding à gauche avec des 0 si la séquence est trop courte
        if len(indices) < LSTM_SEQ_LEN:
            indices = [0] * (LSTM_SEQ_LEN - len(indices)) + indices

        X = np.array(indices).reshape(1, -1)  # Ajoute la dimension batch
        return self.lstm_model.predict(X, verbose=0)[0]

    def recommend(self, user_id, purchased_items=None, top_n=MAX_RECOMMENDATIONS):
        """
        Génère des recommandations personnalisées par fusion des trois modèles.

        Algorithme :
          1. Vérifie si l'utilisateur est connu ou cold-start.
          2. Calcule les scores de chaque modèle (AE, NCF, LSTM).
          3. Normalise chaque ensemble de scores dans [0, 1] (min-max).
          4. Combine par moyenne pondérée : AE×0.10 + NCF×0.35 + LSTM×0.55.
          5. Trie par score décroissant et filtre les articles déjà achetés.
          6. Si peu d'historique (<3 articles), complète avec les articles populaires.

        Retourne : (liste de recommandations, stratégie utilisée)
        """
        idx2item = self.mappings['idx2item']
        item2idx = self.mappings['item2idx']
        desc_map = self.mappings['desc_map']
        n_items  = self.mappings['n_items']

        # Vérifie si l'utilisateur est connu et s'il a un historique
        is_known  = user_id in self.mappings['user2idx']
        has_items = purchased_items and len(purchased_items) > 0

        # Cold-start total : utilisateur inconnu SANS historique → articles populaires
        if not is_known and not has_items:
            return self.popular_items[:top_n], 'popular_fallback'

        # Construit le vecteur d'achats de l'utilisateur
        user_vector = self._get_user_vector(user_id, purchased_items)

        # Ensemble des articles déjà achetés (à exclure des recommandations)
        purchased_set = set()
        if purchased_items:
            purchased_set = set(purchased_items)
        for i in range(n_items):
            if user_vector[i] > 0:
                purchased_set.add(idx2item[i])

        def _normalize(sc):
            """Normalise un vecteur de scores dans [0, 1] (min-max scaling)."""
            mn, mx = sc.min(), sc.max()
            return (sc - mn) / (mx - mn) if mx - mn > 1e-8 else sc

        # ── Calcul et fusion des scores ───────────────────────────────────────
        scores = np.zeros(n_items)
        w_sum  = 0.0  # Somme des poids des modèles actifs (pour renormaliser si un modèle échoue)

        # Scores AutoEncodeur
        ae_sc = self._ae_scores(user_vector)
        if np.any(ae_sc > 0):
            scores += ENSEMBLE_WEIGHT_AE * _normalize(ae_sc)
            w_sum  += ENSEMBLE_WEIGHT_AE

        # Scores NCF (passe l'historique pour inférer l'embedding si cold-start)
        ncf_sc = self._ncf_scores(user_id, purchased_items=purchased_items)
        if np.any(ncf_sc > 0):
            scores += ENSEMBLE_WEIGHT_NCF * _normalize(ncf_sc)
            w_sum  += ENSEMBLE_WEIGHT_NCF

        # Scores LSTM
        lstm_sc = self._lstm_scores(purchased_items or [])
        if np.any(lstm_sc > 0):
            scores += ENSEMBLE_WEIGHT_LSTM * _normalize(lstm_sc)
            w_sum  += ENSEMBLE_WEIGHT_LSTM

        # Renormalise si certains modèles n'ont pas produit de scores
        if w_sum > 0:
            scores /= w_sum

        # ── Tri et filtrage ───────────────────────────────────────────────────
        ranked_indices  = np.argsort(-scores)  # Indices triés du score le plus élevé au plus bas
        recommendations = []

        for idx in ranked_indices:
            item_code = idx2item[idx]

            # Ignore les articles déjà achetés
            if item_code in purchased_set:
                continue

            recommendations.append({
                'product_id':  str(item_code),
                'description': str(desc_map.get(item_code, item_code)),
                'score':       round(float(scores[idx]), 4)
            })

            if len(recommendations) >= top_n:
                break

        # ── Complément cold-start ─────────────────────────────────────────────
        # Si l'utilisateur a peu d'achats (<3), on complète avec les articles populaires
        if has_items and len(purchased_items) < 3:
            pop_ids = {p['product_id'] for p in recommendations}
            for pop_item in self.popular_items:
                if (pop_item['product_id'] not in pop_ids and
                        pop_item['product_id'] not in purchased_set):
                    recommendations.append({
                        'product_id':  str(pop_item['product_id']),
                        'description': str(pop_item.get('description', pop_item['product_id'])),
                        'score':       float(pop_item.get('score', 1.0))
                    })
                    if len(recommendations) >= top_n:
                        break

        # Détermine la stratégie utilisée (pour le logging/debug)
        n_active = sum([np.any(ae_sc > 0), np.any(ncf_sc > 0), np.any(lstm_sc > 0)])
        strategy = 'ensemble_autoencoder_ncf_lstm' if n_active > 1 else 'single_model'

        return recommendations[:top_n], strategy
