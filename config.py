"""
config.py — Fichier de configuration central du module de recommandation IA.
Tous les chemins, hyperparamètres et constantes sont définis ici.
Modifier une valeur ici la propage automatiquement à tous les modules qui l'importent.
"""
import os

# ─────────────────────────────────────────────
# CHEMINS DES FICHIERS
# ─────────────────────────────────────────────

# Répertoire racine du projet (dossier contenant ce fichier)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemin vers le fichier Excel brut utilisé pour l'entraînement et l'évaluation
DATA_PATH = os.path.join(BASE_DIR, "data", "training.xlsx")

# Dossier où sont sauvegardés les modèles entraînés, mappings et résultats
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# Chemin de sauvegarde des règles d'association Apriori en CSV (utilisé par le backend Java)
RULES_CSV_PATH = os.path.join(SAVED_MODELS_DIR, "association_rules.csv")

# Chemin de sauvegarde des mappings utilisateur/article (user2idx, item2idx, desc_map, etc.)
MAPPINGS_PATH = os.path.join(SAVED_MODELS_DIR, "mappings.pkl")

# Chemin de sauvegarde des articles les plus populaires (utilisé en cas de démarrage à froid)
POPULAR_PATH = os.path.join(SAVED_MODELS_DIR, "popular_items.pkl")

# Crée le dossier saved_models s'il n'existe pas encore
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# REPRODUCTIBILITÉ
# ─────────────────────────────────────────────

# Graine aléatoire — définie dans chaque module pour des résultats reproductibles
SEED = 42

# ─────────────────────────────────────────────
# RATIOS DE DÉCOUPAGE DES UTILISATEURS
# ─────────────────────────────────────────────
# Les utilisateurs sont divisés aléatoirement en trois groupes :
#   - Train  (70 %) : utilisé pour entraîner les modèles
#   - Val    (10 %) : utilisé pour l'arrêt anticipé / validation
#   - Test   (20 %) : complètement mis de côté, utilisé uniquement pour l'évaluation finale
# Utiliser 20 % pour le test donne ~868 utilisateurs de test (vs seulement 434 à 10 %).
# NCF et LSTM sont entraînés sur Train + Val (80 %) pour voir plus de données.

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20

# Ratio de découpage temporel par utilisateur — conservé pour la fonction temporal_split()
# dans preprocessing.py au cas où une évaluation temporelle serait nécessaire.
TEMPORAL_TEST_RATIO = 0.20

# ─────────────────────────────────────────────
# RÈGLES D'ASSOCIATION APRIORI
# ─────────────────────────────────────────────

# Fraction minimale de transactions devant contenir un ensemble d'articles
MIN_SUPPORT    = 0.01

# Probabilité minimale P(conséquent | antécédent) pour qu'une règle soit conservée
MIN_CONFIDENCE = 0.5

# Lift minimal (combien de fois plus probable qu'au hasard) — filtre les règles faibles
MIN_LIFT       = 1.5

# ─────────────────────────────────────────────
# HYPERPARAMÈTRES DE L'AUTOENCODEUR
# ─────────────────────────────────────────────
# Architecture : 3665 articles → 512 → 256 → 256 (goulot) → 256 → 512 → 3665
# Entraîné à reconstruire le vecteur binaire d'achats de l'utilisateur (débruitage).

AE_EPOCHS      = 100     # Nombre maximal d'époques (l'arrêt anticipé se déclenche avant)
AE_BATCH_SIZE  = 64      # Nombre d'utilisateurs par mise à jour du gradient
AE_LR          = 0.001   # Taux d'apprentissage Adam
AE_DROPOUT     = 0.1     # Probabilité de dropout après chaque couche Dense encodeur/décodeur
AE_PATIENCE    = 15      # Patience de l'arrêt anticipé (stoppe si val_loss ne s'améliore plus)
AE_POS_WEIGHT  = 15      # Poids des articles achetés dans la perte BCE (gère le déséquilibre)
                          # Inférieur à 40 (ancienne valeur) — réduit les faux positifs
AE_NOISE       = 0.15    # Écart-type du bruit gaussien appliqué aux entrées pendant l'entraînement
                          # Inférieur à 0.3 (ancienne valeur) — corruption plus douce sur vecteurs creux

# ─────────────────────────────────────────────
# HYPERPARAMÈTRES NCF (FILTRAGE COLLABORATIF NEURONAL)
# ─────────────────────────────────────────────
# Architecture : modèle NeuMF double chemin
#   Chemin GMF : embed_user × embed_article (produit élément par élément)
#   Chemin MLP : concat(embed_user, embed_article) → 512 → 256 → 128 → 64
#   Sortie : Dense(1, sigmoïde)

NCF_EPOCHS     = 50      # Nombre maximal d'époques (était 25 — plus d'itérations améliorent le classement)
NCF_BATCH_SIZE = 2048    # Grande taille de lot adaptée à l'entraînement par paires
NCF_LR         = 0.001   # Taux d'apprentissage Adam
NCF_EMBED_DIM  = 128     # Dimension des embeddings utilisateur et article
NCF_NEG_RATIO  = 5       # Échantillons négatifs par paire positive (était 2)
                          # Utilise un échantillonnage pondéré par popularité (négatifs difficiles)

# ─────────────────────────────────────────────
# HYPERPARAMÈTRES LSTM
# ─────────────────────────────────────────────
# Architecture : Embedding → LSTM(256) → LSTM(128) → Dense(128) → Dense(n_articles, softmax)
# Entraîné à prédire le prochain article dans une séquence d'achats.

LSTM_EPOCHS     = 40     # Nombre maximal d'époques
LSTM_BATCH_SIZE = 1024   # Séquences par mise à jour du gradient
LSTM_LR         = 0.001  # Taux d'apprentissage Adam
LSTM_SEQ_LEN    = 10     # Nombre d'articles passés utilisés comme contexte (taille de fenêtre)
LSTM_UNITS      = 256    # Nombre d'unités dans la première couche LSTM (était 128)

# ─────────────────────────────────────────────
# POIDS DE L'ENSEMBLE
# ─────────────────────────────────────────────
# Score final = moyenne pondérée des scores normalisés de chaque modèle.
# Le LSTM domine car il atteint la meilleure précision (~30 %).
# Le NCF apporte le signal de filtrage collaboratif (~17 %).
# L'AE apporte le profil de goût général (~2–5 %).
# La somme des poids doit être égale à 1.0.

ENSEMBLE_WEIGHT_AE   = 0.10
ENSEMBLE_WEIGHT_NCF  = 0.35
ENSEMBLE_WEIGHT_LSTM = 0.55

# ─────────────────────────────────────────────
# ÉVALUATION
# ─────────────────────────────────────────────

# Valeurs de K pour les métriques Précision@K, Rappel@K et NDCG@K
TOP_K_VALUES = [5, 10]

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────

API_HOST            = "0.0.0.0"  # Écoute sur toutes les interfaces réseau
API_PORT            = 8000        # Port exposé par FastAPI / uvicorn
MAX_RECOMMENDATIONS = 10          # Nombre maximum d'articles retournés par appel /recommend
