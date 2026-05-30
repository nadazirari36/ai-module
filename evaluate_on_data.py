"""
evaluate_on_data.py — Évaluation des modèles pré-entraînés sur un nouveau fichier de données.
Charge les modèles depuis saved_models/ (sans réentraînement) et les évalue
sur n'importe quel fichier Excel au même format que Online_Retail.xlsx.

Utilisation :
    python evaluate_on_data.py                        # utilise training.xlsx par défaut
    python evaluate_on_data.py data/autre_fichier.xlsx

Le fichier doit avoir les mêmes colonnes qu'Online_Retail.xlsx :
    InvoiceNo, StockCode, Description, Quantity, InvoiceDate, CustomerID, Country

Gestion des articles / utilisateurs inconnus :
  - AE     : les articles inconnus sont ignorés (vecteur reste à 0)
  - NCF    : les utilisateurs inconnus (non dans user2idx) utilisent le cold-start
  - LSTM   : les articles inconnus sont ignorés dans la séquence
"""
import os
import sys
import pickle
import warnings

# Supprime les avertissements pour un affichage plus propre
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from tensorflow import keras

from config import (SAVED_MODELS_DIR, MAPPINGS_PATH, POPULAR_PATH,
                    SEED, TRAIN_RATIO, VAL_RATIO, LSTM_SEQ_LEN,
                    ENSEMBLE_WEIGHT_AE, ENSEMBLE_WEIGHT_NCF, ENSEMBLE_WEIGHT_LSTM,
                    TOP_K_VALUES)
from evaluate import evaluate_model, build_comparison_table, plot_comparison

np.random.seed(SEED)

# ── Chemin vers le fichier à évaluer ─────────────────────────────────────────
# Utilise l'argument en ligne de commande s'il est fourni, sinon training.xlsx
DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "data", "training.xlsx")
DATA_FILE    = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean(path):
    """
    Charge et nettoie le fichier Excel cible.
    Applique les mêmes étapes de nettoyage que preprocessing.py :
      - Supprime les lignes sans CustomerID
      - Supprime les commandes annulées (InvoiceNo commençant par 'C')
      - Supprime les quantités nulles ou négatives
      - Convertit les types de colonnes
    """
    print(f"Chargement de {os.path.basename(path)} ...")
    df = pd.read_excel(path, engine='openpyxl')
    print(f"  Lignes brutes : {len(df)}")

    df = df.dropna(subset=['CustomerID'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Description'] = df['Description'].fillna(df['StockCode'].astype(str))

    print(f"  Lignes nettoyées : {len(df)}"
          f"  |  clients : {df['CustomerID'].nunique()}"
          f"  |  produits : {df['StockCode'].nunique()}")
    return df


def build_cim(df, all_items):
    """
    Construit la matrice binaire client-article restreinte aux articles connus du modèle.
    Les articles du fichier cible non présents dans le catalogue d'entraînement sont ignorés.
    La matrice est réindexée pour avoir les mêmes colonnes que le catalogue d'entraînement.
    """
    interactions = df.groupby(['CustomerID', 'StockCode']).size().reset_index(name='c')

    # Filtre pour ne garder que les articles connus du modèle entraîné
    interactions = interactions[interactions['StockCode'].isin(all_items)]
    interactions['v'] = 1

    if interactions.empty:
        return pd.DataFrame()

    pivot = interactions.pivot_table(
        index='CustomerID', columns='StockCode', values='v', fill_value=0
    )

    # Réindexe pour avoir les colonnes triées dans le même ordre que item2idx
    pivot = pivot.reindex(columns=sorted(all_items, key=str), fill_value=0)
    return pivot


def ae_scores(ae_model, user_vector):
    """Calcule les scores de reconstruction de l'AE pour un vecteur utilisateur."""
    inp = user_vector.reshape(1, -1)  # Ajoute la dimension batch
    return ae_model.predict(inp, verbose=0)[0]


def ncf_scores(ncf_model, user_id, user2idx, n_items):
    """
    Calcule les scores NCF pour tous les articles pour un utilisateur.
    Retourne des zéros si l'utilisateur n'est pas dans user2idx (inconnu).
    """
    if user_id not in user2idx:
        return np.zeros(n_items)
    uidx     = user2idx[user_id]
    u_arr    = np.full(n_items, uidx)
    i_arr    = np.arange(n_items)
    return ncf_model.predict([u_arr, i_arr], verbose=0, batch_size=1024).flatten()


def lstm_scores(lstm_model, item_list, item2idx, n_items, seq_len=LSTM_SEQ_LEN):
    """
    Calcule les scores LSTM (probabilité du prochain article) pour une séquence.
    Les articles inconnus sont ignorés. La séquence est paddée à gauche si nécessaire.
    """
    if lstm_model is None or not item_list:
        return np.zeros(n_items)

    # Convertit les codes en indices (ignore les articles inconnus)
    indices = [item2idx[it] for it in item_list if it in item2idx]
    if not indices:
        return np.zeros(n_items)

    # Garde les seq_len derniers articles et padde à gauche si nécessaire
    indices = indices[-seq_len:]
    if len(indices) < seq_len:
        indices = [0] * (seq_len - len(indices)) + indices

    return lstm_model.predict(np.array(indices).reshape(1, -1), verbose=0)[0]


def normalize(sc):
    """Normalise un vecteur de scores dans [0, 1] par min-max scaling."""
    mn, mx = sc.min(), sc.max()
    return (sc - mn) / (mx - mn) if mx - mn > 1e-8 else sc


# ─────────────────────────────────────────────────────────────────────────────
# PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── Étape 1 : Chargement des modèles et mappings ──────────────────────────
    print("\n[1] Chargement des modèles et mappings...")

    # Charge les mappings sauvegardés pendant l'entraînement
    with open(MAPPINGS_PATH, 'rb') as f:
        mappings = pickle.load(f)

    item2idx  = mappings['item2idx']   # article_code → indice entier
    idx2item  = mappings['idx2item']   # indice entier → article_code
    user2idx  = mappings['user2idx']   # customer_id → indice entier
    n_items   = mappings['n_items']    # nombre total d'articles distincts
    all_items = set(item2idx.keys())   # ensemble des articles connus du modèle

    # Chargement des fichiers modèles (si disponibles)
    ae_path   = os.path.join(SAVED_MODELS_DIR, 'autoencoder.keras')
    ncf_path  = os.path.join(SAVED_MODELS_DIR, 'ncf.keras')
    lstm_path = os.path.join(SAVED_MODELS_DIR, 'lstm.keras')

    ae_model   = keras.models.load_model(ae_path,   compile=False) if os.path.exists(ae_path)   else None
    ncf_model  = keras.models.load_model(ncf_path,  compile=False) if os.path.exists(ncf_path)  else None
    lstm_model = keras.models.load_model(lstm_path, compile=False) if os.path.exists(lstm_path) else None

    for name, m in [('AE', ae_model), ('NCF', ncf_model), ('LSTM', lstm_model)]:
        print(f"  {name}: {'chargé' if m is not None else 'NON TROUVE (ignoré)'}")

    # ── Étape 2 : Chargement et nettoyage du fichier cible ───────────────────
    print(f"\n[2] Prétraitement de {os.path.basename(DATA_FILE)}...")
    df = load_and_clean(DATA_FILE)

    # ── Étape 3 : Construction de la matrice client-article ───────────────────
    print("\n[3] Construction de la matrice client-article (articles connus uniquement)...")
    cim = build_cim(df, all_items)

    if cim.empty:
        print("  ERREUR : aucun article du fichier cible n'est connu du modèle.")
        return

    # Affiche le taux de couverture du catalogue d'entraînement
    known_item_frac = cim.shape[1] / len(all_items)
    print(f"  Forme de la matrice : {cim.shape}")
    print(f"  Couverture du catalogue : {known_item_frac:.1%} des articles entraînés présents dans ce fichier")

    # ── Étape 4 : Découpage des utilisateurs ──────────────────────────────────
    # Même ratios que train_pipeline.py pour des métriques comparables
    print("\n[4] Découpage des utilisateurs...")
    all_users   = list(cim.index)
    np.random.shuffle(all_users)
    n           = len(all_users)
    n_train     = int(n * TRAIN_RATIO)
    n_val       = int(n * VAL_RATIO)
    train_users = all_users[:n_train]
    test_users  = all_users[n_train + n_val:]
    print(f"  Train : {len(train_users)}  Val : {n_val}  Test : {len(test_users)}")

    # Matrice d'entraînement pour déterminer les articles "déjà vus"
    train_cim    = cim.loc[cim.index.isin(train_users)]

    # Ensemble des utilisateurs connus dans les mappings (leurs embeddings NCF sont entraînés)
    ncf_warm_set = set(user2idx.keys())

    # Helper : cold-start NCF via moyenne des embeddings articles
    def _ncf_cold_start(item_list):
        """
        Calcule un score NCF approché pour un utilisateur dont l'embedding n'est pas entraîné.
        Utilise la moyenne des embeddings GMF des articles achetés comme vecteur utilisateur inféré.
        """
        if ncf_model is None:
            return np.zeros(n_items)
        purchased_idx = [item2idx[it] for it in item_list if it in item2idx]
        if not purchased_idx:
            return np.zeros(n_items)
        try:
            embs = ncf_model.get_layer('item_gmf').embeddings.numpy()  # Embeddings articles GMF
        except Exception:
            return np.zeros(n_items)

        # Embedding utilisateur inféré = moyenne des embeddings des articles achetés
        user_emb = np.mean(embs[purchased_idx], axis=0)
        raw      = embs @ user_emb  # Produit scalaire avec tous les articles
        return 1.0 / (1.0 + np.exp(-raw))  # Sigmoïde → [0, 1]

    # ── Étape 5 : Définition des fonctions de scoring ────────────────────────

    def ae_score_fn(uid):
        """Score AE : reconstruit le vecteur d'achats via l'AutoEncodeur."""
        # Récupère le vecteur d'achats de l'utilisateur
        vec = (cim.loc[uid].values.astype(np.float32)
               if uid in cim.index else np.zeros(n_items, dtype=np.float32))

        # Construit le vecteur complet aligné sur le catalogue d'entraînement (n_items dimensions)
        full_vec = np.zeros(n_items, dtype=np.float32)
        for col, val in zip(cim.columns, vec):
            if col in item2idx:
                full_vec[item2idx[col]] = val

        scores = ae_scores(ae_model, full_vec)
        return {idx2item[i]: float(scores[i]) for i in range(n_items)}

    def ncf_score_fn(uid):
        """
        Score NCF : utilise l'embedding entraîné si l'utilisateur est connu,
        sinon utilise le cold-start par moyenne des embeddings articles.
        """
        if uid in ncf_warm_set:
            # Utilisateur connu → embedding entraîné
            scores = ncf_scores(ncf_model, uid, user2idx, n_items)
        else:
            # Utilisateur inconnu → cold-start par embeddings articles
            items  = df[df['CustomerID'] == uid].sort_values('InvoiceDate')['StockCode'].tolist()
            scores = _ncf_cold_start(items)
        return {idx2item[i]: float(scores[i]) for i in range(n_items)}

    def lstm_score_fn(uid):
        """Score LSTM : prédit le prochain article à partir de l'historique complet."""
        items  = (df[df['CustomerID'] == uid]
                  .sort_values('InvoiceDate')['StockCode'].tolist())
        scores = lstm_scores(lstm_model, items, item2idx, n_items)
        return {idx2item[i]: float(scores[i]) for i in range(n_items)}

    def ensemble_score_fn(uid):
        """Score ensemble : moyenne pondérée normalisée des trois modèles."""
        items   = (df[df['CustomerID'] == uid]
                   .sort_values('InvoiceDate')['StockCode'].tolist())
        vec     = (cim.loc[uid].values.astype(np.float32)
                   if uid in cim.index else np.zeros(n_items, dtype=np.float32))

        # Vecteur complet aligné sur le catalogue d'entraînement
        full_vec = np.zeros(n_items, dtype=np.float32)
        for col, val in zip(cim.columns, vec):
            if col in item2idx:
                full_vec[item2idx[col]] = val

        combined = np.zeros(n_items, dtype=np.float32)
        w_sum    = 0.0

        # Contribution AutoEncodeur
        if ae_model is not None:
            sc = ae_scores(ae_model, full_vec)
            if np.any(sc > 0):
                combined += ENSEMBLE_WEIGHT_AE * normalize(sc)
                w_sum    += ENSEMBLE_WEIGHT_AE

        # Contribution NCF (warm ou cold-start)
        if ncf_model is not None:
            sc = (ncf_scores(ncf_model, uid, user2idx, n_items) if uid in ncf_warm_set
                  else _ncf_cold_start(items))
            if np.any(sc > 0):
                combined += ENSEMBLE_WEIGHT_NCF * normalize(sc)
                w_sum    += ENSEMBLE_WEIGHT_NCF

        # Contribution LSTM
        if lstm_model is not None:
            sc = lstm_scores(lstm_model, items, item2idx, n_items)
            if np.any(sc > 0):
                combined += ENSEMBLE_WEIGHT_LSTM * normalize(sc)
                w_sum    += ENSEMBLE_WEIGHT_LSTM

        if w_sum > 0:
            combined /= w_sum

        return {idx2item[i]: float(combined[i]) for i in range(n_items)}

    # ── Étape 6 : Évaluation ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ÉVALUATION  —  " + os.path.basename(DATA_FILE))
    print("=" * 60)
    all_results = {}

    if ae_model is not None:
        all_results['AutoEncoder'] = evaluate_model(
            'AutoEncoder', ae_score_fn, test_users, cim, train_cim, mappings)

    if ncf_model is not None:
        all_results['NCF'] = evaluate_model(
            'NCF', ncf_score_fn, test_users, cim, train_cim, mappings)

    if lstm_model is not None:
        all_results['LSTM'] = evaluate_model(
            'LSTM', lstm_score_fn, test_users, cim, train_cim, mappings)

    # L'ensemble n'est calculé que si au moins deux modèles sont disponibles
    if len(all_results) > 1:
        all_results['Ensemble'] = evaluate_model(
            'Ensemble', ensemble_score_fn, test_users, cim, train_cim, mappings)

    # ── Étape 7 : Sauvegarde des résultats ───────────────────────────────────
    build_comparison_table(all_results)   # Sauvegarde CSV + affiche le tableau
    plot_comparison(all_results)           # Sauvegarde le graphique en barres
    print(f"\nRésultats sauvegardés dans {SAVED_MODELS_DIR}/")


# ─── Point d'entrée ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
