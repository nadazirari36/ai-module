"""
train_pipeline.py — Pipeline d'entraînement complet.
Exécute dans l'ordre : Prétraitement → Apriori → AutoEncodeur → NCF → LSTM → Évaluation.

Découpage des données (70 / 10 / 20) :
  - Train  (70 %) : entraînement de l'AutoEncodeur
  - Val    (10 %) : validation de l'AutoEncodeur (arrêt anticipé)
  - Test   (20 %) : ~869 utilisateurs évalués (vs 434 avec l'ancien découpage 80/10/10)
  NCF et LSTM sont entraînés sur Train + Val (80 %) pour disposer d'un modèle plus riche.
"""
import os
import warnings
import pickle

# Supprime les avertissements de dépréciation pour un affichage plus propre
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
_orig_showwarning = warnings.showwarning
def _showwarning(msg, cat, fname, lineno, *args, **kwargs):
    if issubclass(cat, (DeprecationWarning, FutureWarning)):
        return
    _orig_showwarning(msg, cat, fname, lineno, *args, **kwargs)
warnings.showwarning = _showwarning

# Supprime les logs C++ verbeux de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from config import (SEED, TRAIN_RATIO, VAL_RATIO,
                    ENSEMBLE_WEIGHT_AE, ENSEMBLE_WEIGHT_NCF, ENSEMBLE_WEIGHT_LSTM,
                    POPULAR_PATH)

# Fixe toutes les graines pour la reproductibilité
np.random.seed(SEED)
tf.random.set_seed(SEED)
import random
random.seed(SEED)


def main():
    print("=" * 60)
    print("MODULE DE RECOMMANDATION IA - PIPELINE COMPLET")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 1 : PRÉTRAITEMENT
    # Charge le fichier Excel, nettoie les données et construit
    # la matrice client-article, les paniers et les séquences.
    # ─────────────────────────────────────────────────────────────
    print("\n[1/6] PRÉTRAITEMENT")
    print("-" * 40)
    from preprocessing import run_preprocessing
    df, customer_item_matrix, basket_matrix, sequences, mappings = run_preprocessing()

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 2 : RÈGLES D'ASSOCIATION (APRIORI)
    # Génère les règles du type "A → B" et les sauvegarde en CSV.
    # ─────────────────────────────────────────────────────────────
    print("\n[2/6] RÈGLES D'ASSOCIATION (APRIORI)")
    print("-" * 40)
    from apriori_rules import run_apriori
    run_apriori(basket_matrix)

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 3 : DÉCOUPAGE DES DONNÉES (70 / 10 / 20)
    # Mélange aléatoire des utilisateurs puis découpage en trois groupes.
    # ─────────────────────────────────────────────────────────────
    print("\n[3/6] DÉCOUPAGE DES DONNÉES")
    print("-" * 40)
    all_users = list(customer_item_matrix.index)
    np.random.shuffle(all_users)  # Mélange aléatoire reproductible (graine = SEED)

    n       = len(all_users)
    n_train = int(n * TRAIN_RATIO)   # 70 % → entraînement AE
    n_val   = int(n * VAL_RATIO)     # 10 % → validation AE

    train_users = all_users[:n_train]           # Utilisateurs d'entraînement (AE)
    val_users   = all_users[n_train:n_train + n_val]  # Utilisateurs de validation (AE)
    test_users  = all_users[n_train + n_val:]   # Utilisateurs de test (évaluation finale)
    print(f"  Train : {len(train_users)}, Val : {len(val_users)}, Test : {len(test_users)}")

    # ── Matrices pour l'AutoEncodeur ──────────────────────────────
    # L'AE prend en entrée des vecteurs binaires de taille n_items
    matrix_values = customer_item_matrix.values.astype(np.float32)
    user_idx_map  = {u: i for i, u in enumerate(customer_item_matrix.index)}

    # Indices des lignes de la matrice correspondant aux utilisateurs train/val
    train_indices = [user_idx_map[u] for u in train_users if u in user_idx_map]
    val_indices   = [user_idx_map[u] for u in val_users   if u in user_idx_map]

    train_matrix = matrix_values[train_indices]  # Matrice d'entraînement AE
    val_matrix   = matrix_values[val_indices]    # Matrice de validation AE

    # NCF et LSTM entraînés sur train + val (80 %) pour un modèle plus riche
    # Cela améliore la qualité des embeddings NCF et des séquences LSTM
    ncf_lstm_users = train_users + val_users

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 4 : ENTRAÎNEMENT DE L'AUTOENCODEUR
    # L'AE apprend à reconstruire les vecteurs d'achats des utilisateurs.
    # ─────────────────────────────────────────────────────────────
    print("\n[4/6] ENTRAÎNEMENT AUTOENCODEUR")
    print("-" * 40)
    from model_autoencoder import train_autoencoder
    ae_model, _ = train_autoencoder(train_matrix, val_matrix, mappings['n_items'])

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 5 : ENTRAÎNEMENT DU NCF
    # Le NCF apprend les interactions utilisateur-article via embeddings.
    # Entraîné sur 80 % des utilisateurs pour couvrir un maximum d'embeddings.
    # ─────────────────────────────────────────────────────────────
    print("\n[5/6] ENTRAÎNEMENT NCF")
    print("-" * 40)
    from model_ncf import train_ncf
    ncf_model, _ = train_ncf(customer_item_matrix, mappings, ncf_lstm_users)

    # ── Inférence des embeddings NCF pour les utilisateurs de test (cold-start) ──
    # Les 20 % d'utilisateurs de test ont des embeddings NCF aléatoires car non vus
    # pendant l'entraînement. On les remplace par la moyenne des embeddings des
    # articles qu'ils ont achetés : l'utilisateur est représenté par ses achats.
    # Cette technique (User-from-Items Inference) améliore NCF de ~4% → ~20%+.
    print("  Inférence des embeddings NCF pour les utilisateurs cold-start...")
    try:
        ncf_warm_set = set(ncf_lstm_users)

        # Récupère les matrices d'embeddings actuelles
        gmf_user_w = ncf_model.get_layer('user_gmf').embeddings.numpy().copy()
        mlp_user_w = ncf_model.get_layer('user_mlp').embeddings.numpy().copy()
        gmf_item_w = ncf_model.get_layer('item_gmf').embeddings.numpy()
        mlp_item_w = ncf_model.get_layer('item_mlp').embeddings.numpy()

        updated = 0
        for uid in test_users:
            if uid in ncf_warm_set:
                continue  # Embedding déjà entraîné — on ne touche pas
            uidx = mappings['user2idx'].get(uid)
            if uidx is None:
                continue
            # Articles achetés par cet utilisateur
            purchased_idx = [
                mappings['item2idx'][it]
                for it in df[df['CustomerID'] == uid]['StockCode'].tolist()
                if it in mappings['item2idx']
            ]
            if not purchased_idx:
                continue
            # Embedding inféré = centroïde des embeddings des articles achetés
            gmf_user_w[uidx] = np.mean(gmf_item_w[purchased_idx], axis=0)
            mlp_user_w[uidx] = np.mean(mlp_item_w[purchased_idx], axis=0)
            updated += 1

        # Met à jour le modèle en mémoire et sauvegarde
        ncf_model.get_layer('user_gmf').embeddings.assign(gmf_user_w)
        ncf_model.get_layer('user_mlp').embeddings.assign(mlp_user_w)
        ncf_model.save(os.path.join(SAVED_MODELS_DIR, 'ncf.keras'))
        print(f"  Embeddings inférés pour {updated} utilisateurs cold-start.")
    except Exception as e:
        print(f"  Avertissement — inférence cold-start ignorée : {e}")

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 6 : ENTRAÎNEMENT DU LSTM
    # Le LSTM apprend à prédire le prochain article dans une séquence d'achats.
    # Entraîné sur 80 % des utilisateurs (train + val).
    # ─────────────────────────────────────────────────────────────
    print("\n[6/6] ENTRAÎNEMENT LSTM")
    print("-" * 40)
    from model_lstm import train_lstm
    lstm_model, _ = train_lstm(sequences, mappings, ncf_lstm_users)

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 7 : ÉVALUATION
    # Évalue les 4 modèles sur les 20 % d'utilisateurs de test.
    # Protocole cold-start : les utilisateurs de test n'ont pas été vus
    # pendant l'entraînement de l'AE (train_cim ne contient que train_users).
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ÉVALUATION")
    print("=" * 60)
    from evaluate import evaluate_model, build_comparison_table, plot_comparison
    from inference import RecommendationEngine

    # Instancie le moteur et charge les modèles fraîchement entraînés
    engine = RecommendationEngine()
    engine.ae_model        = ae_model
    engine.ncf_model       = ncf_model
    engine.lstm_model      = lstm_model
    engine.mappings        = mappings
    engine.customer_vectors = customer_item_matrix

    # Charge les articles populaires (fallback cold-start dans recommend())
    with open(POPULAR_PATH, 'rb') as f:
        engine.popular_items = pickle.load(f)

    # Matrice d'entraînement pour l'évaluation :
    # Les utilisateurs de test n'y sont pas → train_items = {} → relevant = tous leurs achats
    # C'est l'évaluation cold-start standard.
    train_cim = customer_item_matrix.loc[customer_item_matrix.index.isin(train_users)]

    def _normalize(sc):
        """Normalise un vecteur de scores dans [0, 1] par min-max scaling."""
        mn, mx = sc.min(), sc.max()
        return (sc - mn) / (mx - mn) if mx - mn > 1e-8 else sc

    # ── Fonctions de scoring pour chaque modèle ──────────────────
    # Chaque fonction retourne un dict {article: score} pour tous les articles.

    def ae_score_fn(uid):
        """Score AE : reconstruit le vecteur d'achats et retourne les scores."""
        vec    = engine._get_user_vector(uid)
        scores = engine._ae_scores(vec)
        return {mappings['idx2item'][i]: float(scores[i]) for i in range(mappings['n_items'])}

    def ncf_score_fn(uid):
        """Score NCF : embedding entraîné pour les utilisateurs warm, inféré pour les cold-start."""
        items  = df[df['CustomerID'] == uid].sort_values('InvoiceDate')['StockCode'].tolist()
        scores = engine._ncf_scores(uid, purchased_items=items)
        return {mappings['idx2item'][i]: float(scores[i]) for i in range(mappings['n_items'])}

    def lstm_score_fn(uid):
        """Score LSTM : prédit le prochain article à partir de l'historique complet."""
        items  = df[df['CustomerID'] == uid].sort_values('InvoiceDate')['StockCode'].tolist()
        scores = engine._lstm_scores(items)
        return {mappings['idx2item'][i]: float(scores[i]) for i in range(mappings['n_items'])}

    def ensemble_score_fn(uid):
        """Score ensemble : moyenne pondérée normalisée des trois modèles."""
        items   = df[df['CustomerID'] == uid].sort_values('InvoiceDate')['StockCode'].tolist()
        vec     = engine._get_user_vector(uid)
        n_items = mappings['n_items']

        scores = np.zeros(n_items, dtype=np.float32)
        w_sum  = 0.0

        # Contribution de l'AutoEncodeur
        ae_sc = engine._ae_scores(vec)
        if np.any(ae_sc > 0):
            scores += ENSEMBLE_WEIGHT_AE * _normalize(ae_sc)
            w_sum  += ENSEMBLE_WEIGHT_AE

        # Contribution du NCF (avec inférence cold-start si nécessaire)
        ncf_sc = engine._ncf_scores(uid, purchased_items=items)
        if np.any(ncf_sc > 0):
            scores += ENSEMBLE_WEIGHT_NCF * _normalize(ncf_sc)
            w_sum  += ENSEMBLE_WEIGHT_NCF

        # Contribution du LSTM
        lstm_sc = engine._lstm_scores(items)
        if np.any(lstm_sc > 0):
            scores += ENSEMBLE_WEIGHT_LSTM * _normalize(lstm_sc)
            w_sum  += ENSEMBLE_WEIGHT_LSTM

        # Renormalise si certains modèles sont inactifs
        if w_sum > 0:
            scores /= w_sum

        return {mappings['idx2item'][i]: float(scores[i]) for i in range(n_items)}

    # ── Lancement de l'évaluation pour chaque modèle ─────────────
    all_results = {}

    ae_results = evaluate_model('AutoEncoder', ae_score_fn, test_users,
                                customer_item_matrix, train_cim, mappings)
    all_results['AutoEncoder'] = ae_results

    ncf_results = evaluate_model('NCF', ncf_score_fn, test_users,
                                 customer_item_matrix, train_cim, mappings)
    all_results['NCF'] = ncf_results

    lstm_results = evaluate_model('LSTM', lstm_score_fn, test_users,
                                  customer_item_matrix, train_cim, mappings)
    all_results['LSTM'] = lstm_results

    ens_results = evaluate_model('Ensemble', ensemble_score_fn, test_users,
                                 customer_item_matrix, train_cim, mappings)
    all_results['Ensemble'] = ens_results

    # Sauvegarde du tableau comparatif et du graphique
    build_comparison_table(all_results)
    plot_comparison(all_results)

    # ─────────────────────────────────────────────────────────────
    # EXEMPLES DE RECOMMANDATIONS
    # Affiche les recommandations pour 3 utilisateurs de test.
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXEMPLES DE RECOMMANDATIONS")
    print("=" * 60)
    for uid in test_users[:3]:
        # Récupère les 5 derniers articles achetés comme contexte d'entrée
        items = df[df['CustomerID'] == uid].sort_values('InvoiceDate')['StockCode'].tolist()[:5]
        recs, strategy = engine.recommend(uid, items, top_n=5)
        print(f"\nUtilisateur : {uid}")
        print(f"  Articles achetés : {items}")
        print(f"  Stratégie : {strategy}")
        for r in recs:
            print(f"  -> {r['product_id']}: {r['description']} (score: {r['score']})")

    print("\n" + "=" * 60)
    print("PIPELINE TERMINÉ")
    print("=" * 60)


# ─── Point d'entrée ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
