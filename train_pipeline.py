"""
Full Training Pipeline.
Runs preprocessing → Apriori → AutoEncoder → NCF → LSTM → Evaluation.
"""
import sys
import os
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
_orig_showwarning = warnings.showwarning
def _showwarning(msg, cat, fname, lineno, *args, **kwargs):
    if issubclass(cat, (DeprecationWarning, FutureWarning)):
        return
    _orig_showwarning(msg, cat, fname, lineno, *args, **kwargs)
warnings.showwarning = _showwarning

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from config import SEED, ENSEMBLE_WEIGHT_AE, ENSEMBLE_WEIGHT_NCF, ENSEMBLE_WEIGHT_LSTM

# Set all seeds
np.random.seed(SEED)
tf.random.set_seed(SEED)
import random
random.seed(SEED)


def main():
    print("=" * 60)
    print("AI RECOMMENDATION MODULE - FULL PIPELINE")
    print("=" * 60)

    # 1. Preprocessing
    print("\n[1/6] PREPROCESSING")
    print("-" * 40)
    from preprocessing import run_preprocessing
    df, customer_item_matrix, basket_matrix, sequences, mappings = run_preprocessing()

    # 2. Association Rules
    print("\n[2/6] ASSOCIATION RULES (APRIORI)")
    print("-" * 40)
    from apriori_rules import run_apriori
    rules = run_apriori(basket_matrix)

    # 3. Data Split
    print("\n[3/6] DATA SPLIT")
    print("-" * 40)
    all_users = list(customer_item_matrix.index)
    np.random.shuffle(all_users)

    n = len(all_users)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_users = all_users[:n_train]
    val_users = all_users[n_train:n_train + n_val]
    test_users = all_users[n_train + n_val:]
    print(f"  Train: {len(train_users)}, Val: {len(val_users)}, Test: {len(test_users)}")

    # Build train/val/test matrices for AutoEncoder
    matrix_values = customer_item_matrix.values.astype(np.float32)
    user_idx_map = {u: i for i, u in enumerate(customer_item_matrix.index)}

    train_indices = [user_idx_map[u] for u in train_users if u in user_idx_map]
    val_indices = [user_idx_map[u] for u in val_users if u in user_idx_map]
    test_indices = [user_idx_map[u] for u in test_users if u in user_idx_map]

    train_matrix = matrix_values[train_indices]
    val_matrix = matrix_values[val_indices]

    # 4. Train AutoEncoder
    print("\n[4/6] AUTOENCODER TRAINING")
    print("-" * 40)
    from model_autoencoder import train_autoencoder
    ae_model, ae_history = train_autoencoder(train_matrix, val_matrix, mappings['n_items'])

    # 5. Train NCF
    print("\n[5/6] NCF TRAINING")
    print("-" * 40)
    from model_ncf import train_ncf
    ncf_model, ncf_history = train_ncf(customer_item_matrix, mappings, train_users)

    # 6. Train LSTM
    print("\n[6/6] LSTM TRAINING")
    print("-" * 40)
    from model_lstm import train_lstm
    lstm_model, lstm_history = train_lstm(sequences, mappings, train_users)

    # 7. Evaluation
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    from evaluate import evaluate_model, build_comparison_table, plot_comparison
    from inference import RecommendationEngine

    engine = RecommendationEngine()
    engine.ae_model = ae_model
    engine.ncf_model = ncf_model
    engine.lstm_model = lstm_model
    engine.mappings = mappings
    engine.customer_vectors = customer_item_matrix

    # Build train matrix for filtering
    train_cim = customer_item_matrix.loc[customer_item_matrix.index.isin(train_users)]

    all_results = {}

    # Evaluate AutoEncoder
    def ae_score_fn(uid):
        vec = engine._get_user_vector(uid)
        scores = engine._ae_scores(vec)
        return {mappings['idx2item'][i]: float(scores[i]) for i in range(mappings['n_items'])}

    ae_results = evaluate_model('AutoEncoder', ae_score_fn, test_users,
                                customer_item_matrix, train_cim, mappings)
    all_results['AutoEncoder'] = ae_results

    # Evaluate NCF
    def ncf_score_fn(uid):
        scores = engine._ncf_scores(uid)
        return {mappings['idx2item'][i]: float(scores[i]) for i in range(mappings['n_items'])}

    ncf_results = evaluate_model('NCF', ncf_score_fn, test_users,
                                  customer_item_matrix, train_cim, mappings)
    all_results['NCF'] = ncf_results

    # Evaluate LSTM
    def lstm_score_fn(uid):
        if uid not in customer_item_matrix.index:
            return {mappings['idx2item'][i]: 0.0 for i in range(mappings['n_items'])}
        items = df[df['CustomerID'] == uid].sort_values('InvoiceDate')['StockCode'].tolist()
        scores = engine._lstm_scores(items)
        return {mappings['idx2item'][i]: float(scores[i]) for i in range(mappings['n_items'])}

    lstm_results = evaluate_model('LSTM', lstm_score_fn, test_users,
                                   customer_item_matrix, train_cim, mappings)
    all_results['LSTM'] = lstm_results

    def _normalize(sc):
        mn, mx = sc.min(), sc.max()
        return (sc - mn) / (mx - mn) if mx - mn > 1e-8 else sc

    # Evaluate Ensemble — weighted by known model quality: LSTM > NCF >> AE
    def ensemble_score_fn(uid):
        items = df[df['CustomerID'] == uid].sort_values('InvoiceDate')['StockCode'].tolist()
        vec = engine._get_user_vector(uid)
        n_items = mappings['n_items']

        scores = np.zeros(n_items, dtype=np.float32)
        w_sum = 0.0

        ae_sc = engine._ae_scores(vec)
        if np.any(ae_sc > 0):
            scores += ENSEMBLE_WEIGHT_AE * _normalize(ae_sc)
            w_sum += ENSEMBLE_WEIGHT_AE

        ncf_sc = engine._ncf_scores(uid)
        if np.any(ncf_sc > 0):
            scores += ENSEMBLE_WEIGHT_NCF * _normalize(ncf_sc)
            w_sum += ENSEMBLE_WEIGHT_NCF

        lstm_sc = engine._lstm_scores(items)
        if np.any(lstm_sc > 0):
            scores += ENSEMBLE_WEIGHT_LSTM * _normalize(lstm_sc)
            w_sum += ENSEMBLE_WEIGHT_LSTM

        if w_sum > 0:
            scores /= w_sum

        return {mappings['idx2item'][i]: float(scores[i]) for i in range(n_items)}

    ens_results = evaluate_model('Ensemble', ensemble_score_fn, test_users,
                                  customer_item_matrix, train_cim, mappings)
    all_results['Ensemble'] = ens_results

    # Comparison table and plots
    table = build_comparison_table(all_results)
    plot_comparison(all_results)

    # Sample predictions
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    sample_users = test_users[:3]
    for uid in sample_users:
        items = df[df['CustomerID'] == uid].sort_values('InvoiceDate')['StockCode'].tolist()[:5]
        recs, strategy = engine.recommend(uid, items, top_n=5)
        print(f"\nUser: {uid}")
        print(f"  Input items: {items}")
        print(f"  Strategy: {strategy}")
        for r in recs:
            print(f"    → {r['product_id']}: {r['description']} (score: {r['score']})")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
