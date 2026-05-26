"""
Model Evaluation Module.
Computes Precision@K, Recall@K, NDCG@K for all models.
"""
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import TOP_K_VALUES, SAVED_MODELS_DIR, SEED

np.random.seed(SEED)


def precision_at_k(recommended, relevant, k):
    """Precision@K: fraction of recommended items that are relevant."""
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(relevant))
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended, relevant, k):
    """Recall@K: fraction of relevant items that are recommended."""
    rec_k = recommended[:k]
    hits = len(set(rec_k) & set(relevant))
    return hits / len(relevant) if len(relevant) > 0 else 0.0


def ndcg_at_k(recommended, relevant, k):
    """NDCG@K: normalized discounted cumulative gain."""
    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)

    # Ideal DCG
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(model_name, score_fn, test_users, customer_item_matrix,
                   train_customer_item_matrix, mappings, k_values=TOP_K_VALUES):
    """Evaluate a model across test users."""
    print(f"Evaluating {model_name}...")
    metrics = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in k_values}
    evaluated = 0

    for uid in test_users:
        if uid not in customer_item_matrix.index:
            continue

        # Ground truth: items in test but not in train
        test_items = set(customer_item_matrix.loc[uid][customer_item_matrix.loc[uid] == 1].index)
        if uid in train_customer_item_matrix.index:
            train_items = set(train_customer_item_matrix.loc[uid][train_customer_item_matrix.loc[uid] == 1].index)
        else:
            train_items = set()

        relevant = test_items - train_items
        if len(relevant) == 0:
            continue

        # Get scores and rank
        try:
            scores = score_fn(uid)
        except Exception:
            continue

        # Filter out training items
        item_scores = []
        for item_code, score in scores.items():
            if item_code not in train_items:
                item_scores.append((item_code, score))

        item_scores.sort(key=lambda x: x[1], reverse=True)
        recommended = [item for item, _ in item_scores]

        for k in k_values:
            metrics[k]['precision'].append(precision_at_k(recommended, relevant, k))
            metrics[k]['recall'].append(recall_at_k(recommended, relevant, k))
            metrics[k]['ndcg'].append(ndcg_at_k(recommended, relevant, k))

        evaluated += 1

    results = {}
    for k in k_values:
        results[k] = {
            'precision': np.mean(metrics[k]['precision']) if metrics[k]['precision'] else 0,
            'recall': np.mean(metrics[k]['recall']) if metrics[k]['recall'] else 0,
            'ndcg': np.mean(metrics[k]['ndcg']) if metrics[k]['ndcg'] else 0,
        }
    print(f"  Evaluated on {evaluated} users")
    return results


def build_comparison_table(all_results):
    """Build and save comparison table."""
    rows = []
    for model_name, results in all_results.items():
        for k, metrics in results.items():
            rows.append({
                'Model': model_name,
                'K': k,
                'Precision@K': f"{metrics['precision']:.4f}",
                'Recall@K': f"{metrics['recall']:.4f}",
                'NDCG@K': f"{metrics['ndcg']:.4f}",
            })

    table = pd.DataFrame(rows)
    save_path = os.path.join(SAVED_MODELS_DIR, 'evaluation_results.csv')
    table.to_csv(save_path, index=False)
    print(f"\nComparison table saved to {save_path}")
    print(table.to_string(index=False))
    return table


def plot_comparison(all_results):
    """Plot comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metric_names = ['precision', 'recall', 'ndcg']
    titles = ['Precision@K', 'Recall@K', 'NDCG@K']

    for ax, metric, title in zip(axes, metric_names, titles):
        k = 10
        models = []
        values = []
        for model_name, results in all_results.items():
            if k in results:
                models.append(model_name)
                values.append(results[k][metric])

        bars = ax.bar(models, values, color=['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0'])
        ax.set_title(f'{title} (K=10)')
        ax.set_ylabel(title)
        ax.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(SAVED_MODELS_DIR, 'model_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to {save_path}")
