"""
Market Basket Analysis using Apriori algorithm.
Generates association rules and visualizations.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from config import MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT, RULES_CSV_PATH, SAVED_MODELS_DIR, SEED
import os

np.random.seed(SEED)


def generate_rules(basket_matrix, min_support=MIN_SUPPORT):
    """Run Apriori and extract association rules."""
    print("Running Apriori algorithm...")
    # For large datasets, limit columns to top items to keep memory manageable
    item_freq = basket_matrix.sum().sort_values(ascending=False)
    top_items = item_freq.head(200).index
    basket_subset = basket_matrix[top_items]
    print(f"  Using top {len(top_items)} items for Apriori")

    # Use slightly higher support for tractability
    effective_support = max(min_support, 0.02)
    frequent_itemsets = apriori(basket_subset, min_support=effective_support, use_colnames=True)
    print(f"  Frequent itemsets found: {len(frequent_itemsets)}")

    if len(frequent_itemsets) == 0:
        print("  No frequent itemsets found. Try lowering min_support.")
        return pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
    print(f"  Total rules generated: {len(rules)}")

    # Filter by lift and confidence
    filtered = rules[(rules['lift'] > MIN_LIFT) & (rules['confidence'] > MIN_CONFIDENCE)]
    print(f"  Filtered rules (lift>{MIN_LIFT}, conf>{MIN_CONFIDENCE}): {len(filtered)}")

    return filtered.sort_values('lift', ascending=False).reset_index(drop=True)


def save_rules(rules, path=RULES_CSV_PATH):
    """Save rules to CSV for Java backend."""
    if len(rules) == 0:
        print("  No rules to save.")
        return
    # Convert frozensets to strings for CSV
    rules_save = rules.copy()
    rules_save['antecedents'] = rules_save['antecedents'].apply(lambda x: ', '.join(str(i) for i in x))
    rules_save['consequents'] = rules_save['consequents'].apply(lambda x: ', '.join(str(i) for i in x))
    rules_save.to_csv(path, index=False)
    print(f"  Saved {len(rules_save)} rules to {path}")


def plot_rules_heatmap(rules, top_n=20):
    """Plot heatmap of top association rules (confidence x lift)."""
    if len(rules) == 0:
        print("  No rules to plot.")
        return

    top = rules.head(top_n).copy()
    top['rule'] = top.apply(
        lambda r: f"{', '.join(str(i) for i in list(r['antecedents'])[:2])} → {', '.join(str(i) for i in list(r['consequents'])[:2])}",
        axis=1
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_data = top[['rule', 'confidence', 'lift']].set_index('rule')
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    ax.set_title('Top Association Rules: Confidence & Lift')
    plt.tight_layout()
    save_path = os.path.join(SAVED_MODELS_DIR, 'apriori_heatmap.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Heatmap saved to {save_path}")


def run_apriori(basket_matrix):
    """Full Apriori pipeline."""
    rules = generate_rules(basket_matrix)
    save_rules(rules)
    plot_rules_heatmap(rules)
    return rules


if __name__ == '__main__':
    from preprocessing import run_preprocessing
    _, _, basket_matrix, _, _ = run_preprocessing()
    rules = run_apriori(basket_matrix)
    if len(rules) > 0:
        print(f"\nSample rules:")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))
