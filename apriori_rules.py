"""
apriori_rules.py — Analyse des paniers d'achat avec l'algorithme Apriori.
Génère des règles d'association du type : "les clients qui achètent A et B
achètent aussi C avec une probabilité X".
Ces règles sont sauvegardées en CSV pour être consommées par le backend Java.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sans affichage (serveur sans écran)
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from config import MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT, RULES_CSV_PATH, SAVED_MODELS_DIR, SEED
import os

np.random.seed(SEED)


def generate_rules(basket_matrix, min_support=MIN_SUPPORT):
    """
    Exécute l'algorithme Apriori et extrait les règles d'association.

    Étapes :
      1. Sélectionne les 200 articles les plus fréquents (mémoire limitée pour Apriori).
      2. Utilise un support minimum effectif d'au moins 2 % pour la faisabilité.
      3. Génère les ensembles fréquents (itemsets avec support >= min_support).
      4. Dérive les règles d'association à partir des ensembles fréquents.
      5. Filtre par lift et confiance pour ne garder que les règles significatives.

    Paramètres :
      basket_matrix : matrice booléenne (factures × articles) de preprocessing.py
      min_support   : fraction minimale de transactions contenant un itemset

    Retourne : DataFrame pandas avec les règles triées par lift décroissant.
    """
    print("Exécution de l'algorithme Apriori...")

    # Étape 1 : limite aux 200 articles les plus vendus pour la faisabilité mémoire
    # Apriori a une complexité exponentielle — trop d'articles = explosion combinatoire
    item_freq     = basket_matrix.sum().sort_values(ascending=False)
    top_items     = item_freq.head(200).index
    basket_subset = basket_matrix[top_items]
    print(f"  Utilisation des top {len(top_items)} articles pour Apriori")

    # Étape 2 : support effectif — au moins 2 % pour rester tractable
    effective_support = max(min_support, 0.02)

    # Étape 3 : génération des ensembles fréquents
    frequent_itemsets = apriori(basket_subset, min_support=effective_support, use_colnames=True)
    print(f"  Ensembles fréquents trouvés : {len(frequent_itemsets)}")

    if len(frequent_itemsets) == 0:
        print("  Aucun ensemble fréquent trouvé. Essayez de baisser min_support.")
        return pd.DataFrame()

    # Étape 4 : génération des règles (confiance minimum de 30 % pour inclure plus de règles)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
    print(f"  Règles totales générées : {len(rules)}")

    # Étape 5 : filtrage par lift et confiance selon la configuration
    # lift > MIN_LIFT (1.5)     : la règle est 1.5x plus probable que le hasard
    # confidence > MIN_CONFIDENCE (0.5) : au moins 50 % de probabilité conditionnelle
    filtered = rules[(rules['lift'] > MIN_LIFT) & (rules['confidence'] > MIN_CONFIDENCE)]
    print(f"  Règles filtrées (lift>{MIN_LIFT}, conf>{MIN_CONFIDENCE}) : {len(filtered)}")

    return filtered.sort_values('lift', ascending=False).reset_index(drop=True)


def save_rules(rules, path=RULES_CSV_PATH):
    """
    Sauvegarde les règles d'association en CSV.
    Les frozensets (antécédents et conséquents) sont convertis en chaînes de caractères
    pour la compatibilité CSV et le backend Java.
    """
    if len(rules) == 0:
        print("  Aucune règle à sauvegarder.")
        return

    rules_save = rules.copy()

    # Convertit frozenset({'A', 'B'}) → "A, B"
    rules_save['antecedents'] = rules_save['antecedents'].apply(
        lambda x: ', '.join(str(i) for i in x)
    )
    rules_save['consequents'] = rules_save['consequents'].apply(
        lambda x: ', '.join(str(i) for i in x)
    )

    rules_save.to_csv(path, index=False)
    print(f"  {len(rules_save)} règles sauvegardées dans {path}")


def plot_rules_heatmap(rules, top_n=20):
    """
    Trace une heatmap des top règles d'association (confiance × lift).
    L'axe Y = règle sous forme "antécédent → conséquent".
    Les deux métriques affichées sont :
      - Confiance : P(conséquent | antécédent)
      - Lift      : combien de fois plus probable que le hasard
    """
    if len(rules) == 0:
        print("  Aucune règle à visualiser.")
        return

    # Sélectionne les top_n règles (déjà triées par lift décroissant)
    top = rules.head(top_n).copy()

    # Crée une étiquette lisible pour chaque règle
    top['rule'] = top.apply(
        lambda r: (
            f"{', '.join(str(i) for i in list(r['antecedents'])[:2])}"
            f" -> "
            f"{', '.join(str(i) for i in list(r['consequents'])[:2])}"
        ),
        axis=1
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_data = top[['rule', 'confidence', 'lift']].set_index('rule')

    # Heatmap avec annotations numériques, palette rouge-jaune
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    ax.set_title('Top Règles d\'Association : Confiance & Lift')
    plt.tight_layout()

    save_path = os.path.join(SAVED_MODELS_DIR, 'apriori_heatmap.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Heatmap sauvegardée dans {save_path}")


def run_apriori(basket_matrix):
    """
    Lance le pipeline complet Apriori :
      1. Génère les règles d'association
      2. Sauvegarde en CSV
      3. Trace la heatmap de visualisation
    Retourne les règles sous forme de DataFrame.
    """
    rules = generate_rules(basket_matrix)
    save_rules(rules)
    plot_rules_heatmap(rules)
    return rules


# ─── Point d'entrée direct ───────────────────────────────────────────────────
if __name__ == '__main__':
    from preprocessing import run_preprocessing
    _, _, basket_matrix, _, _ = run_preprocessing()
    rules = run_apriori(basket_matrix)
    if len(rules) > 0:
        print(f"\nExemple de règles :")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))
