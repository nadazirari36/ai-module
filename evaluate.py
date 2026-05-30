"""
evaluate.py — Module d'évaluation des modèles de recommandation.
Calcule les métriques standard : Précision@K, Rappel@K et NDCG@K.
Ces métriques mesurent la qualité des K premières recommandations générées.
"""
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Supprime les logs verbeux de TensorFlow

import matplotlib
matplotlib.use('Agg')  # Backend sans affichage (serveur sans écran)
import matplotlib.pyplot as plt
from config import TOP_K_VALUES, SAVED_MODELS_DIR, SEED

# Fixe la graine aléatoire pour la reproductibilité
np.random.seed(SEED)


def precision_at_k(recommended, relevant, k):
    """
    Précision@K : fraction des K articles recommandés qui sont pertinents.
    Formule : |recommandés[:K] ∩ pertinents| / K
    Exemple : si on recommande [A, B, C, D, E] et que l'utilisateur a acheté [B, D, F],
              Précision@5 = 2/5 = 0.4 (B et D sont dans les 5 recommandations)
    """
    rec_k = recommended[:k]  # Garde uniquement les K premières recommandations
    hits  = len(set(rec_k) & set(relevant))  # Intersection : articles recommandés ET pertinents
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended, relevant, k):
    """
    Rappel@K : fraction des articles pertinents qui apparaissent dans les K recommandations.
    Formule : |recommandés[:K] ∩ pertinents| / |pertinents|
    Exemple : si l'utilisateur a acheté 10 articles et que 3 apparaissent dans le top-10,
              Rappel@10 = 3/10 = 0.3
    Note : le rappel est naturellement plus faible quand le catalogue est grand.
    """
    rec_k = recommended[:k]
    hits  = len(set(rec_k) & set(relevant))
    return hits / len(relevant) if len(relevant) > 0 else 0.0


def ndcg_at_k(recommended, relevant, k):
    """
    NDCG@K (Normalized Discounted Cumulative Gain) : mesure la qualité du classement.
    Pénalise les bonnes recommandations qui apparaissent tard dans la liste.
    Un article pertinent en position 1 vaut plus qu'en position 10.
    Formule :
      DCG  = Σ (1 / log2(i+2)) pour chaque article pertinent à la position i (base 0)
      IDCG = DCG idéal (tous les pertinents en tête de liste)
      NDCG = DCG / IDCG  → normalisé entre 0 et 1
    Exemple : [B, A, C, D, E] avec pertinents={A, C}
      DCG  = 1/log2(2) + 1/log2(3) = 1.0 + 0.63 = 1.63   (A en pos 1, C en pos 2)
      IDCG = 1/log2(2) + 1/log2(3) = 1.63 (idéal car 2 pertinents en top-2)
      NDCG = 1.63/1.63 = 1.0 (classement parfait)
    """
    rec_k = recommended[:k]
    dcg   = 0.0

    # Calcule le DCG : somme des gains pondérés par la position
    for i, item in enumerate(rec_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # log2(2)=1 en pos 0, log2(3)≈0.63 en pos 1, etc.

    # Calcule le IDCG : DCG idéal si tous les pertinents étaient en tête
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(model_name, score_fn, test_users, customer_item_matrix,
                   train_customer_item_matrix, mappings, k_values=TOP_K_VALUES):
    """
    Évalue un modèle sur l'ensemble des utilisateurs de test.

    Protocole d'évaluation (démarrage à froid) :
      - Les utilisateurs de test ne sont PAS dans la matrice d'entraînement.
      - Donc train_items = {} et relevant = tous leurs achats.
      - Le modèle doit recommander des articles qu'un nouvel utilisateur achètera,
        en se basant uniquement sur ce qu'il a acheté jusqu'ici.

    Paramètres :
      score_fn              : fonction(user_id) → dict {article: score}
      test_users            : liste des identifiants utilisateurs à évaluer
      customer_item_matrix  : matrice complète (tous les achats, vérité terrain)
      train_customer_item_matrix : matrice d'entraînement (achats déjà vus → filtrés)
      k_values              : liste de valeurs de K (ex. [5, 10])

    Retourne : dict {k: {'precision': float, 'recall': float, 'ndcg': float}}
    """
    print(f"Évaluation de {model_name}...")
    # Initialise les listes de métriques par valeur de K
    metrics   = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in k_values}
    evaluated = 0  # Compteur d'utilisateurs effectivement évalués

    for uid in test_users:
        # Ignore les utilisateurs absents de la matrice complète (ne devrait pas arriver)
        if uid not in customer_item_matrix.index:
            continue

        # ── Vérité terrain ────────────────────────────────────────────────────
        # Articles pertinents = tous les achats de l'utilisateur
        test_items = set(customer_item_matrix.loc[uid][customer_item_matrix.loc[uid] == 1].index)

        # Articles déjà vus pendant l'entraînement (à filtrer des recommandations)
        if uid in train_customer_item_matrix.index:
            train_items = set(train_customer_item_matrix.loc[uid][
                train_customer_item_matrix.loc[uid] == 1].index)
        else:
            train_items = set()  # Utilisateur cold-start : aucun historique d'entraînement

        # Ensemble pertinent = achats non vus pendant l'entraînement
        relevant = test_items - train_items
        if len(relevant) == 0:
            continue  # Rien à prédire → on saute cet utilisateur

        # ── Prédiction ────────────────────────────────────────────────────────
        try:
            scores = score_fn(uid)  # Dict {article: score} pour tous les articles
        except Exception:
            continue  # Si le modèle plante pour cet utilisateur → on le saute

        # Filtre les articles déjà vus en entraînement (ne pas recommander ce qu'on sait déjà)
        item_scores = [
            (item_code, score)
            for item_code, score in scores.items()
            if item_code not in train_items
        ]

        # Trie par score décroissant → liste ordonnée de recommandations
        item_scores.sort(key=lambda x: x[1], reverse=True)
        recommended = [item for item, _ in item_scores]

        # ── Calcul des métriques ──────────────────────────────────────────────
        for k in k_values:
            metrics[k]['precision'].append(precision_at_k(recommended, relevant, k))
            metrics[k]['recall'].append(recall_at_k(recommended, relevant, k))
            metrics[k]['ndcg'].append(ndcg_at_k(recommended, relevant, k))

        evaluated += 1

    # Moyenne des métriques sur tous les utilisateurs évalués
    results = {}
    for k in k_values:
        results[k] = {
            'precision': np.mean(metrics[k]['precision']) if metrics[k]['precision'] else 0,
            'recall':    np.mean(metrics[k]['recall'])    if metrics[k]['recall']    else 0,
            'ndcg':      np.mean(metrics[k]['ndcg'])      if metrics[k]['ndcg']      else 0,
        }
    print(f"  Évalué sur {evaluated} utilisateurs")
    return results


def build_comparison_table(all_results):
    """
    Construit et sauvegarde le tableau comparatif des métriques de tous les modèles.
    Sauvegarde en CSV dans saved_models/evaluation_results.csv.
    Retourne le DataFrame pandas.
    """
    rows = []
    for model_name, results in all_results.items():
        for k, metrics in results.items():
            rows.append({
                'Modèle':       model_name,
                'K':            k,
                'Précision@K':  f"{metrics['precision']:.4f}",
                'Rappel@K':     f"{metrics['recall']:.4f}",
                'NDCG@K':       f"{metrics['ndcg']:.4f}",
            })

    table     = pd.DataFrame(rows)
    save_path = os.path.join(SAVED_MODELS_DIR, 'evaluation_results.csv')
    table.to_csv(save_path, index=False)
    print(f"\nTableau comparatif sauvegardé dans {save_path}")
    print(table.to_string(index=False))
    return table


def plot_comparison(all_results):
    """
    Trace et sauvegarde le graphique en barres comparant tous les modèles.
    Trois sous-graphiques : Précision@10, Rappel@10, NDCG@10.
    Sauvegarde dans saved_models/model_comparison.png.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metric_names = ['precision', 'recall', 'ndcg']
    titles       = ['Précision@K', 'Rappel@K', 'NDCG@K']

    for ax, metric, title in zip(axes, metric_names, titles):
        k = 10  # Affiche uniquement les métriques à K=10
        models = []
        values = []
        for model_name, results in all_results.items():
            if k in results:
                models.append(model_name)
                values.append(results[k][metric])

        # Couleurs distinctes pour chaque modèle
        bars = ax.bar(models, values,
                      color=['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0'])
        ax.set_title(f'{title} (K=10)')
        ax.set_ylabel(title)
        ax.tick_params(axis='x', rotation=45)

        # Affiche la valeur numérique au-dessus de chaque barre
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(SAVED_MODELS_DIR, 'model_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Graphique comparatif sauvegardé dans {save_path}")
