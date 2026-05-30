"""
preprocessing.py — Module de prétraitement des données.
Charge les données brutes, nettoie les erreurs, construit la matrice client-article,
les paniers par facture, les séquences d'achats, et les mappings d'indices.
"""
import numpy as np
import pandas as pd
import pickle
import os
from config import DATA_PATH, SAVED_MODELS_DIR, MAPPINGS_PATH, POPULAR_PATH, SEED, TEMPORAL_TEST_RATIO

# Fixe la graine aléatoire pour la reproductibilité
np.random.seed(SEED)


def load_and_clean(path=DATA_PATH):
    """
    Charge le fichier Excel brut et applique le pipeline de nettoyage.
    Étapes :
      1. Supprime les lignes sans CustomerID (transactions anonymes)
      2. Supprime les commandes annulées (InvoiceNo commençant par 'C')
      3. Supprime les quantités nulles ou négatives (retours, erreurs de saisie)
      4. Convertit les types de colonnes
      5. Remplace les descriptions manquantes par le code article
    """
    print("Chargement des données brutes...")
    df = pd.read_excel(path, engine='openpyxl')
    print(f"  Lignes brutes : {len(df)}")

    # Étape 1 : on ne peut pas identifier l'utilisateur sans CustomerID
    df = df.dropna(subset=['CustomerID'])
    print(f"  Après suppression des CustomerID manquants : {len(df)}")

    # Étape 2 : les factures annulées commencent par 'C' (ex. C536379)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    print(f"  Après suppression des annulations : {len(df)}")

    # Étape 3 : quantités négatives ou nulles = retours ou erreurs
    df = df[df['Quantity'] > 0]
    print(f"  Après suppression des quantités non positives : {len(df)}")

    # Étape 4 : conversion des types pour cohérence
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)   # ex. 17850.0 → '17850'
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])          # chaîne → datetime

    # Étape 5 : certains articles n'ont pas de description, on utilise le code
    df['Description'] = df['Description'].fillna(df['StockCode'].astype(str))

    print(f"  Lignes nettoyées : {len(df)}")
    print(f"  Clients uniques : {df['CustomerID'].nunique()}")
    print(f"  Produits uniques : {df['StockCode'].nunique()}")
    return df


def build_customer_item_matrix(df):
    """
    Construit la matrice binaire client-article.
    Lignes = clients (CustomerID), colonnes = articles (StockCode).
    Valeur = 1 si le client a déjà acheté cet article, 0 sinon.
    Utilisée par : AutoEncoder (entrée), NCF (paires positives), évaluation (vérité terrain).
    """
    print("Construction de la matrice client-article...")

    # Regroupe par (client, article) et marque comme acheté (1)
    interactions = df.groupby(['CustomerID', 'StockCode']).size().reset_index(name='count')
    interactions['purchased'] = 1

    # Pivote en une matrice 2D (clients × articles)
    pivot = interactions.pivot_table(
        index='CustomerID', columns='StockCode', values='purchased', fill_value=0
    )
    print(f"  Forme de la matrice : {pivot.shape}")
    return pivot


def build_invoice_baskets(df):
    """
    Construit la matrice de paniers par facture pour l'algorithme Apriori.
    Lignes = factures (InvoiceNo), colonnes = articles (StockCode).
    Valeur = True si l'article est dans la facture, False sinon.
    Format booléen requis par mlxtend.
    """
    print("Construction des paniers par facture...")
    baskets = df.groupby(['InvoiceNo', 'StockCode']).size().reset_index(name='count')
    baskets['purchased'] = 1
    basket_matrix = baskets.pivot_table(
        index='InvoiceNo', columns='StockCode', values='purchased', fill_value=0
    )
    # Conversion en booléen pour mlxtend
    basket_matrix = basket_matrix.astype(bool)
    print(f"  Forme de la matrice paniers : {basket_matrix.shape}")
    return basket_matrix


def build_purchase_sequences(df, max_seq_len=10):
    """
    Construit les séquences d'achats ordonnées par client pour le LSTM.
    Chaque séquence = liste d'articles achetés dans l'ordre chronologique.
    Les articles consécutifs identiques sont dédupliqués pour réduire la redondance.
    Seuls les clients avec au moins 2 articles sont conservés (minimum pour apprendre).
    Retourne : liste de tuples (customer_id, [article1, article2, ...])
    """
    print("Construction des séquences d'achats...")
    df_sorted = df.sort_values(['CustomerID', 'InvoiceDate'])

    sequences = []
    for cid, group in df_sorted.groupby('CustomerID'):
        items = group['StockCode'].tolist()

        # Déduplique les articles consécutifs identiques
        # ex. [A, A, B, C, C] → [A, B, C]
        deduped = [items[0]]
        for item in items[1:]:
            if item != deduped[-1]:
                deduped.append(item)

        # On garde uniquement les séquences avec au moins 2 articles distincts
        if len(deduped) >= 2:
            sequences.append((cid, deduped))

    print(f"  Clients avec séquences (>=2 articles) : {len(sequences)}")
    return sequences


def build_mappings(df):
    """
    Construit les mappings bidirectionnels entre IDs et indices numériques.
    Les modèles (NCF, LSTM) travaillent avec des indices entiers, pas des chaînes.
    Retourne un dictionnaire contenant :
      - user2idx   : {customer_id → indice entier}
      - idx2user   : {indice entier → customer_id}
      - item2idx   : {stock_code  → indice entier}
      - idx2item   : {indice entier → stock_code}
      - desc_map   : {stock_code  → description textuelle}
      - n_users    : nombre total de clients distincts
      - n_items    : nombre total d'articles distincts
    """
    users = sorted(df['CustomerID'].unique())
    items = sorted(df['StockCode'].unique(), key=str)

    # Mappings utilisateurs : client_id ↔ indice
    user2idx = {u: i for i, u in enumerate(users)}
    idx2user = {i: u for u, i in user2idx.items()}

    # Mappings articles : stock_code ↔ indice
    item2idx = {it: i for i, it in enumerate(items)}
    idx2item = {i: it for it, i in item2idx.items()}

    # Dictionnaire de descriptions pour l'affichage des recommandations
    desc_map = df.drop_duplicates('StockCode').set_index('StockCode')['Description'].to_dict()

    mappings = {
        'user2idx': user2idx, 'idx2user': idx2user,
        'item2idx': item2idx, 'idx2item': idx2item,
        'desc_map': desc_map,
        'n_users': len(users), 'n_items': len(items)
    }
    return mappings


def save_popular_items(df, top_n=10):
    """
    Calcule et sauvegarde les articles les plus populaires.
    Utilisés comme recommandations de secours pour les nouveaux utilisateurs
    qui n'ont aucun historique d'achat (démarrage à froid total).
    Popularité mesurée par la somme des quantités vendues.
    """
    popular = (df.groupby('StockCode')
               .agg(count=('Quantity', 'sum'), description=('Description', 'first'))
               .sort_values('count', ascending=False)
               .head(top_n))
    popular_list = [
        {'product_id': idx, 'description': row['description'], 'score': 1.0}
        for idx, row in popular.iterrows()
    ]
    with open(POPULAR_PATH, 'wb') as f:
        pickle.dump(popular_list, f)
    print(f"  Top {top_n} articles populaires sauvegardés.")
    return popular_list


def temporal_split(df, test_ratio=TEMPORAL_TEST_RATIO):
    """
    Découpage temporel par utilisateur : les dernières test_ratio factures de chaque
    utilisateur sont mises de côté comme données de test.

    Avantage : tous les utilisateurs participent à l'entraînement, donc NCF dispose
    d'embeddings entraînés pour tout le monde.
    Les utilisateurs avec une seule facture vont entièrement en train et sont ignorés
    lors de l'évaluation (ensemble 'relevant' vide).

    Retourne : (train_df, test_df)
    """
    print(f"Construction du découpage temporel par utilisateur (test_ratio={test_ratio})...")
    train_rows, test_rows = [], []

    for uid, user_df in df.groupby('CustomerID'):
        user_df_sorted = user_df.sort_values('InvoiceDate')

        # Trie les factures par date de première ligne
        invoice_dates = user_df_sorted.groupby('InvoiceNo')['InvoiceDate'].min().sort_values()
        invoices = invoice_dates.index.tolist()

        # Impossible de découper avec une seule facture → tout va en train
        if len(invoices) < 2:
            train_rows.append(user_df_sorted)
            continue

        # Calcule combien de factures vont en test (au moins 1)
        n_test = max(1, int(len(invoices) * test_ratio))
        train_invs = set(invoices[:-n_test])   # factures anciennes → train
        test_invs  = set(invoices[-n_test:])   # factures récentes → test

        train_rows.append(user_df_sorted[user_df_sorted['InvoiceNo'].isin(train_invs)])
        test_rows.append(user_df_sorted[user_df_sorted['InvoiceNo'].isin(test_invs)])

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df  = pd.concat(test_rows,  ignore_index=True) if test_rows else pd.DataFrame(columns=df.columns)

    print(f"  Lignes train : {len(train_df)}  |  Lignes test : {len(test_df)}")
    print(f"  Utilisateurs en train : {train_df['CustomerID'].nunique()}")
    print(f"  Utilisateurs avec données de test : {test_df['CustomerID'].nunique() if len(test_df) > 0 else 0}")
    return train_df, test_df


def run_preprocessing():
    """
    Lance le pipeline complet de prétraitement et sauvegarde les artefacts.
    Étapes :
      1. Charge et nettoie les données brutes
      2. Construit la matrice client-article binaire
      3. Construit la matrice de paniers par facture (pour Apriori)
      4. Construit les séquences d'achats (pour LSTM)
      5. Construit les mappings d'indices
      6. Sauvegarde les mappings et les articles populaires
    Retourne : (df, customer_item_matrix, basket_matrix, sequences, mappings)
    """
    df = load_and_clean()
    customer_item_matrix = build_customer_item_matrix(df)
    basket_matrix        = build_invoice_baskets(df)
    sequences            = build_purchase_sequences(df)
    mappings             = build_mappings(df)

    # Sauvegarde des mappings sur disque (requis par l'API en production)
    with open(MAPPINGS_PATH, 'wb') as f:
        pickle.dump(mappings, f)
    print(f"  Mappings sauvegardés dans {MAPPINGS_PATH}")

    # Sauvegarde des articles populaires (fallback démarrage à froid)
    save_popular_items(df)

    return df, customer_item_matrix, basket_matrix, sequences, mappings


# ─── Point d'entrée direct ───────────────────────────────────────────────────
if __name__ == '__main__':
    df, cim, bm, seqs, maps = run_preprocessing()
    print(f"\nPrétraitement terminé.")
    print(f"  Données nettoyées : {len(df)} lignes")
    print(f"  Matrice client-article : {cim.shape}")
    print(f"  Matrice paniers : {bm.shape}")
    print(f"  Séquences : {len(seqs)}")
    print(f"  Utilisateurs : {maps['n_users']}, Articles : {maps['n_items']}")
