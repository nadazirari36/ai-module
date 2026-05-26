"""
Data Preprocessing Module.
Cleans raw data, builds customer-item matrix and purchase sequences.
"""
import numpy as np
import pandas as pd
import pickle
import os
from config import DATA_PATH, SAVED_MODELS_DIR, MAPPINGS_PATH, POPULAR_PATH, SEED

np.random.seed(SEED)


def load_and_clean(path=DATA_PATH):
    """Load raw xlsx and apply cleaning pipeline."""
    print("Loading raw data...")
    df = pd.read_excel(path, engine='openpyxl')
    print(f"  Raw rows: {len(df)}")

    # Remove missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    print(f"  After dropping missing CustomerID: {len(df)}")

    # Remove cancelled orders (InvoiceNo starting with 'C')
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    print(f"  After removing cancellations: {len(df)}")

    # Remove negative or zero quantities
    df = df[df['Quantity'] > 0]
    print(f"  After removing non-positive quantities: {len(df)}")

    # Fix types
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Fill missing Description with StockCode
    df['Description'] = df['Description'].fillna(df['StockCode'].astype(str))

    print(f"  Clean rows: {len(df)}")
    print(f"  Unique customers: {df['CustomerID'].nunique()}")
    print(f"  Unique products: {df['StockCode'].nunique()}")
    return df


def build_customer_item_matrix(df):
    """Build binary customer-item purchase matrix."""
    print("Building customer-item matrix...")
    interactions = df.groupby(['CustomerID', 'StockCode']).size().reset_index(name='count')
    interactions['purchased'] = 1

    pivot = interactions.pivot_table(
        index='CustomerID', columns='StockCode', values='purchased', fill_value=0
    )
    print(f"  Matrix shape: {pivot.shape}")
    return pivot


def build_invoice_baskets(df):
    """Build invoice-level baskets for Apriori."""
    print("Building invoice baskets...")
    baskets = df.groupby(['InvoiceNo', 'StockCode']).size().reset_index(name='count')
    baskets['purchased'] = 1
    basket_matrix = baskets.pivot_table(
        index='InvoiceNo', columns='StockCode', values='purchased', fill_value=0
    )
    # Convert to boolean for mlxtend
    basket_matrix = basket_matrix.astype(bool)
    print(f"  Basket matrix shape: {basket_matrix.shape}")
    return basket_matrix


def build_purchase_sequences(df, max_seq_len=10):
    """Build ordered purchase sequences per customer for LSTM."""
    print("Building purchase sequences...")
    df_sorted = df.sort_values(['CustomerID', 'InvoiceDate'])

    sequences = []
    for cid, group in df_sorted.groupby('CustomerID'):
        items = group['StockCode'].tolist()
        # Deduplicate consecutive items
        deduped = [items[0]]
        for item in items[1:]:
            if item != deduped[-1]:
                deduped.append(item)
        if len(deduped) >= 2:
            sequences.append((cid, deduped))

    print(f"  Customers with sequences (>=2 items): {len(sequences)}")
    return sequences


def build_mappings(df):
    """Build ID-to-index mappings for users and items."""
    users = sorted(df['CustomerID'].unique())
    items = sorted(df['StockCode'].unique(), key=str)

    user2idx = {u: i for i, u in enumerate(users)}
    idx2user = {i: u for u, i in user2idx.items()}
    item2idx = {it: i for i, it in enumerate(items)}
    idx2item = {i: it for it, i in item2idx.items()}

    # Item descriptions
    desc_map = df.drop_duplicates('StockCode').set_index('StockCode')['Description'].to_dict()

    mappings = {
        'user2idx': user2idx, 'idx2user': idx2user,
        'item2idx': item2idx, 'idx2item': idx2item,
        'desc_map': desc_map,
        'n_users': len(users), 'n_items': len(items)
    }
    return mappings


def save_popular_items(df, top_n=10):
    """Compute and save top popular items for cold-start."""
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
    print(f"  Saved top {top_n} popular items.")
    return popular_list


def run_preprocessing():
    """Run full preprocessing pipeline and save artifacts."""
    df = load_and_clean()
    customer_item_matrix = build_customer_item_matrix(df)
    basket_matrix = build_invoice_baskets(df)
    sequences = build_purchase_sequences(df)
    mappings = build_mappings(df)

    # Save mappings
    with open(MAPPINGS_PATH, 'wb') as f:
        pickle.dump(mappings, f)
    print(f"  Saved mappings to {MAPPINGS_PATH}")

    save_popular_items(df)

    return df, customer_item_matrix, basket_matrix, sequences, mappings


if __name__ == '__main__':
    df, cim, bm, seqs, maps = run_preprocessing()
    print(f"\nPreprocessing complete.")
    print(f"  Clean data: {len(df)} rows")
    print(f"  Customer-item matrix: {cim.shape}")
    print(f"  Basket matrix: {bm.shape}")
    print(f"  Sequences: {len(seqs)}")
    print(f"  Users: {maps['n_users']}, Items: {maps['n_items']}")
