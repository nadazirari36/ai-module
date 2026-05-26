"""
Neural Collaborative Filtering (NCF).
Embeds users and items, learns interaction through MLP.
"""
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import (NCF_EPOCHS, NCF_BATCH_SIZE, NCF_LR, NCF_EMBED_DIM,
                    NCF_NEG_RATIO, SAVED_MODELS_DIR, SEED)

np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_ncf(n_users, n_items, embed_dim=NCF_EMBED_DIM):
    """Build NeuMF: GMF path (element-wise product) + MLP path, combined for stronger ranking."""
    user_input = keras.Input(shape=(1,), name='user_input')
    item_input = keras.Input(shape=(1,), name='item_input')

    # GMF path — captures linear interactions via element-wise product
    user_gmf = layers.Embedding(n_users, embed_dim, embeddings_regularizer=regularizers.l2(1e-5), name='user_gmf')(user_input)
    user_gmf = layers.Flatten()(user_gmf)
    item_gmf = layers.Embedding(n_items, embed_dim, embeddings_regularizer=regularizers.l2(1e-5), name='item_gmf')(item_input)
    item_gmf = layers.Flatten()(item_gmf)
    gmf_out = layers.Multiply()([user_gmf, item_gmf])

    # MLP path — captures non-linear interactions
    user_mlp = layers.Embedding(n_users, embed_dim, embeddings_regularizer=regularizers.l2(1e-5), name='user_mlp')(user_input)
    user_mlp = layers.Flatten()(user_mlp)
    item_mlp = layers.Embedding(n_items, embed_dim, embeddings_regularizer=regularizers.l2(1e-5), name='item_mlp')(item_input)
    item_mlp = layers.Flatten()(item_mlp)
    x = layers.Concatenate()([user_mlp, item_mlp])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    mlp_out = x

    # Combine GMF + MLP
    combined = layers.Concatenate()([gmf_out, mlp_out])
    output = layers.Dense(1, activation='sigmoid')(combined)

    model = keras.Model([user_input, item_input], output, name='neumf')
    return model


def generate_training_pairs(customer_item_matrix, user2idx, item2idx, neg_ratio=NCF_NEG_RATIO):
    """Generate positive and negative training pairs (memory-efficient)."""
    print("  Generating NCF training pairs...")
    users, items, labels = [], [], []
    all_items_set = set(range(len(item2idx)))
    for uid, row in customer_item_matrix.iterrows():
        if uid not in user2idx:
            continue
        uidx = user2idx[uid]
        purchased = [item2idx[c] for c in row[row == 1].index if c in item2idx]

        if len(purchased) == 0:
            continue

        for iidx in purchased:
            users.append(uidx)
            items.append(iidx)
            labels.append(1)

        # Negative sampling
        unpurchased = list(all_items_set - set(purchased))
        n_neg = min(len(purchased) * neg_ratio, len(unpurchased))
        if n_neg > 0:
            neg_samples = np.random.choice(unpurchased, size=n_neg, replace=False)
            for iidx in neg_samples:
                users.append(uidx)
                items.append(iidx)
                labels.append(0)

    users = np.array(users, dtype=np.int32)
    items = np.array(items, dtype=np.int32)
    labels = np.array(labels, dtype=np.float32)
    print(f"    Total pairs: {len(labels)} (pos: {int(labels.sum())}, neg: {int((1-labels).sum())})")
    return users, items, labels


def train_ncf(customer_item_matrix, mappings, train_users):
    """Train NCF model."""
    print("Training NCF...")
    n_users = mappings['n_users']
    n_items = mappings['n_items']

    # Filter to training users
    train_matrix = customer_item_matrix.loc[
        customer_item_matrix.index.isin(train_users)
    ]

    users, items, labels = generate_training_pairs(
        train_matrix, mappings['user2idx'], mappings['item2idx']
    )

    # Shuffle
    perm = np.random.permutation(len(labels))
    users, items, labels = users[perm], items[perm], labels[perm]

    # Split into train/val
    split = int(len(labels) * 0.9)
    train_data = ([users[:split], items[:split]], labels[:split])
    val_data = ([users[split:], items[split:]], labels[split:])

    model = build_ncf(n_users, n_items)
    optimizer = keras.optimizers.Adam(learning_rate=NCF_LR)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    history = model.fit(
        train_data[0], train_data[1],
        epochs=NCF_EPOCHS,
        batch_size=NCF_BATCH_SIZE,
        validation_data=val_data,
        callbacks=[lr_scheduler, early_stop],
        verbose=1
    )

    save_path = os.path.join(SAVED_MODELS_DIR, 'ncf.keras')
    model.save(save_path)
    print(f"  NCF saved to {save_path}")

    plot_loss(history, 'NCF')
    return model, history


def plot_loss(history, model_name):
    """Plot training vs validation loss."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(SAVED_MODELS_DIR, f'{model_name.lower()}_loss.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Loss plot saved to {save_path}")


def predict_ncf(model, user_idx, n_items):
    """Get scores for all items for a given user."""
    user_array = np.full(n_items, user_idx)
    item_array = np.arange(n_items)
    scores = model.predict([user_array, item_array], verbose=0, batch_size=1024).flatten()
    return scores


if __name__ == '__main__':
    from preprocessing import run_preprocessing
    _, cim, _, _, mappings = run_preprocessing()
    users = list(cim.index)
    train_users = users[:int(len(users) * 0.8)]
    model, hist = train_ncf(cim, mappings, train_users)
    print("NCF training complete.")
