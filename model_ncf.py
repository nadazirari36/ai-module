"""
model_ncf.py — Filtrage Collaboratif Neuronal (Neural Collaborative Filtering).
Implémente le modèle NeuMF (Neural Matrix Factorization) qui combine :
  - GMF (Generalized Matrix Factorization) : capture les interactions linéaires
    via le produit élément par élément des embeddings utilisateur et article.
  - MLP (Multi-Layer Perceptron) : capture les interactions non linéaires complexes
    via la concaténation des embeddings passée dans des couches denses.
Les deux chemins sont fusionnés pour une prédiction finale de la probabilité d'achat.
"""
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Supprime les logs verbeux de TensorFlow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib
matplotlib.use('Agg')  # Backend sans affichage (serveur sans écran)
import matplotlib.pyplot as plt
from config import (NCF_EPOCHS, NCF_BATCH_SIZE, NCF_LR, NCF_EMBED_DIM,
                    NCF_NEG_RATIO, SAVED_MODELS_DIR, SEED)

# Fixe les graines pour des résultats reproductibles
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_ncf(n_users, n_items, embed_dim=NCF_EMBED_DIM):
    """
    Construit le modèle NeuMF : chemin GMF + chemin MLP, fusionnés pour un meilleur classement.
    Architecture :
      Entrées : indice utilisateur (scalaire), indice article (scalaire)

      Chemin GMF (linéaire) :
        embed_user_gmf(128) × embed_item_gmf(128) → vecteur 128D

      Chemin MLP (non linéaire) :
        concat(embed_user_mlp(128), embed_item_mlp(128)) [256D]
        → Dense(512) + BatchNorm + Dropout(0.3)
        → Dense(256) + BatchNorm + Dropout(0.3)
        → Dense(128) + BatchNorm + Dropout(0.2)
        → Dense(64)

      Fusion : concat(gmf_out[128], mlp_out[64]) → Dense(1, sigmoïde)
    """
    # ── Entrées ───────────────────────────────────────────────────────────────
    user_input = keras.Input(shape=(1,), name='user_input')
    item_input = keras.Input(shape=(1,), name='item_input')

    # ── Chemin GMF : interactions linéaires via produit élément par élément ──
    # Embedding utilisateur pour le chemin GMF
    user_gmf = layers.Embedding(n_users, embed_dim,
                                 embeddings_regularizer=regularizers.l2(1e-5),
                                 name='user_gmf')(user_input)
    user_gmf = layers.Flatten()(user_gmf)  # (batch, 1, embed_dim) → (batch, embed_dim)

    # Embedding article pour le chemin GMF
    item_gmf = layers.Embedding(n_items, embed_dim,
                                 embeddings_regularizer=regularizers.l2(1e-5),
                                 name='item_gmf')(item_input)
    item_gmf = layers.Flatten()(item_gmf)

    # Produit de Hadamard (élément par élément) : capture la compatibilité linéaire
    gmf_out = layers.Multiply()([user_gmf, item_gmf])  # → (batch, embed_dim)

    # ── Chemin MLP : interactions non linéaires via couches denses ───────────
    # Embeddings séparés pour le chemin MLP (appris indépendamment du GMF)
    user_mlp = layers.Embedding(n_users, embed_dim,
                                 embeddings_regularizer=regularizers.l2(1e-5),
                                 name='user_mlp')(user_input)
    user_mlp = layers.Flatten()(user_mlp)

    item_mlp = layers.Embedding(n_items, embed_dim,
                                 embeddings_regularizer=regularizers.l2(1e-5),
                                 name='item_mlp')(item_input)
    item_mlp = layers.Flatten()(item_mlp)

    # Concaténation des embeddings utilisateur et article (256D)
    x = layers.Concatenate()([user_mlp, item_mlp])

    # Couches denses avec BatchNorm pour stabiliser l'entraînement
    # Architecture élargie (512 → 256 → 128 → 64) par rapport à la version initiale
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)   # Normalise les activations → convergence plus rapide
    x = layers.Dropout(0.3)(x)           # Régularisation : 30 % des neurones désactivés aléatoirement

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)           # Dropout plus faible dans les couches profondes

    x = layers.Dense(64, activation='relu')(x)
    mlp_out = x  # Représentation finale du chemin MLP (64D)

    # ── Fusion GMF + MLP ──────────────────────────────────────────────────────
    combined = layers.Concatenate()([gmf_out, mlp_out])  # 128 + 64 = 192D
    output   = layers.Dense(1, activation='sigmoid')(combined)  # Score ∈ [0, 1]

    model = keras.Model([user_input, item_input], output, name='neumf')
    return model


def generate_training_pairs(customer_item_matrix, user2idx, item2idx, neg_ratio=NCF_NEG_RATIO):
    """
    Génère les paires d'entraînement positives et négatives avec échantillonnage
    négatif pondéré par la popularité (négatifs difficiles).

    Principe :
      - Paires positives : (utilisateur, article acheté, label=1)
      - Paires négatives : (utilisateur, article NON acheté, label=0)
        → Les articles populaires ont plus de chances d'être choisis comme négatifs,
          forçant le modèle à discriminer les vraies préférences des articles simplement populaires.

    neg_ratio = 5 : 5 négatifs pour chaque positif (était 2).
    Plus de négatifs → meilleure calibration du classement.
    """
    print("  Génération des paires d'entraînement NCF...")

    # ── Calcul de la popularité des articles ──────────────────────────────────
    # Nombre d'utilisateurs ayant acheté chaque article (par indice)
    n_items = len(item2idx)
    item_pop = np.zeros(n_items, dtype=np.float64)
    for col in customer_item_matrix.columns:
        if col in item2idx:
            item_pop[item2idx[col]] = float(customer_item_matrix[col].sum())

    # Lissage de Laplace : évite les probabilités nulles pour les articles peu vus
    item_pop += 1.0
    item_pop /= item_pop.sum()  # Normalise en distribution de probabilité

    # Tableau de tous les indices d'articles
    all_indices = np.arange(n_items, dtype=np.int32)
    users, items, labels = [], [], []

    for uid, row in customer_item_matrix.iterrows():
        # Ignore les utilisateurs sans mapping (ne devraient pas exister)
        if uid not in user2idx:
            continue
        uidx = user2idx[uid]

        # Articles achetés par cet utilisateur (indices)
        purchased_set = set(item2idx[c] for c in row[row == 1].index if c in item2idx)
        purchased = list(purchased_set)

        if not purchased:
            continue  # Ignore les utilisateurs sans achat

        # ── Paires positives ──────────────────────────────────────────────────
        for iidx in purchased:
            users.append(uidx)
            items.append(iidx)
            labels.append(1)   # label = 1 : l'utilisateur a acheté cet article

        # ── Paires négatives pondérées par popularité ─────────────────────────
        # Masque booléen : True pour les articles NON achetés
        unpurchased_mask = np.ones(n_items, dtype=bool)
        for iidx in purchased:
            unpurchased_mask[iidx] = False
        unpurchased_arr = all_indices[unpurchased_mask]

        # Nombre de négatifs = min(ratio × positifs, articles disponibles)
        n_neg = min(len(purchased) * neg_ratio, len(unpurchased_arr))

        if n_neg > 0:
            # Probabilités de sélection proportionnelles à la popularité
            neg_weights = item_pop[unpurchased_arr]
            neg_weights = neg_weights / neg_weights.sum()

            # Sélection sans remise pondérée par popularité
            neg_samples = np.random.choice(unpurchased_arr, size=n_neg,
                                            replace=False, p=neg_weights)
            for iidx in neg_samples:
                users.append(uidx)
                items.append(int(iidx))
                labels.append(0)   # label = 0 : article non acheté (négatif)

    # Conversion en tableaux numpy pour Keras
    users  = np.array(users,  dtype=np.int32)
    items  = np.array(items,  dtype=np.int32)
    labels = np.array(labels, dtype=np.float32)
    print(f"    Total paires : {len(labels)} (pos : {int(labels.sum())}, neg : {int((1-labels).sum())})")
    return users, items, labels


def train_ncf(customer_item_matrix, mappings, train_users):
    """
    Entraîne le modèle NCF et sauvegarde les poids.
    Paramètres :
      customer_item_matrix : matrice binaire client-article (DataFrame pandas)
      mappings             : dictionnaire des mappings (user2idx, item2idx, etc.)
      train_users          : liste des utilisateurs à inclure dans l'entraînement
                             (typiquement train + val = 80 % de tous les utilisateurs)
    """
    print("Entraînement du NCF...")
    n_users = mappings['n_users']
    n_items = mappings['n_items']

    # Filtre la matrice aux seuls utilisateurs d'entraînement
    train_matrix = customer_item_matrix.loc[
        customer_item_matrix.index.isin(train_users)
    ]

    # Génère les paires positives et négatives d'entraînement
    users, items, labels = generate_training_pairs(
        train_matrix, mappings['user2idx'], mappings['item2idx']
    )

    # Mélange aléatoire pour éviter les biais d'ordre
    perm = np.random.permutation(len(labels))
    users, items, labels = users[perm], items[perm], labels[perm]

    # Division 90 % train / 10 % validation (pour le suivi de la perte)
    split = int(len(labels) * 0.9)
    train_data = ([users[:split],  items[:split]],  labels[:split])
    val_data   = ([users[split:],  items[split:]],  labels[split:])

    # Construction et compilation du modèle
    model     = build_ncf(n_users, n_items)
    optimizer = keras.optimizers.Adam(learning_rate=NCF_LR)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Réduit le taux d'apprentissage si la val_loss stagne
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    # Arrêt anticipé avec patience augmentée (8 au lieu de 5)
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=8, restore_best_weights=True
    )

    # Entraînement
    history = model.fit(
        train_data[0], train_data[1],
        epochs=NCF_EPOCHS,
        batch_size=NCF_BATCH_SIZE,
        validation_data=val_data,
        callbacks=[lr_scheduler, early_stop],
        verbose=1
    )

    # Sauvegarde du modèle
    save_path = os.path.join(SAVED_MODELS_DIR, 'ncf.keras')
    model.save(save_path)
    print(f"  NCF sauvegardé dans {save_path}")

    plot_loss(history, 'NCF')
    return model, history


def plot_loss(history, model_name):
    """
    Trace et sauvegarde la courbe de perte entraînement vs validation.
    Utile pour détecter le sur-apprentissage : si val_loss remonte alors que
    train_loss continue de baisser, le modèle surapprend.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'],     label='Perte Entraînement')
    plt.plot(history.history['val_loss'], label='Perte Validation')
    plt.title(f'{model_name} - Perte Entraînement vs Validation')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(SAVED_MODELS_DIR, f'{model_name.lower()}_loss.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Courbe de perte sauvegardée dans {save_path}")


def predict_ncf(model, user_idx, n_items):
    """
    Calcule les scores NCF pour tous les articles pour un utilisateur donné.
    Paramètre :
      user_idx : indice entier de l'utilisateur dans l'embedding (user2idx[user_id])
    Retourne :
      scores : tableau numpy de taille n_items (probabilité d'achat pour chaque article)
    """
    # Crée un tableau rempli de l'indice utilisateur (un score par article)
    user_array = np.full(n_items, user_idx)
    item_array = np.arange(n_items)  # Tous les indices d'articles

    # Prédiction en batch (batch_size=1024 pour économiser la mémoire)
    scores = model.predict([user_array, item_array], verbose=0, batch_size=1024).flatten()
    return scores


# ─── Point d'entrée direct ───────────────────────────────────────────────────
if __name__ == '__main__':
    from preprocessing import run_preprocessing
    _, cim, _, _, mappings = run_preprocessing()
    users = list(cim.index)
    train_users = users[:int(len(users) * 0.8)]
    model, hist = train_ncf(cim, mappings, train_users)
    print("Entraînement NCF terminé.")
