"""
model_lstm.py — Modèle LSTM pour la recommandation séquentielle.
Principe : apprend à prédire le prochain article acheté en fonction
des derniers articles de la séquence d'achats de l'utilisateur.
C'est le modèle le plus performant (~30 % de précision@5) car il capture
les patterns temporels et séquentiels des comportements d'achat.
"""
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Supprime les logs verbeux de TensorFlow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')  # Backend sans affichage (serveur sans écran)
import matplotlib.pyplot as plt
from config import (LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LR, LSTM_SEQ_LEN,
                    LSTM_UNITS, SAVED_MODELS_DIR, SEED)

# Fixe les graines pour des résultats reproductibles
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_lstm(n_items, seq_len=LSTM_SEQ_LEN, units=LSTM_UNITS, embed_dim=64):
    """
    Construit le modèle LSTM empilé pour la prédiction du prochain article.
    Architecture :
      Entrée   : séquence de seq_len indices d'articles (fenêtre glissante)
      Embedding : indice → vecteur dense de 64 dimensions (mask_zero=True ignore le padding)
      LSTM(256) : capture les dépendances à long terme + retourne toutes les sorties
      Dropout(0.3)
      LSTM(128) : affine la représentation + retourne uniquement la dernière sortie
      Dropout(0.3)
      Dense(128, relu) : couche intermédiaire
      Dropout(0.2)
      Dense(n_items, softmax) : distribution de probabilité sur tous les articles

    LSTM_UNITS=256 (augmenté de 128) → plus de capacité pour les séquences longues.
    mask_zero=True : ignore les positions paddées (0) dans le calcul de l'attention LSTM.
    """
    inputs = keras.Input(shape=(seq_len,))

    # Embedding : convertit les indices d'articles en vecteurs denses
    # mask_zero=True : les 0 de padding sont ignorés par le LSTM
    x = layers.Embedding(n_items + 1, embed_dim, mask_zero=True)(inputs)

    # Premier LSTM : return_sequences=True → passe toute la séquence au LSTM suivant
    x = layers.LSTM(units, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)

    # Deuxième LSTM : return_sequences=False → ne retourne que la dernière sortie
    x = layers.LSTM(units // 2, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    # Couche dense intermédiaire pour affiner la représentation
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Couche de sortie : probabilité pour chaque article d'être le prochain acheté
    output = layers.Dense(n_items, activation='softmax')(x)

    model = keras.Model(inputs, output, name='lstm_seq')
    return model


def prepare_sequences(sequences, item2idx, seq_len=LSTM_SEQ_LEN):
    """
    Convertit les séquences d'achats brutes en paires (entrée, cible) pour l'entraînement.
    Principe de fenêtre glissante :
      Séquence : [A, B, C, D, E]
      Paires générées :
        ([A], B), ([A, B], C), ([A, B, C], D), ([A, B, C, D], E)
    Les fenêtres sont paddées à gauche avec des 0 pour atteindre seq_len.
    Les articles inconnus (non présents dans item2idx) sont ignorés.
    """
    print("  Préparation des séquences LSTM...")
    X, y = [], []
    n_items = len(item2idx)

    for cid, items in sequences:
        # Convertit les codes articles en indices entiers (ignore les articles inconnus)
        indices = [item2idx[it] for it in items if it in item2idx]
        if len(indices) < 2:
            continue  # Minimum 2 articles pour créer une paire (entrée, cible)

        # Génère toutes les sous-séquences possibles avec fenêtre glissante
        for i in range(1, len(indices)):
            # Contexte : les i derniers articles (ou seq_len si i > seq_len)
            start = max(0, i - seq_len)
            seq   = indices[start:i]     # Sous-séquence de contexte
            target = indices[i]           # Article à prédire

            # Padding à gauche avec des 0 pour atteindre seq_len
            if len(seq) < seq_len:
                seq = [0] * (seq_len - len(seq)) + seq

            X.append(seq)
            y.append(target)

    X = np.array(X)
    y = np.array(y)
    print(f"    Paires de séquences : {len(X)}")
    return X, y


def train_lstm(sequences, mappings, train_users):
    """
    Entraîne le modèle LSTM et sauvegarde les poids.
    Paramètres :
      sequences   : liste de tuples (customer_id, [article1, article2, ...])
      mappings    : dictionnaire des mappings (item2idx, n_items, etc.)
      train_users : ensemble des utilisateurs à inclure dans l'entraînement
                    (typiquement train + val = 80 % de tous les utilisateurs)
    """
    print("Entraînement du LSTM...")
    n_items  = mappings['n_items']
    item2idx = mappings['item2idx']

    # Filtre les séquences aux seuls utilisateurs d'entraînement
    train_seqs = [(cid, items) for cid, items in sequences if cid in set(train_users)]

    # Prépare les paires (séquence_contexte, article_cible)
    X, y = prepare_sequences(train_seqs, item2idx)
    if len(X) == 0:
        print("  Aucune séquence valide pour l'entraînement LSTM.")
        return None, None

    # Convertit les cibles en encodage one-hot (requis par categorical_crossentropy)
    # ex. indice 42 → vecteur de taille n_items avec 1 à la position 42
    y_cat = keras.utils.to_categorical(y, num_classes=n_items)

    # Division 90 % train / 10 % validation
    split  = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y_cat[:split], y_cat[split:]

    # Construction et compilation du modèle
    model     = build_lstm(n_items)
    optimizer = keras.optimizers.Adam(learning_rate=LSTM_LR)
    # Perte adaptée à la prédiction multi-classes (n_items classes)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Réduit le taux d'apprentissage si la val_loss stagne
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    # Arrêt anticipé
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    # Entraînement
    history = model.fit(
        X_train, y_train,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[lr_scheduler, early_stop],
        verbose=1
    )

    # Sauvegarde du modèle
    save_path = os.path.join(SAVED_MODELS_DIR, 'lstm.keras')
    model.save(save_path)
    print(f"  LSTM sauvegardé dans {save_path}")

    plot_loss(history, 'LSTM')
    return model, history


def plot_loss(history, model_name):
    """Trace et sauvegarde la courbe de perte entraînement vs validation."""
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


def predict_lstm(model, item_sequence, item2idx, n_items, seq_len=LSTM_SEQ_LEN):
    """
    Prédit la distribution de probabilité du prochain article à partir d'une séquence.
    Paramètre :
      item_sequence : liste de codes articles dans l'ordre chronologique
    Retourne :
      scores : tableau numpy de taille n_items (probabilité pour chaque article)
    """
    # Convertit les codes en indices (0 pour les articles inconnus)
    indices = [item2idx.get(it, 0) for it in item_sequence]

    # Garde uniquement les seq_len derniers articles (fenêtre la plus récente)
    indices = indices[-seq_len:]

    # Padding à gauche avec des 0 si la séquence est trop courte
    if len(indices) < seq_len:
        indices = [0] * (seq_len - len(indices)) + indices

    X = np.array(indices).reshape(1, -1)   # Ajoute la dimension batch
    scores = model.predict(X, verbose=0)[0]  # Supprime la dimension batch
    return scores


# ─── Point d'entrée direct ───────────────────────────────────────────────────
if __name__ == '__main__':
    from preprocessing import run_preprocessing
    _, cim, _, sequences, mappings = run_preprocessing()
    users = list(cim.index)
    train_users = users[:int(len(users) * 0.8)]
    model, hist = train_lstm(sequences, mappings, train_users)
    print("Entraînement LSTM terminé.")
