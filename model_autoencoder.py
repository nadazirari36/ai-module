"""
model_autoencoder.py — AutoEncodeur de débruitage pour le filtrage collaboratif.
Principe : reconstruit le vecteur binaire d'achats de l'utilisateur à partir
d'une version bruitée, forçant le modèle à apprendre des représentations robustes
plutôt que de simplement mémoriser l'identité.
Les articles avec un score de reconstruction élevé sont les candidats à recommander.
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
from config import (AE_EPOCHS, AE_BATCH_SIZE, AE_LR, AE_DROPOUT, AE_PATIENCE,
                    AE_POS_WEIGHT, AE_NOISE, SAVED_MODELS_DIR, SEED)

# Fixe les graines pour des résultats reproductibles
np.random.seed(SEED)
tf.random.set_seed(SEED)


def weighted_bce(pos_weight=10.0):
    """
    Perte de type entropie croisée binaire avec poids sur les positifs.
    Problème : la matrice d'achats est très creuse (peu de 1 parmi des milliers de 0).
    Sans pondération, le modèle apprend à tout prédire à 0 et obtient malgré tout
    une bonne précision globale.
    Solution : on pénalise davantage les faux négatifs (articles achetés mais non prédits).
    AE_POS_WEIGHT = 15 (réduit de 40) → moins de faux positifs, meilleure précision.
    """
    def loss(y_true, y_pred):
        # Évite log(0) en clippant les prédictions dans ]0, 1[
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Poids = 1 pour les négatifs, pos_weight pour les positifs
        weights = 1.0 + (pos_weight - 1.0) * y_true

        # Calcul de la BCE standard
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))

        # Applique les poids et retourne la moyenne
        return tf.reduce_mean(weights * bce)
    return loss


def build_autoencoder(n_items, dropout=AE_DROPOUT):
    """
    Construit l'AutoEncodeur de débruitage avec un goulot d'étranglement élargi.
    Architecture :
      Entrée  : vecteur binaire de taille n_items (3665 articles)
      Encodeur : 3665 → 512 → 256 → 256 (goulot d'étranglement)
      Décodeur : 256 → 256 → 512 → 3665 (sigmoïde)

    Le bruit gaussien (AE_NOISE=0.15) corrompt l'entrée pendant l'entraînement,
    obligeant le modèle à reconstruire à partir d'un signal incomplet.
    Le goulot à 256 (au lieu de 128) offre plus de capacité pour capturer
    les co-achats d'articles parmi les 3665 produits.
    """
    inputs = keras.Input(shape=(n_items,))

    # Bruit gaussien appliqué uniquement pendant l'entraînement (désactivé en inférence)
    # AE_NOISE=0.15 (réduit de 0.3) — corruption plus douce sur vecteurs binaires creux
    noisy = layers.GaussianNoise(AE_NOISE)(inputs)

    # ── Encodeur ──────────────────────────────────────────────────────────────
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(noisy)
    x = layers.BatchNormalization()(x)  # Stabilise l'entraînement, accélère la convergence
    x = layers.Dropout(dropout)(x)      # Régularisation : désactive aléatoirement des neurones

    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    # Goulot d'étranglement : représentation compressée de l'utilisateur (256 dimensions)
    # Élargi de 128 → 256 pour mieux capturer les motifs de co-achat
    encoded = layers.Dense(256, activation='relu')(x)

    # ── Décodeur (symétrique à l'encodeur) ───────────────────────────────────
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    # Sortie : score de probabilité d'achat pour chaque article (sigmoïde → [0, 1])
    decoded = layers.Dense(n_items, activation='sigmoid')(x)

    model = keras.Model(inputs, decoded, name='autoencoder')
    return model


def train_autoencoder(train_matrix, val_matrix, n_items):
    """
    Entraîne l'AutoEncodeur et sauvegarde le modèle.
    Paramètres :
      train_matrix : tableau numpy (n_train_users, n_items) — vecteurs d'achats en entrée ET cible
      val_matrix   : tableau numpy (n_val_users,  n_items) — pour l'arrêt anticipé
      n_items      : nombre total d'articles distincts
    L'AE est entraîné de façon non supervisée : l'entrée ET la cible sont le même vecteur.
    Il apprend à reconstruire ce qu'un utilisateur a acheté.
    """
    print("Entraînement de l'AutoEncodeur...")
    model = build_autoencoder(n_items)

    # Optimiseur Adam avec taux d'apprentissage initial
    optimizer = keras.optimizers.Adam(learning_rate=AE_LR)

    # Perte BCE pondérée pour gérer la rareté des achats positifs
    model.compile(optimizer=optimizer, loss=weighted_bce(pos_weight=AE_POS_WEIGHT))

    # Réduit le taux d'apprentissage si la val_loss stagne (facteur 0.5)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
    )

    # Arrêt anticipé : stoppe si val_loss ne s'améliore pas pendant AE_PATIENCE époques
    # restore_best_weights=True : récupère les poids de la meilleure époque
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=AE_PATIENCE, restore_best_weights=True
    )

    # Entraînement : entrée = vecteur bruité (via GaussianNoise), cible = vecteur original
    history = model.fit(
        train_matrix, train_matrix,          # entrée et cible identiques (auto-encodage)
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        validation_data=(val_matrix, val_matrix),
        callbacks=[lr_scheduler, early_stop],
        verbose=1
    )

    # Sauvegarde du modèle entraîné
    save_path = os.path.join(SAVED_MODELS_DIR, 'autoencoder.keras')
    model.save(save_path)
    print(f"  AutoEncodeur sauvegardé dans {save_path}")

    plot_loss(history, 'AutoEncoder')
    return model, history


def plot_loss(history, model_name):
    """
    Trace et sauvegarde la courbe de perte entraînement vs validation.
    Utile pour diagnostiquer le sur-apprentissage (overfitting) ou le sous-apprentissage.
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


def predict_autoencoder(model, user_vector):
    """
    Prédit les scores de recommandation pour un utilisateur donné.
    Paramètre :
      user_vector : tableau numpy binaire de taille n_items
                    (1 = article acheté, 0 = non acheté)
    Retourne :
      scores : tableau numpy de taille n_items
               (valeur élevée = l'AE prédit que cet article correspond au profil de l'utilisateur)
    """
    user_input = np.array(user_vector).reshape(1, -1)  # Ajoute la dimension batch
    scores = model.predict(user_input, verbose=0)[0]    # Supprime la dimension batch
    return scores


# ─── Point d'entrée direct ───────────────────────────────────────────────────
if __name__ == '__main__':
    from preprocessing import run_preprocessing
    _, customer_item_matrix, _, _, mappings = run_preprocessing()

    matrix = customer_item_matrix.values.astype(np.float32)
    n = len(matrix)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)

    train_m = matrix[:n_train]
    val_m   = matrix[n_train:n_train + n_val]

    model, hist = train_autoencoder(train_m, val_m, mappings['n_items'])
    print("Entraînement AutoEncodeur terminé.")
