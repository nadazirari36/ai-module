"""
AutoEncoder for Collaborative Filtering.
Reconstructs binary purchase vectors to surface unvisited items.
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
from config import (AE_EPOCHS, AE_BATCH_SIZE, AE_LR, AE_DROPOUT, AE_PATIENCE,
                    SAVED_MODELS_DIR, SEED)

np.random.seed(SEED)
tf.random.set_seed(SEED)


def weighted_bce(pos_weight=10.0):
    """BCE with higher weight on positive (purchased) interactions to handle sparse data."""
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        weights = 1.0 + (pos_weight - 1.0) * y_true
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(weights * bce)
    return loss


def build_autoencoder(n_items, dropout=AE_DROPOUT):
    """Build Denoising AutoEncoder — GaussianNoise on input forces robust latent representations on sparse data."""
    inputs = keras.Input(shape=(n_items,))

    # Denoising: corrupt inputs during training so the model can't memorise identity
    noisy = layers.GaussianNoise(0.3)(inputs)

    # Encoder
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(noisy)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    encoded = layers.Dense(128, activation='relu')(x)

    # Decoder
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    decoded = layers.Dense(n_items, activation='sigmoid')(x)

    model = keras.Model(inputs, decoded, name='autoencoder')
    return model


def train_autoencoder(train_matrix, val_matrix, n_items):
    """Train AutoEncoder and save model."""
    print("Training AutoEncoder...")
    model = build_autoencoder(n_items)

    optimizer = keras.optimizers.Adam(learning_rate=AE_LR)
    model.compile(optimizer=optimizer, loss=weighted_bce(pos_weight=40.0))

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=AE_PATIENCE, restore_best_weights=True
    )

    history = model.fit(
        train_matrix, train_matrix,
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        validation_data=(val_matrix, val_matrix),
        callbacks=[lr_scheduler, early_stop],
        verbose=1
    )

    save_path = os.path.join(SAVED_MODELS_DIR, 'autoencoder.keras')
    model.save(save_path)
    print(f"  AutoEncoder saved to {save_path}")

    # Plot loss curves
    plot_loss(history, 'AutoEncoder')
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


def predict_autoencoder(model, user_vector):
    """Get reconstruction scores for a user."""
    user_input = np.array(user_vector).reshape(1, -1)
    scores = model.predict(user_input, verbose=0)[0]
    return scores


if __name__ == '__main__':
    from preprocessing import run_preprocessing
    _, customer_item_matrix, _, _, mappings = run_preprocessing()

    matrix = customer_item_matrix.values.astype(np.float32)
    n = len(matrix)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_m = matrix[:n_train]
    val_m = matrix[n_train:n_train + n_val]

    model, hist = train_autoencoder(train_m, val_m, mappings['n_items'])
    print("AutoEncoder training complete.")
