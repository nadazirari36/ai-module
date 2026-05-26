"""
LSTM Sequential Model.
Predicts next likely product based on purchase sequence.
"""
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import (LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LR, LSTM_SEQ_LEN,
                    LSTM_UNITS, SAVED_MODELS_DIR, SEED)

np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_lstm(n_items, seq_len=LSTM_SEQ_LEN, units=LSTM_UNITS, embed_dim=64):
    """Build stacked LSTM model for sequence prediction."""
    inputs = keras.Input(shape=(seq_len,))

    x = layers.Embedding(n_items + 1, embed_dim, mask_zero=True)(inputs)
    x = layers.LSTM(units, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(units // 2, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(n_items, activation='softmax')(x)

    model = keras.Model(inputs, output, name='lstm_seq')
    return model


def prepare_sequences(sequences, item2idx, seq_len=LSTM_SEQ_LEN):
    """Convert purchase sequences to fixed-length input/target pairs."""
    print("  Preparing LSTM sequences...")
    X, y = [], []
    n_items = len(item2idx)

    for cid, items in sequences:
        # Convert to indices
        indices = [item2idx[it] for it in items if it in item2idx]
        if len(indices) < 2:
            continue

        # Create sliding windows
        for i in range(1, len(indices)):
            start = max(0, i - seq_len)
            seq = indices[start:i]
            target = indices[i]

            # Pad sequence
            if len(seq) < seq_len:
                seq = [0] * (seq_len - len(seq)) + seq

            X.append(seq)
            y.append(target)

    X = np.array(X)
    y = np.array(y)
    print(f"    Sequence pairs: {len(X)}")
    return X, y


def train_lstm(sequences, mappings, train_users):
    """Train LSTM model."""
    print("Training LSTM...")
    n_items = mappings['n_items']
    item2idx = mappings['item2idx']

    # Filter sequences to training users
    train_seqs = [(cid, items) for cid, items in sequences if cid in set(train_users)]

    X, y = prepare_sequences(train_seqs, item2idx)
    if len(X) == 0:
        print("  No valid sequences for LSTM training.")
        return None, None

    # Convert targets to categorical
    y_cat = keras.utils.to_categorical(y, num_classes=n_items)

    # Split
    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y_cat[:split], y_cat[split:]

    model = build_lstm(n_items)
    optimizer = keras.optimizers.Adam(learning_rate=LSTM_LR)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[lr_scheduler, early_stop],
        verbose=1
    )

    save_path = os.path.join(SAVED_MODELS_DIR, 'lstm.keras')
    model.save(save_path)
    print(f"  LSTM saved to {save_path}")

    plot_loss(history, 'LSTM')
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


def predict_lstm(model, item_sequence, item2idx, n_items, seq_len=LSTM_SEQ_LEN):
    """Predict next-item probabilities from a sequence."""
    indices = [item2idx.get(it, 0) for it in item_sequence]
    # Take last seq_len items
    indices = indices[-seq_len:]
    # Pad
    if len(indices) < seq_len:
        indices = [0] * (seq_len - len(indices)) + indices
    X = np.array(indices).reshape(1, -1)
    scores = model.predict(X, verbose=0)[0]
    return scores


if __name__ == '__main__':
    from preprocessing import run_preprocessing
    _, cim, _, sequences, mappings = run_preprocessing()
    users = list(cim.index)
    train_users = users[:int(len(users) * 0.8)]
    model, hist = train_lstm(sequences, mappings, train_users)
    print("LSTM training complete.")
