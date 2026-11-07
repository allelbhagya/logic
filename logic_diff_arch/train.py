import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import confusion_matrix, classification_report

tf.keras.utils.set_random_seed(42)

with open("data/logic_gates_mixed.json") as f:
    data = json.load(f)

labels = {"AND": 0, "OR": 1, "XOR": 2}
acts = {"sigmoid": 0, "tanh": 1, "relu": 2}

MAX_IN = 8
MAX_TOK = 9
MAX_LAYERS = 3
FEAT = MAX_IN + 2
D = 16

holdout = sys.argv[1] if len(sys.argv) > 1 else None


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


N = len(data)
X = np.zeros((N, MAX_TOK, FEAT), dtype=np.float32)
L = np.zeros((N, MAX_TOK), dtype=np.int32)
M = np.zeros((N, MAX_TOK + 1), dtype=bool)
A = np.zeros((N, 1), dtype=np.int32)
y = np.zeros(N, dtype=np.int32)
names = np.array([s["arch_name"] for s in data])

M[:, 0] = True

for n, sample in enumerate(data):

    n_layers = len(sample["architecture"]) - 1
    k = 0

    for i in range(n_layers):
        w = np.array(sample["weights"][f"w{i + 1}"])
        b = np.array(sample["biases"][f"b{i + 1}"])

        for j in range(w.shape[0]):
            X[n, k, :w.shape[1]] = slog(w[j])
            X[n, k, MAX_IN] = slog(b[j])
            X[n, k, MAX_IN + 1] = float(i == n_layers - 1)
            L[n, k] = i
            M[n, k + 1] = True
            k += 1

    A[n, 0] = acts[sample["activation"]]
    y[n] = labels[sample["gate"]]

print("X shape:", X.shape)
print("y shape:", y.shape)

if holdout:
    val_idx = np.where(names == holdout)[0]
    train_idx = np.where(names != holdout)[0]
    print("holdout arch:", holdout)
else:
    indices = np.random.permutation(N)
    split = int(0.8 * N)
    train_idx = indices[:split]
    val_idx = indices[split:]


def take(idx):
    return [X[idx], L[idx], M[idx], A[idx]], y[idx]


x_train, y_train = take(train_idx)
x_val, y_val = take(val_idx)


def transformer_block(x, attn_mask):
    attention = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=8,
        dropout=0.0
    )(x, x, attention_mask=attn_mask)

    x = layers.LayerNormalization()(x + attention)

    ff = layers.Dense(32, activation="gelu")(x)
    ff = layers.Dense(16)(ff)

    x = layers.LayerNormalization()(x + ff)

    return x


tok_in = keras.Input(shape=(MAX_TOK, FEAT))
lid_in = keras.Input(shape=(MAX_TOK,), dtype="int32")
mask_in = keras.Input(shape=(MAX_TOK + 1,), dtype="bool")
act_in = keras.Input(shape=(1,), dtype="int32")

x = layers.Add()([
    layers.Dense(D)(tok_in),
    layers.Embedding(MAX_LAYERS, D)(lid_in)
])

cls = layers.Embedding(len(acts), D)(act_in)
x = layers.Concatenate(axis=1)([cls, x])

attn_mask = layers.Lambda(
    lambda m: tf.tile(m[:, None, :], [1, MAX_TOK + 1, 1])
)(mask_in)

x = transformer_block(x, attn_mask)
x = layers.Lambda(lambda t: t[:, 0])(x)

outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model([tok_in, lid_in, mask_in, act_in], outputs)

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)

loss, accuracy = model.evaluate(x_val, y_val)

print("val acc:", accuracy)

y_pred = np.argmax(model.predict(x_val), axis=1)

print("\nconfusion matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nclassification report:")
print(classification_report(
    y_val,
    y_pred,
    target_names=["AND", "OR", "XOR"],
    zero_division=0
))

print("per arch acc:")
val_names = names[val_idx]
for name in np.unique(val_names):
    sel = val_names == name
    print(f"{name}: {(y_pred[sel] == y_val[sel]).mean():.3f} ({sel.sum()})")

Path("model").mkdir(exist_ok=True)
model.save("model/logic_transformer.keras")