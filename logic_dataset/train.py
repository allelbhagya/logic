
import json
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

tf.keras.utils.set_random_seed(42)

with open("data/logic_gates.json") as f:
    data = json.load(f)

labels = {
    "AND": 0,
    "OR": 1,
    "XOR": 2
}

X = []
y = []

for sample in data:

    W1 = sample["weights"]["W1"]
    W2 = sample["weights"]["W2"]

    b1 = sample["biases"]["b1"]
    b2 = sample["biases"]["b2"]

    # 3 tokens per network / 3 tokens per sample
    tokens = [
        [W1[0][0], W1[0][1], b1[0], 0],
        [W1[1][0], W1[1][1], b1[1], 0],
        [W2[0][0], W2[0][1], b2[0], 1]
    ]

    X.append(tokens)
    y.append(labels[sample["gate"]])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("X shape:", X.shape)
print("y shape:", y.shape)


indices = np.random.permutation(len(X))

split = int(0.8 * len(X))

train_idx = indices[:split]
val_idx = indices[split:]

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]


def transformer_block(x):
    # base mha
    attention = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=8,
        dropout=0.0
    )(x, x)

    # layer norm + residual
    x = layers.LayerNormalization()(
        x + attention
    )

    ## ff - mlp
    ff = layers.Dense(32, activation="gelu")(x)
    ff = layers.Dense(16)(ff)

    x = layers.LayerNormalization()(
        x + ff
    )

    return x


inputs = keras.Input(shape=(3, 4))

x = layers.Dense(16)(inputs)

x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)

outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=4,
    verbose=1
)

loss, accuracy = model.evaluate(X_val, y_val)

print("val acc:", accuracy)

model.save("logic_transformer.keras")
