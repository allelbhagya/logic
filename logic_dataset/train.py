
import json
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import confusion_matrix, classification_report

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

    w1 = sample["weights"]["w1"]
    w2 = sample["weights"]["w2"]

    b1 = sample["biases"]["b1"]
    b2 = sample["biases"]["b2"]

    # 3 tokens per network / 3 tokens per sample
    tokens = [
        [w1[0][0], w1[0][1], b1[0], 0],
        [w1[1][0], w1[1][1], b1[1], 0],
        [w2[0][0], w2[0][1], b2[0], 1]
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
    epochs=50,
    batch_size=4,
    verbose=1
)

loss, accuracy = model.evaluate(X_val, y_val)

print("val acc:", accuracy)

y_pred = np.argmax(model.predict(X_val), axis=1)

print("\nconfusion matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nclassification report:")
print(classification_report(
    y_val,
    y_pred,
    target_names=["AND", "OR", "XOR"],
    zero_division=0
))

model.save("model/logic_transformer.keras")
