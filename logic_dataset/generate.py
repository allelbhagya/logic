import numpy as np
import json
from pathlib import Path

rng = np.random.default_rng(42)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float64)

gates = {
    "AND": np.array([0, 0, 0, 1]),
    "OR":  np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0])
}


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def forward(X, w1, b1, w2, b2):
    h = sigmoid(X @ w1.T + b1)
    output = sigmoid(h @ w2.T + b2)
    return output.squeeze(-1)


def train_network(X, y, epochs=10000, lr=0.5):
    """
    arch 3 layers: 
    2 input nodes in 1st layer
    2 nodes in hidden layer
    1 output node in last layer
    """

    w1 = rng.normal(0, 1, (2, 2))
    b1 = np.zeros(2)

    w2 = rng.normal(0, 1, (1, 2))
    b2 = np.zeros(1)

    for epoch in range(epochs):

        h = sigmoid(X @ w1.T + b1)
        output = sigmoid(h @ w2.T + b2)

        dz2 = (output - y.reshape(-1, 1)) / len(X)

        dw2 = dz2.T @ h
        db2 = dz2.sum(axis=0)

        dh = dz2 @ w2
        dz1 = dh * h * (1 - h)

        dw1 = dz1.T @ X
        db1 = dz1.sum(axis=0)

        w1 -= lr * dw1
        b1 -= lr * db1

        w2 -= lr * dw2
        b2 -= lr * db2

    return w1, b1, w2, b2


def generate_sample(gate, max_attempts=30):

    y = gates[gate]

    for attempt in range(max_attempts):

        w1, b1, w2, b2 = train_network(X, y)

        predictions = forward(X, w1, b1, w2, b2)
        predicted_labels = (predictions >= 0.5).astype(int)

        if np.array_equal(predicted_labels, y):
            return {
                "architecture": [2, 2, 1],
                "activation": "sigmoid",
                "weights": {
                    "w1": w1.tolist(),
                    "w2": w2.tolist()
                },
                "biases": {
                    "b1": b1.tolist(),
                    "b2": b2.tolist()
                },
                "gate": gate,
                "predictions": predictions.tolist()
            }

    return None


def generate_dataset(samples_per_gate=100):

    dataset = []

    for gate in gates:

        generated = 0
        attempts = 0

        while generated < samples_per_gate:

            sample = generate_sample(gate)
            attempts += 1

            if sample is not None:
                dataset.append(sample)
                generated += 1

        print(f"{gate}: generated {generated} samples")

    rng.shuffle(dataset)

    return dataset

dataset = generate_dataset(samples_per_gate=100)

Path("data").mkdir(exist_ok=True)

with open("data/logic_gates.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("\nTotal samples:", len(dataset))
