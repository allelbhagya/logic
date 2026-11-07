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

architectures = {
    "2-2-1_sigmoid":     {"architecture": [2, 2, 1],    "activation": "sigmoid"},
    "2-3-1_sigmoid":     {"architecture": [2, 3, 1],    "activation": "sigmoid"},
    "2-4-1_sigmoid":     {"architecture": [2, 4, 1],    "activation": "sigmoid"},
    "2-4-1_tanh":        {"architecture": [2, 4, 1],    "activation": "tanh"},
    "2-8-1_tanh":        {"architecture": [2, 8, 1],    "activation": "tanh"},
    "2-4-1_relu":        {"architecture": [2, 4, 1],    "activation": "relu"},
    "2-3-3-1_tanh":      {"architecture": [2, 3, 3, 1], "activation": "tanh"},
    "2-4-4-1_sigmoid":   {"architecture": [2, 4, 4, 1], "activation": "sigmoid"},
}


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def activate(x, name):
    if name == "sigmoid":
        return sigmoid(x)
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(0, x)
    raise ValueError(f"unknown activation: {name}")


def activate_grad(a, name):
    if name == "sigmoid":
        return a * (1 - a)
    if name == "tanh":
        return 1 - a ** 2
    if name == "relu":
        return (a > 0).astype(np.float64)
    raise ValueError(f"unknown activation: {name}")


def forward(X, weights, biases, activation):
    a = X
    n_layers = len(weights)

    for i in range(n_layers):
        z = a @ weights[i].T + biases[i]
        a = sigmoid(z) if i == n_layers - 1 else activate(z, activation)

    return a.squeeze(-1)


def train_network(X, y, architecture, activation, epochs=10000, lr=0.5):

    weights = []
    biases = []

    for i in range(len(architecture) - 1):
        weights.append(rng.normal(0, 1, (architecture[i + 1], architecture[i])))
        biases.append(np.zeros(architecture[i + 1]))

    n_layers = len(weights)

    for epoch in range(epochs):

        # forward, keep activations of every layer for backprop
        activations = [X]
        for i in range(n_layers):
            z = activations[-1] @ weights[i].T + biases[i]
            a = sigmoid(z) if i == n_layers - 1 else activate(z, activation)
            activations.append(a)

        output = activations[-1]

        # backward
        dz = (output - y.reshape(-1, 1)) / len(X)

        for i in reversed(range(n_layers)):
            dw = dz.T @ activations[i]
            db = dz.sum(axis=0)

            if i > 0:
                dh = dz @ weights[i]
                dz = dh * activate_grad(activations[i], activation)

            weights[i] -= lr * dw
            biases[i] -= lr * db

    return weights, biases


def generate_sample(gate, architecture, activation, max_attempts=30):

    y = gates[gate]

    for attempt in range(max_attempts):

        weights, biases = train_network(X, y, architecture, activation)

        predictions = forward(X, weights, biases, activation)
        predicted_labels = (predictions >= 0.5).astype(int)

        if np.array_equal(predicted_labels, y):
            return {
                "architecture": architecture,
                "activation": activation,
                "weights": {f"w{i + 1}": w.tolist() for i, w in enumerate(weights)},
                "biases": {f"b{i + 1}": b.tolist() for i, b in enumerate(biases)},
                "gate": gate,
                "predictions": predictions.tolist()
            }

    return None


def generate_dataset(arch_name, samples_per_gate=100):

    config = architectures[arch_name]
    dataset = []

    for gate in gates:

        generated = 0
        failed = 0

        while generated < samples_per_gate:

            sample = generate_sample(gate, config["architecture"], config["activation"])

            if sample is None:
                failed += 1
                if failed >= 5:
                    break
                continue

            sample["arch_name"] = arch_name
            dataset.append(sample)
            generated += 1

        print(f"{arch_name} | {gate}: generated {generated} samples")

    return dataset


def generate_all(samples_per_gate=100):

    per_arch = {}
    mixed = []

    for arch_name in architectures:
        per_arch[arch_name] = generate_dataset(arch_name, samples_per_gate)
        mixed.extend(per_arch[arch_name])

    rng.shuffle(mixed)

    return per_arch, mixed


if __name__ == "__main__":

    per_arch, mixed = generate_all(samples_per_gate=100)

    Path("data/by_arch").mkdir(parents=True, exist_ok=True)

    for arch_name, dataset in per_arch.items():
        with open(f"data/by_arch/{arch_name}.json", "w") as f:
            json.dump(dataset, f, indent=2)

    with open("data/logic_gates_mixed.json", "w") as f:
        json.dump(mixed, f, indent=2)

    print("\nTotal samples:", len(mixed))