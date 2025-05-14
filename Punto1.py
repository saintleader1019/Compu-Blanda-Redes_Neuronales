import numpy as np
import matplotlib.pyplot as plt

# --- Funciones de activación ---
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # estabilidad numérica
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8), axis=1)

# --- Inicialización ---
def init_params(input_size, hidden_size, output_size):
    W1 = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# --- Forward ---
def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1.T + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2.T + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# --- Backward ---
def backward(X, Y, Z1, A1, A2, W2):
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = (dZ2.T @ A1) / m
    db2 = np.mean(dZ2, axis=0, keepdims=True)
    dA1 = dZ2 @ W2
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (dZ1.T @ X) / m
    db1 = np.mean(dZ1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2

# --- Entrenamiento ---
def train(X, Y, input_size=2, hidden_size=20, output_size=3, lr=0.1, epochs=1000):
    W1, b1, W2, b2 = init_params(input_size, hidden_size, output_size)
    losses = []
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
        loss = np.mean(cross_entropy(Y, A2))
        losses.append(loss)
        dW1, db1, dW2, db2 = backward(X, Y, Z1, A1, A2, W2)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return W1, b1, W2, b2, losses

# --- Predicción ---
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)

# --- One-hot ---
def one_hot_encode(rgb_list):
    color_map = {(1, 0, 0): 0, (0, 1, 0): 1, (0, 0, 1): 2}
    indices = [color_map[tuple(c)] for c in rgb_list]
    one_hot = np.zeros((len(indices), 3))
    one_hot[np.arange(len(indices)), indices] = 1
    return one_hot

index_to_color = {0: 'purple', 1: 'orange', 2: 'green'}

# === Datos ===
inputs = [
    (0.0000, 0.3929), (0.5484, 0.7500), (0.0645, 0.5714), (0.5806, 0.5714), (0.2258, 0.8929),
    (0.4839, 0.2500), (0.3226, 0.2143), (0.7742, 0.8214), (0.4516, 0.5000), (0.4194, 0.0357),
    (0.4839, 0.2500), (0.3226, 0.7143), (0.5806, 0.5000), (0.5484, 0.1071), (0.6129, 0.6429),
    (0.6774, 0.1786), (0.2258, 0.8214), (0.7419, 0.1429), (0.6452, 1.0000), (0.8387, 0.2500),
    (0.9677, 0.3214), (0.3226, 0.4643), (0.3871, 0.5357), (0.3548, 0.1429), (0.3548, 0.6429),
    (0.1935, 0.4643), (0.4516, 0.3929), (0.4839, 0.6071), (0.6129, 0.6786), (0.2258, 0.6071),
    (0.5161, 0.3214), (0.5484, 0.6786), (0.3871, 0.8571), (0.6452, 0.6071), (0.1935, 0.3929),
    (0.6452, 0.3929), (0.6774, 0.4643), (0.3226, 0.2857), (0.7419, 0.7143), (0.7419, 0.3214),
    (1.0000, 0.3929), (0.8065, 0.3929), (0.1935, 0.5000), (0.1613, 0.8214), (0.2903, 0.9286),
    (0.3548, 0.0000), (0.2903, 0.6786), (0.5484, 0.9643), (0.4194, 0.1786), (0.2581, 0.2500),
    (0.3226, 0.7143), (0.5161, 0.3929), (0.2903, 0.6429), (0.5484, 0.9286), (0.2581, 0.3214),
    (0.0968, 0.5000), (0.6129, 0.7857), (0.0968, 0.3214), (0.6452, 0.9286), (0.8065, 0.7500)
]
targets_rgb = [
    (1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0),
    (1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 0, 0), (0, 0, 1), (1, 0, 0),
    (1, 0, 0), (0, 1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 0, 1),
    (0, 1, 0), (0, 1, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0),
    (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1), (0, 0, 1), (1, 0, 0), (0, 0, 1), (0, 0, 1), (1, 0, 0), (1, 0, 0),
    (0, 0, 1), (0, 1, 0), (0, 0, 1), (0, 0, 1), (1, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 0), (0, 0, 1), (0, 0, 1)
]

# === Datos de prueba ===
test_inputs = [
    (0.0000, 0.3929), (0.0645, 0.5714), (0.0968, 0.3214), (0.0968, 0.5000), (0.2581, 0.3214),
    (0.1935, 0.4643), (0.2581, 0.2500), (0.1935, 0.3929), (0.3226, 0.2143), (0.4839, 0.2500),
    (0.3226, 0.4643), (0.3871, 0.5357), (0.3548, 0.6429), (0.4516, 0.5000), (0.4516, 0.3929),
    (0.5161, 0.3929), (0.5484, 0.7500), (0.6129, 0.6786), (0.5161, 0.3214), (0.5484, 0.6786),
    (0.1935, 0.5000), (0.2258, 0.6071), (0.3226, 0.7143), (0.2903, 0.6786), (0.3226, 0.7143),
    (0.2258, 0.8214), (0.2903, 0.6429), (0.6129, 0.7857), (0.7742, 0.8214), (0.8065, 0.7500)
]
test_targets_rgb = [(1, 0, 0)] * 10 + [(0, 1, 0)] * 10 + [(0, 0, 1)] * 10

# === Preprocesamiento ===
X_train = np.array(inputs)
Y_train = one_hot_encode(targets_rgb)
X_test = np.array(test_inputs)
Y_test = one_hot_encode(test_targets_rgb)

# === Entrenamiento ===
W1, b1, W2, b2, losses = train(X_train, Y_train)

# === Evaluación ===
preds = predict(X_test, W1, b1, W2, b2)
true_labels = np.argmax(Y_test, axis=1)
accuracy = np.mean(preds == true_labels)
print(f"\nExactitud en el conjunto de prueba: {accuracy * 100:.2f}%")

print("\n=== RESULTADOS DE PRUEBA ===")
for coord, pred_idx, true_idx in zip(X_test, preds, true_labels):
    pred_color = index_to_color[pred_idx]
    true_color = index_to_color[true_idx]
    print(f"Coordenada ({float(coord[0]):.4f}, {float(coord[1]):.4f}) → Predicho: {pred_color} | Real: {true_color}")

# === Visualización de pérdida ===
plt.plot(losses)
plt.title("Evolución del error (entropía cruzada)")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.grid()
plt.show()

# === Visualización de clasificación ===
plt.figure(figsize=(6, 6))
colors = ['purple', 'orange', 'green']
for i, color in enumerate(colors):
    idxs = preds == i
    plt.scatter(X_test[idxs, 0], X_test[idxs, 1], label=f'{color} (pred)', alpha=0.6)

plt.title("Clasificación de puntos de prueba")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
plt.show()

