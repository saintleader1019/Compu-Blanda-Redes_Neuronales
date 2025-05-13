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

# --- Inicialización de pesos y sesgos ---
def init_params(input_size, hidden_size, output_size):
    W1 = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# --- Forward propagation ---
def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1.T + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2.T + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# --- Backpropagation ---
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

# --- Entrenamiento del modelo ---
def train(X, Y, input_size=2, hidden_size=10, output_size=3, lr=0.1, epochs=1000):
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

# --- Predicción de clases ---
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)

# --- Codificación One-Hot ---
def one_hot_encode(rgb_list):
    color_map = {(1, 0, 0): 0, (0, 1, 0): 1, (0, 0, 1): 2}
    indices = [color_map[c] for c in rgb_list]
    one_hot = np.zeros((len(indices), 3))
    one_hot[np.arange(len(indices)), indices] = 1
    return one_hot

# --- Diccionario inverso para interpretación de resultados ---
index_to_color = {0: 'purple', 1: 'orange', 2: 'green'}

# === DATOS ===
# Entrenamiento
inputs = [...]
targets_rgb = [...]

# Prueba
test_inputs = [...]
test_targets_rgb = [...]

# Convertir a NumPy
X_train = np.array(inputs)
Y_train = one_hot_encode(targets_rgb)
X_test = np.array(test_inputs)
Y_test = one_hot_encode(test_targets_rgb)

# === ENTRENAMIENTO ===
W1, b1, W2, b2, losses = train(X_train, Y_train, hidden_size=20, epochs=1000, lr=0.1)

# === PREDICCIÓN Y EVALUACIÓN ===
preds = predict(X_test, W1, b1, W2, b2)
true_labels = np.argmax(Y_test, axis=1)
accuracy = np.mean(preds == true_labels)
print(f"\nExactitud en el conjunto de prueba: {accuracy * 100:.2f}%")

# === DETALLE POR COORDENADA ===
print("\n=== RESULTADOS DE PRUEBA ===")
for coord, pred_idx, true_idx in zip(X_test, preds, true_labels):
    pred_color = index_to_color[pred_idx]
    true_color = index_to_color[true_idx]
    print(f"Coordenada {tuple(coord)} → Predicho: {pred_color} | Real: {true_color}")

# === GRAFICAR PÉRDIDA ===
plt.plot(losses)
plt.title("Evolución del error (entropía cruzada)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()

# === GRAFICAR CLASIFICACIÓN FINAL ===
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
