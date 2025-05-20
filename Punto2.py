import numpy as np
import pandas as pd

# Activaciones
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
    return -np.sum(y_true * np.log(y_pred), axis=1)

# Inicialización
def init_params(input_size, hidden_size, output_size):
    W1 = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# Forward
def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1.T + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2.T + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Backward
def backward(X, Y, Z1, A1, A2, W2):
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = (dZ2.T @ A1) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = dZ2 @ W2
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (dZ1.T @ X) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

# Entrenamiento
def train(X, Y, input_size=784, hidden_size=20, output_size=10, lr=0.1, epochs=1000):
    W1, b1, W2, b2 = init_params(input_size, hidden_size, output_size)
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
        loss = np.mean(cross_entropy(Y, A2))
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
        dW1, db1, dW2, db2 = backward(X, Y, Z1, A1, A2, W2)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    return W1, b1, W2, b2

# Predicción
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)

# One-hot
def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

# === Main: Cargar datos y entrenar ===
# Leer train.csv (asegúrate de tenerlo en el mismo directorio)
train_df = pd.read_csv("test.csv")  # <- cambia a ruta absoluta si hace falta

X = train_df.drop(columns=["label"]).values / 255.0
y = train_df["label"].values
Y = one_hot_encode(y)

# Usamos los primeros 10000 datos como muestra de entrenamiento
X_sample = X[:10000]
Y_sample = Y[:10000]
y_sample = y[:10000]

# Entrenar red neuronal
W1, b1, W2, b2 = train(X_sample, Y_sample, epochs=1000)

# Evaluar modelo
preds = predict(X_sample, W1, b1, W2, b2)
accuracy = np.mean(preds == y_sample)
print(f"\nPrecisión en conjunto de entrenamiento: {accuracy * 100:.2f}%")
