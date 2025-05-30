import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

np.random.seed(0)

# --- Funciones de activación ---
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

# --- Inicialización de parámetros ---
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
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = dZ2 @ W2
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (dZ1.T @ X) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

# --- Entrenamiento ---
def train(X, Y, input_size=784, hidden_size=20, output_size=10, lr=0.3, epochs=3000):
    W1, b1, W2, b2 = init_params(input_size, hidden_size, output_size)
    losses = []
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
        loss = np.mean(cross_entropy(Y, A2))
        losses.append(loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
        dW1, db1, dW2, db2 = backward(X, Y, Z1, A1, A2, W2)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    return W1, b1, W2, b2, losses

# --- Predicción ---
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)

# --- One-hot encoding ---
def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

# === Main: Cargar datos y entrenar ===
train_df = pd.read_csv("train.csv")
X = train_df.drop(columns=["label"]).values / 255.0
y = train_df["label"].values
Y = one_hot_encode(y)

# Usamos los primeros 5000 datos como muestra
X_sample = X[:42000]
Y_sample = Y[:42000]
y_sample = y[:42000]

# === Entrenamiento o carga del modelo ===
if os.path.exists("modelo_entrenado.npz"):
    data = np.load("modelo_entrenado.npz")
    W1 = data["W1"]
    b1 = data["b1"]
    W2 = data["W2"]
    b2 = data["b2"]
    losses = data["losses"]
    print("Modelo cargado desde archivo ✅")
else:
    W1, b1, W2, b2, losses = train(X_sample, Y_sample, epochs=3000)
    np.savez("modelo_entrenado.npz", W1=W1, b1=b1, W2=W2, b2=b2, losses=losses)
    print("Modelo entrenado y guardado ✅")

# === Evaluar modelo ===
preds = predict(X_sample, W1, b1, W2, b2)
accuracy = np.mean(preds == y_sample)
print(f"\nPrecisión en conjunto de entrenamiento: {accuracy * 100:.2f}%")

# === Gráfico de pérdida ===
plt.plot(losses)
plt.title("Evolución del error durante el entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.grid()
plt.savefig("grafica_perdida.png")
plt.show()

# === Visualizar algunas predicciones ===
for i in range(10):
    img = X_sample[i].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(f"Etiqueta real: {y_sample[i]} — Predicción: {preds[i]}")
    plt.axis('off')
    plt.savefig(f"imagen_prediccion_{i}.png")
    plt.show()

# === Verificación de predicciones ===
print("\nVerificación de predicciones:")
for i in range(10):
    correcto = "✅" if preds[i] == y_sample[i] else "❌"
    print(f"{correcto} Imagen {i}: real = {y_sample[i]}, predicción = {preds[i]}")

# === Predicción sobre test.csv ===
if os.path.exists("test.csv"):
    test_df = pd.read_csv("test.csv")
    X_test = test_df.values / 255.0
    preds_test = predict(X_test, W1, b1, W2, b2)

    for i in range(10):
        img = X_test[i].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(f"Predicción: {preds_test[i]}")
        plt.axis('off')
        plt.savefig(f"imagen_prediccion_test_{i}.png")
        plt.show()
        
    submission = pd.DataFrame({
        "ImageId": np.arange(1, len(preds_test) + 1),
        "Label": preds_test
    })
    submission.to_csv("submission.csv", index=False)
    print("\nArchivo submission.csv generado con predicciones sobre test.csv ✅")

else:
    print("\n⚠️ test.csv no fue encontrado. Asegúrate de colocarlo en el mismo directorio.")
