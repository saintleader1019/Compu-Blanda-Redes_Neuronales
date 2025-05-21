# 🧠 Clasificador Multiclase con Red Neuronal desde Cero

Este proyecto implementa una red neuronal multicapa (MLP) desde cero en Python y NumPy, como parte del curso **Computación Blanda**. La red es capaz de clasificar puntos 2D en tres categorías: **purple**, **orange** y **green**, usando aprendizaje supervisado.

---

## 🎯 Objetivo

Construir un modelo de red neuronal simple capaz de aprender patrones espaciales y clasificar puntos de acuerdo a ejemplos previos, sin usar librerías avanzadas de machine learning.

---

## 🧩 ¿Cómo funciona?

Una red neuronal es una estructura de capas que procesa datos. Este modelo:

- Toma **coordenadas 2D** como entrada (`x`, `y`)
- Las pasa por una **capa oculta** con 20 neuronas y activación ReLU
- Finaliza con una **capa de salida** con 3 neuronas, una por clase
- Usa **softmax** para predecir una probabilidad para cada clase
- Aprende con **descenso por gradiente** y **función de pérdida de entropía cruzada**

---

## 📊 Datos

Se entrenó la red con:

- **60 puntos de entrenamiento** con sus etiquetas de color
- **30 puntos de prueba** para validar qué tan bien aprendió

---

## ⚙️ Entrenamiento

- **Entradas**: coordenadas `(x, y)`
- **Salidas esperadas**: etiquetas codificadas en one-hot (como `[1, 0, 0]`)
- **Épocas**: 1000
- **Tasa de aprendizaje**: 0.1
- **Pérdida**: Entropía cruzada
- **Optimizador**: Descenso por gradiente (manual)

---

## 🧠 Variables importantes

| Variable | Significado |
|----------|-------------|
| `inputs` | Lista de coordenadas de entrenamiento |
| `targets_rgb` | Lista de etiquetas RGB (clases reales) |
| `W1`, `W2` | Pesos de la red entre capas |
| `b1`, `b2` | Sesgos de cada capa |
| `Z1`, `A1` | Resultados intermedios de la capa oculta |
| `Z2`, `A2` | Resultados de la capa de salida |
| `preds` | Predicciones del modelo sobre los puntos de prueba |
| `losses` | Historial de pérdida para graficar la mejora |

---

## 🖥️ Resultados

- Se muestra en consola la predicción del color para cada coordenada de prueba
- Se imprime también el color real para comparar
- Se genera un gráfico con:
  - Evolución del error (pérdida)
  - Clasificación visual de los puntos en el plano 2D

---

## 📌 Ejemplo de salida

- Coordenada (0.2581, 0.3214) → Predicho: purple | Real: purple
- Coordenada (0.6129, 0.6786) → Predicho: orange | Real: orange
- Coordenada (0.7742, 0.8214) → Predicho: green | Real: green


---

## 📈 Visualizaciones

1. **Curva de pérdida** – muestra si el modelo está aprendiendo bien
2. **Gráfico de puntos clasificados** – coloreados según la predicción

---

## ✅ Conclusión

Este proyecto demuestra cómo una red neuronal simple puede aprender a clasificar puntos en el espacio usando solo operaciones matemáticas y NumPy, sin usar bibliotecas externas de IA.

---


## 🧠 Reconocimiento de Dígitos Manuscritos (MNIST)

### 🎯 Objetivo

Implementar una red neuronal para reconocer imágenes de dígitos (0–9) escritos a mano, usando el dataset `train.csv` de [Kaggle - Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer).

### ⚙️ Arquitectura

- **Entradas**: 784 valores (28×28 píxeles aplanados)
- **Capa oculta**: 20 neuronas (ReLU)
- **Capa de salida**: 10 neuronas (Softmax)
- **Inicialización**: pesos ∈ [-0.5, 0.5], bias = 0
- **Pérdida**: Entropía cruzada multiclase
- **Entrenamiento**: Descenso por gradiente

### 🧪 Evaluación

- Se entrenó con 10.000 ejemplos para mayor rapidez
- Precisión sobre los datos de entrenamiento superó el 90%
- Se graficaron 10 ejemplos con predicción vs etiqueta real
- Se guardaron los pesos entrenados para no reentrenar cada vez

### 🗂️ Archivos útiles

- `modelo_entrenado.npz`: contiene los pesos y sesgos (`W1`, `b1`, `W2`, `b2`)
- `grafica_perdida.png`: curva de error durante el entrenamiento
- `imagen_prediccion_X.png`: predicción visual para algunas muestras

---

## ✅ Reutilización del modelo

El modelo se guarda automáticamente luego del entrenamiento, y puede reutilizarse así:

```python
data = np.load("modelo_entrenado.npz")
W1, b1, W2, b2 = data["W1"], data["b1"], data["W2"], data["b2"]



## 📌 Ejemplo de salida


### 👨‍💻 Autor

- **Santiago Herrera Muñoz**