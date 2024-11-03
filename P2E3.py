import numpy as np
from scipy.special import lambertw

# Constantes
I0 = 1e-12
eta = 1
VT = 0.026 
R = 100
Vcc_values = [3, 6, 9]

# Calculo de VD
def calculate_VD_steps(Vcc):
    a = Vcc + R * I0  # Calculamos 'a'
    b = R * I0 / (eta * VT)  # Calculamos 'b'
    exponent = (Vcc + R * I0) / (eta * VT)  # Calculamos el exponent
    W_argument = b * np.exp(exponent)  # Calculamos el argumento de la funcion Lambert
    W_result = lambertw(W_argument).real  # La evaluamos
    VD = a - eta * VT * W_result  # Calculo final de VD
    return VD

# Calculamos el VD de cada Vcc
VD = {Vcc: calculate_VD_steps(Vcc) for Vcc in Vcc_values}
# print(VD)


# Datos y objetivos
X = np.array([[3], [6], [9]])   # Entradas Vcc (3 filas x 1 columna)
Y = np.array([[VD[3]], [VD[6]], [VD[9]]])  # Objetivo VD
alpha = 1  # Tasa de aprendizaje

# Pesos y sesgos iniciales
# Primera capa
W1 = np.array([[0.05], [0.15], [-0.20]])  # (3, 1)
b1 = np.array([[0.23], [-0.10], [0.17]])  # (3, 1)

# Segunda capa
W2 = np.array([[0.8, -0.6, 0.5], [0.7, 0.9, -0.6]])  # (2, 3)
b2 = np.array([[0.45], [-0.34]])  # (2, 1)

# Capa de salida
W3 = np.array([[0.8, 0.8]])  # (1, 2)
b3 = np.array([[0.7]])  # (1, 1)

# Funciones de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Propagación hacia adelante
def forward(X):
    # Primera capa
    Z1 = W1 @ X.T + b1  # Ajustamos para que las dimensiones sean compatibles
    A1 = sigmoid(Z1)

    # Segunda capa
    Z2 = W2 @ A1 + b2
    A2 = relu(Z2)

    # Capa de salida
    Z3 = W3 @ A2 + b3
    A3 = Z3  # Lineal, sin activación

    return Z1, A1, Z2, A2, Z3, A3

# Coste (MSE)
def compute_cost(A3, Y):
    m = Y.shape[1]
    cost = np.sum((A3 - Y) ** 2) / (2 * m)
    return cost

# Propagación hacia atrás (backpropagation)
def backward(X, Y, Z1, A1, Z2, A2, Z3, A3):
    m = X.shape[1]

    # Derivada de la capa de salida
    dZ3 = A3 - Y  # dC/dZ3
    dW3 = dZ3 @ A2.T / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    # Derivadas de la segunda capa
    dA2 = W3.T @ dZ3
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = dZ2 @ A1.T / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    # Derivadas de la primera capa
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = dZ1 @ X / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return dW1, db1, dW2, db2, dW3, db3

# Actualización de pesos
def update_parameters(dW1, db1, dW2, db2, dW3, db3):
    global W1, b1, W2, b2, W3, b3
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W3 -= alpha * dW3
    b3 -= alpha * db3

# Una época de entrenamiento
Z1, A1, Z2, A2, Z3, A3 = forward(X)
cost = compute_cost(A3, Y.T)
dW1, db1, dW2, db2, dW3, db3 = backward(X, Y.T, Z1, A1, Z2, A2, Z3, A3)
update_parameters(dW1, db1, dW2, db2, dW3, db3)

print("Coste:", cost)
print("Predicciones finales:", A3.T)
