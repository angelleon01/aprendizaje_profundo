from math import e

import matplotlib.pyplot as plt
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

# Datos y objetivos
X = np.array([[3], [6], [9]])  # Entradas Vcc (3 filas x 1 columna)
Y = np.array([[VD[3]], [VD[6]], [VD[9]]])  # Objetivo VD

# Pesos y sesgos iniciales
# Primera capa
Ws = np.array([[0.05], [0.15], [-0.20]])  # (3x1)
bs = np.array([[0.23], [-0.10], [0.17]])  # (3x1)

# Segunda capa
Wu = np.array([[0.8, -0.6, 0.5], [0.7, 0.9, -0.6]])  # (2x3)
bu = np.array([[0.45], [-0.34]])  # (2x1)

# Capa de salida
Wy = np.array([[0.8, 0.5]])  # (1x2)
by = np.array([[0.7]])  # (1x1)


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
    S = Ws @ X.T + bs  # Ajustamos para que las dimensiones sean compatibles
    T = sigmoid(S)

    # Segunda capa
    U = Wu @ T + bu
    V = relu(U)

    # Capa de salida
    Y_pred = Wy @ V + by

    return S, T, U, V, Y_pred


# Coste (MSE)
def compute_cost(Y, Y_pred):
    m = Y.shape[1]
    cost = np.sum((Y - Y_pred) ** 2) / m
    return cost


# Propagación hacia atrás (backpropagation)
def backward(X, Y, S, T, U, V, Y_pred):
    m = X.shape[1]

    # Derivada de la capa de salida
    dY_pred = Y_pred - Y  # dC/dY_pred
    dW3 = dY_pred @ V.T / m
    db3 = np.sum(dY_pred, axis=1, keepdims=True) / m

    # Derivadas de la segunda capa
    dV = Wy.T @ dY_pred
    dU = dV * relu_derivative(U)
    dW2 = dU @ T.T / m
    db2 = np.sum(dU, axis=1, keepdims=True) / m

    # Derivadas de la primera capa
    dT = Wu.T @ dU
    dS = dT * sigmoid_derivative(S)
    dW1 = dS @ X / m
    db1 = np.sum(dS, axis=1, keepdims=True) / m

    return dW1, db1, dW2, db2, dW3, db3


# Actualización de pesos
def update_parameters(dW1, db1, dW2, db2, dW3, db3, alpha=0.01, momentum=0):
    global Ws, bs, Wu, bu, Wy, by

    # Momentum terms
    Ws_m = np.zeros_like(Ws)
    bs_m = np.zeros_like(bs)
    Wu_m = np.zeros_like(Wu)
    bu_m = np.zeros_like(bu)
    Wy_m = np.zeros_like(Wy)
    by_m = np.zeros_like(by)

    Ws_m = momentum * Ws_m + (1 - momentum) * dW1
    bs_m = momentum * bs_m + (1 - momentum) * db1
    Wu_m = momentum * Wu_m + (1 - momentum) * dW2
    bu_m = momentum * bu_m + (1 - momentum) * db2
    Wy_m = momentum * Wy_m + (1 - momentum) * dW3
    by_m = momentum * by_m + (1 - momentum) * db3

    Ws -= alpha * Ws_m
    bs -= alpha * bs_m
    Wu -= alpha * Wu_m
    bu -= alpha * bu_m
    Wy -= alpha * Wy_m
    by -= alpha * by_m


def diode_tension(X):
    S, T, U, V, Y_pred = forward(X)
    return Y_pred.T


def main():
    # Número de épocas y tasa de aprendizaje
    n_epochs = 1  # Ajusta esto al número de épocas deseado
    lr = 0.01  # Ajusta según tus necesidades
    momentum = 0.5
    costs = []
    for epoch in range(n_epochs):
        # Forward pass
        S, T, U, V, Y_pred = forward(X)

        # Cálculo del coste
        cost = compute_cost(Y_pred, Y.T)
        costs.append(cost)

        # Backward pass
        Ws, bs, Wu, bu, Wy, by = backward(X, Y.T, S, T, U, V, Y_pred)

        # Actualización de parámetros
        if epoch == 0:
            update_parameters(Ws, bs, Wu, bu, Wy, by, lr, 0)
        else:
            update_parameters(Ws, bs, Wu, bu, Wy, by, lr, momentum)

    # Cálculo del coste
    S, T, U, V, Y_pred = forward(X)
    cost = compute_cost(Y_pred, Y.T)
    costs.append(cost)

    # Resultados finales
    print("Final cost:", cost)
    Vd = Y_pred.T
    print("Final predictions:", Vd)

    # Plot de la función de coste
    plt.plot(range(n_epochs + 1), costs)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Cost evolution")
    plt.show()

    # Get Vr using the trained model
    I = I0 * (e ** (Vd / (eta * VT)) - 1)
    Vr = R * I

    print("VD:", Vd)

    print("VR:", Vr)


if __name__ == "__main__":
    main()
