import numpy as np
from math import e

# Parameters
m = 3
lr = 0.01

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input data
X = np.array([[0 * e**0],
              [1 * e**1],
              [2 * e**2]])

Y = np.array([[0],
              [1],
              [2]])

Wu = np.array([[0.15, -0.10, 0.12]])
bu = np.array([[0.3, -0.2, 0.07]])

Wy = np.array([[1.4],
               [7.8],
               [3.4]])

by = np.array([[0.5]])

# Forward pass
V_3 = Wu
V_2 = bu
V_1 = Wy
V0 = by
V1 = X @ V_3 + np.ones((m, 1)) @ V_2
V2 = sigmoid(V1)
V3 = V2 @ V_1 + np.ones((m, 1)) @ V0
V4 = (1 / m) * ((Y - V3).T @ (Y - V3))
f = V4

# Initial values
print("Coste inicial:", f)
print("Wu inicial:", Wu)
print("bu inicial:", bu)
print("Wy inicial:\n", Wy)
print("by inicial:", by)

# Backward pass
V4_ = 1
V3_ = -2/m * (Y - V3)
V2_ = V3_ @ V_1.T
V1_ = V2_ * V2 * (np.ones((V2.shape[0], V2.shape[1])) - V2)
V0_ = np.ones((m, 1)).T @ V3_
V_1_ = V2.T @ V3_
V_2_ = np.ones((m, 1)).T @ V1_
V_3_ = X.T @ V1_

# Update weights and biases
Wu -= lr * V_3_
bu -= lr * V_2_
Wy -= lr * V_1_
by -= lr * V0_

# Forward pass
V_3 = Wu
V_2 = bu
V_1 = Wy
V0 = by
V1 = X @ V_3 + np.ones((m, 1)) @ V_2
V2 = sigmoid(V1)
V3 = V2 @ V_1 + np.ones((m, 1)) @ V0
V4 = (1 / m) * ((Y - V3).T @ (Y - V3))
f = V4

# Values after one epoch
print("Coste final:", f)
print("Wu final:", Wu)
print("bu final:", bu)
print("Wy final:\n", Wy)
print("by final:", by)

