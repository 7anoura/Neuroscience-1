import numpy as np

np.random.seed(42)
w1 = np.random.uniform(-0.5, 0.5)
w2 = np.random.uniform(-0.5, 0.5)
w3 = np.random.uniform(-0.5, 0.5)
w4 = np.random.uniform(-0.5, 0.5)

b1 = 0.5
b2 = 0.7

def tanh(x):
    return np.tanh(x)

x1, x2 = 0.6, -0.2  # Example inputs

h1 = tanh(x1 * w1 + x2 * w2 + b1)
h2 = tanh(x1 * w3 + x2 * w4 + b1)

output = tanh(h1 * w1 + h2 * w2 + b2)

print(f"Final neural network output: {output:.5f}")
