import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

np.random.seed(42)
W1 = np.random.uniform(-0.5, 0.5, (2, 3))
W2 = np.random.uniform(-0.5, 0.5, (3, 1))
b1 = np.random.uniform(-0.5, 0.5, (1, 3))
b2 = np.random.uniform(-0.5, 0.5, (1, 1))

x = np.array([[0.1, 0.2]])
y = np.array([[0.3]])

hidden_input = np.dot(x, W1) + b1
hidden_output = tanh(hidden_input)
final_input = np.dot(hidden_output, W2) + b2
final_output = tanh(final_input)

print("Forward output before training:", final_output)

learning_rate = 0.2
epochs = 2000

for epoch in range(epochs):
    hidden_input = np.dot(x, W1) + b1
    hidden_output = tanh(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = tanh(final_input)

    error = y - final_output

    d_final = error * tanh_derivative(final_output)  # تدرج الطبقة النهائية
    d_hidden = d_final.dot(W2.T) * tanh_derivative(hidden_output)  # تدرج الطبقة المخفية

    W2 += learning_rate * hidden_output.T.dot(d_final)
    W1 += learning_rate * x.T.dot(d_hidden)
    b2 += learning_rate * np.sum(d_final, axis=0, keepdims=True)
    b1 += learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Backpropagation output =", final_output)

hidden_input = np.dot(x, W1) + b1
hidden_output = tanh(hidden_input)
final_input = np.dot(hidden_output, W2) + b2
final_output = tanh(final_input)

print("Forward output after training:", final_output)
