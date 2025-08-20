import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

dataset = pd.read_excel(r"dataset.xlsx")
x = dataset[['x']].values
y = dataset[['y']].values

split_idx = int(0.7 * len(x))

x_train = x[:split_idx]
y_train = y[:split_idx]
x_val = x[split_idx:]
y_val = y[split_idx:]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

x_train_norm = scaler_X.fit_transform(x_train)
x_val_norm = scaler_X.transform(x_val)
y_train_norm = scaler_y.fit_transform(y_train)
y_val_norm = scaler_y.transform(y_val)

def sigmoid(z):
    z = np.clip(z, -250, 250)
    return 1/ (1+ np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)** 2)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 0.0001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weight1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(0.5 / self.input_size)
        self.weight2 = np.random.randn(self.hidden_size, self.hidden_size) * np.sqrt(0.5 / self.hidden_size)
        self.weight3 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(0.5 / self.hidden_size)

        self.bias1 = np.zeros ((1, self.hidden_size))
        self.bias2 = np.zeros ((1, self.hidden_size))
        self.bias3 = np.zeros ((1, self.output_size))


    def predict(self, x):
        hidden_layer1 = x.dot(self.weight1) + self.bias1
        activation_hidden_layer1 = sigmoid(hidden_layer1)
        hidden_layer2 = activation_hidden_layer1.dot(self.weight2) + self.bias2
        activation_hidden_layer2 = sigmoid(hidden_layer2)
        output = activation_hidden_layer2.dot(self.weight3) + self.bias3
        return output


    def fit(self, x, y, x_val=None, y_val=None, epochs=1000):
        for epoch in range(epochs):

            #forward
            hidden_layer1 = x.dot(self.weight1) + self.bias1
            activation_hidden_layer1 = sigmoid(hidden_layer1)
            
            hidden_layer2 = activation_hidden_layer1.dot(self.weight2) + self.bias2
            activation_hidden_layer2 = sigmoid(hidden_layer2)
            
            output = activation_hidden_layer2.dot(self.weight3) + self.bias3

            output_error = output - y

            #backpropagation
            d_weight3 = activation_hidden_layer2.T.dot(output_error)
            d_bias3 = np.sum(output_error, axis=0, keepdims=True) 

            hidden_layer2_error = output_error.dot(self.weight3.T) * sigmoid_derivative(hidden_layer2)
            
            d_weight2 = activation_hidden_layer1.T.dot(hidden_layer2_error)
            d_bias2 = np.sum(hidden_layer2_error, axis=0, keepdims=True)

            hidden_layer1_error = hidden_layer2_error.dot(self.weight2.T) * sigmoid_derivative(hidden_layer1)

            d_weight1 = x.T.dot(hidden_layer1_error)
            d_bias1 = np.sum(hidden_layer1_error, axis=0, keepdims=True)

            #gradient clipping
            max_grad = 1.0
            d_weight1 = np.clip(d_weight1, -max_grad, max_grad)
            d_weight2 = np.clip(d_weight2, -max_grad, max_grad)
            d_weight3 = np.clip(d_weight3, -max_grad, max_grad)
            d_bias1 = np.clip(d_bias1, -max_grad, max_grad)
            d_bias2 = np.clip(d_bias2, -max_grad, max_grad)
            d_bias3 = np.clip(d_bias3, -max_grad, max_grad)

            self.weight3 -= self.learning_rate * d_weight3
            self.bias3 -= self.learning_rate * d_bias3
            self.weight2 -= self.learning_rate * d_weight2
            self.bias2 -= self.learning_rate * d_bias2
            self.weight1 -= self.learning_rate * d_weight1
            self.bias1 -= self.learning_rate * d_bias1

            #printing progress
            if epoch % 100 == 0:
                train_loss = mse_loss(y, output)
                print (f"Epoch {epoch}, Train Loss: {train_loss:.6f}")

                val_pred = self.predict(x_val)
                val_loss = mse_loss(y_val, val_pred)
                print (f"Epoch {epoch}, Validation Loss: {val_loss:.6f}")

mlp = MLP(input_size= 1, hidden_size=3, output_size=1, learning_rate=0.0001)
mlp.fit(x_train_norm, y_train_norm, x_val_norm, y_val_norm, epochs=1000)

y_pred_norm = mlp.predict(x_val_norm)
y_pred = scaler_y.inverse_transform (y_pred_norm)

mse = mse_loss(y_val, y_pred)
print(f"Validation MSE: {mse:.4f}")

ss_res = np.sum((y_val - y_pred) **2)
ss_tot = np.sum((y_val - np.mean(y_val))**2)

r2_score = 1 - (ss_res / ss_tot)
print(f"R2-score: {r2_score:.4f}")


