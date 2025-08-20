import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and prepare data
df = pd.read_excel(r"dataset.xlsx")
X = df[['x']].values 
y = df[['y']].values 
split_idx = int(0.7 * len(X))

X_train = X[:split_idx]
y_train = y[:split_idx]
X_val = X[split_idx:]
y_val = y[split_idx:]

# Data normalization
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_norm = scaler_X.fit_transform(X_train)
X_val_norm = scaler_X.transform(X_val)
y_train_norm = scaler_y.fit_transform(y_train)
y_val_norm = scaler_y.transform(y_val)

def sigmoid(z):
    # More stable sigmoid with tighter clipping
    z = np.clip(z, -250, 250)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class MLP_1Hidden:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001): 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Smaller Xavier initialization
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(0.5 / self.input_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(0.5 / self.hidden_size)
        
        self.bias1 = np.zeros((1, self.hidden_size))
        self.bias2 = np.zeros((1, self.output_size))
    
    def fit(self, X, y, X_val=None, y_val=None, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            hidden_layer1 = X.dot(self.weights1) + self.bias1
            activation_hidden_layer1 = sigmoid(hidden_layer1)
            output = activation_hidden_layer1.dot(self.weights2) + self.bias2
            
            # Backpropagation
            output_error = output - y
            
            # Output layer gradients
            d_weights2 = activation_hidden_layer1.T.dot(output_error)
            d_bias2 = np.sum(output_error, axis=0, keepdims=True)
            
            # Hidden layer gradients
            hidden1_error = output_error.dot(self.weights2.T) * sigmoid_derivative(hidden_layer1)
            
            d_weights1 = X.T.dot(hidden1_error)
            d_bias1 = np.sum(hidden1_error, axis=0, keepdims=True)
            
            # Gradient clipping to prevent explosion
            max_grad = 1.0
            d_weights1 = np.clip(d_weights1, -max_grad, max_grad)
            d_weights2 = np.clip(d_weights2, -max_grad, max_grad)
            d_bias1 = np.clip(d_bias1, -max_grad, max_grad)
            d_bias2 = np.clip(d_bias2, -max_grad, max_grad)
            
            # Update weights and biases
            self.weights2 -= self.learning_rate * d_weights2
            self.bias2 -= self.learning_rate * d_bias2
            self.weights1 -= self.learning_rate * d_weights1
            self.bias1 -= self.learning_rate * d_bias1
            
            # Print progress
            if epoch % 100 == 0:
                train_loss = mse_loss(y, output)
                print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}")
                
                if X_val is not None and y_val is not None:
                    val_pred = self.predict(X_val)
                    val_loss = mse_loss(y_val, val_pred)
                    print(f"Epoch {epoch}, Validation Loss: {val_loss:.6f}")
                    
                # Early stopping if loss becomes too large
                if train_loss > 1e6:
                    print(f"Training stopped early at epoch {epoch} due to exploding gradients")
                    break
    
    def predict(self, X):
        hidden_layer1 = X.dot(self.weights1) + self.bias1
        activation_hidden_layer1 = sigmoid(hidden_layer1)
        output = activation_hidden_layer1.dot(self.weights2) + self.bias2
        return output

mlp_1hidden = MLP_1Hidden(input_size=1, hidden_size=4, output_size=1, learning_rate=0.001)
mlp_1hidden.fit(X_train_norm, y_train_norm, X_val_norm, y_val_norm, epochs=1000)

# Make predictions and evaluate
y_pred_norm = mlp_1hidden.predict(X_val_norm)
y_pred = scaler_y.inverse_transform(y_pred_norm)

mse = mse_loss(y_val, y_pred)
print(f"Validation MSE: {mse:.4f}")

ss_res = np.sum((y_val - y_pred) ** 2)
ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
r2_score = 1 - (ss_res / ss_tot)
print(f"R² Score: {r2_score:.4f}")