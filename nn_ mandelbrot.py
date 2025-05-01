import numpy as np
import matplotlib.pyplot as plt

# Activation functions and derivatives
def sigmoid(x):
    # Clip to avoid overflow in exp
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Layer class
class Layer:
    def __init__(self, input_size, output_size, activation='sigmoid'):
        # Xavier/Glorot initialization for better convergence
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))
        self.bias = np.zeros((1, output_size))
        
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        else:
            raise ValueError("Unsupported activation function")

# Neural Network class
class NeuralNetwork:
    def __init__(self, layer_sizes, activations, learning_rate=0.01, epochs=5000):
        assert len(layer_sizes) - 1 == len(activations), "Mismatch between layers and activations"
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1], activations[i]) 
                      for i in range(len(activations))]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []
        
    def forward(self, X):
        self.layer_inputs = [X]
        self.outputs = [X]
        
        for layer in self.layers:
            z = np.dot(self.outputs[-1], layer.weights) + layer.bias
            self.layer_inputs.append(z)
            a = layer.activation(z)
            self.outputs.append(a)
            
        return self.outputs[-1]
    
    def backward(self, X, Y):
        m = X.shape[0]
        error = Y - self.outputs[-1]
        deltas = [error]
        
        # Calculate deltas for each layer
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            delta = deltas[0]
            
            if i < len(self.layers) - 1:  # Not output layer
                delta = np.dot(deltas[0], self.layers[i+1].weights.T) * layer.activation_derivative(self.outputs[i+1])
                deltas.insert(0, delta)
            
            # Update weights and biases
            layer.weights += self.learning_rate * np.dot(self.outputs[i].T, delta) / m
            layer.bias += self.learning_rate * np.sum(delta, axis=0, keepdims=True) / m
        
        # Return binary cross-entropy loss
        epsilon = 1e-15
        output = self.outputs[-1]
        output = np.clip(output, epsilon, 1 - epsilon)
        loss = -np.mean(Y * np.log(output) + (1 - Y) * np.log(1 - output))
        return loss
    
    def train(self, X, Y, batch_size=32):
        m = X.shape[0]
        for epoch in range(self.epochs):
            # Mini-batch training
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                
                self.forward(X_batch)
                loss = self.backward(X_batch, Y_batch)
            
            # Track loss with full dataset
            if epoch % 100 == 0:
                predictions = self.forward(X)
                epsilon = 1e-15
                predictions = np.clip(predictions, epsilon, 1 - epsilon)
                loss = -np.mean(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))
                self.loss_history.append(loss)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return self.forward(X)

# Generate Mandelbrot data with normalized inputs
def generate_mandelbrot_data(xmin=-2, xmax=1, ymin=-1.5, ymax=1.5, width=40, height=40, max_iter=100):
    X, Y = [], []
    
    for ix in range(width):
        for iy in range(height):
            x0 = xmin + (xmax - xmin) * ix / (width - 1)
            y0 = ymin + (ymax - ymin) * iy / (height - 1)
            
            x, y = 0.0, 0.0
            iteration = 0
            
            while x*x + y*y <= 4 and iteration < max_iter:
                xtemp = x*x - y*y + x0
                y = 2*x*y + y0
                x = xtemp
                iteration += 1
            
            # Normalize inputs to [-1, 1] range for better training
            x0_norm = 2 * (x0 - xmin) / (xmax - xmin) - 1
            y0_norm = 2 * (y0 - ymin) / (ymax - ymin) - 1
            
            X.append([x0_norm, y0_norm])
            # Binary label: 1 if in set (didn't escape), 0 if not in set (escaped)
            Y.append([1 if iteration == max_iter else 0])
    
    return np.array(X), np.array(Y)

# Setup
np.random.seed(42)  # For reproducibility
width, height = 40, 40
X, Y = generate_mandelbrot_data(width=width, height=height)

# Create and train the neural network
nn = NeuralNetwork(
    layer_sizes=[2, 20, 40, 40, 10, 1],  # More neurons and layers
    activations=['relu', 'relu','sigmoid', 'relu', 'sigmoid'],
    learning_rate=0.01,
    epochs=20000
)

print(f"Training on {X.shape[0]} points...")
nn.train(X, Y, batch_size=64)

# Plot the loss history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(0, nn.epochs, 100), nn.loss_history)
plt.title('Loss History')
plt.xlabel('Epoch (x100)')
plt.ylabel('Loss')

# Predict and visualize
print("Generating predictions...")
predictions = nn.predict(X)
predictions_grid = predictions.reshape(height, width)
true_labels = Y.reshape(height, width)

plt.subplot(1, 2, 2)
plt.imshow(predictions_grid, cmap='hot', origin='lower', 
           extent=[-2, 1, -1.5, 1.5])
plt.colorbar(label='Probability of being in set')
plt.title("NN Predicted Mandelbrot Set")

plt.tight_layout()
plt.show()

# Compare true vs predicted
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(true_labels, cmap='hot', origin='lower', 
           extent=[-2, 1, -1.5, 1.5])
plt.colorbar(label='In set (1) or not (0)')
plt.title("True Mandelbrot Set")

plt.subplot(1, 2, 2)
plt.imshow(predictions_grid > 0.5, cmap='hot', origin='lower',
           extent=[-2, 1, -1.5, 1.5])
plt.colorbar(label='Predicted: In set (1) or not (0)')
plt.title("NN Predictions (Thresholded at 0.5)")

plt.tight_layout()
plt.show()