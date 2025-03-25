import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_neurons, hidden_neurons, output_neurons, learning_rate=0.5, epochs=10000):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        np.random.seed(0)
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
        self.bias_hidden = np.random.uniform(-1, 1, (1, hidden_neurons))
        self.bias_output = np.random.uniform(-1, 1, (1, output_neurons))
    
    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        
        self.final_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_layer_input)
        
        return self.final_output
    
    def backward(self, X, Y):
        error = Y - self.final_output
        d_output = error * sigmoid_derivative(self.final_output)
        
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * sigmoid_derivative(self.hidden_layer_output)
        
        self.weights_hidden_output += self.hidden_layer_output.T.dot(d_output) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        
        self.weights_input_hidden += X.T.dot(d_hidden) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate
        
        return np.mean(np.abs(error))
    
    def train(self, X, Y):
        for epoch in range(self.epochs):
            self.forward(X)
            loss = self.backward(X, Y)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        output = self.forward(X)
        return np.round(output)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(input_neurons=2, hidden_neurons=2, output_neurons=1)
nn.train(X, Y)

print("Final Output:")
print(nn.predict(X))