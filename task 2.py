import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        #weights and biases
        self.W_input_hidden = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
        self.W_hidden_output = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
        self.b_hidden = np.random.uniform(-0.5, 0.5, (hidden_size, 1))
        self.b_output = np.random.uniform(-0.5, 0.5, (output_size, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.input_layer = X.reshape(-1, 1)
        self.hidden_layer = self.sigmoid(np.dot(self.W_input_hidden, self.input_layer) + self.b_hidden)
        self.output_layer = self.sigmoid(np.dot(self.W_hidden_output, self.hidden_layer) + self.b_output)
        return self.output_layer
    
    def backward(self, X, Y):
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        
        #error
        output_error = Y - self.output_layer
        output_delta = output_error * self.sigmoid_derivative(self.output_layer)
        
        hidden_error = np.dot(self.W_hidden_output.T, output_delta)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer)
        
        #update weights and biases
        self.W_hidden_output += self.learning_rate * np.dot(output_delta, self.hidden_layer.T)
        self.b_output += self.learning_rate * output_delta
        self.W_input_hidden += self.learning_rate * np.dot(hidden_delta, X.T)
        self.b_hidden += self.learning_rate * hidden_delta
    
    def train(self, X, Y, epochs=10000):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, Y)
            if epoch % 1000 == 0:
                loss = np.mean(0.5 * (Y - self.output_layer) ** 2)
                print(f"Epoch {epoch}: Loss = {loss:.6f}")

#inputs and outputs
X = np.array([0.05, 0.10])
Y = np.array([0.01, 0.99])

#train the neural network
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=2, learning_rate=0.5)
nn.train(X, Y, epochs=5000)


output = nn.forward(X)
print("output:",output.flatten())