import numpy as np
from sklearn.datasets import fetch_openml

def load_mnist_data():
    """
    Load and preprocess the MNIST dataset.
    
    Returns:
        tuple: (x_train, y_train, x_test, y_test) containing the preprocessed data
    """
    # Load data from OpenML
    mnist = fetch_openml('mnist_784', version=1)
    x, y = mnist.data, mnist.target
    
    # Normalize pixel values to [0, 1]
    x = x.astype('float32') / 255.0
    
    # Split into training and test sets
    x_train, x_test = x[:60000], x[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    return x_train.values, y_train.values.astype(np.int8), x_test.values, y_test.values.astype(np.int8)

class NeuralNetwork():
    """
    Simple feedforward neural network for MNIST digit recognition.
    
    Attributes:
        input_size (int): Number of input features
        hidden_size (int): Number of neurons in the hidden layer
        output_size (int): Number of output classes
        weights_input_hidden (np.ndarray): Weights from input to hidden layer
        weights_hidden_output (np.ndarray): Weights from hidden to output layer
    """
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with small random values
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.01
    
    def sigmoid(self, x):
        """
        Sigmoid activation function.
        
        Args:
            x (np.ndarray): Input array
            
        Returns:
            np.ndarray: Sigmoid activated output
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid function.
        
        Args:
            x (np.ndarray): Input array
            
        Returns:
            np.ndarray: Derivative of the sigmoid activated output
        """
        return x * (1 - x)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (np.ndarray): Input data
            
        Returns:
            np.ndarray: Output of the network
        """
        self.hidden_layer_input = np.dot(x, self.weights_input_hidden)
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.output = self.sigmoid(self.output_layer_input)
        
        return self.output
    
    def backward(self, x, y, learning_rate=0.01):
        """
        Backward pass and weight update.
        
        Args:
            x (np.ndarray): Input data
            y (np.ndarray): True labels
            learning_rate (float): Learning rate for weight updates
        """
        # Calculate the error
        output_error = y - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)
        
        # Calculate hidden layer error
        hidden_layer_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(self.hidden_layer_output)
        
        # Update weights
        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_delta) * learning_rate
        self.weights_input_hidden += np.dot(x.T, hidden_layer_delta) * learning_rate

    def train(self, x, y, epochs=1000, learning_rate=0.01):
        """
        Train the neural network.
        
        Args:
            x (np.ndarray): Training data
            y (np.ndarray): True labels
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for weight updates
        """
        for epoch in range(epochs):
            self.forward(x)
            self.backward(x, y, learning_rate)
            
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - self.output))
                print(f'Epoch {epoch}, Loss: {loss}')
    
    def predict(self, x):
        """
        Make predictions on new data.
        
        Args:
            x (np.ndarray): Input data
            
        Returns:
            np.ndarray: Predicted class labels
        """
        output = self.forward(x)
        return np.argmax(output, axis=1)
    

def main():
    # Load data
    x_train, y_train, x_test, y_test = load_mnist_data()
    
    # One-hot encode labels
    y_train_one_hot = np.eye(10)[y_train]
    
    # Initialize and train the model
    model = NeuralNetwork()
    model.train(x_train, y_train_one_hot, epochs=1000, learning_rate=0.01)
    
    # Make predictions on test data
    predictions = model.predict(x_test)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()