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
        
        # Initialize weights with He initialization for better convergence
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2/self.input_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2/self.hidden_size)
        
        # Add biases
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
        
        # For tracking metrics
        self.train_losses = []
        self.test_accuracies = []
    
    def sigmoid(self, x):
        """
        Sigmoid activation function.
        
        Args:
            x (np.ndarray): Input array
            
        Returns:
            np.ndarray: Sigmoid activated output
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow
    
    def relu(self, x):
        """
        ReLU activation function for better performance.
        
        Args:
            x (np.ndarray): Input array
            
        Returns:
            np.ndarray: ReLU activated output
        """
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """
        Derivative of the ReLU function.
        
        Args:
            x (np.ndarray): Input array
            
        Returns:
            np.ndarray: Derivative of the ReLU activated output
        """
        return np.where(x > 0, 1, 0)
    
    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid function.
        
        Args:
            x (np.ndarray): Input array
            
        Returns:
            np.ndarray: Derivative of the sigmoid activated output
        """
        return x * (1 - x)
    
    def softmax(self, x):
        """
        Softmax activation function for better classification output.
        
        Args:
            x (np.ndarray): Input array
            
        Returns:
            np.ndarray: Softmax activated output
        """
        # Shift x for numerical stability
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (np.ndarray): Input data
            
        Returns:
            np.ndarray: Output of the network
        """
        # Pass through hidden layer with ReLU activation
        self.hidden_layer_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.relu(self.hidden_layer_input)
        
        # Pass through output layer with softmax activation
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = self.softmax(self.output_layer_input)
        
        return self.output
    
    def backward(self, x, y, learning_rate=0.01):
        """
        Backward pass and weight update.
        
        Args:
            x (np.ndarray): Input data
            y (np.ndarray): True labels (one-hot encoded)
            learning_rate (float): Learning rate for weight updates
        """
        batch_size = x.shape[0]
        
        # Calculate output layer error (cross-entropy derivative with softmax)
        output_error = self.output - y
        
        # Calculate hidden layer error
        hidden_layer_error = np.dot(output_error, self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * self.relu_derivative(self.hidden_layer_output)
        
        # Update weights and biases
        self.weights_hidden_output -= (learning_rate * np.dot(self.hidden_layer_output.T, output_error) / batch_size)
        self.bias_output -= learning_rate * np.mean(output_error, axis=0, keepdims=True)
        
        self.weights_input_hidden -= (learning_rate * np.dot(x.T, hidden_layer_delta) / batch_size)
        self.bias_hidden -= learning_rate * np.mean(hidden_layer_delta, axis=0, keepdims=True)
    
    def train_minibatch(self, x, y, batch_size=32, epochs=5, learning_rate=0.01, x_test=None, y_test=None):
        """
        Train the neural network using mini-batch gradient descent.
        
        Args:
            x (np.ndarray): Training data
            y (np.ndarray): True labels (one-hot encoded)
            batch_size (int): Size of mini-batches
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for weight updates
            x_test (np.ndarray): Test data for tracking accuracy
            y_test (np.ndarray): Test labels for tracking accuracy
        """
        n_samples = x.shape[0]
        n_batches = n_samples // batch_size
        
        # Clear previous metrics
        self.train_losses = []
        self.test_accuracies = []
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Process mini-batches
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward and backward pass
                self.forward(x_batch)
                self.backward(x_batch, y_batch, learning_rate)
                
                # Calculate batch loss (cross-entropy)
                batch_loss = -np.mean(np.sum(y_batch * np.log(np.clip(self.output, 1e-10, 1.0)), axis=1))
                epoch_loss += batch_loss
            
            # Calculate average loss for the epoch
            avg_epoch_loss = epoch_loss / n_batches
            self.train_losses.append(avg_epoch_loss)
            
            # Calculate test accuracy if test data is provided
            if x_test is not None and y_test is not None:
                predictions = self.predict(x_test)
                test_y_indices = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
                accuracy = np.mean(predictions == test_y_indices)
                self.test_accuracies.append(accuracy)
                
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}, Test Accuracy: {accuracy:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}')
    
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
    
    def calculate_accuracy(self, x, y):
        """
        Calculate accuracy on given data.
        
        Args:
            x (np.ndarray): Input data
            y (np.ndarray): True labels (can be one-hot encoded or indices)
            
        Returns:
            float: Accuracy as a fraction
        """
        predictions = self.predict(x)
        y_indices = np.argmax(y, axis=1) if y.ndim > 1 else y
        return np.mean(predictions == y_indices)


def main():
    """
    Main function to run the MNIST digit recognition with the optimized scratch implementation.
    """
    print("Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = load_mnist_data()
    
    # One-hot encode labels
    y_train_one_hot = np.eye(10)[y_train.astype(int)]
    y_test_one_hot = np.eye(10)[y_test.astype(int)]
    
    print("Initializing model...")
    model = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
    
    print("Training model...")
    # Train with mini-batches
    model.train_minibatch(
        x_train, 
        y_train_one_hot, 
        batch_size=64,  # Using mini-batches
        epochs=5,       # Matching other implementations
        learning_rate=0.1,  # Higher learning rate for faster convergence
        x_test=x_test,
        y_test=y_test_one_hot
    )
    
    # Calculate final accuracy
    final_accuracy = model.calculate_accuracy(x_test, y_test)
    print(f'Final Test Accuracy: {final_accuracy:.4f}')
    
    # Visualize training metrics
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(model.train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(model.test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()