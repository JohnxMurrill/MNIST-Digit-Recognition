import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

def load_mnist_data(batch_size=64):
    """
    Load and preprocess the MNIST dataset.
    
    Args:
        batch_size (int): Number of samples per batch
        
    Returns:
        tuple: (train_loader, test_loader) containing the DataLoader objects
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load training data
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Download and load test data
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader


class MNISTModel(nn.Module):
    """
    Convolutional Neural Network for MNIST digit recognition.
    
    Architecture:
    - 2 Convolutional layers with ReLU activation and max pooling
    - 2 Fully connected layers
    - Output layer with 10 classes (digits 0-9)
    """
    
    def __init__(self):
        """Initialize the network architecture"""
        super(MNISTModel, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input image batch of shape (batch_size, 1, 28, 28)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10)
        """
        # Pass through first convolutional block
        x = self.conv1(x)  # Output: (batch_size, 32, 14, 14)
        
        # Pass through second convolutional block
        x = self.conv2(x)  # Output: (batch_size, 64, 7, 7)
        
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64 * 7 * 7)
        
        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x


def train_model(model, train_loader, test_loader, epochs=5, lr=0.001):
    """
    Train the model on the MNIST dataset.
    
    Args:
        model (nn.Module): PyTorch model to train
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        epochs (int): Number of training epochs
        lr (float): Learning rate
        
    Returns:
        tuple: (trained_model, train_losses, test_accuracies) - the trained model and performance metrics
    """
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # For tracking metrics
    train_losses = []
    test_accuracies = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Iterate over batches
        for i, (images, labels) in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss
            running_loss += loss.item()
            
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Evaluate on test set
        accuracy = evaluate_model(model, test_loader)
        test_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.4f}')
    
    return model, train_losses, test_accuracies


def evaluate_model(model, data_loader):
    """
    Evaluate the model on the given data loader.
    
    Args:
        model (nn.Module): PyTorch model to evaluate
        data_loader (DataLoader): Data loader for evaluation
        
    Returns:
        float: Accuracy of the model on the given data
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total
    return accuracy


def visualize_results(train_losses, test_accuracies, num_samples=10, test_loader=None, model=None):
    """
    Visualize training metrics and sample predictions.
    
    Args:
        train_losses (list): Training loss history
        test_accuracies (list): Test accuracy history
        num_samples (int): Number of test samples to visualize
        test_loader (DataLoader): Test data loader
        model (nn.Module): Trained model for predictions
    """
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()
    
    # If test_loader and model are provided, visualize some predictions
    if test_loader is not None and model is not None:
        model.eval()
        
        # Get a batch of test images
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
        
        # Plot the images and predictions
        fig = plt.figure(figsize=(15, 3))
        for i in range(num_samples):
            ax = fig.add_subplot(1, num_samples, i + 1)
            img = images[i].squeeze().numpy()
            ax.imshow(img, cmap='gray')
            title = f"Pred: {predicted[i]}, True: {labels[i]}"
            ax.set_title(title, color=("green" if predicted[i] == labels[i] else "red"))
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function to execute the MNIST digit recognition pipeline.
    """
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size=64)
    
    # Initialize model
    print("Initializing model...")
    model = MNISTModel()
    
    # Train model
    print("Training model...")
    trained_model, train_losses, test_accuracies = train_model(
        model, 
        train_loader, 
        test_loader,
        epochs=5
    )
    
    # Evaluate final model
    final_accuracy = evaluate_model(trained_model, test_loader)
    print(f"Final Test Accuracy: {final_accuracy:.4f}")
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(
        train_losses, 
        test_accuracies,
        num_samples=10,
        test_loader=test_loader,
        model=trained_model
    )
    
    # Save model
    torch.save(trained_model.state_dict(), "mnist_pytorch_model.pth")
    print("Model saved to 'mnist_pytorch_model.pth'")


if __name__ == "__main__":
    main()