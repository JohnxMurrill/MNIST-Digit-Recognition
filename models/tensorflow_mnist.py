import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_mnist_data():
    """
    Load and preprocess the MNIST dataset.
    
    Returns:
        tuple: (x_train, y_train, x_test, y_test) containing the preprocessed data
    """
    # Load data from Keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to add channel dimension [samples, height, width, channels]
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test


def create_model():
    """
    Create a CNN model for MNIST digit recognition.
    
    Returns:
        tf.keras.Model: Compiled Keras model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten layer
        layers.Flatten(),
        
        # Fully connected layers
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, x_train, y_train, x_test, y_test, epochs=5, batch_size=64):
    """
    Train the TensorFlow model on the MNIST dataset.
    
    Args:
        model (tf.keras.Model): Keras model to train
        x_train (ndarray): Training images
        y_train (ndarray): Training labels (one-hot encoded)
        x_test (ndarray): Test images
        y_test (ndarray): Test labels (one-hot encoded)
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (trained_model, history) - the trained model and training history
    """
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    return model, history


def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model on the test dataset.
    
    Args:
        model (tf.keras.Model): Trained Keras model
        x_test (ndarray): Test images
        y_test (ndarray): Test labels (one-hot encoded)
        
    Returns:
        tuple: (loss, accuracy) - evaluation metrics
    """
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy


def visualize_results(history, x_test, y_test, model, num_samples=10):
    """
    Visualize training metrics and sample predictions.
    
    Args:
        history (tf.keras.callbacks.History): Training history
        x_test (ndarray): Test images
        y_test (ndarray): Test labels (one-hot encoded)
        model (tf.keras.Model): Trained model
        num_samples (int): Number of test samples to visualize
    """
    # Plot training metrics
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper right')
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc='lower right')
    
    plt.tight_layout()
    plt.show()
    
    # Visualize some predictions
    # Get random samples from the test set
    indices = np.random.choice(len(x_test), num_samples)
    sample_images = x_test[indices]
    sample_labels = np.argmax(y_test[indices], axis=1)
    
    # Make predictions
    predictions = model.predict(sample_images)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Plot the images and predictions
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
        title = f"Pred: {predicted_labels[i]}, True: {sample_labels[i]}"
        plt.title(title, color=("green" if predicted_labels[i] == sample_labels[i] else "red"))
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to execute the MNIST digit recognition pipeline using TensorFlow.
    """
    # Load and preprocess data
    print("Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = load_mnist_data()
    print(f"Data shapes - x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}")
    
    # Create model
    print("Creating model...")
    model = create_model()
    model.summary()
    
    # Train model
    print("Training model...")
    trained_model, history = train_model(
        model, 
        x_train, y_train, 
        x_test, y_test, 
        epochs=5
    )
    
    # Evaluate model
    loss, accuracy = evaluate_model(trained_model, x_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(history, x_test, y_test, trained_model)
    
    # Save model
    trained_model.save("mnist_tensorflow_model.h5")
    print("Model saved to 'mnist_tensorflow_model.h5'")


if __name__ == "__main__":
    main()