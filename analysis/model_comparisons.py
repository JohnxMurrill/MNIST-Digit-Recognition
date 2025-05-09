import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import tensorflow as tf

# Import the implementation files
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import torch_mnist, scratch_mnist, tensorflow_mnist

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)


def compare_implementations(num_epochs=5, batch_size=64):
    """
    Compare PyTorch, TensorFlow, and Scratch implementations of MNIST digit recognition.
    
    This function runs all three implementations, tracks training time, convergence rates,
    and final performance, then visualizes the results for comparison.
    
    Args:
        num_epochs (int): Number of training epochs for all implementations
        batch_size (int): Batch size to use for training (where applicable)
        
    Returns:
        dict: Dictionary containing comparison metrics for all implementations
    """
    results = {
        'pytorch': {
            'training_time': 0,
            'train_losses': [],
            'test_accuracies': [],
            'final_accuracy': 0
        },
        'tensorflow': {
            'training_time': 0,
            'train_losses': [],
            'test_accuracies': [],
            'final_accuracy': 0
        },
        'scratch': {
            'training_time': 0,
            'train_losses': [],
            'test_accuracies': [],
            'final_accuracy': 0
        }
    }
    
    print("=" * 50)
    print(f"Comparing MNIST implementations over {num_epochs} epochs")
    print("=" * 50)
    
    # ===================== PyTorch Implementation =====================
    print("\nRunning PyTorch implementation...")
    
    # Load data
    train_loader, test_loader = torch_mnist.load_mnist_data(batch_size=batch_size)
    
    # Initialize model
    pytorch_model = torch_mnist.MNISTModel()
    
    # Train model and time it
    start_time = time.time()
    trained_model, train_losses, test_accuracies = torch_mnist.train_model(
        pytorch_model, 
        train_loader, 
        test_loader,
        epochs=num_epochs
    )
    end_time = time.time()
    
    # Calculate final accuracy
    final_accuracy = torch_mnist.evaluate_model(trained_model, test_loader)
    
    # Store results
    results['pytorch']['training_time'] = end_time - start_time
    results['pytorch']['train_losses'] = train_losses
    results['pytorch']['test_accuracies'] = test_accuracies
    results['pytorch']['final_accuracy'] = final_accuracy
    
    print(f"PyTorch training completed in {results['pytorch']['training_time']:.2f} seconds")
    print(f"Final test accuracy: {final_accuracy:.4f}")
    
    # ===================== TensorFlow Implementation =====================
    print("\nRunning TensorFlow implementation...")
    
    # Load data
    x_train, y_train, x_test, y_test = tensorflow_mnist.load_mnist_data()
    
    # Create model
    tensorflow_model = tensorflow_mnist.create_model()
    
    # Custom callback to track metrics per epoch
    class MetricsCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super(MetricsCallback, self).__init__()
            self.train_losses = []
            self.test_accuracies = []
            
        def on_epoch_end(self, epoch, logs=None):
            self.train_losses.append(logs.get('loss'))
            self.test_accuracies.append(logs.get('val_accuracy'))
    
    metrics_callback = MetricsCallback()
    
    # Train model and time it
    start_time = time.time()
    history = tensorflow_model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(x_test, y_test),
        callbacks=[metrics_callback],
        verbose=1
    )
    end_time = time.time()
    
    # Calculate final metrics
    loss, accuracy = tensorflow_model.evaluate(x_test, y_test, verbose=0)
    
    # Store results
    results['tensorflow']['training_time'] = end_time - start_time
    results['tensorflow']['train_losses'] = metrics_callback.train_losses
    results['tensorflow']['test_accuracies'] = metrics_callback.test_accuracies
    results['tensorflow']['final_accuracy'] = accuracy
    
    print(f"TensorFlow training completed in {results['tensorflow']['training_time']:.2f} seconds")
    print(f"Final test accuracy: {accuracy:.4f}")
    
    # ===================== Scratch Implementation =====================
    print("\nRunning Scratch (NumPy) implementation...")
    
    # Load data
    x_train, y_train, x_test, y_test = scratch_mnist.load_mnist_data()
    
    # For our scratch implementation, we need to one-hot encode the labels
    # but make a copy to avoid modifying the original data
    y_train_one_hot = np.eye(10)[y_train.astype(int)]
    
    # Create a custom class that extends the scratch implementation to track metrics
    class TrackedNeuralNetwork(scratch_mnist.NeuralNetwork):
        def __init__(self, input_size=784, hidden_size=128, output_size=10):
            super().__init__(input_size, hidden_size, output_size)
            self.train_losses = []
            self.test_accuracies = []
            
        def train_with_tracking(self, x_train, y_train, x_test, y_test, epochs=1000, learning_rate=0.01):
            """Extended training method that tracks metrics per epoch"""
            for epoch in range(epochs):
                # Forward and backward pass (standard training)
                self.forward(x_train)
                self.backward(x_train, y_train, learning_rate)
                
                # Calculate and track loss
                loss = np.mean(np.square(y_train - self.output))
                self.train_losses.append(loss)
                
                # Calculate and track test accuracy
                predictions = self.predict(x_test)
                accuracy = np.mean(predictions == np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test)
                self.test_accuracies.append(accuracy)
                
                if epoch % 1 == 0:  # Print every epoch
                    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
    
    # Initialize model
    scratch_model = TrackedNeuralNetwork(input_size=784, hidden_size=128, output_size=10)
    
    # Train model and time it (with fewer epochs to make it comparable to the other implementations)
    start_time = time.time()
    scratch_model.train_with_tracking(
        x_train, 
        y_train_one_hot, 
        x_test, 
        y_test,
        epochs=num_epochs,  # Match epochs with other implementations
        learning_rate=0.1   # Higher learning rate for faster convergence
    )
    end_time = time.time()
    
    # Final test prediction
    final_predictions = scratch_model.predict(x_test)
    final_accuracy = np.mean(final_predictions == y_test.astype(int))
    
    # Store results
    results['scratch']['training_time'] = end_time - start_time
    results['scratch']['train_losses'] = scratch_model.train_losses
    results['scratch']['test_accuracies'] = scratch_model.test_accuracies
    results['scratch']['final_accuracy'] = final_accuracy
    
    print(f"Scratch implementation training completed in {results['scratch']['training_time']:.2f} seconds")
    print(f"Final test accuracy: {final_accuracy:.4f}")
    
    return results

def visualize_comparison(results):
    """
    Visualize the comparison results between PyTorch, TensorFlow, and Scratch implementations.
    
    Args:
        results (dict): Dictionary containing comparison metrics for all implementations
    """
    plt.figure(figsize=(15, 12))
    
    # Define colors and implementation names
    colors = {'pytorch': '#EE4C2C', 'tensorflow': '#FF9E0B', 'scratch': '#3498DB'}
    framework_names = {'pytorch': 'PyTorch (CNN)', 'tensorflow': 'TensorFlow (CNN)', 'scratch': 'NumPy (MLP)'}
    
    # Plot training loss comparison
    plt.subplot(2, 2, 1)
    for impl, color in colors.items():
        plt.plot(results[impl]['train_losses'], label=framework_names[impl], color=color)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot test accuracy comparison
    plt.subplot(2, 2, 2)
    for impl, color in colors.items():
        plt.plot(results[impl]['test_accuracies'], label=framework_names[impl], color=color)
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot training time comparison
    plt.subplot(2, 2, 3)
    frameworks = list(framework_names.values())
    times = [results['pytorch']['training_time'], results['tensorflow']['training_time'], results['scratch']['training_time']]
    bars = plt.bar(frameworks, times, color=list(colors.values()))
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=15)
    for i, v in enumerate(times):
        plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot final accuracy comparison
    plt.subplot(2, 2, 4)
    accuracies = [results['pytorch']['final_accuracy'], results['tensorflow']['final_accuracy'], results['scratch']['final_accuracy']]
    bars = plt.bar(frameworks, accuracies, color=list(colors.values()))
    plt.title('Final Test Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=15)
    # Set y-axis to start from a reasonable value to better show differences
    plt.ylim([min(accuracies) - 0.01, 1.0])
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.005, f"{v:.4f}", ha='center')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison visualization saved to 'comparison_results.png'")

def calculate_epochs_to_threshold(results, threshold=0.97):
    """
    Calculate how many epochs it takes for each implementation to reach a threshold accuracy.
    
    Args:
        results (dict): Dictionary containing comparison metrics for all frameworks
        threshold (float): Accuracy threshold to measure convergence speed
        
    Returns:
        dict: Epochs required to reach threshold for each framework
    """
    convergence = {}
    
    # Check convergence for each implementation
    for impl in ['pytorch', 'tensorflow', 'scratch']:
        for i, acc in enumerate(results[impl]['test_accuracies']):
            if acc >= threshold:
                convergence[impl] = i + 1
                break
        else:
            convergence[impl] = "Did not reach threshold"
    
    return convergence

def analyze_learning_rates(num_epochs=5, batch_sizes=[32, 64, 128]):
    """
    Analyze how different batch sizes affect learning rates and convergence.
    
    Args:
        num_epochs (int): Number of training epochs
        batch_sizes (list): List of batch sizes to test
        
    Returns:
        dict: Results for different batch sizes for all frameworks
    """
    batch_results = {}
    
    for batch_size in batch_sizes:
        print(f"\n{'-' * 30}")
        print(f"Testing with batch size: {batch_size}")
        print(f"{'-' * 30}")
        
        batch_results[batch_size] = compare_implementations(num_epochs=num_epochs, batch_size=batch_size)
    
    # Visualize batch size comparison
    plt.figure(figsize=(18, 15))
    
    # Plot training time vs batch size
    plt.subplot(3, 2, 1)
    x = np.arange(len(batch_sizes))
    width = 0.25
    pytorch_times = [batch_results[bs]['pytorch']['training_time'] for bs in batch_sizes]
    tensorflow_times = [batch_results[bs]['tensorflow']['training_time'] for bs in batch_sizes]
    scratch_times = [batch_results[bs]['scratch']['training_time'] for bs in batch_sizes]
    
    plt.bar(x - width, pytorch_times, width, label='PyTorch (CNN)', color='#EE4C2C')
    plt.bar(x, tensorflow_times, width, label='TensorFlow (CNN)', color='#FF9E0B')
    plt.bar(x + width, scratch_times, width, label='NumPy (MLP)', color='#3498DB')
    plt.xlabel('Batch Size')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs Batch Size')
    plt.xticks(x, batch_sizes)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot final accuracy vs batch size
    plt.subplot(3, 2, 2)
    pytorch_accs = [batch_results[bs]['pytorch']['final_accuracy'] for bs in batch_sizes]
    tensorflow_accs = [batch_results[bs]['tensorflow']['final_accuracy'] for bs in batch_sizes]
    scratch_accs = [batch_results[bs]['scratch']['final_accuracy'] for bs in batch_sizes]
    
    plt.bar(x - width, pytorch_accs, width, label='PyTorch (CNN)', color='#EE4C2C')
    plt.bar(x, tensorflow_accs, width, label='TensorFlow (CNN)', color='#FF9E0B')
    plt.bar(x + width, scratch_accs, width, label='NumPy (MLP)', color='#3498DB')
    plt.xlabel('Batch Size')
    plt.ylabel('Final Accuracy')
    plt.title('Final Accuracy vs Batch Size')
    plt.xticks(x, batch_sizes)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot convergence rates for PyTorch
    plt.subplot(3, 2, 3)
    for bs in batch_sizes:
        plt.plot(batch_results[bs]['pytorch']['test_accuracies'], label=f'Batch Size {bs}')
    plt.title('PyTorch (CNN) Convergence by Batch Size')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot convergence rates for TensorFlow
    plt.subplot(3, 2, 4)
    for bs in batch_sizes:
        plt.plot(batch_results[bs]['tensorflow']['test_accuracies'], label=f'Batch Size {bs}')
    plt.title('TensorFlow (CNN) Convergence by Batch Size')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot convergence rates for Scratch
    plt.subplot(3, 2, 5)
    for bs in batch_sizes:
        plt.plot(batch_results[bs]['scratch']['test_accuracies'], label=f'Batch Size {bs}')
    plt.title('NumPy (MLP) Convergence by Batch Size')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot architecture comparison (CNN vs MLP)
    plt.subplot(3, 2, 6)
    # Average across batch sizes for cleaner comparison
    pytorch_avg = np.mean([batch_results[bs]['pytorch']['test_accuracies'] for bs in batch_sizes], axis=0)
    tensorflow_avg = np.mean([batch_results[bs]['tensorflow']['test_accuracies'] for bs in batch_sizes], axis=0)
    scratch_avg = np.mean([batch_results[bs]['scratch']['test_accuracies'] for bs in batch_sizes], axis=0)
    
    plt.plot(pytorch_avg, label='PyTorch (CNN)', color='#EE4C2C')
    plt.plot(tensorflow_avg, label='TensorFlow (CNN)', color='#FF9E0B')
    plt.plot(scratch_avg, label='NumPy (MLP)', color='#3498DB')
    plt.title('CNN vs MLP Architecture Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('batch_size_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Batch size comparison saved to 'batch_size_comparison.png'")
    
    return batch_results

def hardware_comparison():
    """
    Print hardware information to contextualize performance comparisons.
    """
    import platform
    
    print("\nHardware Information:")
    print(f"System: {platform.system()} {platform.version()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # PyTorch specific
    if torch.cuda.is_available():
        print("\nPyTorch GPU Information:")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("\nPyTorch: Running on CPU")
    
    # TensorFlow specific
    print("\nTensorFlow Device Information:")
    physical_devices = tf.config.list_physical_devices()
    print("Devices:", physical_devices)
    print(f"TensorFlow Version: {tf.__version__}")

def model_parameter_comparison():
    """
    Code to add to the comparison script to calculate and compare the number of
    parameters in each model implementation.
    """
    def count_parameters():
        """
        Calculate and compare the number of parameters in each model.
        """
        print("\nModel Parameter Comparison:")
        print("=" * 50)
        
        # PyTorch CNN parameters
        pytorch_model = torch_mnist.MNISTModel()
        pytorch_params = sum(p.numel() for p in pytorch_model.parameters())
        print(f"PyTorch CNN Parameters: {pytorch_params:,}")
        
        # Detail the CNN architecture parameters
        print("  - Conv1: 1->32 filters (3x3): 32 * (3*3*1 + 1) = 320 parameters")
        print("  - Conv2: 32->64 filters (3x3): 64 * (3*3*32 + 1) = 18,496 parameters")
        print("  - FC1: 7*7*64->128: 128 * (7*7*64 + 1) = 401,536 parameters")
        print("  - FC2: 128->10: 10 * (128 + 1) = 1,290 parameters")
        
        # TensorFlow CNN parameters
        tf_model = tensorflow_mnist.create_model()
        tf_params = tf_model.count_params()
        print(f"TensorFlow CNN Parameters: {tf_params:,}")
        
        # Scratch MLP parameters
        scratch_model = scratch_mnist.NeuralNetwork()
        # input->hidden weights + hidden biases + hidden->output weights + output biases
        scratch_params = (784 * 128) + 128 + (128 * 10) + 10
        print(f"Scratch MLP Parameters: {scratch_params:,}")
        print("  - Input->Hidden: 784 * 128 = 100,352 parameters")
        print("  - Hidden Biases: 128 parameters")
        print("  - Hidden->Output: 128 * 10 = 1,280 parameters")
        print("  - Output Biases: 10 parameters")
        
        return {
            'pytorch': pytorch_params,
            'tensorflow': tf_params,
            'scratch': scratch_params
        }
    
    # Call this function in the main code
    param_counts = count_parameters()
    
    # Visualize parameter counts
    plt.figure(figsize=(10, 6))
    implementations = ['PyTorch CNN', 'TensorFlow CNN', 'NumPy MLP']
    params = [param_counts['pytorch'], param_counts['tensorflow'], param_counts['scratch']]
    
    plt.bar(implementations, params, color=['#EE4C2C', '#FF9E0B', '#3498DB'])
    plt.title('Model Parameters Comparison')
    plt.ylabel('Number of Parameters')
    plt.yscale('log')  # Log scale to better show differences
    
    # Add parameter count labels
    for i, v in enumerate(params):
        plt.text(i, v * 1.1, f"{v:,}", ha='center')
    
    plt.tight_layout()
    plt.savefig('parameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def memory_usage_tracking():
    """
    Code to add to the comparison script to track memory usage during training.
    """
    import tracemalloc
    import psutil
    
    def measure_memory_usage(func, *args, **kwargs):
        """
        Measure peak memory usage during function execution.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            tuple: (function_result, peak_memory_bytes)
        """
        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        start_memory = process.memory_info().rss
        
        # Run the function
        result = func(*args, **kwargs)
        
        # Get peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        end_memory = process.memory_info().rss
        memory_used = end_memory - start_memory + peak
        
        # Stop memory tracking
        tracemalloc.stop()
        
        return result, memory_used
    
    def training_with_memory_tracking():
        """
        Example of how to use memory tracking during model training
        """
        memory_results = {
            'pytorch': 0,
            'tensorflow': 0,
            'scratch': 0
        }
        
        # PyTorch memory tracking
        print("\nTraining PyTorch model with memory tracking...")
        train_loader, test_loader = torch_mnist.load_mnist_data(batch_size=64)
        pytorch_model = torch_mnist.MNISTModel()
        
        _, pytorch_memory = measure_memory_usage(
            torch_mnist.train_model,
            pytorch_model, 
            train_loader, 
            test_loader,
            epochs=3  # Fewer epochs for quicker test
        )
        memory_results['pytorch'] = pytorch_memory
        
        # TensorFlow memory tracking
        print("\nTraining TensorFlow model with memory tracking...")
        x_train, y_train, x_test, y_test = tensorflow_mnist.load_mnist_data()
        tf_model = tensorflow_mnist.create_model()
        
        _, tf_memory = measure_memory_usage(
            tf_model.fit,
            x_train, y_train,
            batch_size=64,
            epochs=3,  # Fewer epochs for quicker test
            validation_data=(x_test, y_test),
            verbose=1
        )
        memory_results['tensorflow'] = tf_memory
        
        # Scratch memory tracking
        print("\nTraining Scratch model with memory tracking...")
        scratch_x_train, scratch_y_train, scratch_x_test, scratch_y_test = scratch_mnist.load_mnist_data()
        scratch_y_train_one_hot = np.eye(10)[scratch_y_train.astype(int)]
        scratch_model = scratch_mnist.NeuralNetwork()
        
        _, scratch_memory = measure_memory_usage(
            scratch_model.train_minibatch,  # Assuming modified version with minibatch training
            scratch_x_train, 
            scratch_y_train_one_hot,
            batch_size=64,
            epochs=3,  # Fewer epochs for quicker test
            x_test=scratch_x_test,
            y_test=scratch_y_test
        )
        memory_results['scratch'] = scratch_memory
        
        # Report and visualize memory usage
        print("\nMemory Usage Results:")
        for impl, memory in memory_results.items():
            print(f"{impl}: {memory / (1024 * 1024):.2f} MB")
        
        # Visualize memory usage
        plt.figure(figsize=(10, 6))
        implementations = ['PyTorch CNN', 'TensorFlow CNN', 'NumPy MLP']
        memory_values = [
            memory_results['pytorch'] / (1024 * 1024), 
            memory_results['tensorflow'] / (1024 * 1024), 
            memory_results['scratch'] / (1024 * 1024)
        ]
        
        plt.bar(implementations, memory_values, color=['#EE4C2C', '#FF9E0B', '#3498DB'])
        plt.title('Memory Usage Comparison')
        plt.ylabel('Memory Usage (MB)')
        
        # Add memory usage labels
        for i, v in enumerate(memory_values):
            plt.text(i, v * 1.05, f"{v:.2f} MB", ha='center')
        
        plt.tight_layout()
        plt.savefig('memory_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return memory_results

def prediction_time_comparison():
    """
    Code to add to compare inference speed for each model implementation.
    """
    def measure_inference_speed():
        """
        Measure and compare inference speed for each implementation.
        """
        print("\nInference Speed Comparison:")
        print("=" * 50)
        
        # Load test data for all implementations
        # PyTorch
        _, test_loader = torch_mnist.load_mnist_data(batch_size=1)  # Batch size 1 for fair comparison
        pytorch_model = torch_mnist.MNISTModel()
        # Load a trained model if available, otherwise use untrained model
        try:
            pytorch_model.load_state_dict(torch.load("mnist_pytorch_model.pth"))
        except:
            pass
        pytorch_model.eval()
        
        # TensorFlow
        _, _, x_test, y_test = tensorflow_mnist.load_mnist_data()
        try:
            tf_model = tf.keras.models.load_model("mnist_tensorflow_model.h5")
        except:
            tf_model = tensorflow_mnist.create_model()
        
        # Scratch
        scratch_x_train, scratch_y_train, scratch_x_test, scratch_y_test = scratch_mnist.load_mnist_data()
        scratch_model = scratch_mnist.NeuralNetwork()
        
        # Measure PyTorch inference time (average over 100 samples)
        pytorch_times = []
        for images, _ in test_loader:
            start_time = time.time()
            with torch.no_grad():
                _ = pytorch_model(images)
            pytorch_times.append(time.time() - start_time)
            if len(pytorch_times) >= 100:
                break
        
        # Measure TensorFlow inference time (average over 100 samples)
        tf_times = []
        for i in range(100):
            sample = x_test[i:i+1]
            start_time = time.time()
            _ = tf_model.predict(sample, verbose=0)
            tf_times.append(time.time() - start_time)
        
        # Measure Scratch inference time (average over 100 samples)
        scratch_times = []
        for i in range(100):
            sample = scratch_x_test[i:i+1]
            start_time = time.time()
            _ = scratch_model.predict(sample)
            scratch_times.append(time.time() - start_time)
        
        # Calculate average inference times
        avg_pytorch_time = np.mean(pytorch_times) * 1000  # Convert to ms
        avg_tf_time = np.mean(tf_times) * 1000  # Convert to ms
        avg_scratch_time = np.mean(scratch_times) * 1000  # Convert to ms
        
        print(f"PyTorch CNN average inference time: {avg_pytorch_time:.2f} ms per sample")
        print(f"TensorFlow CNN average inference time: {avg_tf_time:.2f} ms per sample")
        print(f"NumPy MLP average inference time: {avg_scratch_time:.2f} ms per sample")
        
        # Visualize inference times
        plt.figure(figsize=(10, 6))
        implementations = ['PyTorch CNN', 'TensorFlow CNN', 'NumPy MLP']
        inference_times = [avg_pytorch_time, avg_tf_time, avg_scratch_time]
        
        plt.bar(implementations, inference_times, color=['#EE4C2C', '#FF9E0B', '#3498DB'])
        plt.title('Inference Time Comparison')
        plt.ylabel('Time per Sample (ms)')
        
        # Add time labels
        for i, v in enumerate(inference_times):
            plt.text(i, v * 1.05, f"{v:.2f} ms", ha='center')
        
        plt.tight_layout()
        plt.savefig('inference_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'pytorch': avg_pytorch_time,
            'tensorflow': avg_tf_time,
            'scratch': avg_scratch_time
        }

if __name__ == "__main__":
    # Print hardware information
    hardware_comparison()
    
    # Simple comparison with default parameters
    print("\n\n" + "=" * 50)
    print("Running basic comparison")
    print("=" * 50)
    results = compare_implementations(num_epochs=5)
    visualize_comparison(results)
    
    # Calculate convergence speed
    threshold = 0.97  # 97% accuracy threshold
    convergence = calculate_epochs_to_threshold(results, threshold)
    print(f"\nEpochs to reach {threshold*100}% accuracy:")
    print(f"PyTorch (CNN): {convergence['pytorch']}")
    print(f"TensorFlow (CNN): {convergence['tensorflow']}")
    print(f"NumPy (MLP): {convergence['scratch']}")
    
    # Add model architecture comparison
    print("\nArchitecture Comparison:")
    print("=" * 50)
    print("CNN (PyTorch/TensorFlow):")
    print("- Uses convolutional layers that capture spatial patterns")
    print("- Has two Conv2D layers (32 and 64 filters) with max pooling")
    print("- Contains about 1.2M parameters")
    print("\nMLP (NumPy from scratch):")
    print("- Simple feedforward network with one hidden layer")
    print("- No convolutional layers, treats image as flat vector")
    print("- Contains about 101k parameters (784×128 + 128×10)")
    print("=" * 50)
    
    # Analyze effect of batch size (optional)
    use_batch_analysis = input("\nRun batch size analysis? (y/n): ").lower() == 'y'
    if use_batch_analysis:
        batch_results = analyze_learning_rates(num_epochs=5, batch_sizes=[32, 64, 128])