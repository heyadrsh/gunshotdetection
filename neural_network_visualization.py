import sys
import matplotlib.pyplot as plt
import numpy as np

# Function to create a simple neural network diagram

def plot_neural_network():
    plt.figure(figsize=(10, 6))
    plt.title('Neural Network Diagram')

    # Example structure of a neural network
    layers = [3, 5, 4, 2]  # Input layer, hidden layers, output layer
    layer_colors = ['#FFDDC1', '#FFABAB', '#FFC3A0', '#D5AAFF']

    for i, layer_size in enumerate(layers):
        layer_x = i * 2  # Space layers apart
        for j in range(layer_size):
            plt.scatter(layer_x, j, s=500, color=layer_colors[i], edgecolor='black')
            if i > 0:
                for k in range(layers[i - 1]):
                    plt.plot([layer_x - 2, layer_x], [k, j], color='gray', alpha=0.5)

    plt.xlim(-1, len(layers) * 2)
    plt.ylim(-1, max(layers) + 1)
    plt.axis('off')  # Hide axes
    plt.show()

if __name__ == '__main__':
    plot_neural_network() 