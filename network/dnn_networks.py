import torch
import torch.nn as nn
from typing import Sequence, Optional

from .layers import FeedforwardLayer

class DNN(nn.Module):
    """
    Standard Implementation of a Deep Neural Network (DNN) 
    """
    def __init__(self,
                 layer_sizes: Sequence[int],
                 activation: nn.Module = nn.ReLU(),
                 output_activation: Optional[nn.Module] = None):
        """
        Initialization of DNN.

        :param layer_sizes: List with all layer sizes [input_size, hidden1_size, ..., output_size]
        :param activation: Activation function for Hidden Layers
        :param output_activation: Activation function for the output (None for Regression)
        """
        super().__init__()
        self.layer_sizes = layer_sizes
        n_layers = len(layer_sizes)
        
        ######################################################################################################
        # Create forward layers
        ######################################################################################################
        self.forward_layers = nn.ModuleList()
        for i in range(n_layers - 1):
            act = output_activation if i == n_layers - 2 else activation
            layer = FeedforwardLayer(in_features=layer_sizes[i],
                                     out_features=layer_sizes[i + 1],
                                     activation=act)
            self.forward_layers.append(layer)

    def forward(self, x):
        """Forward pass through the network. Gradients are detached in FeedforwardLayer"""
        h = x
        for layer in self.forward_layers:
            h = layer.forward(h)
        return h
    
    def extract_hidden_activations(self, inputs):
        """
        Extracts hidden activations from the Dense Neural Network.
        
        :param model: The trained model (an instance of DNN)
        :param inputs: Input tensor to the network
        :return: List of activations for each hidden layer
        """
        hidden_activations = []

        # Perform a forward pass through the model
        h = inputs
        for layer in self.forward_layers:
            h = layer(h)  # Compute forward pass through the model
            hidden_activations.append(layer.output.detach())  # Store the activation (detach from computation graph)

        return hidden_activations

