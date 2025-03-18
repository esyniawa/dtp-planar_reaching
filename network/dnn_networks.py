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
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            act = output_activation if i == n_layers - 2 else activation
            layer = FeedforwardLayer(in_features=layer_sizes[i],
                                     out_features=layer_sizes[i + 1],
                                     activation=act)
            self.layers.append(layer)

    def forward(self, x):
        """Forward pass through the network. Gradients are detached in FeedforwardLayer"""
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h

