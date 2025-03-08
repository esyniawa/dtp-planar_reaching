import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional, Sequence

from .layers import FeedforwardLayer, FeedbackLayer


class DDTPNetwork(nn.Module):
    """
    Direct Difference Target Propagation Network with Difference Reconstruction Loss

    This implements the DDTP-linear variant with direct feedback connections from the output layer to each hidden layer,
    trained using the Difference Reconstruction Loss.
    """
    def __init__(self,
                 layer_sizes: Sequence[int],
                 ff_activation: nn.Module = nn.ELU(),
                 fb_activation: nn.Module = nn.ELU(),
                 output_activation: Optional[nn.Module] = None):
        """
        Initialize the DDTP-linear network.

        :arg layer_sizes: List or tuple of integers describing network architecture
                          [input_size, hidden1_size, ..., output_size]
        :arg ff_activation: Activation function for hidden layers
        :arg output_activation: Activation function for output layer (None for regression)
        """
        super().__init__()
        self.layer_sizes = layer_sizes
        n_layers = len(layer_sizes)

        # Create forward layers
        self.forward_layers = nn.ModuleList()
        for i in range(n_layers - 1):
            act = output_activation if i == n_layers - 2 else ff_activation
            layer = FeedforwardLayer(in_features=layer_sizes[i],
                                     out_features=layer_sizes[i + 1],
                                     activation=act)
            self.forward_layers.append(layer)

        # Create feedback layers (direct connections from output to each hidden layer)
        self.feedback_layers = nn.ModuleList()
        # No feedback for output layer, only for hidden layers
        for i in range(n_layers - 2):
            # From output layer to hidden layer i = Direct DTP
            self.feedback_layers.append(FeedbackLayer(output_size=layer_sizes[-1],
                                                      target_size=layer_sizes[i + 1],
                                                      activation=fb_activation))

    def forward(self, x):
        """Forward pass through the network. Gradients are detached in FeedforwardLayer"""
        h = x
        for layer in self.forward_layers:
            h = layer(h)
        return h

    def compute_targets(self, output: torch.Tensor, output_target: torch.Tensor):
        """
        Propagate targets from output layer to all hidden layers

        :arg output: Actual output layer activations
        :arg output_target: Target output layer activations
        :return: List of targets for each hidden layer
        """
        targets = []
        for i, fb_layer in enumerate(self.feedback_layers):
            hidden_true = self.forward_layers[i].output
            target = fb_layer.compute_target(output=output,
                                             output_target=output_target,
                                             hidden_true=hidden_true)
            targets.append(target)
        return targets

    def local_loss(self, targets: List[torch.Tensor]):
        """
        Compute local loss for each layer based on targets
        This implements the local layer losses L_i = ||h_hat_i - h_i||^2

        :arg targets: List of targets for each hidden layer
        :return: List of local losses for each hidden layer
        """
        losses = []
        for i, target in enumerate(targets):
            hidden_true = self.forward_layers[i].output
            loss = torch.mean((target - hidden_true) ** 2)
            losses.append(loss)
        return losses

    def drl_loss(self, idx: int, sigma: float = 0.1):
        """
        Compute Difference Reconstruction Loss for a feedback layer

        :arg idx: Index of the feedback layer to train
        :arg sigma: Standard deviation of noise perturbation
        :return: The DRL loss for the specified feedback layer
        """
        assert 0 <= idx < len(self.feedback_layers), "Invalid feedback layer index"

        hidden_layer = self.forward_layers[idx]
        feedback_layer = self.feedback_layers[idx]

        # Get true activations
        hidden_true = hidden_layer.output
        output_true = self.forward_layers[-1].output

        # Add noise to hidden activations (noise corruption)
        noise = torch.randn_like(hidden_true) * sigma
        hidden_noisy = hidden_true + noise  # (h_i + noise)

        # Forward propagate the noise through the network to the output
        h = hidden_noisy
        for layer in self.forward_layers[idx + 1:]:
            h = layer(h)
        output_noisy = h

        # Reconstruct the hidden activations using feedback connections with difference correction
        # g_L,i(f_i,L(h_i + noise), h_L, h_i)
        hidden_reconstructed = feedback_layer.feedback(output_noisy) + (
                    hidden_true - feedback_layer.feedback(output_true))

        # Compute the reconstruction error (DRL loss)
        # Loss = || g_L,i(f_i,L(h_i + noise), h_L, h_i) - (h_i + noise) ||^2
        loss = torch.mean((hidden_reconstructed - hidden_noisy) ** 2)

        return loss

