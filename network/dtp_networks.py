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

        ######################################################################################################
        # Create forward layers
        ######################################################################################################
        self.forward_layers = nn.ModuleList()
        for i in range(n_layers - 1):
            act = output_activation if i == n_layers - 2 else ff_activation
            layer = FeedforwardLayer(in_features=layer_sizes[i],
                                     out_features=layer_sizes[i + 1],
                                     activation=act)
            self.forward_layers.append(layer)

        ######################################################################################################
        # Create feedback layers (direct connections from output to each hidden layer)
        ######################################################################################################
        self.feedback_layers = nn.ModuleList()
        # No feedback for output layer, only for hidden layers
        for i in range(n_layers - 2):
            # From output layer to hidden layer i = Direct DTP
            layer = FeedbackLayer(output_size=layer_sizes[-1],
                                  target_size=layer_sizes[i + 1],
                                  activation=fb_activation)
            self.feedback_layers.append(layer)

    def forward(self, x):
        """Forward pass through the network. Gradients are detached in FeedforwardLayer"""
        h = x
        for layer in self.forward_layers:
            h = layer.forward(h)
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
        
        # Check that index is valid number inside the layers
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
        # g_L,i(f_(i,L)(h_i + noise), h_L, h_i)
        hidden_reconstructed = feedback_layer.feedback(output_noisy) + (
                    hidden_true - feedback_layer.feedback(output_true))

        # Compute the reconstruction error (DRL loss)
        # Loss = || g_L,i(f_i,L(h_i + noise), h_L, h_i) - (h_i + noise) ||^2
        loss = torch.mean((hidden_reconstructed - hidden_noisy) ** 2)
        

        return loss


class DDTPRHLNetwork(nn.Module):
    """
    Direct Difference Target Propagation Network with Random Hidden Layer (RHL)

    This implements a DDTP network with a shared random hidden layer in the
    feedback pathway from the output layer to each hidden layer.
    """

    def __init__(self,
                 layer_sizes: Sequence[int],
                 ff_activation: nn.Module = nn.ELU(),
                 fb_activation: nn.Module = nn.ELU(),
                 output_activation: Optional[nn.Module] = None,
                 random_hidden_size: int = 1024):
        """
        Initialize the DDTP-RHL network.

        :arg layer_sizes: List or tuple of integers describing network architecture
        :arg ff_activation: Activation function for hidden layers
        :arg fb_activation: Activation function for feedback connections
        :arg output_activation: Activation function for output layer (None for regression)
        :arg random_hidden_size: Size of the random hidden layer in feedback path
        """
        super().__init__()

        self.layer_sizes = layer_sizes
        n_layers = len(layer_sizes)
        self.fb_activation = fb_activation

        # Create forward layers
        self.forward_layers = nn.ModuleList()
        for i in range(n_layers - 1):
            act = output_activation if i == n_layers - 2 else ff_activation
            layer = FeedforwardLayer(in_features=layer_sizes[i],
                                     out_features=layer_sizes[i + 1],
                                     activation=act)
            self.forward_layers.append(layer)

        # Create shared random hidden layer in feedback path
        self.random_layer = nn.Linear(layer_sizes[-1], random_hidden_size)
        nn.init.xavier_normal_(self.random_layer.weight)
        nn.init.zeros_(self.random_layer.bias)

        # Create feedback layers (from random hidden layer to each hidden layer)
        self.feedback_layers = nn.ModuleList()
        for i in range(n_layers - 2):  # No feedback for output layer
            feedback = nn.Linear(random_hidden_size, layer_sizes[i + 1])
            nn.init.xavier_normal_(feedback.weight)
            nn.init.zeros_(feedback.bias)
            self.feedback_layers.append(feedback)

    def forward(self, x):
        """Forward pass through the network."""
        h = x
        for layer in self.forward_layers:
            h = layer(h)
        return h

    def _feedback(self, output, idx):
        """Apply feedback from output through random hidden layer to target hidden layer."""
        # First pass through random hidden layer
        random_hidden = self.fb_activation(self.random_layer(output))
        # Then through the feedback layer to target
        return self.fb_activation(self.feedback_layers[idx](random_hidden))

    def compute_targets(self, output: torch.Tensor, output_target: torch.Tensor):
        """
        Propagate targets from output layer to all hidden layers
        """
        targets = []
        for i in range(len(self.feedback_layers)):
            hidden_true = self.forward_layers[i].output

            # Compute difference target with feedback pathway
            target_from_output = self._feedback(output_target, i)
            reconstruction = self._feedback(output, i)

            # Target with difference correction
            target = target_from_output + (hidden_true - reconstruction)
            targets.append(target)

        return targets

    def local_loss(self, targets: List[torch.Tensor]):
        """
        Compute local loss for each layer based on targets
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
        """
        assert 0 <= idx < len(self.feedback_layers), "Invalid feedback layer index"

        hidden_layer = self.forward_layers[idx]

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
        hidden_reconstructed = self._feedback(output_noisy, idx) + (
                hidden_true - self._feedback(output_true, idx))

        # Compute the reconstruction error (DRL loss)
        loss = torch.mean((hidden_reconstructed - hidden_noisy) ** 2)

        return loss
