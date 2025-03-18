import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class DTPConfig:
    """Configuration for DTP."""
    beta: float = 0.9
    noise_scale: float = 0.1
    K_iterations: int = 1
    learning_rate: float = 0.001


class DTPLayer(nn.Module):
    """Enhanced DTP layer with internal gradient and loss calculations."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: str = 'tanh',
                 config: DTPConfig = DTPConfig(),
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        
        self.activation_type = activation
        self.config = config

        # Forward weights and bias (out_features x in_features)
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        # Feedback weights and bias (in_features x out_features)
        self.feedback_weights = nn.Parameter(torch.Tensor(in_features, out_features))
        self.feedback_bias = nn.Parameter(torch.Tensor(in_features)) if bias else None

        # Initialize parameters
        self.reset_parameters()

        # Cache for intermediate values
        self.input = None
        self.pre_activation = None
        self.post_activation = None
        self.target = None
        self.reconstruction_loss = None

    def reset_parameters(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        if self.activation_type in ['sigmoid', 'tanh']:
            bound = 1 / torch.sqrt(torch.tensor(self.in_features).float())
            nn.init.uniform_(self.weights, -bound, bound)
            nn.init.uniform_(self.feedback_weights, -bound, bound)
        elif self.activation_type in ['identity', None]:
            # For linear layers, Xavier uniform initialization is appropriate
            bound = 1 / torch.sqrt(torch.tensor(self.in_features).float())
            nn.init.uniform_(self.weights, -bound, bound)
            nn.init.uniform_(self.feedback_weights, -bound, bound)
        else:
            if self.activation_type == 'elu':
                gain = nn.init.calculate_gain('relu')  # Use ReLU gain cause ELU doesn't have a gain
            else:
                gain = nn.init.calculate_gain(self.activation_type)

            nn.init.kaiming_uniform_(self.weights, a=gain)
            nn.init.kaiming_uniform_(self.feedback_weights, a=gain)

        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if self.feedback_bias is not None:
            nn.init.zeros_(self.feedback_bias)

    def forward_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward activation function."""
        if self.activation_type == 'relu':
            return F.relu(x)
        elif self.activation_type == 'tanh':
            return torch.tanh(x)
        elif self.activation_type == 'elu':
            return F.elu(x)
        elif self.activation_type == 'sigmoid':
            return torch.sigmoid(x)
        else:
            return x

    def compute_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian of the activation function."""
        if self.activation_type == 'relu':
            return (x > 0).float()
        elif self.activation_type == 'tanh':
            return 1 - torch.tanh(x) ** 2
        elif self.activation_type == 'elu':
            jac = torch.ones_like(x)
            mask = x <= 0
            jac[mask] = F.elu(x[mask]) + 1
            return jac
        elif self.activation_type == 'sigmoid':
            sig = torch.sigmoid(x)
            return sig * (1 - sig)
        else:
            return torch.ones_like(x)

    def compute_forward_gradients(self, h_target: torch.Tensor) -> None:
        """Compute gradients for the forward parameters."""
        # Get Jacobian approximation for activation function
        jacobian = self.compute_jacobian(self.pre_activation)

        # Compute teaching signal
        teaching_signal = 2 * jacobian * (self.post_activation - h_target)

        # Compute gradients
        if self.bias is not None:
            bias_grad = teaching_signal.mean(0)
            self.bias.grad = bias_grad.detach()

        # Compute weight gradients maintaining batch dimension
        weights_grad = torch.bmm(
            teaching_signal.unsqueeze(2),
            self.input.unsqueeze(1)
        ).mean(0)
        self.weights.grad = weights_grad.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer."""
        self.input = x.detach()  # Cache input
        # Linear transformation (batch_size x in_features) @ (in_features x out_features)^T
        self.pre_activation = F.linear(x, self.weights, self.bias)
        self.post_activation = self.forward_activation(self.pre_activation)
        return self.post_activation

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        """Backward pass through the layer."""
        return F.linear(x, self.feedback_weights, self.feedback_bias)

    def compute_target(self,
                       current_activation: torch.Tensor,
                       next_layer_target: torch.Tensor,
                       next_layer_activation: torch.Tensor) -> torch.Tensor:
        """Compute layer target using difference target propagation."""
        current_activation = current_activation.detach()
        next_layer_target = next_layer_target.detach()
        next_layer_activation = next_layer_activation.detach()

        target = self.backward(next_layer_target) - (self.backward(next_layer_activation) - current_activation)
        self.target = target.detach()  # Cache target
        return target

    def compute_local_loss(self, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute local loss for the layer."""
        if self.post_activation is None:
            raise ValueError("Forward pass must be performed before computing loss")

        # Use provided target or cached target
        if target is not None:
            loss_target = target
        elif self.target is not None:
            loss_target = self.target
        else:
            raise ValueError("No target provided or cached")

        return 0.5 * ((self.post_activation - loss_target) ** 2).sum(1).mean()

    def train_feedback_weights(self) -> float:
        """Train feedback weights using difference reconstruction loss."""
        if self.input is None or self.post_activation is None:
            raise ValueError("Forward pass must be performed before training feedback weights")

        final_loss = 0.0
        for _ in range(self.config.K_iterations):
            # Generate noise samples
            noise_input = torch.randn_like(self.input) * self.config.noise_scale
            noise_output = torch.randn_like(self.post_activation) * self.config.noise_scale

            # Forward pass with noise
            noisy_input = self.input + noise_input
            noisy_forward = self(noisy_input)

            # Get baseline reconstruction
            baseline_reconstruction = self.backward(self.post_activation)

            # Reconstruction through forward-feedback loop
            reconstruction_noise = self.backward(noisy_forward)

            # Output perturbation reconstruction
            perturbed_output = self.post_activation + noise_output
            reconstruction_output = self.backward(perturbed_output)

            # First term: input perturbation loss
            first_term = -2 * (noise_input * (reconstruction_noise - baseline_reconstruction)).sum(1).mean()

            # Second term: output perturbation loss
            second_term = torch.square(reconstruction_output - baseline_reconstruction).sum(1).mean()

            # Total reconstruction loss
            reconstruction_loss = first_term + second_term

            # Compute gradients
            grads = torch.autograd.grad(reconstruction_loss,
                                        [self.feedback_weights] + (
                                            [self.feedback_bias] if self.feedback_bias is not None else []),
                                        retain_graph=False)

            # Update feedback weights
            with torch.no_grad():
                self.feedback_weights -= self.config.learning_rate * grads[0]
                if self.feedback_bias is not None:
                    self.feedback_bias -= self.config.learning_rate * grads[1]

            final_loss = reconstruction_loss.item()

        return final_loss


class DTPNetwork(nn.Module):
    """Neural network using enhanced DTP layers."""

    def __init__(self,
                 layer_sizes: List[int],
                 activation: str = 'tanh',
                 config: DTPConfig = DTPConfig(),
                 bias: bool = True,
                 final_activation: Optional[str] = None):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            layer = DTPLayer(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                activation=activation if i < len(layer_sizes) - 2 else final_activation,
                config=config,
                bias=bias
            )
            self.layers.append(layer)
            

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass returning both output and intermediate activations."""
        activations = [x]
        current = x

        for layer in self.layers:
            current = layer(current)
            activations.append(current)

        return current, activations

    def compute_targets(self,
                        activations: List[torch.Tensor],
                        final_target: torch.Tensor) -> List[torch.Tensor]:
        """Compute layer-wise targets using DTP."""
        targets = [None] * len(activations)
        targets[-1] = final_target

        for i in reversed(range(len(self.layers))):
            if i == 0:  # No target needed for input layer
                continue

            targets[i] = self.layers[i].compute_target(
                activations[i],
                targets[i + 1],
                activations[i + 1]
            )

        return targets

    def train_step(self, input: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a single training step."""
        # Forward pass
        output, activations = self(input)

        # Compute initial loss and gradient
        mse_loss = 0.5 * ((output - target) ** 2).sum(dim=1).mean()
        grad = torch.autograd.grad(mse_loss,
                                   output,
                                   only_inputs=True,
                                   create_graph=False)[0]

        # Compute output target
        output_target = output - self.layers[-1].config.beta * grad

        # Train feedback weights first
        feedback_loss = 0.0
        for layer in self.layers[:-1]:  # Skip last layer
            feedback_loss += layer.train_feedback_weights()

        # Compute targets for all layers
        targets = self.compute_targets(activations, output_target)

        # Compute forward loss using the computed targets
        forward_loss = torch.tensor(0.0, device=input.device)
        for i, layer in enumerate(self.layers[:-1]):
            forward_loss += layer.compute_local_loss(targets[i + 1])

        return forward_loss, torch.tensor(feedback_loss)