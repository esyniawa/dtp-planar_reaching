import torch
import torch.nn as nn


class FeedforwardLayer(nn.Module):
    """
    Feedforward layer with activation, storing activations for local learning.

    This class implements a single layer in the feedforward path of the network.
    It stores both input and output activations for use in local loss computation.
    """

    def __init__(self, in_features, out_features, activation=nn.Tanh()):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

        # Forward pass
        self.input = None
        self.output = None

        # Initialize weights according to the paper recommendations
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        """Forward pass through the layer, storing activations"""
        self.input = x.detach()  # Store input (detached to avoid gradient tracking)
        pre_activation = self.linear(x)
        output = self.activation(pre_activation) if self.activation else pre_activation
        self.output = output.detach()  # Store output (detached to avoid gradient tracking)
        return output


class FeedbackLayer(nn.Module):
    """
    Direct feedback connection from output layer to a hidden layer with activation function.

    This implements the direct linear connections in DDTP-linear, where each hidden layer receives direct feedback
    from the output layer.
    """

    def __init__(self, output_size: int, target_size: int, activation: nn.Module = nn.Tanh()):
        super().__init__()
        # Direct linear connection from output to hidden layer (DDTP-linear)
        self.feedback = nn.Linear(output_size, target_size, bias=True)
        self.activation = activation

        # Initialize weights according to the paper recommendations
        nn.init.xavier_normal_(self.feedback.weight)
        nn.init.zeros_(self.feedback.bias)

    def forward(self, output: torch.Tensor):
        """
        Forward pass of feedback connections
        :arg: The output layer activations
        :return:Feedback signal for the corresponding hidden layer
        """
        pre_activation = self.feedback(output)
        return self.activation(pre_activation) if self.activation else pre_activation

    def compute_target(self, output: torch.Tensor, output_target: torch.Tensor, hidden_true: torch.Tensor):
        """
        Compute difference target propagation target

        This implements the equation:
            h_hat_i = g_i(h_hat_L) + h_i - g_i(h_L)

            g_i(h_hat_L) is the feedback mapping from the output target
            h_i is the actual hidden layer activation
            g_i(h_L) is the feedback mapping from the actual output

        :arg output: Actual output layer activations (h_L)
        :arg output_target: Target output activations (h_hat_L)
        :arg hidden_true: Actual hidden layer activations (h_i)
        :return: The target for the hidden layer
        """
        # g_i(h_hat_L): Feedback from output target
        target_from_output = self.forward(output_target)

        # g_i(h_L): Feedback from actual output
        reconstruction = self.forward(output)

        # Add difference correction (h_i - g_i(h_L))
        target = target_from_output + (hidden_true - reconstruction)

        return target
