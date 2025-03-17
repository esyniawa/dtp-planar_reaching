import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple,Optional
from tqdm import tqdm

# script imports
from network.dtp_networks import DDTPNetwork, DDTPRHLNetwork
from environment import MovementBuffer, inverse_target_transform, create_batch
from kinematics.planar_arms import PlanarArms


def argument_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', type=str, default="right", choices=["right", "left"])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_batches', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--rhl_size', type=int, default=None)  # None for DDTP: linear
    parser.add_argument('--trainings_buffer_size', type=int, default=5_000)
    parser.add_argument('--validation_interval', type=int, default=25)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--lr_forward', type=float, default=1e-4)
    parser.add_argument('--lr_feedback', type=float, default=5e-5)
    parser.add_argument('--feedback_weight_decay', type=float, default=1e-6)
    parser.add_argument('--target_stepsize', type=float, default=0.9)  # former beta param
    parser.add_argument('--sigma', type=float, default=0.1)  # maybe try 0.05
    parser.add_argument('--feedback_training_iterations', type=int, default=3)
    parser.add_argument('--plot_history', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    return args, parser


def create_ddtp_network(
        layer_dims: List[int] | Tuple[int],
        ff_activation: str = "elu",
        fb_activation: str = "elu",
        final_activation: Optional[str] = None,
        rhl_size: Optional[int] = None,
) -> DDTPNetwork | DDTPRHLNetwork:
    """Create a DDTP network with the specified architecture."""
    activation_map = {
        "elu": nn.ELU(),
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "none": None
    }

    ff_activation = activation_map.get(ff_activation, nn.ELU())
    fb_activation = activation_map.get(fb_activation, nn.ELU())
    output_act = activation_map.get(final_activation, None)

    if rhl_size is not None:
        return DDTPRHLNetwork(
            layer_sizes=layer_dims,
            ff_activation=ff_activation,
            fb_activation=fb_activation,
            output_activation=output_act,
            random_hidden_size=rhl_size
        )
    else:
        return DDTPNetwork(
            layer_sizes=layer_dims,
            ff_activation=ff_activation,
            fb_activation=fb_activation,
            output_activation=output_act
        )


def train_epoch(
        network: DDTPNetwork | DDTPRHLNetwork,
        buffer: MovementBuffer,
        num_batches: int,
        batch_size: int,
        lr_forward: float = 0.001,
        lr_feedback: float = 0.003,
        feedback_weight_decay: float = 1e-5,
        target_stepsize: float = 0.9,  # beta
        sigma: float = 0.1,  # noise std
        feedback_training_iterations: int = 3,
        device: torch.device = torch.device('cpu')
) -> Dict[str, float]:

    """Train the network for one epoch using the movement buffer."""
    # Create separate optimizers for forward and feedback weights
    forward_params = []
    for layer in network.forward_layers:
        forward_params.extend(list(layer.parameters()))

    feedback_params = []
    for layer in network.feedback_layers:
        feedback_params.extend(list(layer.parameters()))

    forward_optimizer = optim.Adam(forward_params, lr=lr_forward)
    feedback_optimizer = optim.Adam(feedback_params, lr=lr_feedback, weight_decay=feedback_weight_decay)
    mse_loss = nn.MSELoss()  # use MSE loss for regression tasks

    total_forward_loss = 0.0
    total_feedback_loss = 0.0

    for _ in range(num_batches):
        # Generate a batch
        inputs, targets, _ = buffer.get_batches(batch_size=batch_size)
        inputs, targets = inputs.to(device), targets.to(device)

        # ===== Phase 1: Train fb weights with DRL =====
        feedback_loss = 0.0
        for _ in range(feedback_training_iterations):
            # Forward pass (needed to compute hidden layer activations)
            output = network.forward(inputs)

            feedback_optimizer.zero_grad()

            # Compute DRL loss for each feedback layer
            fb_loss = 0
            for i in range(len(network.feedback_layers)):
                fb_loss += network.drl_loss(i, sigma)

            # Update feedback weights
            fb_loss.backward()
            feedback_optimizer.step()
            feedback_loss += fb_loss.item() / num_batches

        feedback_loss /= feedback_training_iterations

        # ===== Phase 2: Train ff weights with local losses =====
        output = network(inputs)  # (recomputed after feedback weight updates)

        # compute output loss
        output_loss = mse_loss(output, targets)

        # compute output targets
        output_target = output - target_stepsize * (output - targets)
        forward_optimizer.zero_grad()

        # Compute targets for each hidden layer
        hidden_targets = network.compute_targets(output, output_target)

        # compute local losses for each hidden layer
        local_losses = network.local_loss(hidden_targets)
        total_local_loss = sum(local_losses) / num_batches

        # Update forward weights based on local losses
        total_local_loss.backward()
        forward_optimizer.step()

        total_forward_loss += output_loss.item()
        total_feedback_loss += feedback_loss

    return {
        'forward_loss': total_forward_loss,
        'feedback_loss': total_feedback_loss
    }


def train_network(
        network: DDTPNetwork | DDTPRHLNetwork,
        num_epochs: int,
        num_batches: int,
        batch_size: int,
        trainings_buffer_size: int,
        arm: str,
        device: torch.device,
        lr_forward: float = 0.001,
        lr_feedback: float = 0.003,
        feedback_weight_decay: float = 1e-5,
        target_stepsize: float = 0.1,
        sigma: float = 0.01,
        feedback_training_iterations: int = 3,
        validation_interval: int = 10
) -> Dict[str, List[float]]:
    """Train the network for multiple epochs with validation."""

    # Initialize dataset
    trainings_buffer = MovementBuffer(
        arm=arm,
        buffer_size=trainings_buffer_size,
        device=device
    )

    # Initialize history
    history = {
        'forward_loss': [],
        'feedback_loss': [],
        'validation_error': []
    }

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Fill buffer with movements
        trainings_buffer.fill_buffer()

        # Train for one epoch
        epoch_losses = train_epoch(
            network=network,
            buffer=trainings_buffer,
            num_batches=num_batches,
            batch_size=batch_size,
            lr_forward=lr_forward,
            lr_feedback=lr_feedback,
            feedback_weight_decay=feedback_weight_decay,
            target_stepsize=target_stepsize,
            sigma=sigma,
            feedback_training_iterations=feedback_training_iterations,
            device=device
        )

        history['forward_loss'].append(epoch_losses['forward_loss'])
        history['feedback_loss'].append(epoch_losses['feedback_loss'])

        # Clear buffer
        trainings_buffer.clear_buffer()

        # Run validation periodically
        if (epoch + 1) % validation_interval == 0:
            val_error = evaluate_reaching(
                network=network,
                num_tests=50,
                arm=arm,
                device=device
            )
            history['validation_error'].append(val_error)
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}")
            tqdm.write(f"Forward Loss: {epoch_losses['forward_loss']:.4f}")
            tqdm.write(f"Feedback Loss: {epoch_losses['feedback_loss']:.4f}")
            tqdm.write(f"Validation Error: {val_error:.2f}mm\n")

    return history


def evaluate_reaching(
        network: DDTPNetwork | DDTPRHLNetwork,
        num_tests: int,
        arm: str,
        device: torch.device
) -> float:
    """Evaluate the network's reaching accuracy."""
    network.eval()
    total_error = 0.0

    with torch.no_grad():
        for _ in range(num_tests):
            # Generate a single test movement
            inputs, targets, initial_thetas = create_batch(
                arm=arm,
                device=device
            )
            inputs, targets = inputs.to(device), targets.to(device)

            # Get network prediction
            outputs = network(inputs)

            # Convert network outputs and targets back to radians
            target_delta_thetas = inverse_target_transform(targets.cpu().numpy())
            pred_delta_thetas = inverse_target_transform(outputs.cpu().numpy())

            # Calculate reaching error
            target_xy = PlanarArms.forward_kinematics(
                arm=arm,
                thetas=PlanarArms.clip_values(initial_thetas + target_delta_thetas[0], radians=True),
                radians=True,
                check_limits=False
            )[:, -1]
            pred_xy = PlanarArms.forward_kinematics(
                arm=arm,
                thetas=PlanarArms.clip_values(initial_thetas + pred_delta_thetas[0], radians=True),
                radians=True,
                check_limits=False
            )[:, -1]

            error = np.linalg.norm(target_xy - pred_xy)
            total_error += error

    return total_error / num_tests


if __name__ == "__main__":
    args, _ = argument_parser()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    print(f'Pytorch version: {torch.__version__} running on {device}')

    # Create network
    layer_sizes = [4, 128, 128, 2]
    network = create_ddtp_network(
        layer_dims=layer_sizes,
        ff_activation='elu',
        final_activation=None,
        rhl_size=args.rhl_size
    )
    network = network.to(device)

    # Training
    history = train_network(
        network=network,
        num_epochs=args.num_epochs,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        trainings_buffer_size=args.trainings_buffer_size,
        arm=args.arm,
        device=device,
        validation_interval=args.validation_interval,
        lr_forward=args.lr_forward,
        lr_feedback=args.lr_feedback,
        feedback_weight_decay=args.feedback_weight_decay,
        target_stepsize=args.target_stepsize,
        sigma=args.sigma,
        feedback_training_iterations=args.feedback_training_iterations
    )

    # Final evaluation
    final_error = evaluate_reaching(
        network=network,
        num_tests=1_000,
        arm=args.arm,
        device=device
    )
    print(f"Final reaching error: {final_error:.2f}mm")

    # Plot history
    if args.plot_history:
        import os
        import matplotlib.pyplot as plt

        plot_folder = "figures/"
        os.makedirs(plot_folder, exist_ok=True)

        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
        axs[0].plot(history['forward_loss'], label='Forward Loss')
        axs[1].plot(history['feedback_loss'], label='Feedback Loss')
        axs[2].plot(history['validation_error'], label='Validation Error (mm)')
        for ax in axs:
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(True)
        axs[0].set_ylabel('Forward Loss')
        axs[1].set_ylabel('Feedback Loss')
        axs[2].set_ylabel('Error (mm)')
        plt.tight_layout()
        plt.savefig(plot_folder + f"ddtp_history_{args.arm}.png")
        plt.close(fig)

        # Save model
        model_folder = "models/"
        os.makedirs(model_folder, exist_ok=True)
        torch.save(network.state_dict(), model_folder + f"ddtp_network_{args.arm}.pt")
