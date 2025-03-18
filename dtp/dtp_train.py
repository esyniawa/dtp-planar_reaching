import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from dtp import DTPNetwork, DTPConfig
from environment import MovementBuffer, inverse_target_transform, create_batch
from kinematics.planar_arms import PlanarArms



def dtp_train_epoch(
        network: DTPNetwork,
        optimizer: torch.optim.Optimizer,
        buffer: MovementBuffer,
        num_batches: int,
        batch_size: int,
) -> Dict[str, float]:
    """Train the network for one epoch using the movement buffer."""
    network.train()
    total_forward_loss = 0.0
    total_feedback_loss = 0.0

    for _ in range(num_batches):
        # Generate a batch
        inputs, targets, _ = buffer.get_batches(batch_size=batch_size)

        # Train step
        optimizer.zero_grad()
        forward_loss, feedback_loss = network.train_step(inputs, targets)

        # Update weights
        forward_loss.backward()
        optimizer.step()

        total_forward_loss += forward_loss.item()
        total_feedback_loss += feedback_loss.item()

    return {
        'forward_loss': total_forward_loss / num_batches,
        'feedback_loss': total_feedback_loss / num_batches
    }


def dtp_train_network(
        network: DTPNetwork,
        num_epochs: int,
        num_batches: int,
        batch_size: int,
        trainings_buffer_size: int,
        arm: str,
        device: torch.device,
        learning_rate: float = 1e-4,
        validation_interval: int = 10
) -> Dict[str, List[float]]:
    """Train the network for multiple epochs with validation."""
    # Initialize optimizer
    optimizer = torch.optim.SGD(
        network.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-5
    )

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
        epoch_losses = dtp_train_epoch(
            network=network,
            optimizer=optimizer,
            buffer=trainings_buffer,
            num_batches=num_batches,
            batch_size=batch_size,
        )

        history['forward_loss'].append(epoch_losses['forward_loss'])
        history['feedback_loss'].append(epoch_losses['feedback_loss'])

        # Clear buffer
        trainings_buffer.clear_buffer()

        # Run validation periodically
        if (epoch + 1) % validation_interval == 0:
            val_error = dtp_evaluate_reaching(
                network=network,
                num_tests=50,
                arm=arm,
                device=device
            )
            history['validation_error'].append(val_error)
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}")
            tqdm.write(f"Forward Loss: {epoch_losses['forward_loss']:.4f}")
            tqdm.write(f"Feedback Loss: {epoch_losses['feedback_loss']:.4f}")
            tqdm.write(f"Validation Error: {val_error:.2f} mm\n")

    return history


def dtp_evaluate_reaching(
        network: DTPNetwork,
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

            # Get network prediction
            outputs, _ = network.forward(inputs)

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