import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple,Optional
from tqdm import tqdm

# script imports
from network.dtp_networks import DDTPNetwork, DDTPRHLNetwork
from network.dnn_networks import DNN
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
    parser.add_argument('--dnn_lr', type=float, default=0.001)
    parser.add_argument('--dnn_criterion', type=nn.Module, default=nn.MSELoss())
    parser.add_argument('--feedback_training_iterations', type=int, default=3)
    parser.add_argument('--plot_history', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    return args, parser

###################################################################################
# Dense Neural Network
###################################################################################
def create_dnn(
        layer_dims: List[int] | Tuple[int],
        activation: str = "elu",
        output_activation: Optional[str] = None,
) -> DNN:
    """Create a Dense Neural Network with the specified architecture."""
    activation_map = {
        "elu": nn.ELU(),
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "linear": None, 
        "none": None
    }

    # Set feedforward activation to a given one -> in case the given one is not registered set ELU
    activation = activation_map.get(activation, nn.ELU())
    # Set output activation to a given one -> in case the given one is not registered set a linear activation
    output_act = activation_map.get(output_activation, None)
    
    return DNN(
            layer_sizes=layer_dims,
            activation=activation,
            output_activation=output_act
        )
    
def train_dnn_epoch(
        network: DNN,
        buffer: MovementBuffer,
        num_batches: int,
        batch_size: int,
        criterion: nn.Module = nn.MSELoss(),    # use MSE loss for regression tasks
        lr: float = 0.001,
        device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """Train a standard DNN for one epoch using a movement buffer."""
    
    # Create optimizer
    params = []
    for layer in network.layers:
        params.extend(list(layer.parameters()))
        
    optimizer = optim.Adam(params, lr=lr)  
    total_loss = 0.0
    
    
    for _ in range(num_batches):
        # Generate a batch
        inputs, targets, _ = buffer.get_batches(batch_size=batch_size)
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = network(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return {
        'loss': total_loss / num_batches
        }


###################################################################################
# DDTP Network
###################################################################################
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
        "linear": None, 
        "none": None
    }

    # Set feedforward activation to a given one -> in case the given one is not registered set ELU
    ff_activation = activation_map.get(ff_activation, nn.ELU())
    # Set feedback activation to a given one -> in case the given one is not registered set ELU
    fb_activation = activation_map.get(fb_activation, nn.ELU())
    # Set output activation to a given one -> in case the given one is not registered set a linear activation
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


def ddtp_train_epoch(
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

        # ===== Phase 1: Step forward and Train fb weights with DRL =====
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


def train_networks(
        ddtp: DDTPNetwork | DDTPRHLNetwork,
        dnn: DNN,
        num_epochs: int,
        num_batches: int,
        batch_size: int,
        trainings_buffer_size: int,
        arm: str,
        device: torch.device,
        ddtp_lr_forward: float = 0.001,
        ddtp_lr_feedback: float = 0.003,
        ddtp_feedback_weight_decay: float = 1e-5,
        target_stepsize: float = 0.1,
        sigma: float = 0.01,
        feedback_training_iterations: int = 3,
        dnn_lr: float = 0.001,
        dnn_criterion: nn.Module = nn.MSELoss(),
        validation_interval: int = 10
) -> Dict[str, List[float]]:
    """Train the network for multiple epochs with validation."""

    # Initialize dataset
    trainings_buffer = MovementBuffer(
        arm=arm,
        buffer_size=trainings_buffer_size,
        device=device
    )

    # Initialize history of ddtp and dnn
    ddtp_history = {
        'forward_loss': [],
        'feedback_loss': [],
        'ddtp_validation_error': []
    }
    
    dnn_history = {
        'dnn_loss': [],
        'dnn_validation_error': []
    }

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Fill buffer with movements
        trainings_buffer.fill_buffer()

        ################################################################################
        # Train for one epoch
        ################################################################################
        
        # DDTP Training
        ddtp_epoch_losses = ddtp_train_epoch(
            network=ddtp,
            buffer=trainings_buffer,
            num_batches=num_batches,
            batch_size=batch_size,
            lr_forward=ddtp_lr_forward,
            lr_feedback=ddtp_lr_feedback,
            feedback_weight_decay=ddtp_feedback_weight_decay,
            target_stepsize=target_stepsize,
            sigma=sigma,
            feedback_training_iterations=feedback_training_iterations,
            device=device
        )

        ddtp_history['forward_loss'].append(ddtp_epoch_losses['forward_loss'])
        ddtp_history['feedback_loss'].append(ddtp_epoch_losses['feedback_loss'])
        
        # DNN Training
        dnn_epoch_loss = train_dnn_epoch(
            network=dnn,
            buffer=trainings_buffer,
            num_batches=num_batches,
            batch_size=batch_size,
            criterion=dnn_criterion,   
            lr=dnn_lr,
            device=device
        )
        
        dnn_history['dnn_loss'].append(dnn_epoch_loss['loss'])
        
        # Clear buffer
        trainings_buffer.clear_buffer()

        ############################################################################
        # Run validation periodically
        ############################################################################
        
        # DDTP Evaluation
        if (epoch + 1) % validation_interval == 0:
            ddtp_val_error = evaluate_reaching(
                network=ddtp,
                num_tests=50,
                arm=arm,
                device=device
            )
            
            ddtp_history['ddtp_validation_error'].append(ddtp_val_error)
            tqdm.write("DDTP Results.......")
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}")
            tqdm.write(f"Forward Loss: {ddtp_epoch_losses['forward_loss']:.4f}")
            tqdm.write(f"Feedback Loss: {ddtp_epoch_losses['feedback_loss']:.4f}")
            tqdm.write(f"DDTP Validation Error: {ddtp_val_error:.2f} mm\n")
        
        # DNN Evaluation
        if (epoch + 1) % validation_interval == 0:
            dnn_val_error = evaluate_reaching(
                network=dnn,
                num_tests=50,
                arm=arm,
                device=device
            )
            
            dnn_history['dnn_validation_error'].append(dnn_val_error)
            tqdm.write("DNN Results.......")
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}")
            tqdm.write(f"Loss: {dnn_epoch_loss['loss']:.4f}")
            tqdm.write(f"DNN Validation Error: {dnn_val_error:.2f} mm\n")

    return ddtp_history, dnn_history


def evaluate_reaching(
        network: DDTPNetwork | DDTPRHLNetwork | DNN,
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
    
    
    ################################################################################################################
    # Initalizations
    ################################################################################################################
    args, _ = argument_parser()

    # Initialize device
    device = torch.device(args.device)
    
    # Initialize seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Take CPU for computations in case cuda is not available
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    print(f'Pytorch version: {torch.__version__} running on {device}')
    
    
    #################################################################################################################
    # Network definitions
    #################################################################################################################

    # Create Direct DTP network
    ddtp_layer_sizes = [4, 128, 128, 2]
    ddtp_network = create_ddtp_network(
        layer_dims=ddtp_layer_sizes,
        ff_activation='elu',
        fb_activation='elu',
        final_activation=None,
        rhl_size=args.rhl_size
    )
    ddtp_network = ddtp_network.to(device)
    
    # Create Dense Neural Network
    dnn_layer_sizes = [4, 128, 128, 2]
    dnn = create_dnn(
        layer_dims=dnn_layer_sizes,
        activation="elu",
        output_activation=None,
    )

    ##################################################################################################################
    # Training
    ##################################################################################################################
    ddtp_history, dnn_history = train_networks(
        ddtp=ddtp_network,
        dnn=dnn,
        num_epochs=args.num_epochs,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        trainings_buffer_size=args.trainings_buffer_size,
        arm=args.arm,
        device=device,
        validation_interval=args.validation_interval,
        ddtp_lr_forward=args.lr_forward,
        ddtp_lr_feedback=args.lr_feedback,
        ddtp_feedback_weight_decay=args.feedback_weight_decay,
        target_stepsize=args.target_stepsize,
        sigma=args.sigma,
        dnn_lr=args.dnn_lr,
        dnn_criterion=args.dnn_criterion,
        feedback_training_iterations=args.feedback_training_iterations
    )
    
    ###################################################################################################################
    # Evaluation
    ###################################################################################################################

    # Final DDTP evaluation
    final_ddtp_error = evaluate_reaching(
        network=ddtp_network,
        num_tests=1_000,
        arm=args.arm,
        device=device
    )
    print(f"Final DDTP reaching error: {final_ddtp_error:.2f} mm")
    
    # Final DNN evaluation
    final_dnn_error = evaluate_reaching(
        network=dnn,
        num_tests=1_000,
        arm=args.arm,
        device=device
    )
    print(f"Final DNN reaching error: {final_dnn_error:.2f} mm")

    # Plot history
    if args.plot_history:
        import os
        import matplotlib.pyplot as plt
        import datetime

        current_date = datetime.datetime.now().strftime('%Y_%m_%d')
        plot_folder = os.path.join(current_date,"figures")
        os.makedirs(plot_folder, exist_ok=True)

        ##############################################################################################
        # DDTP
        ##############################################################################################
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
        axs[0].plot(ddtp_history['forward_loss'], label='Forward Loss')
        axs[1].plot(ddtp_history['feedback_loss'], label='Feedback Loss')
        axs[2].plot(ddtp_history['ddtp_validation_error'], label='Validation Error (mm)')
        for ax in axs:
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(True)
        axs[0].set_ylabel('Forward Loss')
        axs[1].set_ylabel('Feedback Loss')
        axs[2].set_ylabel('Error (mm)')
        plt.tight_layout()
        plt.savefig(os.path.joint(plot_folder, f"ddtp_history_{args.arm}.png"))
        plt.close(fig)

        # Save model
        model_folder = os.path.join(current_date,"models")
        os.makedirs(model_folder, exist_ok=True)
        torch.save(ddtp_network.state_dict(), os.path.join(model_folder, f"ddtp_network_{args.arm}.pt"))

    
    
        ###############################################################################################
        # DNN 
        ###############################################################################################
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
        axs[0].plot(dnn_history['dnn_loss'], label='Loss')
        axs[1].plot(dnn_history['dnn_validation_error'], label='Validation Error (mm)')
        for ax in axs:
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(True)
        axs[0].set_ylabel('Loss')
        axs[1].set_ylabel('Error (mm)')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, f"dnn_history_{args.arm}.png"))
        plt.close(fig)

        # Save model
        torch.save(ddtp_network.state_dict(), os.path.join(model_folder, f"dnn_network_{args.arm}.pt"))