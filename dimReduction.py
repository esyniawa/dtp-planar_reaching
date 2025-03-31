import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from umap import UMAP

from network.dtp_networks import DDTPNetwork, DDTPRHLNetwork
from network.dnn_networks import DNN

import os

def dimReduction(network: DDTPNetwork | DDTPRHLNetwork | DNN, 
                 current_date: str,  
                 hidden_activations: list[list], 
                 target_angles: list, 
                 method='PCA'):
    """
    Reduces dimensions of hidden activations and visualizes them with color-coding for shoulder and elbow joint angle changes.
    
    :param network: Network, which dimensionality becomes reduced
    :param current_date: Current daytime
    :param network_name: Used network for the analysis
    :param hidden_activations: Hidden layer activations (batch_size, hidden_dim)
    :param target_angles: Target joint angle changes (batch_size, 2) - [shoulder_change, elbow_change]
    :param method: 'PCA' or 'UMAP' for dimensionality reduction
    """
    
    # Define type of network
    network_type = type(network).__name__
    
    # Choose method
    if method == 'PCA':
        reducer = PCA(n_components=2)
    elif method == 'UMAP':
        reducer = UMAP(n_components=2)
    else:
        raise ValueError("Method must be 'PCA' or 'UMAP'")
    
    
    # Forward propagate through the layers
    for i in range(len(network.layer_sizes)-1):
    
        reduced = reducer.fit_transform(torch.from_numpy(np.array(hidden_activations[i])).squeeze(1))
        
        # Normalize target_angles for coloring
        shoulder_changes = torch.from_numpy(np.array(target_angles)).squeeze(1)[:,0] # Target change for shoulder
        elbow_changes = torch.from_numpy(np.array(target_angles)).squeeze(1)[:,1]   # Target change for elbow
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # 1 row, 2 columns for the subplots

        # Shoulder Changes Plot
        scatter1 = axes[0].scatter(reduced[:, 0], reduced[:, 1], c=shoulder_changes, cmap='Blues', alpha=0.6)
        fig.colorbar(scatter1, ax=axes[0], label='Shoulder Angle Change (radians)')
        axes[0].set_xlabel(f'{method} Component 1')
        axes[0].set_ylabel(f'{method} Component 2')
        axes[0].set_title(f'{network_type} - Layer_{i+1} - Shoulder Changes ({method})')
        axes[0].grid(True)
        
        # Elbow Changes Plot
        scatter2 = axes[1].scatter(reduced[:, 0], reduced[:, 1], c=elbow_changes, cmap='YlOrRd', alpha=0.6)
        fig.colorbar(scatter2, ax=axes[1], label='Elbow Angle Change (radians)')
        axes[1].set_xlabel(f'{method} Component 1')
        axes[1].set_ylabel(f'{method} Component 2')
        axes[1].set_title(f'{network_type} - Layer_{i+1} - Elbow Changes ({method})')
        axes[1].grid(True)
        
        # Save the figure
        save_dir = os.path.join(current_date, "figures")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{network_type}_layer{i+1}_{method}.png")
        plt.savefig(save_path)
        plt.close()



def plotActivations(network: DDTPNetwork | DDTPRHLNetwork | DNN, 
                    current_date: str, 
                    hidden_activations: list[list]):
    """
    Plots the average activation per neuron in the hidden layer as a bar graph.
    
    :param network: Network, which dimensionality becomes reduced
    :param current_date: Current daytime
    :param hidden_activations: Hidden layer activations 
    """
    
    # Define type of network
    network_type = type(network).__name__
        
    
    # Forward propagate through the layers
    for i in range(len(network.layer_sizes)-1):
                
        # Calculate the average activation per neuron (across all samples in the batch)
        avg_activations = np.mean(hidden_activations[i], axis=0)  # Shape will be (hidden_dim,)
        
        # Calculate the overall mean activation (single scalar)
        overall_mean = np.mean(hidden_activations[i])  
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(avg_activations.flatten())), avg_activations.flatten(), color='teal', label='Mean per Neuron')

        
        # Add overall mean as a green dotted line
        ax.axhline(y=overall_mean, color='green', linestyle='--', label='Overall Mean')
        
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Average Activation')
        ax.set_title(f'{network_type} - Layer_{i+1} - Average Activations')
        ax.grid(True)

        # Save the figure
        save_dir = os.path.join(current_date, "figures")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{network_type}_layer{i+1}_avg_activations.png")
        plt.savefig(save_path)
        plt.close()



def sensitivity_analysis(network: DDTPNetwork | DDTPRHLNetwork | DNN, 
                         original_output: list, 
                         hidden_activations: list[list], 
                         target_angles: list,
                         current_date: str):
    """
    Perform sensitivity analysis by perturbing hidden activations and measuring the effect on output.
    Generates three plots: Total sensitivity (full width) and separate sensitivity for shoulder & elbow.
    
    :param network: The neural network model
    :param original_output: Output of the original network
    :param hidden_activations: Activations of the hidden layers 
    :param target_angles: Target joint angle changes (batch_size, 2) - [shoulder_change, elbow_change]
    :param current_date: Current date for saving results
    """

    # Define type of network
    network_type = type(network).__name__
    
       
    
    
    target_angles = torch.stack(target_angles).squeeze(1)
    original_output = torch.stack(original_output).squeeze(1)

    # Compute original loss values
    original_loss_total = compute_loss(original_output, target_angles)
    original_loss_shoulder = compute_loss(original_output[:, 0], target_angles[:, 0])
    original_loss_elbow = compute_loss(original_output[:, 1], target_angles[:, 1])

    
    
    # Forward propagate through the layers
    for i in range(len(network.layer_sizes)-1):
        
        importance_scores_total = []
        importance_scores_shoulder = []
        importance_scores_elbow = []

        # Loop through each neuron and perturb it
        for j in range(hidden_activations[i][0].shape[1]):
            perturbed_activations = torch.stack(hidden_activations[i]).squeeze(1).clone()
            perturbed_activations[:, j] = 0  # Set neuron to zero
            
            x = perturbed_activations
            
            # New predictions
            for r in range(len(network.layer_sizes)-i-2): # Must go through forward method that offen depending on which layer it is
                x = network.forward_layers[i+r+1].forward(x) # layers
                
            perturbed_output = x

            # Compute loss differences
            perturbed_loss_total = compute_loss(perturbed_output, target_angles)
            perturbed_loss_shoulder = compute_loss(perturbed_output[:, 0], target_angles[:, 0])
            perturbed_loss_elbow = compute_loss(perturbed_output[:, 1], target_angles[:, 1])

            # Compute importance scores
            importance_scores_total.append(perturbed_loss_total - original_loss_total)
            importance_scores_shoulder.append(perturbed_loss_shoulder - original_loss_shoulder)
            importance_scores_elbow.append(perturbed_loss_elbow - original_loss_elbow)

        #############################################################################################
        # Option 1
        #############################################################################################
        # Create a subplot 
        fig, axes = plt.subplots(3, 1, figsize=(24, 16))

        # Plot total sensitivity (occupying full width)
        axes[0].bar(range(len(importance_scores_total)), torch.stack(importance_scores_total).detach().cpu().numpy(), color='black', alpha=0.7)
        axes[0].set_xlabel("Neuron Index")
        axes[0].set_ylabel("Importance Score")
        axes[0].set_title(f"Total Neuron Sensitivity - {network_type} - Layer {i+1}")
        axes[0].grid(True)
        

        # Plot shoulder sensitivity
        axes[1].bar(range(len(importance_scores_shoulder)), torch.stack(importance_scores_shoulder).detach().cpu().numpy(), color='blue', alpha=0.7)
        axes[1].set_xlabel("Neuron Index")
        axes[1].set_ylabel("Importance Score")
        axes[1].set_title(f"Neuron Sensitivity - Shoulder - {network_type} - Layer {i+1}")
        axes[1].grid(True)

        # Plot elbow sensitivity
        axes[2].bar(range(len(importance_scores_elbow)), torch.stack(importance_scores_elbow).detach().cpu().numpy(), color='red', alpha=0.7)
        axes[2].set_xlabel("Neuron Index")
        axes[2].set_ylabel("Importance Score")
        axes[2].set_title(f"Neuron Sensitivity - Elbow - {network_type} - Layer {i+1}")
        axes[2].grid(True)

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        save_dir = os.path.join(current_date, "figures")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{network_type}_layer{i+1}_sensitivity.png")
        plt.savefig(save_path)
        plt.close()
        
        ##############################################################################################################
        # Option 2
        ##############################################################################################################
        barWidth = 0.3
        fig, ax = plt.subplots(figsize=(12,8))
        
        x = range(len(importance_scores_total))
        
        # Plot grouped bars
        ax.bar([i - barWidth for i in x], torch.stack(importance_scores_total).detach().cpu().numpy(), width=barWidth, color='black', alpha=0.7, label="Total")
        ax.bar(x, torch.stack(importance_scores_shoulder).detach().cpu().numpy(), width=barWidth, color='blue', alpha=0.7, label="Shoulder")
        ax.bar([i + barWidth for i in x], torch.stack(importance_scores_elbow).detach().cpu().numpy(), width=barWidth, color='red', alpha=0.7, label="Elbow")

        # Set labels, title, and grid
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Importance Score")
        ax.set_title(f"Neuron Sensitivity Comparison - {network_type} - Layer {i+1}")
        ax.grid(True)
        ax.legend()

        # Adjust layout and save the figure
        plt.tight_layout()
        save_dir = os.path.join(current_date, "figures")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{network_type}_layer{i+1}_sensitivity_opt2.png")
        plt.savefig(save_path)
        plt.close()
        
        
        #return importance_scores_total, importance_scores_shoulder, importance_scores_elbow


def compute_loss(predicted, target):
    """
    Compute the loss between predicted and target joint angle changes.
    
    :param predicted: Predicted joint angle changes (batch_size)
    :param target: Target joint angle changes (batch_size)
    :return: Loss value (mean squared error)
    """
    return torch.mean((predicted - target) ** 2)

