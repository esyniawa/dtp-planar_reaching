import matplotlib.pyplot as plt
import os
import numpy as np



#######################################################################################################
# Visualize Validation Error
#######################################################################################################
def visualize_validation_error(
    network_type: str, 
    errors: list, 
    current_date: str):

    plt.figure(figsize=(10, 6))
    plt.plot(errors, label='Validation Error', color='blue')
    plt.xlabel('Test Samples')
    plt.ylabel('Error')
    plt.title(f'{network_type} - Validation Error over Test Samples')
    plt.axhline(y=np.mean(errors), color='green', linestyle='--', label='Mean Error')
    plt.legend()
    plt.grid()

    # Save the figure
    save_dir = os.path.join(current_date, "figures")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{network_type}_error_validation.png")
    plt.savefig(save_path)
    plt.close()