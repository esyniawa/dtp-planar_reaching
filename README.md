# Planar Reaching with Difference Target Propagation and Dense Neural Networks

This repository implements a learning framework for planar reaching movements using Difference Target Propagation (DTP) and Dense Neural Networks (DNN). The implementation includes forward kinematics for a 2-joint planar arm, a DTP network and a DNN for learning reaching movements. Furthermore, it indicates analysis methods such as dimensionality reduction and sensitivity analysis.

## Overview

The project consists of three main components:

1. **Planar Arm Kinematics**: A 2-joint robotic arm model with:
   - Forward kinematics for end-effector positioning
   - Inverse kinematics for joint angle calculation
   - Workspace constraints and joint limits

2. **Difference Target Propagation Network**: A neural network that learns to generate reaching movements with:
   - Direct DTP
   - Feedback weight learning through local target computation
   - Feedforward weight through DRL

3. **Dense Neural Network**: A neural network that learns to generate reaching movements with
   - Forward Propagation
   - Backpropagation
   - Mean Squared Error (MSE) loss and Adam optimizer

## Project Structure

```
esyniawa-dtp-planar_reaching/
├── LICENSE               # MIT License
├── environment.py        # Environment setup and data handling
├── main.py               # Training script and network creation (example)
├── analysis.py           # Scripts for analysis methods: dimReduction, plotActivations and sensitivity_analysis
├── kinematics/          
│   ├── planar_arms.py    # Planar arm implementation
│   └── utils.py          # Kinematics utilities
├── network/
│   ├── dnn_networks.py           # DNN implementation
│   ├── dtp_networks.py           # DDTP networks implementation
│   └── layers.py                 # Feedforward and feedback layer definitions
├── visualizations/
│   └── validation_error.py            # Script for visualization error plots
└── models/                                  # Folder for saving the trained models
```

## Implementation Details

### Planar Arm
- Two-joint arm with shoulder and elbow
- Configurable link lengths and joint limits
- Support for both left and right arm configurations

### DTP Network
- Customizable layer architecture
- Forward and feedback weight optimization
- Local target computation for each layer
- MSE-based reconstruction loss
- Built-in support for batch processing

### DNN
- Customizable layer architecture
- Forward weight optimization
- MSE-Loss
- Adam optimizer
- Built-in support for batch processing

## Dependencies

- PyTorch
- Matplotlib
- tqdm
- pandas
- umap
- sklearn

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Based on research in target propagation and biological learning mechanisms, particularly:
- Lee et al. (2015) ["Difference Target Propagation"](https://link.springer.com/chapter/10.1007/978-3-319-23528-8_31)
- Meulemans et al. (2020) ["A Theoretical Framework for Target Propagation"](https://proceedings.neurips.cc/paper_files/paper/2020/hash/e7a425c6ece20cbc9056f98699b53c6f-Abstract.html)
