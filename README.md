# PINN_torch

## Introduction
This repository, "PINN_torch", records simple code examples of using PyTorch to solve fluid - related Physics - Informed Neural Networks (PINN) problems. PINN combines neural networks with physics laws, offering a new way to handle complex fluid - dynamic issues.

### What are Physics - Informed Neural Networks (PINN)?
In scenarios with limited data, most advanced machine - learning methods like deep, convolutional, or recurrent neural networks lack robustness and can't guarantee convergence. Yet, in fluid dynamics and other physical systems, there's a lot of **prior knowledge** that modern machine learning often ignores.

PINNs are a type of neural network that integrates physical laws. They use partial differential equations (PDEs) governing the physical phenomena of the observed data, which capture symmetries and conservation principles.

PINNs use deep neural networks as universal function approximators. By **encoding physical information into the learning algorithm**, PINNs amplify the data's information. This helps the network quickly find the right solution and generalize well with few training examples.

The neural network in a PINN takes input variables (e.g., spatial and temporal coordinates in fluid problems) and outputs solutions. The loss function has two parts:
1. **Data loss**: Measures the difference between predictions and observed data.
2. **Physics loss**: Ensures physical equations are satisfied at collocation points. Minimizing the total loss makes the network learn solutions consistent with data and physics.

### Why PINN?
- **Data efficiency**: Collecting large fluid - related datasets is difficult and costly. PINNs use physical knowledge to reduce data needs and make accurate predictions with limited data.
- **Generalization**: Based on physical principles, PINNs generalize well to unseen fluid data and different scenarios, suitable for various fluid problems.
- **Interpretability**: Compared to black - box neural networks, PINNs are more interpretable as physical equations in the loss function help understand fluid mechanisms.

## Repository Structure
This repository has several Python scripts using PyTorch for fluid - related PINN problems. Each script focuses on a different fluid - dynamic problem with detailed code comments.

### Getting Started
To run the code examples, you need:
- Python 3.9 or higher
- PyTorch
- NumPy
- Matplotlib (for visualization)

Install the required packages with `pip`:
```bash
pip install torch numpy matplotlib
