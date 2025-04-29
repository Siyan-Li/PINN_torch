# Physics-Informed Neural Networks (PINNs) for Heat Equation Parameter Identification

This repository contains two Python programs demonstrating the use of Physics-Informed Neural Networks (PINNs) to identify an unknown parameter in the heat equation. The goal is to verify whether PINNs can accurately recover the thermal conductivity coefficient α from known solutions, initial conditions, and boundary conditions.

## Project Structure

1. **`generate_heat_equation_solution.py`**  
   Solves the 1D heat equation with a known α (for example 0.2 or 0.8) using numerical methods, generating:
   - Initial conditions
   - Boundary conditions
   - A reference solution (ground truth)

2. **`pinn_heat_equation.py`**  
   Uses PINNs to estimate the thermal conductivity coefficient α by incorporating:
   - Physics constraints (the heat equation)
   - Initial and boundary conditions
   - Sampled points from the reference solution

## Key Features

- **Two-Phase Optimization**: Separate optimizers and learning rate schedulers for:
  1. Neural network parameters
  2. The thermal conductivity parameter α

- **Hyperparameter Tuning**:
  - Adjusted physics loss weight to balance data fidelity and physical constraints
  - Custom learning rate strategies for α and network parameters
  - Parameter constraints to ensure physical plausibility

- **Validation Cases**:
  - Successfully tested with α = 0.2 and α = 0.8
  - Achieved relatively high accuracy in parameter recovery (< 10％)
