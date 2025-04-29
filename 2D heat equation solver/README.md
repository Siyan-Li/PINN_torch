# 2D Heat Equation Solver with Numerical and Physics-Informed Neural Network (PINN) Methods

This project contains two Python programs for solving the 2D heat equation using finite difference method and Physics-Informed Neural Networks (PINNs), with a particular focus on **constant heat source terms** and **Robin boundary conditions**. The project demonstrates how to generate high-fidelity numerical solutions and use PINNs to infer key physical parameters from known solutions.

## Project Structure

1. **`2d_heat_equation_solver.py`**  
   Solves the 2D heat equation using finite difference method, generating:
   - Numerical solutions of temperature fields considering constant heat sources
   - Accurate implementation of Robin boundary conditions
   - Support for different thermal conductivity coefficients α (e.g., 0.2 and 0.8)
   - Saving temperature field data for subsequent PINN training

2. **`2d_heat_equation_pinn.py`**  
   Uses PINNs to infer the thermal conductivity coefficient α from numerical solutions:
   - Constructs a neural network to approximate the temperature field u(x,y,t)
   - Treats α as a trainable parameter
   - Optimizes both data fitting and physical constraints simultaneously
   - Sets adaptive optimization strategies for network parameters and α respectively

## Mathematical Model and Implementation Details

### 2D Heat Equation

The equation is given by:

$$
\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) + Q
$$

where:
- $u(x,y,t)$ is the temperature field
- $\alpha$ is the thermal conductivity coefficient to be identified
- $Q$ is the constant heat source term

### Robin Boundary Conditions

The boundary conditions are given by:

$$
\frac{\partial u}{\partial n} + h \cdot u = g
$$

where:
- $\frac{\partial u}{\partial n}$ is the derivative of the temperature field along the outward normal direction of the boundary
- $h$ and $g$ are known functions

