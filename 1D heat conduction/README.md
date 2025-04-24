# 1D Heat Conduction

The code in this folder is based on the one - dimensional heat conduction equation and implements the control of different boundary conditions, including Dirichlet, Neumann, and Robin boundary conditions.

## One - Dimensional Heat Conduction Equation

The one - dimensional heat conduction equation describes the heat conduction process over time in a one - dimensional space. Its general form is:

$$
\frac{\partial u}{\partial t}=\alpha\frac{\partial^{2} u}{\partial x^{2}}
$$

where $u(x,t)$ is the temperature distribution, $\alpha$ is the thermal diffusivity, $x$ is the spatial coordinate, and $t$ is the time.

## Boundary Conditions

### 1. Dirichlet Boundary Conditions
Dirichlet boundary conditions specify the temperature values at the boundaries. For a one - dimensional heat conduction problem, the temperature values $u(x_1,t) = u_{bc1}$ and $u(x_2,t)=u_{bc2}$ are usually specified at the two boundaries $x = x_1$ and $x = x_2$.

#### Formulas
- At the boundary $x = x_1$: $u(x_1,t)=u_{bc1}$
- At the boundary $x = x_2$: $u(x_2,t)=u_{bc2}$

#### Code Implementation
```python
loss_bc1 = torch.mean((u_bc1 - u_bc1_value) ** 2)
loss_bc2 = torch.mean((u_bc2 - u_bc2_value) ** 2)
```
Here, `u_bc1` and `u_bc2` are the boundary temperatures predicted by the model, and `u_bc1_value` and `u_bc2_value` are the given boundary temperature values. `loss_bc1` and `loss_bc2` are the loss functions for the two boundaries respectively, which are calculated using the mean squared error.

### 2. Neumann Boundary Conditions
Neumann boundary conditions specify the heat flux at the boundaries. The heat flux is related to the temperature gradient. For a one - dimensional problem, at the boundaries $x = x_1$ and $x = x_2$, the heat fluxes are $q_0$ and $q_1$ respectively.

#### Formulas
- At the boundary $x = x_1$: $\frac{\partial u}{\partial x}\big|_{x = x_1}=q_0$
- At the boundary $x = x_2$: $\frac{\partial u}{\partial x}\big|_{x = x_2}=q_1$

#### Code Implementation
```python
loss_bc1 = torch.mean((du_dx_bc1 - q0) ** 2)
loss_bc2 = torch.mean((du_dx_bc2 - q1) ** 2)
```
Here, `du_dx_bc1` and `du_dx_bc2` are the temperature gradients at the boundaries predicted by the model, and `q0` and `q1` are the given heat fluxes. `loss_bc1` and `loss_bc2` are the loss functions for the two boundaries respectively, calculated using the mean squared error.

### 3. Robin Boundary Conditions
Robin boundary conditions consider both the temperature and the heat flux at the boundaries, which is a combination of Dirichlet and Neumann boundary conditions.

#### Formulas
- At the boundary $x = x_1$: $\alpha\frac{\partial u}{\partial x}\big|_{x = x_1}+h u(x_1,t)=q_0$
- At the boundary $x = x_2$: $\alpha\frac{\partial u}{\partial x}\big|_{x = x_2}+h u(x_2,t)=q_1$

where $\alpha$ is the thermal diffusivity, $h$ is the heat transfer coefficient.

#### Code Implementation
```python
loss_bc1 = torch.mean((alpha * du_dx_bc1 + h * u_bc1 - q0) ** 2)
loss_bc2 = torch.mean((alpha * du_dx_bc2 + h * u_bc2 - q1) ** 2)
```
Here, `du_dx_bc1` and `du_dx_bc2` are the temperature gradients at the boundaries predicted by the model, `u_bc1` and `u_bc2` are the boundary temperatures predicted by the model, `q0` and `q1` are the given heat fluxes, $\alpha$ is the thermal diffusivity, and $h$ is the heat transfer coefficient. `loss_bc1` and `loss_bc2` are the loss functions for the two boundaries respectively, calculated using the mean squared error.
