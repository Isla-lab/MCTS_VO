import numpy as np
from scipy.optimize import minimize

# Define the time array from 0 to 100
t = np.linspace(0, 100, 10000)


# Define the function A(t) based on the trigonometric components
def A(t, omega):
    cos_omega_t = np.cos(omega * t)
    cos_2omega_t = np.cos(2 * omega * t)
    sin_omega_t = np.sin(omega * t)
    sin_2omega_t = np.sin(2 * omega * t)

    A_t = np.sqrt((cos_omega_t + 4 * cos_2omega_t) ** 2 + (-sin_omega_t + 4 * sin_2omega_t) ** 2)
    return A_t


# Define the objective function for optimization
def objective(omega):
    A_t = A(t, omega)
    A_max = np.max(A_t)
    return abs(omega * A_max - 0.2)


# Initial guess for omega
initial_guess = 0.01

# Perform the optimization
result = minimize(objective, initial_guess, bounds=[(0, None)], method='L-BFGS-B')

# Extract the optimal omega
optimal_omega = result.x[0]
A_t_optimal = A(t, optimal_omega)
A_max_optimal = np.max(A_t_optimal)
v_max_optimal = optimal_omega * A_max_optimal

print(optimal_omega, A_max_optimal, v_max_optimal)
