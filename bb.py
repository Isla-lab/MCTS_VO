# Import numpy and matplotlib libraries
import numpy as np
import matplotlib.pyplot as plt

# Define the mean and standard deviation of the normal distribution
mean = 0
std = 0.38*2

# Define the range of x values from -pi to pi
x = np.linspace(-np.pi, np.pi, 100)

# Compute the probability density function of the normal distribution
y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

# Plot the normal distribution curve
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Normal distribution with mean {mean} and std {std}')
plt.show()