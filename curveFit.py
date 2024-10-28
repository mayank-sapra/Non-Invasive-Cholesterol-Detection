import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define your function
def func(t, A1, A2, A3, A4, A5, A6):
    return A1 * np.exp(-A2 * t) + A3 * np.exp(-A4 * t) * np.cos(A5 * t + A6)

# Example time data (t) and signal data (P)
# Use your actual t and P data here
t_data = np.arange(0, 71, 1)  # Time data from 0 to 70
P_data = np.array([272, 273, 278, 287, 302, 323, 351, 382, 413, 448,
                   482, 514, 542, 562, 580, 594, 602, 605, 603, 597,
                   589, 577, 565, 549, 533, 516, 498, 482, 465, 451,
                   438, 427, 421, 415, 412, 414, 417, 421, 426, 429,
                   432, 434, 434, 433, 431, 427, 421, 413, 405, 397,
                   388, 379, 369, 359, 351, 342, 334, 325, 318, 311,
                   304, 298, 292, 286, 282, 277, 272, 268, 265, 264,
                   263])  # Corresponding P(t) values

# Initial guesses for the parameters (A1, A2, A3, A4, A5, A6)
initial_guesses = [600, 0.1, 500, 0.05, 1, 0]

# Perform curve fitting
params, covariance = curve_fit(func, t_data, P_data, p0=initial_guesses)

# Extract fitted parameters
A1, A2, A3, A4, A5, A6 = params

# Print fitted parameters
print(f"Fitted parameters:\nA1 = {A1}\nA2 = {A2}\nA3 = {A3}\nA4 = {A4}\nA5 = {A5}\nA6 = {A6}")

# Plot the original data
plt.plot(t_data, P_data, 'b-', label='Data')

# Plot the fitted curve
t_fit = np.linspace(0, 70, 1000)  # Fine time values for plotting the fitted curve
P_fit = func(t_fit, *params)
plt.plot(t_fit, P_fit, 'r-', label='Fitted Curve')

plt.xlabel('Time')
plt.ylabel('P(t)')
plt.legend()
plt.show()
