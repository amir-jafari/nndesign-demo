import numpy as np
import matplotlib.pyplot as plt   # MATLAB plotting functions
from SequenceUtilities import plot_poles, plot_response, input_output

# System 1

# Assign the weights and bias
IW = [0, 0.5, 0.5]
LW = []
b = 0

# Create the FIR network
system = input_output(IW, LW, b)

# Define the input sequence
p = [0, 1, 2, 3, 2, 1, 0, 0, 0]

# Simulate the system
a = system.process(p)

# Define the time points
t = np.arange(len(p))

# Plot the response
plot_response(a,t)
plt.show()

# System 2

# Assign the weights and bias
IW = [0, 1]
LW = [0.707, -0.25]
b = 0
denominator = [1, -0.707, 0.25]

# Create the IIR network
system = input_output(IW, LW, b)

# Define the input sequence
p = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Simulate the system
a = system.process(p)

# Define the time points
t = np.arange(len(p))

# Plot the response
plot_response(a,t)
plt.show()

# Then use np.roots to find the poles
poles = np.roots(denominator)

# Print out the poles
print('System poles:')
print(poles)

# Convert poles to magnitude and phase
magnitudes = np.abs(poles)
phases = np.angle(poles)*180/np.pi
print('Pole magnitudes:')
print(magnitudes)
print('Pole phases:')
print(phases)

# Plot the poles
plot_poles(poles)
plt.show()



