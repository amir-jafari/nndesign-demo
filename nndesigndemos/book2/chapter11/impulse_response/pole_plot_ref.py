import numpy as np
import matplotlib.pyplot as plt


def plot_poles(poles):
    # Plot poles in the complex plane
    plt.figure(figsize=(6, 6))
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.gca().set_aspect('equal', 'box')

    # Plot unit circle for reference
    theta = np.linspace(0, 2*np.pi, 400)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle')

    # Plot poles
    plt.plot(poles.real, poles.imag, 'rx', markersize=10, label='Poles')

    plt.title("Poles in the Complex Plane")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # First we get the denominator of the transfer function
    denominator = [1, -1.62, 1]

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

    plot_poles(poles)