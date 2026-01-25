import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys
from matplotlib import cm


def linear_regression_steepest_descent(x, y, learning_rate, iterations):
    """
    Solve a single variable linear regression
    roblem using steepest descent algorithm.

    Args:
        x: Array of independent variables
        y: Array of dependent variables
        learning_rate: Learning rate for gradient descent
        iterations: Number of iterations to perform

    Returns:
        w: Final slope of the linear regression line
        b: Final intercept of the linear regression line
        errors: List of errors at each iteration
        w_history: List of w values at each iteration
        b_history: List of b values at each iteration
    """
    # Convert inputs to numpy arrays
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Initialize parameters
    w = 0.0
    b = 0.0

    # Number of samples
    n = len(x)

    # To store errors and parameters at each iteration
    errors = []
    w_history = [w]
    b_history = [b]

    # Perform steepest descent
    for i in range(iterations):
        # Compute predictions
        y_pred = w * x + b

        # Compute error (sum of squared errors)
        error = np.sum((y_pred - y) ** 2) / n
        errors.append(error)

        # Compute gradients
        dw = (2/n) * np.sum((y_pred - y) * x)
        db = (2/n) * np.sum(y_pred - y)

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Store updated parameters
        w_history.append(w)
        b_history.append(b)

    return w, b, errors, w_history, b_history


def compute_error_surface(x, y, w_range, b_range):
    """
    Compute the sum squared error for a grid of w, b values.

    Args:
        x: Array of independent variables
        y: Array of dependent variables
        w_range: Array of w values
        b_range: Array of b values

    Returns:
        error_surface: 2D array of error values
    """
    n = len(x)
    error_surface = np.zeros((len(w_range), len(b_range)))

    for i, w in enumerate(w_range):
        for j, b in enumerate(b_range):
            y_pred = w * x + b
            error = np.sum((y_pred - y) ** 2) / n
            error_surface[i, j] = error

    return error_surface


def plot_results(x, y, w, b, errors):
    """
    Plot the regression line and the error curve.

    Args:
        x: Array of independent variables
        y: Array of dependent variables
        w: Slope of the regression line
        b: Intercept of the regression line
        errors: List of errors at each iteration
    """
    plt.figure(figsize=(12, 5))

    # Plot data and regression line
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, color='blue', label='Data points')

    # Generate points for the regression line
    x_line = np.linspace(min(x), max(x), 100)
    y_line = w * x_line + b

    plt.plot(x_line, y_line, color='red', label=f'Regression line: y = {w:.4f}x + {b:.4f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()

    # Plot error curve
    plt.subplot(1, 2, 2)
    plt.plot(range(len(errors)), errors)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title('Error vs. Iteration')

    plt.tight_layout()
    plt.show()


def animate_regression(x, y, w_history, b_history, errors):
    """
    Create an animation showing how the regression line changes at each iteration.
    Also shows a contour plot of the sum squared error in the w, b plane,
    and the movement of the w, b location by a circle as they are updated by the algorithm.
    Additionally shows a 3D surface plot of the error surface with the marker moving along the surface.

    Args:
        x: Array of independent variables
        y: Array of dependent variables
        w_history: List of w values at each iteration
        b_history: List of b values at each iteration
        errors: List of errors at each iteration
    """
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')

    # Plot data points (these won't change)
    ax1.scatter(x, y, color='blue', label='Data points')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Linear Regression')

    # Generate x values for the regression line
    x_line = np.linspace(min(x), max(x), 100)

    # Initial regression line
    line, = ax1.plot([], [], 'r-', label='Regression line')

    # Text to display current parameters and iteration (positioned at bottom-right to avoid regression line)
    param_text = ax1.text(0.95, 0.05, '', transform=ax1.transAxes, 
                          verticalalignment='bottom', horizontalalignment='right')

    # Error plot
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_title('Error vs. Iteration')

    # Set axis limits for error plot
    ax2.set_xlim(0, len(errors))
    ax2.set_ylim(0, max(errors) * 1.1)

    # Error line
    error_line, = ax2.plot([], [], 'b-')

    # Add legend to the first subplot
    ax1.legend()

    # Create contour plot of error surface
    # Determine appropriate ranges for w and b based on their histories
    w_min, w_max = min(w_history), max(w_history)
    b_min, b_max = min(b_history), max(b_history)

    # Add some padding to the ranges
    w_padding = (w_max - w_min) * 0.5
    b_padding = (b_max - b_min) * 0.5
    w_min -= w_padding
    w_max += w_padding
    b_min -= b_padding
    b_max += b_padding

    # Create grid for w and b
    w_range = np.linspace(w_min, w_max, 100)
    b_range = np.linspace(b_min, b_max, 100)

    # Compute error surface
    error_surface = compute_error_surface(x, y, w_range, b_range)

    # Create contour plot with curves instead of color map
    W, B = np.meshgrid(w_range, b_range)

    # Compute more evenly spaced contour levels using logarithmic scale
    min_error = np.min(error_surface)
    max_error = np.max(error_surface)
    # Use logarithmic spacing for levels to get more even spacing visually
    levels = np.logspace(np.log10(min_error + 1e-10), np.log10(max_error), 10)

    contour = ax3.contour(W, B, error_surface.T, levels=levels, colors='k')
    ax3.clabel(contour, inline=True, fontsize=8)

    # Set labels for contour plot
    ax3.set_xlabel('w (slope)')
    ax3.set_ylabel('b (intercept)')
    ax3.set_title('Error Surface (2D)')

    # Set equal scales for the contour plot so steepest descent path is perpendicular to contour lines
    ax3.set_aspect('equal')

    # Add marker for current w, b position on contour plot
    point, = ax3.plot([], [], 'ro', markersize=8)

    # Create 3D surface plot
    surf = ax4.plot_surface(W, B, error_surface.T, cmap=cm.viridis, alpha=0.8, linewidth=0, antialiased=True)

    # Set labels for 3D surface plot
    ax4.set_xlabel('w (slope)')
    ax4.set_ylabel('b (intercept)')
    ax4.set_zlabel('Error')
    ax4.set_title('Error Surface (3D)')

    # Set number of tick marks on each axis to 4
    x_ticks = np.linspace(min(w_range), max(w_range), 4)
    y_ticks = np.linspace(min(b_range), max(b_range), 4)
    z_ticks = np.linspace(min_error, max_error, 4)

    ax4.set_xticks(x_ticks)
    ax4.set_yticks(y_ticks)
    ax4.set_zticks(z_ticks)

    # Format tick labels to display two digits after the decimal point
    ax4.set_xticklabels([f'{x:.2f}' for x in x_ticks])
    ax4.set_yticklabels([f'{y:.2f}' for y in y_ticks])
    ax4.set_zticklabels([f'{z:.2f}' for z in z_ticks])

    # Put the Z axis tick marks and label on the left side of the 3D plot
    ax4.zaxis.set_rotate_label(False)
    ax4.zaxis.set_ticks_position('lower')
    ax4.zaxis.set_label_position('lower')

    # Add a colorbar for the 3D surface
    fig.colorbar(surf, ax=ax4, shrink=0.5, aspect=5)

    # Add marker for current w, b position on 3D surface
    # Use plot instead of scatter for better visibility in 3D
    point3d, = ax4.plot([], [], [], 'ro', markersize=10, markeredgecolor='black', zorder=10)

    # Tight layout
    plt.tight_layout()

    def init():
        """Initialize the animation."""
        line.set_data([], [])
        error_line.set_data([], [])
        point.set_data([], [])
        point3d.set_data_3d([], [], [])
        param_text.set_text('')
        return line, error_line, point, point3d, param_text

    def update(frame):
        """Update the animation for each frame."""
        # Update regression line
        w = w_history[frame]
        b = b_history[frame]
        y_line = w * x_line + b
        line.set_data(x_line, y_line)

        # Update parameter text
        param_text.set_text(f'Iteration: {frame}\nw = {w:.4f}, b = {b:.4f}')

        # Update error plot
        error_line.set_data(range(frame + 1), errors[:frame + 1])

        # Update marker position on contour plot
        point.set_data([w], [b])

        # Calculate the error value for the current w, b
        n = len(x)
        y_pred = w * x + b
        error = np.sum((y_pred - y) ** 2) / n

        # Update marker position on 3D surface plot
        # For 3D line plots with markers, we use set_data_3d
        point3d.set_data_3d(np.array([w]), np.array([b]), np.array([error]))

        return line, error_line, point, point3d, param_text

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(errors),
                                  init_func=init, blit=True, interval=100)

    plt.show()

    return ani


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Linear Regression using Steepest Descent')

    parser.add_argument('--learning-rate', type=float, default=0.2,
                        help='Learning rate for steepest descent (default: 0.2)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations to perform (default: 100)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting of results')
    parser.add_argument('--animate', action='store_true',
                        help='Create an animation showing the regression line changing at each iteration')
    parser.add_argument('--use-sample-data', action='store_true',
                        help='Use sample data instead of providing x and y values')
    parser.add_argument('--x', type=float, nargs='+',
                        help='List of x values (independent variables)')
    parser.add_argument('--y', type=float, nargs='+',
                        help='List of y values (dependent variables)')
    parser.add_argument('--input-file', type=str,
                        help='Path to input file with x,y values (one pair per line)')

    return parser.parse_args()


def load_data_from_file(file_path):
    """Load x and y data from a file."""
    x = []
    y = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                values = line.strip().split(',')
                if len(values) >= 2:
                    try:
                        x_val = float(values[0])
                        y_val = float(values[1])
                        x.append(x_val)
                        y.append(y_val)
                    except ValueError:
                        # Skip lines that can't be converted to float
                        continue
    except Exception as e:
        print(f"Error loading data from file: {e}")
        sys.exit(1)

    if not x or not y:
        print("No valid data found in the input file.")
        sys.exit(1)

    return x, y


def main():
    """Main function to run linear regression with steepest descent."""
    args = parse_arguments()

    # Get data
    if args.use_sample_data:
        # Generate sample data
        np.random.seed(42)
        x = np.linspace(0, 1, 100)
        y = 2 * x - 0.5 + np.random.normal(0, .1, 100)  # True: w=2, b=-0.5
        print("Using sample data (true parameters: w=2, b=-0.5)")
    elif args.input_file:
        # Load data from file
        x, y = load_data_from_file(args.input_file)
        print(f"Loaded {len(x)} data points from {args.input_file}")
    elif args.x is not None and args.y is not None:
        # Use provided x and y values
        if len(args.x) != len(args.y):
            print("Error: x and y must have the same length")
            sys.exit(1)
        x = args.x
        y = args.y
        print(f"Using {len(x)} provided data points")
    else:
        print("Error: Must provide data (--use-sample-data, --input-file, or --x and --y)")
        sys.exit(1)

    # Get hyperparameters
    learning_rate = args.learning_rate
    iterations = args.iterations

    print(f"Running steepest descent with learning_rate={learning_rate}, iterations={iterations}")

    # Run linear regression
    w, b, errors, w_history, b_history = linear_regression_steepest_descent(x, y, learning_rate, iterations)

    # Print results
    print(f"Estimated parameters: w = {w:.4f}, b = {b:.4f}")
    print(f"Final error: {errors[-1]:.4f}")

    # Plot results or animate based on command-line options
    if not args.no_plot:
        if args.animate:
            print("Creating animation of regression line...")
            animate_regression(x, y, w_history, b_history, errors)
        else:
            plot_results(x, y, w, b, errors)


if __name__ == "__main__":
    main()
#python3 linear_regression_steepest_descent.py --use-sample-data --animate
