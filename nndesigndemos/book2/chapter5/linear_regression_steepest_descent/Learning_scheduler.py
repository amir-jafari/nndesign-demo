from PyQt6 import QtWidgets, QtCore
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH


class LearningRateScheduler:
    """
    Learning rate scheduler that:
    - Starts at lr_0
    - Increases linearly for lr_0_steps to lr_max
    - Holds at lr_max for lr_max_steps
    - Decays geometrically at rate lr_decay toward lr_min
    """
    def __init__(self, lr_0=0.001, lr_0_steps=10, lr_max=0.01, lr_max_steps=10, lr_min=0.000001, lr_decay=0.85):
        self.lr_0 = lr_0
        self.lr_max = lr_max
        self.lr_0_steps = lr_0_steps
        self.lr_max_steps = lr_max_steps
        self.lr_min = lr_min
        self.lr_decay = lr_decay

    def __call__(self, epoch):
        if epoch < self.lr_0_steps:
            # Linear warmup phase
            lr = self.lr_0 + (self.lr_max - self.lr_0) * epoch / self.lr_0_steps
        elif epoch < self.lr_0_steps + self.lr_max_steps:
            # Plateau phase
            lr = self.lr_max
        else:
            # Exponential decay phase
            lr = self.lr_min + (self.lr_max - self.lr_min) * self.lr_decay ** (epoch - self.lr_0_steps - self.lr_max_steps)
        return lr


class LearningScheduler(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(LearningScheduler, self).__init__(w_ratio, h_ratio, dpi, main_menu=2)

        self.fill_chapter("Learning Scheduler", 5,
                          "\nAdjust the learning rate\nscheduler parameters\nusing the sliders.\n\n"
                          "The demo shows how\ndifferent learning rate\nschedules affect the\nconvergence of gradient\ndescent on a linear\nregression problem.",
                          PACKAGE_PATH + "Chapters/5_D/Logo_Ch_5.svg", None, 2,
                          icon_move_left=120, description_coords=(535, 70, 450, 350))

        # Generate sample data for linear regression
        np.random.seed(42)
        self.x_data = np.linspace(0, 1, 100)
        self.y_data = 2 * self.x_data - 0.5 + np.random.normal(0, 0.1, 100)  # True: w=2, b=-0.5

        # Create three plots - positioned for better visibility
        # Plot 1 & 2: Top row side by side
        plot_start_x = 5
        plot_start_y = 95

        plot_width = 265
        plot_height = 280

        plot_spacing_x = 240

        self.make_plot(1, (plot_start_x, plot_start_y, plot_width, plot_height))   # Learning rate schedule
        self.make_plot(2, (plot_start_x + plot_spacing_x, plot_start_y, plot_width, plot_height))  # Training error curve
        self.make_plot(3, (plot_start_x+10, plot_start_y + 250, 450, 340))  # Linear regression fit

        self.figure.set_tight_layout(True)
        self.figure2.set_tight_layout(True)
        self.figure3.set_tight_layout(True)

        # Slider parameters with 10x magnitude ranges
        # Position after description text (ends ~y=350) and before bottom
        slider_y_start = 360
        slider_spacing = 60

        # lr_0: 0.0001 to 0.01 (default 0.001) - 10x range up and down
        self.make_slider("slider_lr0", QtCore.Qt.Orientation.Horizontal, (1, 100),
                         QtWidgets.QSlider.TickPosition.TicksBelow, 1, 50,
                         (self.x_chapter_usual, slider_y_start, self.w_chapter_slider, 50),
                         self.slider_update, "label_lr0",
                         "lr_0: 0.0010", (self.x_chapter_usual + 10, slider_y_start - 25, self.w_chapter_slider, 50))

        # lr_0_steps: 1 to 100 (default 10)
        self.make_slider("slider_lr0_steps", QtCore.Qt.Orientation.Horizontal, (1, 100),
                         QtWidgets.QSlider.TickPosition.TicksBelow, 1, 10,
                         (self.x_chapter_usual, slider_y_start + slider_spacing, self.w_chapter_slider, 50),
                         self.slider_update, "label_lr0_steps",
                         "lr_0_steps: 10", (self.x_chapter_usual + 10, slider_y_start + slider_spacing - 25, self.w_chapter_slider, 50))

        # lr_max: 0.001 to 0.1 (default 0.01) - 10x range up and down
        self.make_slider("slider_lr_max", QtCore.Qt.Orientation.Horizontal, (1, 100),
                         QtWidgets.QSlider.TickPosition.TicksBelow, 1, 50,
                         (self.x_chapter_usual, slider_y_start + 2 * slider_spacing, self.w_chapter_slider, 50),
                         self.slider_update, "label_lr_max",
                         "lr_max: 0.0100", (self.x_chapter_usual + 10, slider_y_start + 2 * slider_spacing - 25, self.w_chapter_slider, 50))

        # lr_max_steps: 1 to 100 (default 10)
        self.make_slider("slider_lr_max_steps", QtCore.Qt.Orientation.Horizontal, (1, 100),
                         QtWidgets.QSlider.TickPosition.TicksBelow, 1, 10,
                         (self.x_chapter_usual, slider_y_start + 3 * slider_spacing, self.w_chapter_slider, 50),
                         self.slider_update, "label_lr_max_steps",
                         "lr_max_steps: 10", (self.x_chapter_usual + 10, slider_y_start + 3 * slider_spacing - 25, self.w_chapter_slider, 50))

        # lr_decay: 0.50 to 0.99 (default 0.85)
        self.make_slider("slider_lr_decay", QtCore.Qt.Orientation.Horizontal, (50, 99),
                         QtWidgets.QSlider.TickPosition.TicksBelow, 1, 85,
                         (self.x_chapter_usual, slider_y_start + 4 * slider_spacing, self.w_chapter_slider, 50),
                         self.slider_update, "label_lr_decay",
                         "lr_decay: 0.85", (self.x_chapter_usual + 10, slider_y_start + 4 * slider_spacing - 25, self.w_chapter_slider, 50))

        self.max_epochs = 100
        self.graph()

    def get_params(self):
        """Extract current parameter values from sliders"""
        # lr_0: logarithmic scale from 0.0001 to 0.01
        lr_0 = 0.0001 * (10 ** (self.slider_lr0.value() / 50))

        lr_0_steps = self.slider_lr0_steps.value()

        # lr_max: logarithmic scale from 0.001 to 0.1
        lr_max = 0.001 * (10 ** (self.slider_lr_max.value() / 50))

        lr_max_steps = self.slider_lr_max_steps.value()

        # lr_decay: linear scale from 0.50 to 0.99
        lr_decay = self.slider_lr_decay.value() / 100.0

        lr_min = 0.000001  # Fixed

        return lr_0, lr_0_steps, lr_max, lr_max_steps, lr_min, lr_decay

    def slider_update(self):
        """Update labels and redraw when sliders change"""
        lr_0, lr_0_steps, lr_max, lr_max_steps, lr_min, lr_decay = self.get_params()

        self.label_lr0.setText(f"lr_0: {lr_0:.4f}")
        self.label_lr0_steps.setText(f"lr_0_steps: {lr_0_steps}")
        self.label_lr_max.setText(f"lr_max: {lr_max:.4f}")
        self.label_lr_max_steps.setText(f"lr_max_steps: {lr_max_steps}")
        self.label_lr_decay.setText(f"lr_decay: {lr_decay:.2f}")

        self.graph()

    def linear_regression_with_scheduler(self, scheduler, max_epochs):
        """
        Perform linear regression using steepest descent with a learning rate scheduler.

        Returns:
            w, b: final parameters
            errors: list of MSE at each epoch
            w_history, b_history: parameter histories
            lr_history: learning rate at each epoch
        """
        x = self.x_data
        y = self.y_data

        # Initialize parameters
        w = 0.0
        b = 0.0
        n = len(x)

        # Storage
        errors = []
        w_history = [w]
        b_history = [b]
        lr_history = []

        # Training loop
        for epoch in range(max_epochs):
            # Get learning rate from scheduler
            lr = scheduler(epoch)
            lr_history.append(lr)

            # Compute predictions
            y_pred = w * x + b

            # Compute error (MSE)
            error = np.sum((y_pred - y) ** 2) / n
            errors.append(error)

            # Compute gradients
            dw = (2/n) * np.sum((y_pred - y) * x)
            db = (2/n) * np.sum(y_pred - y)

            # Update parameters
            w = w - lr * dw
            b = b - lr * db

            # Store updated parameters
            w_history.append(w)
            b_history.append(b)

        return w, b, errors, w_history, b_history, lr_history

    def graph(self):
        """Update all plots"""
        # Get current parameters
        lr_0, lr_0_steps, lr_max, lr_max_steps, lr_min, lr_decay = self.get_params()

        # Create scheduler
        scheduler = LearningRateScheduler(lr_0, lr_0_steps, lr_max, lr_max_steps, lr_min, lr_decay)

        # Run training
        w, b, errors, w_history, b_history, lr_history = self.linear_regression_with_scheduler(
            scheduler, self.max_epochs)

        # Plot 1: Learning Rate Schedule
        self.figure.clf()
        ax1 = self.figure.add_subplot(111)
        epochs = np.arange(self.max_epochs)
        ax1.plot(epochs, lr_history, 'b-', linewidth=1.5)
        ax1.set_xlabel('Epoch', fontsize=9)
        ax1.set_ylabel('Learning Rate', fontsize=9)
        ax1.set_title('LR Schedule', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, self.max_epochs - 1])
        ax1.tick_params(labelsize=8)

        # Highlight the three phases
        if lr_0_steps > 0:
            ax1.axvspan(0, lr_0_steps - 1, alpha=0.15, color='green', label='Warmup')
        if lr_max_steps > 0:
            ax1.axvspan(lr_0_steps, lr_0_steps + lr_max_steps - 1, alpha=0.15, color='yellow', label='Plateau')
        ax1.axvspan(lr_0_steps + lr_max_steps, self.max_epochs - 1, alpha=0.15, color='red', label='Decay')
        ax1.legend(loc='upper right', fontsize=7)

        # Plot 2: Training Error Curve
        self.figure2.clf()
        ax2 = self.figure2.add_subplot(111)
        ax2.plot(epochs, errors, 'r-', linewidth=1.5)
        ax2.set_xlabel('Epoch', fontsize=9)
        ax2.set_ylabel('MSE', fontsize=9)
        ax2.set_title('Training Error', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, self.max_epochs - 1])
        ax2.set_ylim([0, max(errors) * 1.1])
        ax2.tick_params(labelsize=8)

        # Plot 3: Linear Regression Fit
        self.figure3.clf()
        ax3 = self.figure3.add_subplot(111)

        # Plot data points
        ax3.scatter(self.x_data, self.y_data, color='blue', alpha=0.5, s=15, label='Data')

        # Plot final regression line
        x_line = np.linspace(min(self.x_data), max(self.x_data), 100)
        y_line = w * x_line + b
        ax3.plot(x_line, y_line, 'r-', linewidth=1.5,
                label=f'Fit: y={w:.2f}x+{b:.2f}')

        # Plot true line
        y_true = 2 * x_line - 0.5
        ax3.plot(x_line, y_true, 'g--', linewidth=1.5, alpha=0.7,
                label='True: y=2x-0.5')

        ax3.set_xlabel('x', fontsize=9)
        ax3.set_ylabel('y', fontsize=9)
        ax3.set_title(f'Linear Regression (MSE: {errors[-1]:.4f})', fontsize=10)
        ax3.legend(loc='upper left', fontsize=7)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=8)

        # Draw all canvases
        self.canvas.draw()
        self.canvas2.draw()
        self.canvas3.draw()
