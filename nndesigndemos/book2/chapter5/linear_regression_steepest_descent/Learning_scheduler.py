from PyQt6 import QtWidgets, QtCore
import numpy as np
import warnings
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
        plot_start_x = 0
        plot_start_y = 95

        plot_width = 265
        plot_height = 280

        plot_spacing_x = 240

        self.make_plot(1, (plot_start_x, plot_start_y, plot_width, plot_height))   # Learning rate schedule
        self.make_plot(2, (plot_start_x + plot_spacing_x, plot_start_y, plot_width, plot_height))  # Training error curve
        self.make_plot(3, (plot_start_x, plot_start_y + 250, 310, 340))  # Linear regression fit (moved left, narrower)

        self.figure.set_tight_layout(True)
        self.figure2.set_tight_layout(True)
        self.figure3.set_tight_layout(True)

        # Slider parameters with 10x magnitude ranges
        # Position after description text (ends ~y=350) and before bottom
        slider_y_start = 360
        slider_spacing = 60

        # lr_0: 0.001 to 0.1 (default 0.01) - linear
        self.make_slider("slider_lr0", QtCore.Qt.Orientation.Horizontal, (1, 100),
                         QtWidgets.QSlider.TickPosition.TicksBelow, 1, 10,
                         (self.x_chapter_usual, slider_y_start, self.w_chapter_slider, 50),
                         self.slider_update, "label_lr0",
                         "lr_0: 0.01", (self.x_chapter_usual + 10, slider_y_start - 25, self.w_chapter_slider, 50))

        # lr_0_steps: 1 to 200 (default 40)
        self.make_slider("slider_lr0_steps", QtCore.Qt.Orientation.Horizontal, (1, 200),
                         QtWidgets.QSlider.TickPosition.TicksBelow, 1, 40,
                         (self.x_chapter_usual, slider_y_start + slider_spacing, self.w_chapter_slider, 50),
                         self.slider_update, "label_lr0_steps",
                         "lr_0_steps: 40", (self.x_chapter_usual + 10, slider_y_start + slider_spacing - 25, self.w_chapter_slider, 50))

        # lr_max: 0.01 to 1.0 (default 0.2) - linear
        self.make_slider("slider_lr_max", QtCore.Qt.Orientation.Horizontal, (1, 100),
                         QtWidgets.QSlider.TickPosition.TicksBelow, 1, 20,
                         (self.x_chapter_usual, slider_y_start + 2 * slider_spacing, self.w_chapter_slider, 50),
                         self.slider_update, "label_lr_max",
                         "lr_max: 0.2", (self.x_chapter_usual + 10, slider_y_start + 2 * slider_spacing - 25, self.w_chapter_slider, 50))

        # lr_max_steps: 1 to 200 (default 40)
        self.make_slider("slider_lr_max_steps", QtCore.Qt.Orientation.Horizontal, (1, 200),
                         QtWidgets.QSlider.TickPosition.TicksBelow, 1, 40,
                         (self.x_chapter_usual, slider_y_start + 3 * slider_spacing, self.w_chapter_slider, 50),
                         self.slider_update, "label_lr_max_steps",
                         "lr_max_steps: 40", (self.x_chapter_usual + 10, slider_y_start + 3 * slider_spacing - 25, self.w_chapter_slider, 50))

        # lr_decay: 0.10 to 0.99 (default 0.85)
        self.make_slider("slider_lr_decay", QtCore.Qt.Orientation.Horizontal, (10, 99),
                         QtWidgets.QSlider.TickPosition.TicksBelow, 1, 85,
                         (self.x_chapter_usual, slider_y_start + 4 * slider_spacing, self.w_chapter_slider, 50),
                         self.slider_update, "label_lr_decay",
                         "lr_decay: 0.85", (self.x_chapter_usual + 10, slider_y_start + 4 * slider_spacing - 25, self.w_chapter_slider, 50))

        # Animation controls - right of bottom plot
        ani_x = 340
        ani_y = 370

        self.animation_enabled = False
        self.make_checkbox('checkbox_animation', 'Enable Animation',
                           (ani_x-10, ani_y, 150, 30),
                           self.toggle_animation, self.animation_enabled)

        self.make_button("btn_train", "Train",
                         (ani_x, ani_y + 35, 120, 30), self.on_train)

        self.make_button("btn_pause", "Pause",
                         (ani_x, ani_y + 70, 120, 30), self.toggle_pause)

        self.make_label("label_speed", "Animation Speed:", (ani_x, ani_y + 110, 150, 20))

        self.make_slider("slider_speed", QtCore.Qt.Orientation.Horizontal, (1, 100),
                         QtWidgets.QSlider.TickPosition.TicksBelow, 10, 70,
                         (ani_x-10, ani_y + 130, 140, 30), self.change_speed)

        self.make_label("label_epoch", "Epoch: -", (ani_x, ani_y + 155, 150, 20))

        # Random seed control - below animation controls
        self.random_seed = 42
        self.make_label("label_seed", "Seed: 42", (ani_x, ani_y + 205, 150, 20))
        self.make_button("btn_random_seed", "New Random Data",
                         (ani_x, ani_y + 225, 140, 30), self.new_random_data)

        # Animation state
        self.animation = None
        self.animation_speed = self._calc_speed(70)
        self.is_animating = False
        self.animation_paused = False
        self.current_epoch = 0

        # Pre-computed training data
        self.lr_history_data = []
        self.errors_data = []
        self.w_history_data = []
        self.b_history_data = []

        self.max_epochs = 200
        self.compute_training()
        self.plot_at_epoch(self.max_epochs - 1)

    @staticmethod
    def _calc_speed(slider_val):
        """Convert slider value (1-100) to animation interval in ms. Higher slider = faster."""
        return max(10, (101 - slider_val) * 5)

    def get_params(self):
        """Extract current parameter values from sliders"""
        # lr_0: linear scale from 0.001 to 0.1
        lr_0 = self.slider_lr0.value() * 0.001

        lr_0_steps = self.slider_lr0_steps.value()

        # lr_max: linear scale from 0.01 to 1.0
        lr_max = self.slider_lr_max.value() * 0.01

        lr_max_steps = self.slider_lr_max_steps.value()

        # lr_decay: linear scale from 0.10 to 0.99
        lr_decay = self.slider_lr_decay.value() / 100.0

        lr_min = 0.000001  # Fixed

        return lr_0, lr_0_steps, lr_max, lr_max_steps, lr_min, lr_decay

    def slider_update(self):
        """Update labels and redraw when sliders change"""
        lr_0, lr_0_steps, lr_max, lr_max_steps, _, lr_decay = self.get_params()

        self.label_lr0.setText(f"lr_0: {lr_0:.3g}")
        self.label_lr0_steps.setText(f"lr_0_steps: {lr_0_steps}")
        self.label_lr_max.setText(f"lr_max: {lr_max:.3g}")
        self.label_lr_max_steps.setText(f"lr_max_steps: {lr_max_steps}")
        self.label_lr_decay.setText(f"lr_decay: {lr_decay:.2f}")

        self.stop_animation()
        self.compute_training()

        if self.animation_enabled:
            self.current_epoch = 0
            self.is_animating = True
            self.animation_paused = False
            self.btn_pause.setText("Pause")
            self.animation = QtCore.QTimer()
            self.animation.timeout.connect(self.animate_step)
            self.animation.start(self.animation_speed)
        else:
            self.plot_at_epoch(self.max_epochs - 1)
            self.label_epoch.setText("Epoch: -")

    def new_random_data(self):
        """Generate new random data points with a new seed"""
        self.random_seed = np.random.randint(0, 100000)
        np.random.seed(self.random_seed)
        self.x_data = np.linspace(0, 1, 100)
        self.y_data = 2 * self.x_data - 0.5 + np.random.normal(0, 0.1, 100)
        self.label_seed.setText(f"Seed: {self.random_seed}")

        self.stop_animation()
        self.compute_training()
        if self.animation_enabled:
            self.current_epoch = 0
            self.is_animating = True
            self.animation_paused = False
            self.btn_pause.setText("Pause")
            self.animation = QtCore.QTimer()
            self.animation.timeout.connect(self.animate_step)
            self.animation.start(self.animation_speed)
        else:
            self.plot_at_epoch(self.max_epochs - 1)
            self.label_epoch.setText("Epoch: -")

    def toggle_animation(self):
        """Toggle whether animation is enabled via checkbox"""
        self.animation_enabled = self.checkbox_animation.checkState().value == 2
        if not self.animation_enabled:
            self.stop_animation()
            self.compute_training()
            self.plot_at_epoch(self.max_epochs - 1)
            self.label_epoch.setText("Epoch: -")

    def compute_training(self):
        """Pre-compute all training epochs and store results"""
        lr_0, lr_0_steps, lr_max, lr_max_steps, lr_min, lr_decay = self.get_params()
        scheduler = LearningRateScheduler(lr_0, lr_0_steps, lr_max, lr_max_steps, lr_min, lr_decay)
        _, _, self.errors_data, self.w_history_data, self.b_history_data, self.lr_history_data = \
            self.linear_regression_with_scheduler(scheduler, self.max_epochs)

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

    def plot_at_epoch(self, epoch):
        """Plot all three charts showing state at a specific epoch"""
        lr_0_steps = self.slider_lr0_steps.value()
        lr_max_steps = self.slider_lr_max_steps.value()

        all_epochs = np.arange(self.max_epochs)

        # Plot 1: Learning Rate Schedule (full curve with marker at current epoch)
        self.figure.clf()
        ax1 = self.figure.add_subplot(111)
        ax1.plot(all_epochs, self.lr_history_data, 'b-', linewidth=1.5)

        # Red marker at current epoch
        ax1.plot(epoch, self.lr_history_data[epoch], 'ro', markersize=8, zorder=5)

        ax1.set_xlabel('Epoch', fontsize=9)
        ax1.set_ylabel('Learning Rate', fontsize=9)
        ax1.set_title('LR Schedule', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, self.max_epochs - 1])
        ax1.tick_params(labelsize=8)

        # Highlight the three phases
        decay_start = lr_0_steps + lr_max_steps
        if lr_0_steps > 0 and lr_0_steps <= self.max_epochs - 1:
            ax1.axvspan(0, min(lr_0_steps - 1, self.max_epochs - 1), alpha=0.15, color='green', label='Warmup')
        if lr_max_steps > 0 and lr_0_steps < self.max_epochs - 1:
            ax1.axvspan(lr_0_steps, min(decay_start - 1, self.max_epochs - 1), alpha=0.15, color='yellow', label='Plateau')
        if decay_start < self.max_epochs - 1:
            ax1.axvspan(decay_start, self.max_epochs - 1, alpha=0.15, color='red', label='Decay')
        ax1.legend(loc='upper right', fontsize=7)

        # Plot 2: Training Error (up to current epoch)
        self.figure2.clf()
        ax2 = self.figure2.add_subplot(111)
        show_epochs = all_epochs[:epoch + 1]
        show_errors = self.errors_data[:epoch + 1]
        ax2.plot(show_epochs, show_errors, 'r-', linewidth=1.5)
        ax2.set_xlabel('Epoch', fontsize=9)
        ax2.set_ylabel('MSE', fontsize=9)
        ax2.set_title('Training Error', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, self.max_epochs - 1])
        ax2.set_ylim([0, max(max(self.errors_data) * 1.1, 0.001)])
        ax2.tick_params(labelsize=8)

        # Plot 3: Linear Regression Fit
        self.figure3.clf()
        ax3 = self.figure3.add_subplot(111)

        # Plot data points
        ax3.scatter(self.x_data, self.y_data, color='blue', alpha=0.5, s=15, label='Data')

        # Plot regression line at current epoch (after update)
        w = self.w_history_data[epoch + 1]
        b = self.b_history_data[epoch + 1]
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
        ax3.set_title(f'Linear Regression (MSE: {self.errors_data[epoch]:.4f})', fontsize=10)
        ax3.legend(loc='upper left', fontsize=7)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([min(self.y_data) - 0.5, max(self.y_data) + 0.5])
        ax3.tick_params(labelsize=8)

        # Draw all canvases
        self.canvas.draw()
        self.canvas2.draw()
        self.canvas3.draw()

    # --- Animation methods ---

    def on_train(self):
        """Start training animation from epoch 0"""
        self.stop_animation()
        self.compute_training()

        if not self.animation_enabled:
            # No animation - just show final result
            self.plot_at_epoch(self.max_epochs - 1)
            self.label_epoch.setText("Epoch: -")
            return

        self.current_epoch = 0
        self.is_animating = True
        self.animation_paused = False
        self.btn_pause.setText("Pause")

        self.animation = QtCore.QTimer()
        self.animation.timeout.connect(self.animate_step)
        self.animation.start(self.animation_speed)

    def animate_step(self):
        """Process the next animation frame"""
        if self.current_epoch >= self.max_epochs:
            self.stop_animation()
            return

        self.plot_at_epoch(self.current_epoch)
        self.label_epoch.setText(f"Epoch: {self.current_epoch + 1}/{self.max_epochs}")
        self.current_epoch += 1

    def stop_animation(self):
        """Stop the animation"""
        if self.animation:
            self.animation.stop()
            self.animation = None
        self.is_animating = False
        self.animation_paused = False
        self.btn_pause.setText("Pause")

    def toggle_pause(self):
        """Toggle between pause and play"""
        if not self.is_animating:
            # If not currently animating but have more epochs, resume
            if 0 < self.current_epoch < self.max_epochs:
                self.is_animating = True
                self.animation_paused = False
                self.btn_pause.setText("Pause")
                self.animation = QtCore.QTimer()
                self.animation.timeout.connect(self.animate_step)
                self.animation.start(self.animation_speed)
            return

        if self.animation_paused:
            # Resume animation
            self.animation_paused = False
            self.btn_pause.setText("Pause")
            if self.animation:
                self.animation.start(self.animation_speed)
        else:
            # Pause animation
            self.animation_paused = True
            self.btn_pause.setText("Play")
            if self.animation:
                self.animation.stop()

    def change_speed(self):
        """Update animation speed from slider"""
        self.animation_speed = self._calc_speed(self.slider_speed.value())
        if self.animation and self.is_animating and not self.animation_paused:
            self.animation.setInterval(self.animation_speed)
