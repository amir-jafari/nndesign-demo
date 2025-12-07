from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH
from nndesigndemos.book2.chapter12.constant_error_carousel.sequence_processing_script import CEC


class ConstantErrorCarousel(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(ConstantErrorCarousel, self).__init__(w_ratio, h_ratio, dpi, main_menu=2)

        self.fill_chapter(f"Constant Error Carousel", 12,
                         "Change input sequence by\nentering values in input\nfields.\n\n"
                         "Click [Update] to apply\nyour changes.\n\n"
                         "Click [Set Default] to\nrestore original values.\n\n"
                         "CEC uses LW1,1 = 1 and\nlinear transfer function\nto maintain long memory.",
                         PACKAGE_PATH + "Logo/Logo_Ch_12.svg", PACKAGE_PATH + 'Figures/nndeep12_CEC_Net_a.svg', 2,
                         icon_move_left=110, icon_move_up=-25, description_coords=(535, 130, 450, 280))

        # Two plots at the bottom
        self.make_plot(1, (10, 430, 250, 250))  # Input sequence
        self.make_plot(2, (260, 430, 250, 250))  # Output sequence

        # Default input sequence
        self.p_str = ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1']
        self.p = np.array(self.p_str, dtype=float)

        self.n = len(self.p_str)

        # Initialize CEC network (parameters are fixed in the CEC class)
        self.cec = CEC()

        self.initialize_table()

        # Display parameters (read-only labels)
        # param_y_start = 360
        # param_spacing = 30
        # self.make_label("label_iw11", f"IW1,1: {self.cec.IW11:.2f}",
        #                (self.x_chapter_usual, param_y_start, self.w_chapter_slider, 25))
        # self.make_label("label_lw11", f"LW1,1: {self.cec.LW11:.2f}",
        #                (self.x_chapter_usual, param_y_start + param_spacing, self.w_chapter_slider, 25))
        # self.make_label("label_bias", f"Bias: {self.cec.bias:.2f}",
        #                (self.x_chapter_usual, param_y_start + 2*param_spacing, self.w_chapter_slider, 25))

        # Update button
        self.make_button("update_button", "Update",
                        (self.x_chapter_button, 460, self.w_chapter_button, self.h_chapter_button),
                        self.update_values)

        # Set Default button
        self.make_button("default_button", "Set Default",
                        (self.x_chapter_button, 510, self.w_chapter_button, self.h_chapter_button),
                        self.set_default_values)

        # Animation controls
        ani_x = 330

        self.animation_enabled = False
        self.make_checkbox('checkbox_animation', 'Enable Animation', (ani_x-12, 120, 150, 30),
                          self.toggle_animation, self.animation_enabled)

        self.make_label("label_animation", "Animation Speed:", (ani_x, 160, 150, 30))
        self.make_slider("slider_animation", QtCore.Qt.Orientation.Horizontal, (100, 1000),
                        QtWidgets.QSlider.TickPosition.TicksBelow, 100, 500,
                        (ani_x-12, 200, 150, 30), self.change_animation_speed)

        self.make_button("btn_play_pause", "Pause",
                        (ani_x, 235, 80, 30), self.toggle_play_pause)

        # Animation variables
        self.animation_speed = 500
        self.is_animating = False
        self.animation = None
        self.current_step = 0
        self.total_steps = 0
        self.animation_frames = []
        self.animation_paused = False

        self.set_default_values()

    def initialize_table(self):
        """Create and setup the sequence table"""
        self.table = QTableWidget(3, self.n, self)
        self.table.setGeometry(20, 290, 480, 143)

        self.table.setVerticalHeaderLabels(['p(t)', 'd(t)', 'a(t)'])

        self.table.setCornerButtonEnabled(False)

        # Make table look nice
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.table.verticalHeader().setDefaultSectionSize(40)

        # Set font size
        font = self.table.font()
        font.setPointSize(10)
        self.table.setFont(font)

        self.table.show()

    def update_table(self):
        """Update table with current sequence data"""
        for i in range(self.n):
            # Input p(t)
            item_p = QTableWidgetItem(str(self.p[i]))
            item_p.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(0, i, item_p)

            # Delay state d(t) - delay_states has n+1 elements, use i+1 for delay after processing p[i]
            item_d = QTableWidgetItem(f"{self.delay_states[i+1]:.2f}")
            item_d.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_d.setFlags(item_d.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_d.setBackground(QtGui.QColor(240, 240, 240))
            self.table.setItem(1, i, item_d)

            # Output a(t)
            item_a = QTableWidgetItem(f"{self.a[i]:.2f}")
            item_a.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_a.setFlags(item_a.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_a.setBackground(QtGui.QColor(240, 240, 240))
            self.table.setItem(2, i, item_a)

    def set_default_values(self):
        """Reset to default values"""
        # Reset P values
        for i in range(self.n):
            item_p = QTableWidgetItem(self.p_str[i])
            item_p.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(0, i, item_p)

        self.update_values()

    def update_values(self):
        """Process input sequence and update display"""
        try:
            # Read input values from table
            for i in range(self.n):
                self.p[i] = float(self.table.item(0, i).text())

            # Reset CEC and process sequence
            self.cec.reset_delay()
            self.a, self.delay_states = self.cec.forward(self.p)

            self.graph()
            self.update_table()

            # Restart animation if it's enabled
            if self.animation_enabled and self.checkbox_animation.checkState().value == 2:
                self.prepare_animation_frames()

        except ValueError:
            # Show error if parsing fails
            QtWidgets.QMessageBox.critical(self, "Input Error",
                "Please enter valid numeric values for all parameters.")

    def graph(self):
        """Update both plots"""
        # Determine value ranges for each plot independently
        min_input = min(self.p)
        max_input = max(self.p)
        min_output = min(self.a)
        max_output = max(self.a)

        # Plot 1: Input sequence
        self.plot_input(self.figure, self.canvas, min_input, max_input)

        # Plot 2: Output sequence
        self.plot_output(self.figure2, self.canvas2, min_output, max_output)

    def plot_input(self, figure, canvas, min_value, max_value, highlight_step=None):
        """Plot input sequence"""
        figure.clf()
        ax = figure.add_subplot(1, 1, 1)

        x = np.arange(len(self.p))

        # Plot input sequence (blue)
        ax.plot(x, self.p, 'bo-', linewidth=2, markersize=6, label='Input p(t)', alpha=0.7)

        # Highlight the current step during animation
        if highlight_step is not None and highlight_step < len(self.p):
            ax.plot(highlight_step, self.p[highlight_step], 'bo', markersize=10, zorder=10)

        # Set axis limits
        ax.set_xlim(-0.5, len(self.p) - 0.5)

        y_max = np.ceil(max_value) + 1 if max_value > 0 else 1
        y_min = np.floor(min_value) - 1 if min_value < 0 else 0
        ax.set_ylim(y_min, y_max)

        # Set ticks - only integers for y-axis
        y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + 1, 1)
        ax.set_yticks(y_ticks)

        x_ticks = np.arange(0, len(self.p), max(1, len(self.p)//8))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(tick)) for tick in x_ticks])

        # Add title and legend
        ax.set_title('Input', fontsize=12, pad=5)
        # ax.set_xlabel('Time Step')
        # ax.set_ylabel('Value')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        canvas.draw()

    def plot_output(self, figure, canvas, min_value, max_value, highlight_step=None):
        """Plot output sequence"""
        figure.clf()
        ax = figure.add_subplot(1, 1, 1)

        x = np.arange(len(self.a))

        # Plot output sequence (red)
        ax.plot(x, self.a, 'ro-', linewidth=2, markersize=6, label='Output a(t)', alpha=0.7)

        # Highlight the current step during animation
        if highlight_step is not None and highlight_step < len(self.a):
            ax.plot(highlight_step, self.a[highlight_step], 'ro', markersize=10, zorder=10)

        # Set axis limits
        ax.set_xlim(-0.5, len(self.a) - 0.5)
        y_max = max_value * 1.1 if max_value > 0 else 1
        y_min = min_value * 1.1 if min_value < 0 else -0.1
        ax.set_ylim(y_min, y_max)

        # Set ticks
        x_ticks = np.arange(0, len(self.a), max(1, len(self.a)//8))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(tick)) for tick in x_ticks])

        # Add title and legend
        ax.set_title('Output', fontsize=12, pad=5)
        # ax.set_xlabel('Time Step')
        # ax.set_ylabel('Value')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        canvas.draw()

    def toggle_animation(self):
        """Toggle whether animation is enabled"""
        if self.checkbox_animation.checkState().value != 2:
            self.animation_enabled = False
            self.stop_animation()
            self.update_values()  # Update normally without animation
        else:
            self.animation_enabled = True
            self.prepare_animation_frames()

    def prepare_animation_frames(self):
        """Prepare animation frames for visualizing the sequence processing"""
        if self.is_animating:
            self.stop_animation()

        # Use already calculated results (a and delay_states) and create frames for step-by-step display
        self.animation_frames = []
        for i in range(self.n):
            frame = {
                'step': i,
                'input_value': self.p[i],
                'delay_state': self.delay_states[i+1],  # delay_states[i+1] is delay after processing p[i]
                'output_value': self.a[i]
            }
            self.animation_frames.append(frame)

        # Calculate and store min/max from full arrays to keep y-axis fixed during animation
        self.animation_min_input = min(self.p)
        self.animation_max_input = max(self.p)
        self.animation_min_output = min(self.a)
        self.animation_max_output = max(self.a)

        self.current_step = 0
        self.total_steps = len(self.animation_frames)
        self.is_animating = True

        # Clear the table initially
        self.clear_animation_table()
        self.start_animation()

    def clear_animation_table(self):
        """Clear d(t) and a(t) rows for animation"""
        for i in range(self.n):
            # Clear d(t) row
            item_d = QTableWidgetItem("")
            item_d.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_d.setFlags(item_d.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_d.setBackground(QtGui.QColor(240, 240, 240))
            self.table.setItem(1, i, item_d)

            # Clear a(t) row
            item_a = QTableWidgetItem("")
            item_a.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_a.setFlags(item_a.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_a.setBackground(QtGui.QColor(240, 240, 240))
            self.table.setItem(2, i, item_a)

    def start_animation(self):
        """Start the animation sequence"""
        if not self.animation_frames:
            self.is_animating = False
            return

        self.animation = QtCore.QTimer()
        self.animation.timeout.connect(self.animate_next_step)
        self.animation.start(self.animation_speed)

        self.animation_paused = False
        self.btn_play_pause.setText("Pause")

    def animate_next_step(self):
        """Process the next animation step"""
        if self.current_step >= self.total_steps:
            self.stop_animation()
            return

        frame = self.animation_frames[self.current_step]

        # Update d(t) value for current step
        item_d = QTableWidgetItem(f"{frame['delay_state']:.2f}")
        item_d.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        item_d.setFlags(item_d.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        item_d.setBackground(QtGui.QColor(240, 240, 240))
        self.table.setItem(1, self.current_step, item_d)

        # Update a(t) value for current step
        item_a = QTableWidgetItem(f"{frame['output_value']:.2f}")
        item_a.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        item_a.setFlags(item_a.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        item_a.setBackground(QtGui.QColor(240, 240, 240))
        self.table.setItem(2, self.current_step, item_a)

        # Build current state for plots
        current_a = [self.animation_frames[j]['output_value'] if j <= self.current_step else 0
                    for j in range(self.n)]

        # Temporarily update for plotting
        old_a = self.a.copy()
        self.a = np.array(current_a)

        # Use fixed min/max values calculated from full arrays to keep y-axis stable
        # Update plots with current state and highlight
        self.plot_input(self.figure, self.canvas, self.animation_min_input, self.animation_max_input,
                       highlight_step=self.current_step)
        self.plot_output(self.figure2, self.canvas2, self.animation_min_output, self.animation_max_output,
                        highlight_step=self.current_step)

        # Restore full data
        self.a = old_a

        self.current_step += 1

    def stop_animation(self):
        """Stop the animation sequence"""
        if self.animation:
            self.animation.stop()

        self.is_animating = False
        self.animation_paused = False
        self.btn_play_pause.setText("Pause")

    def change_animation_speed(self):
        """Change the animation speed based on slider value"""
        self.animation_speed = 1100 - self.slider_animation.value()
        if self.animation:
            self.animation.setInterval(self.animation_speed)

    def toggle_play_pause(self):
        """Toggle between pausing and resuming the animation"""
        if not self.is_animating:
            if self.animation_frames and self.checkbox_animation.checkState().value == 2:
                self.start_animation()
            return

        if self.animation_paused:
            # Resume animation
            self.animation_paused = False
            self.btn_play_pause.setText("Pause")
            self.animation.start(self.animation_speed)
        else:
            # Pause animation
            self.animation_paused = True
            self.btn_play_pause.setText("Play")
            if self.animation:
                self.animation.stop()
