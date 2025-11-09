from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH
from nndesigndemos.book2.chapter12.parity_network.paritynetwork import SpecificParityNetwork


class ParityNetworkDemo(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(ParityNetworkDemo, self).__init__(w_ratio, h_ratio, dpi, main_menu=2)

        self.fill_chapter(f"Parity Network", 12, "Click input buttons to\ntoggle bits (0/1).\n\n"
                                                   "The output updates\nautomatically.\n\n"
                                                   "Enable animation to see\nstep-by-step execution.\n\n"
                                                   "Click [Set Default] to\nrestore original values.",
                          PACKAGE_PATH + "Logo/Logo_Ch_12.svg", None, 2,
                          icon_move_left=0, icon_move_up=0, description_coords=(535, 130, 450, 250))

        # Single plot for output sequence
        self.make_plot(1, (25, 430, 500, 250))

        # Default input sequence (max 10 bits)
        self.p_str = ['0', '0', '1', '1', '0', '1', '1', '0', '0', '0']
        self.p = np.array(self.p_str, dtype=int)
        self.max_length = 10

        # Network outputs
        self.a1 = np.zeros((self.max_length, 2))  # Layer 1 output (2 neurons)
        self.a2 = np.zeros(self.max_length)       # Layer 2 output (final)

        # Create toggle buttons for input
        self.create_input_buttons()

        # Create table for displaying sequences
        self.initialize_table()

        # Animation controls
        ani_x = self.x_chapter_button + 15
        ani_y = 450
        self.animation_enabled = False
        self.make_checkbox('checkbox_animation', 'Enable Animation', (ani_x-12, ani_y, 150, 30),
                          self.toggle_animation, self.animation_enabled)

        self.make_label("label_animation", "Animation Speed:", (ani_x, ani_y + 40, 150, 30))
        self.make_slider("slider_animation", QtCore.Qt.Orientation.Horizontal, (100, 1000),
                        QtWidgets.QSlider.TickPosition.TicksBelow, 100, 500,
                        (ani_x-12, ani_y+80, 150, 30), self.change_animation_speed)

        self.make_button("btn_play_pause", "Pause",
                        (ani_x, ani_y+115, 80, 30), self.toggle_play_pause)

        # Animation variables
        self.animation_speed = 500
        self.is_animating = False
        self.animation = None
        self.current_step = 0
        self.total_steps = 0
        self.animation_frames = []
        self.animation_paused = False

        # Control buttons
        button_y = 390
        self.make_button("default_button", "Set Default",
                        (self.x_chapter_button, button_y, self.w_chapter_button, self.h_chapter_button),
                        self.set_default_values)

        # Initialize with default values
        self.set_default_values()

    def create_input_buttons(self):
        """Create 10 toggle buttons for binary input"""
        button_start_x = 20
        button_start_y = 150
        button_width = 45
        button_height = 30
        button_spacing = 48

        self.input_buttons = []

        # Label for input row
        self.make_label("label_input", "Input p(t):", (button_start_x, button_start_y - 25, 100, 20))

        # Create 10 buttons
        for i in range(self.max_length):
            button = QtWidgets.QPushButton(self.p_str[i], self)
            button.setGeometry(button_start_x + i * button_spacing, button_start_y, button_width, button_height)
            button.setStyleSheet(self.get_button_style(int(self.p_str[i])))
            button.clicked.connect(lambda checked, idx=i: self.toggle_input(idx))
            button.show()
            self.input_buttons.append(button)

        # Add time step labels below buttons
        for i in range(self.max_length):
            label = QtWidgets.QLabel(str(i+1), self)
            label.setGeometry(button_start_x + i * button_spacing + 20, button_start_y + 35, 20, 20)
            label.setStyleSheet("font-size: 10px; color: gray;")
            label.show()

    def get_button_style(self, value):
        """Get button style based on value (0 or 1)"""
        if value == 0:
            return """
                QPushButton {
                    background-color: #f0f0f0;
                    border: 2px solid #999;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
            """
        else:
            return """
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: 2px solid #45a049;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """

    def toggle_input(self, idx):
        """Toggle input bit between 0 and 1"""
        self.p[idx] = 1 - self.p[idx]
        self.input_buttons[idx].setText(str(self.p[idx]))
        self.input_buttons[idx].setStyleSheet(self.get_button_style(self.p[idx]))
        # Auto-update the output
        self.update_values()

    def initialize_table(self):
        """Create and setup the sequence table"""
        self.table = QTableWidget(4, self.max_length, self)
        self.table.setGeometry(20, 235, 480, 183)

        self.table.setVerticalHeaderLabels(['p(t)', 'a¹₁(t)', 'a¹₂(t)', 'a²(t)'])

        # Set column headers to show time indices
        headers = [str(i+1) for i in range(self.max_length)]
        self.table.setHorizontalHeaderLabels(headers)

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
        for i in range(self.max_length):
            # Input row
            item_p = QTableWidgetItem(str(self.p[i]))
            item_p.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_p.setFlags(item_p.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_p.setBackground(QtGui.QColor(240, 240, 240))
            self.table.setItem(0, i, item_p)

            # Layer 1, neuron 1 output
            item_a1_1 = QTableWidgetItem(str(int(self.a1[i, 0])))
            item_a1_1.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_a1_1.setFlags(item_a1_1.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_a1_1.setBackground(QtGui.QColor(255, 250, 205))  # Light yellow
            self.table.setItem(1, i, item_a1_1)

            # Layer 1, neuron 2 output
            item_a1_2 = QTableWidgetItem(str(int(self.a1[i, 1])))
            item_a1_2.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_a1_2.setFlags(item_a1_2.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_a1_2.setBackground(QtGui.QColor(255, 250, 205))  # Light yellow
            self.table.setItem(2, i, item_a1_2)

            # Layer 2 output (final)
            item_a2 = QTableWidgetItem(str(int(self.a2[i])))
            item_a2.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_a2.setFlags(item_a2.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_a2.setBackground(QtGui.QColor(200, 255, 200))  # Light green
            self.table.setItem(3, i, item_a2)

    def set_default_values(self):
        """Reset all values to defaults"""
        self.p = np.array(self.p_str, dtype=int)

        # Update button states
        for i in range(self.max_length):
            self.input_buttons[i].setText(str(self.p[i]))
            self.input_buttons[i].setStyleSheet(self.get_button_style(self.p[i]))

        self.update_values()

    def update_values(self):
        """Process sequence through network and update display"""
        try:
            # Create network and process sequence
            network = SpecificParityNetwork()
            self.a1, self.a2 = network.forward(self.p)

            self.graph()
            self.update_table()

            # Restart animation if it's enabled
            if self.animation_enabled and self.checkbox_animation.checkState().value == 2:
                self.prepare_animation_frames()

        except Exception as e:
            print(f"Error in calculations: {e}")

    def graph(self):
        """Update output plot"""
        self.plot_sequence(self.a2, self.figure, self.canvas, r'Output Sequence $a^2(t)$ (Parity)', 0, 1)

    def plot_sequence(self, array, figure, canvas, title, min_value, max_value, highlight_step=None):
        """Plot a sequence with optional step highlighting"""
        figure.clf()
        ax = figure.add_subplot(1, 1, 1)

        figure.subplots_adjust(left=0.15)

        x = np.arange(len(array))

        # Create stem plot
        markerline, stemlines, baseline = ax.stem(x, array)

        # Set stem formatting (blue color)
        plt.setp(markerline, 'color', 'blue', 'markersize', 6)
        plt.setp(stemlines, 'color', 'blue', 'linewidth', 2)

        # Highlight the current step during animation
        if highlight_step is not None and highlight_step < len(array):
            ax.stem([highlight_step], [array[highlight_step]], linefmt='r-', markerfmt='ro', basefmt=' ')
            ax.plot(highlight_step, array[highlight_step], 'ro', markersize=10, zorder=10)

        # Set axis limits
        ax.set_xlim(-0.5, len(array) - 0.5)
        ax.set_ylim(-0.2, 1.2)

        # Set integer ticks
        x_ticks = np.arange(0, len(array))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(tick)+1) for tick in x_ticks])

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['0 (Even)', '1 (Odd)'])

        # Add title and labels
        ax.set_title(title, fontsize=12, pad=5)
        ax.set_xlabel('Time Step', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Update the canvas
        canvas.draw()

    def toggle_animation(self):
        """Toggle whether animation is enabled"""
        if self.checkbox_animation.checkState().value != 2:
            self.animation_enabled = False
            self.stop_animation()
            self.update_values()
        else:
            self.animation_enabled = True
            self.prepare_animation_frames()

    def prepare_animation_frames(self):
        """Prepare animation frames for visualizing the sequence processing"""
        if self.is_animating:
            self.stop_animation()

        # Create network and process step by step
        network = SpecificParityNetwork()

        # Generate animation frames
        self.animation_frames = []
        for i in range(self.max_length):
            a1, a2 = network.step(self.p[i])
            frame = {
                'step': i,
                'input_value': self.p[i],
                'a1_value': a1.copy(),
                'a2_value': a2,
            }
            self.animation_frames.append(frame)

        self.current_step = 0
        self.total_steps = len(self.animation_frames)
        self.is_animating = True

        # Clear the output initially
        self.clear_animation_table()
        self.start_animation()

    def clear_animation_table(self):
        """Clear output rows for animation"""
        for i in range(self.max_length):
            # Keep input row
            item_p = QTableWidgetItem(str(self.p[i]))
            item_p.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_p.setFlags(item_p.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_p.setBackground(QtGui.QColor(240, 240, 240))
            self.table.setItem(0, i, item_p)

            # Clear a¹₁
            item_a1_1 = QTableWidgetItem("")
            item_a1_1.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_a1_1.setFlags(item_a1_1.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_a1_1.setBackground(QtGui.QColor(255, 250, 205))
            self.table.setItem(1, i, item_a1_1)

            # Clear a¹₂
            item_a1_2 = QTableWidgetItem("")
            item_a1_2.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_a1_2.setFlags(item_a1_2.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_a1_2.setBackground(QtGui.QColor(255, 250, 205))
            self.table.setItem(2, i, item_a1_2)

            # Clear a²
            item_a2 = QTableWidgetItem("")
            item_a2.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_a2.setFlags(item_a2.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_a2.setBackground(QtGui.QColor(200, 255, 200))
            self.table.setItem(3, i, item_a2)

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

        # Update a¹₁ value for current step
        item_a1_1 = QTableWidgetItem(str(int(frame['a1_value'][0])))
        item_a1_1.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        item_a1_1.setFlags(item_a1_1.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        item_a1_1.setBackground(QtGui.QColor(255, 250, 205))
        self.table.setItem(1, self.current_step, item_a1_1)

        # Update a¹₂ value for current step
        item_a1_2 = QTableWidgetItem(str(int(frame['a1_value'][1])))
        item_a1_2.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        item_a1_2.setFlags(item_a1_2.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        item_a1_2.setBackground(QtGui.QColor(255, 250, 205))
        self.table.setItem(2, self.current_step, item_a1_2)

        # Update a² value for current step
        item_a2 = QTableWidgetItem(str(int(frame['a2_value'])))
        item_a2.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        item_a2.setFlags(item_a2.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        item_a2.setBackground(QtGui.QColor(200, 255, 200))
        self.table.setItem(3, self.current_step, item_a2)

        # Update plot with current state
        current_a2 = [self.animation_frames[j]['a2_value'] if j <= self.current_step else 0
                     for j in range(self.max_length)]

        # Highlight current step
        self.plot_sequence(current_a2, self.figure, self.canvas,
                          r'Output Sequence $a^2(t)$ (Parity)', 0, 1, highlight_step=self.current_step)

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
