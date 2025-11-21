from PyQt6 import QtWidgets, QtCore, QtGui
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH
from nndesigndemos.book2.chapter12.gated_cec.gated_cec import GatedCEC


class GatedCECDemo(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(GatedCECDemo, self).__init__(w_ratio, h_ratio, dpi, main_menu=2)

        self.fill_chapter(f"Gated CEC", 12, "Click buttons to toggle\ngates between 0 and 1.\n\n"
                                                   "Input gate g^i(t) controls\nhow much input enters\nmemory.\n\n"
                                                   "Feedback gate g^f(t)\ncontrols how much\nprevious state is retained.\n\n"
                                                   "Enable animation to see\nstep-by-step execution.",
                          PACKAGE_PATH + "Logo/Logo_Ch_12.svg", None, 2,
                          icon_move_left=0, icon_move_up=0, description_coords=(535, 130, 450, 300))

        # Create four plots in 2x2 grid
        plot_width = 240
        plot_height = 180
        plot_start_x = 25
        plot_start_y = 280
        plot_spacing_x = 250
        plot_spacing_y = 190

        self.make_plot(1, (plot_start_x, plot_start_y, plot_width, plot_height))
        self.make_plot(2, (plot_start_x + plot_spacing_x, plot_start_y, plot_width, plot_height))
        self.make_plot(3, (plot_start_x, plot_start_y + plot_spacing_y, plot_width, plot_height))
        self.make_plot(4, (plot_start_x + plot_spacing_x, plot_start_y + plot_spacing_y, plot_width, plot_height))

        # Define sequences (matching the PDF)
        self.max_length = 13

        # Fixed input p(t) - all ones
        self.p = np.ones(self.max_length)

        # Default g^i(t): ones from t=0 to t=4, zeros from t=5 to t=6, ones from t=7 to t=12
        self.gi_default = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1])
        self.gi = self.gi_default.copy()

        # Default g^f(t): ones from t=0 to t=6, zeros at t=7, ones from t=8 to t=10, zeros at t=11 to t=12
        self.gf_default = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0])
        self.gf = self.gf_default.copy()

        # Network outputs
        self.output = np.zeros(self.max_length)
        self.states = np.zeros(self.max_length + 1)

        # Create toggle buttons for gates
        self.create_gate_buttons()

        # Animation controls
        ani_x = self.x_chapter_button + 15
        ani_y = 480
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
        button_y = 420
        self.make_button("default_button", "Set Default",
                        (self.x_chapter_button, button_y, self.w_chapter_button, self.h_chapter_button),
                        self.set_default_values)

        # Initialize with default values
        self.set_default_values()

    def create_gate_buttons(self):
        """Create toggle buttons for g^i(t) and g^f(t)"""
        button_start_x = 20
        button_start_y = 150
        button_width = 35
        button_height = 30
        button_spacing = 38
        row_spacing = 60

        self.gi_buttons = []
        self.gf_buttons = []

        # Label for g^i(t) row
        label_gi = QtWidgets.QLabel("Input Gate g^i(t):", self)
        label_gi.setGeometry(button_start_x, button_start_y - 25, 150, 20)
        label_gi.setStyleSheet("font-size: 11px; font-weight: bold;")
        label_gi.show()

        # Create g^i(t) buttons
        for i in range(self.max_length):
            button = QtWidgets.QPushButton(str(int(self.gi[i])), self)
            button.setGeometry(button_start_x + i * button_spacing, button_start_y, button_width, button_height)
            button.setStyleSheet(self.get_button_style(int(self.gi[i])))
            button.clicked.connect(lambda checked, idx=i: self.toggle_gi(idx))
            button.show()
            self.gi_buttons.append(button)

        # Label for g^f(t) row
        label_gf = QtWidgets.QLabel("Feedback Gate g^f(t):", self)
        label_gf.setGeometry(button_start_x, button_start_y + row_spacing - 25, 150, 20)
        label_gf.setStyleSheet("font-size: 11px; font-weight: bold;")
        label_gf.show()

        # Create g^f(t) buttons
        for i in range(self.max_length):
            button = QtWidgets.QPushButton(str(int(self.gf[i])), self)
            button.setGeometry(button_start_x + i * button_spacing, button_start_y + row_spacing, button_width, button_height)
            button.setStyleSheet(self.get_button_style(int(self.gf[i])))
            button.clicked.connect(lambda checked, idx=i: self.toggle_gf(idx))
            button.show()
            self.gf_buttons.append(button)

        # Add time step labels below second row
        for i in range(self.max_length):
            label = QtWidgets.QLabel(str(i), self)
            label.setGeometry(button_start_x + i * button_spacing + 14, button_start_y + row_spacing + 35, 20, 20)
            label.setStyleSheet("font-size: 9px; color: gray;")
            label.show()

    def get_button_style(self, value):
        """Get button style based on value (0 or 1)"""
        if value == 0:
            return """
                QPushButton {
                    background-color: #f0f0f0;
                    border: 2px solid #999;
                    border-radius: 5px;
                    font-size: 12px;
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
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """

    def toggle_gi(self, idx):
        """Toggle g^i gate at index idx"""
        self.gi[idx] = 1 - self.gi[idx]
        self.gi_buttons[idx].setText(str(int(self.gi[idx])))
        self.gi_buttons[idx].setStyleSheet(self.get_button_style(int(self.gi[idx])))
        self.update_values()

    def toggle_gf(self, idx):
        """Toggle g^f gate at index idx"""
        self.gf[idx] = 1 - self.gf[idx]
        self.gf_buttons[idx].setText(str(int(self.gf[idx])))
        self.gf_buttons[idx].setStyleSheet(self.get_button_style(int(self.gf[idx])))
        self.update_values()

    def set_default_values(self):
        """Reset all values to defaults"""
        self.gi = self.gi_default.copy()
        self.gf = self.gf_default.copy()

        # Update button states for g^i
        for i in range(self.max_length):
            self.gi_buttons[i].setText(str(int(self.gi[i])))
            self.gi_buttons[i].setStyleSheet(self.get_button_style(int(self.gi[i])))

        # Update button states for g^f
        for i in range(self.max_length):
            self.gf_buttons[i].setText(str(int(self.gf[i])))
            self.gf_buttons[i].setStyleSheet(self.get_button_style(int(self.gf[i])))

        self.update_values()

    def update_values(self):
        """Process sequence through network and update display"""
        try:
            # Create network and process sequence
            cec = GatedCEC(initial_state=0.0)
            self.output, self.states = cec.forward(self.p, self.gi, self.gf)

            self.graph()

            # Restart animation if it's enabled
            if self.animation_enabled and self.checkbox_animation.checkState().value == 2:
                self.prepare_animation_frames()

        except Exception as e:
            print(f"Error in calculations: {e}")

    def graph(self):
        """Update all plots"""
        # Plot 1: Input p(t)
        self.plot_sequence(self.p, self.figure, self.canvas, r'Input $p(t)$', 1)

        # Plot 2: Input gate g^i(t)
        self.plot_sequence(self.gi, self.figure2, self.canvas2, r'Input Gate $g^i(t)$', 2)

        # Plot 3: Feedback gate g^f(t)
        self.plot_sequence(self.gf, self.figure3, self.canvas3, r'Feedback Gate $g^f(t)$', 3)

        # Plot 4: Output a(t)
        self.plot_sequence(self.output, self.figure4, self.canvas4, r'Output $a(t)$', 4)

    def plot_sequence(self, array, figure, canvas, title, plot_num, highlight_step=None):
        """Plot a sequence with optional step highlighting"""
        figure.clf()
        ax = figure.add_subplot(1, 1, 1)

        figure.subplots_adjust(left=0.175, right=0.95, top=0.88, bottom=0.15)

        x = np.arange(len(array))

        # Create stem plot
        markerline, stemlines, baseline = ax.stem(x, array)

        # Set stem formatting (blue color)
        plt.setp(markerline, 'color', 'blue', 'markersize', 5)
        plt.setp(stemlines, 'color', 'blue', 'linewidth', 1.5)

        # Highlight the current step during animation
        if highlight_step is not None and highlight_step < len(array):
            ax.stem([highlight_step], [array[highlight_step]], linefmt='r-', markerfmt='ro', basefmt=' ')
            ax.plot(highlight_step, array[highlight_step], 'ro', markersize=8, zorder=10)

        # Set axis limits
        ax.set_xlim(-0.5, len(array) - 0.5)

        # Adjust y-limits based on plot type
        if plot_num == 4:  # Output plot may have varying values
            y_max = max(np.max(array), 1.0) if len(array) > 0 else 1.0
            ax.set_ylim(-0.2, y_max + 0.5)
        else:
            ax.set_ylim(-0.2, 1.2)

        # Set ticks
        if len(array) <= 13:
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(tick)) for tick in x], fontsize=8)

        # Add title and labels
        ax.set_title(title, fontsize=10, pad=5)
        ax.set_xlabel('Time Step', fontsize=9)
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
        cec = GatedCEC(initial_state=0.0)

        # Generate animation frames
        self.animation_frames = []
        current_output = []
        current_states = [0.0]  # Initial state

        for i in range(self.max_length):
            # Calculate current state
            prev_state = current_states[-1]
            new_state = self.gf[i] * prev_state + self.gi[i] * self.p[i]
            current_states.append(new_state)
            current_output.append(new_state)

            frame = {
                'step': i,
                'output': np.array(current_output + [0] * (self.max_length - len(current_output))),
                'state': new_state,
            }
            self.animation_frames.append(frame)

        self.current_step = 0
        self.total_steps = len(self.animation_frames)
        self.is_animating = True

        self.start_animation()

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

        # Update all four plots with highlighted step
        # Plot 1: Input p(t) with highlight
        self.plot_sequence(self.p, self.figure, self.canvas, r'Input $p(t)$', 1,
                          highlight_step=self.current_step)

        # Plot 2: Input gate g^i(t) with highlight
        self.plot_sequence(self.gi, self.figure2, self.canvas2, r'Input Gate $g^i(t)$', 2,
                          highlight_step=self.current_step)

        # Plot 3: Feedback gate g^f(t) with highlight
        self.plot_sequence(self.gf, self.figure3, self.canvas3, r'Feedback Gate $g^f(t)$', 3,
                          highlight_step=self.current_step)

        # Plot 4: Output a(t) with highlight (show only computed values so far)
        partial_output = frame['output'][:self.current_step + 1]
        full_output = np.concatenate([partial_output, np.zeros(self.max_length - len(partial_output))])
        self.plot_sequence(full_output, self.figure4, self.canvas4, r'Output $a(t)$', 4,
                          highlight_step=self.current_step)

        self.current_step += 1

    def stop_animation(self):
        """Stop the animation sequence"""
        if self.animation:
            self.animation.stop()

        self.is_animating = False
        self.animation_paused = False
        self.btn_play_pause.setText("Pause")

        # Show final state
        self.graph()

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
