# function with 3 values
# Table p and t. We can change p in the table.
# Two figures. pole plot; impulse response plot.
# animation.

# Three issues needed to be solved:


from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH
from nndesigndemos.book2.chapter11.impulse_response.utils import state_space


class ImpulseResponse(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(ImpulseResponse, self).__init__(w_ratio, h_ratio, dpi, main_menu=2)

        self.fill_chapter(f"Impulse Response", 11, "Change impulse sequence\nby entering values in input\nfields.\n\n"
                                                   "Adjust pole locations\nusing sliders.\n\n"
                                                   "Click [Set Default] to\nrestore original values.",
                          PACKAGE_PATH + "Logo/Logo_Ch_11.svg", PACKAGE_PATH + "Figures/nndeep11_IIR_1_NoEq.svg", 2,
                          icon_move_left=80, icon_move_up=0, description_coords=(535, 130, 450, 220))

        self.make_plot(1, (10, 420, 250, 250))  # Pole plot
        self.make_plot(2, (260, 420, 250, 250))  # Impulse response plot

        # Default values
        self.p_str = ['1', '0', '0', '0', '0', '0', '0', '0']
        self.lw11_str = ['1', '-0.24', '1']  # Three coefficients for lw11
        
        self.p = np.array(self.p_str, dtype=int)
        self.lw11_coeff = np.array(self.lw11_str, dtype=float)
        
        # Fixed parameters
        self.iw11 = np.array([1, 0])
        self.lw21 = np.array([1, 0])
        self.b1 = np.zeros(2)
        self.b2 = 0
        self.a_0 = [0, 0]
        # print(self.iw11, np.array([[1, -0.24],[1, 0]]).transpose(), self.b1, self.lw21, self.b2, self.a_0, self.p)

        self.n = len(self.p_str)
        self.n_outputs = self.n + 1  # Network outputs include initial condition

        self.initialize_table()
        
        # Three coefficient sliders for lw11
        slider_y_start = 360
        slider_spacing = 60
        
        self.make_slider("coeff1_slider", QtCore.Qt.Orientation.Horizontal, (-100, 100), QtWidgets.QSlider.TickPosition.TicksBelow, 10, 
                        int(self.lw11_coeff[0] * 100), (self.x_chapter_usual, slider_y_start, self.w_chapter_slider, 50), 
                        self.slider_update, "label_coeff1", "Coeff 1: 1.00", (self.x_chapter_usual+20, slider_y_start-25, self.w_chapter_slider, 50))
        
        self.make_slider("coeff2_slider", QtCore.Qt.Orientation.Horizontal, (-100, 100), QtWidgets.QSlider.TickPosition.TicksBelow, 10, 
                        int(self.lw11_coeff[1] * 100), (self.x_chapter_usual, slider_y_start + slider_spacing, self.w_chapter_slider, 50), 
                        self.slider_update, "label_coeff2", "Coeff 2: -0.24", (self.x_chapter_usual+20, slider_y_start + slider_spacing-25, self.w_chapter_slider, 50))
                        
        self.make_slider("coeff3_slider", QtCore.Qt.Orientation.Horizontal, (-100, 100), QtWidgets.QSlider.TickPosition.TicksBelow, 10, 
                        int(self.lw11_coeff[2] * 100), (self.x_chapter_usual, slider_y_start + 2*slider_spacing, self.w_chapter_slider, 50), 
                        self.slider_update, "label_coeff3", "Coeff 3: 1.00", (self.x_chapter_usual+20, slider_y_start + 2*slider_spacing-25, self.w_chapter_slider, 50))
        
        # Set Default button
        self.make_button("default_button", "Set Default", 
                        (self.x_chapter_button, 550, self.w_chapter_button, self.h_chapter_button), 
                        self.set_default_values)

        # Animation controls after SVG figure
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
    
    def slider_update(self):
        """Update coefficients from slider values and refresh display"""
        coeff1_val = self.get_slider_value_and_update(self.coeff1_slider, self.label_coeff1, 1/100, 2)
        coeff2_val = self.get_slider_value_and_update(self.coeff2_slider, self.label_coeff2, 1/100, 2)
        coeff3_val = self.get_slider_value_and_update(self.coeff3_slider, self.label_coeff3, 1/100, 2)
        
        self.lw11_coeff = np.array([coeff1_val, coeff2_val, coeff3_val], dtype=float)
        self.update_values()

    def initialize_table(self):
        """Create and setup the sequence table"""
        self.table = QTableWidget(2, self.n_outputs, self)
        self.table.setGeometry(20, 310, 480, 103)

        self.table.setVerticalHeaderLabels(['p(t)', 'a(t)'])

        # Set column headers to show time indices starting from 0
        headers = [str(i) for i in range(self.n_outputs)]
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
        
        # Connect table item changes to real-time update
        self.table.itemChanged.connect(self.on_table_item_changed)

        self.table.show()

    def on_table_item_changed(self, item):
        """Handle real-time table updates"""
        if item.row() == 0:  # Only process p(t) row changes
            try:
                col = item.column()

                # Update the p array (adjust for initial condition offset)
                new_value = int(item.text())
                self.p[col-1] = new_value
                self.update_values()
            except ValueError:
                # Revert to previous value if invalid
                item.setText(str(self.p[col-1]))

    def update_table(self):
        """Update table with current sequence data"""
        # Temporarily disconnect the signal to avoid recursion
        self.table.itemChanged.disconnect()

        for i in range(self.n_outputs):
            if i == 0:
                # First column is initial condition (no input)
                item_p = QTableWidgetItem("None")  # Initial Condition
                item_p.setFlags(item_p.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                item_p.setBackground(QtGui.QColor(240, 240, 240))
            else:
                # Subsequent columns show input values
                item_p = QTableWidgetItem(str(self.p[i-1]))

            item_p.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(0, i, item_p)

            item_a = QTableWidgetItem(f"{self.a[i]:.2f}")
            item_a.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_a.setFlags(item_a.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_a.setBackground(QtGui.QColor(240, 240, 240))
            self.table.setItem(1, i, item_a)

        # Reconnect the signal
        self.table.itemChanged.connect(self.on_table_item_changed)

    def set_default_values(self):
        """Reset all values to defaults"""
        # Reset P values
        self.p = np.array(self.p_str, dtype=int)
        self.lw11_coeff = np.array(self.lw11_str, dtype=float)

        # Temporarily disconnect sliders to prevent premature updates
        self.coeff1_slider.valueChanged.disconnect()
        self.coeff2_slider.valueChanged.disconnect() 
        self.coeff3_slider.valueChanged.disconnect()
        
        self.coeff1_slider.setValue(int(self.lw11_coeff[0] * 100))
        self.coeff2_slider.setValue(int(self.lw11_coeff[1] * 100))
        self.coeff3_slider.setValue(int(self.lw11_coeff[2] * 100))
        
        # Update labels
        self.label_coeff1.setText(f"Coeff 1: {float(self.lw11_coeff[0]):.2f}")
        self.label_coeff2.setText(f"Coeff 2: {float(self.lw11_coeff[1]):.2f}")
        self.label_coeff3.setText(f"Coeff 3: {float(self.lw11_coeff[2]):.2f}")

        # Reconnect sliders
        self.coeff1_slider.valueChanged.connect(self.slider_update)
        self.coeff2_slider.valueChanged.connect(self.slider_update)
        self.coeff3_slider.valueChanged.connect(self.slider_update)

        self.update_values()

    def update_values(self):
        """Update network calculations and plots"""
        try:
            # Build lw11 matrix from coefficients
            lw11 = np.array([[self.lw11_coeff[0], self.lw11_coeff[1]], 
                            [self.lw11_coeff[2], 0]]).transpose()
            
            # print(self.iw11, lw11, self.b1, self.lw21, self.b2, self.a_0, self.p)
            # Create network and process sequence
            net = state_space(self.iw11, lw11, self.b1, self.lw21, self.b2, self.a_0)
            self.a = net.process(self.p)

            self.graph()
            self.update_table()
            
            # Restart animation if it's enabled
            if self.animation_enabled and self.checkbox_animation.checkState().value == 2:
                self.prepare_animation_frames()
            
        except Exception as e:
            # Handle any calculation errors
            print(f"Error in calculations: {e}")

    def graph(self):
        """Update both plots"""
        self.plot_poles()
        self.plot_impulse_response()

    def plot_poles(self, highlight_step=None):
        """Plot poles in the complex plane"""
        self.figure.clf()
        ax = self.figure.add_subplot(1, 1, 1)
        
        # Calculate poles from denominator coefficients
        # Denominator polynomial: coeff0*z^2 + coeff1*z + coeff2 = 0
        # Using the three coefficients from sliders
        denominator = [self.lw11_coeff[0], self.lw11_coeff[1], self.lw11_coeff[2]]
        poles = np.roots(denominator)
        
        # Plot unit circle for reference
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, linewidth=1)
        
        # Plot axes
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
        
        # Plot poles
        ax.plot(poles.real, poles.imag, 'rx', markersize=8, markeredgewidth=2)
        
        # Set equal aspect ratio and limits
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        
        # Set consistent tick formatting for both axes
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        
        ax.set_title('Pole Locations', fontsize=12, pad=5)
        # ax.set_xlabel('Real Part')
        # ax.set_ylabel('Imaginary Part')
        ax.grid(True, alpha=0.3)
        
        # Adjust subplot margins to show axis labels
        # self.figure.subplots_adjust(left=0.2, bottom=0.2)
        
        self.canvas.draw()

    def plot_impulse_response(self, highlight_step=None):
        """Plot impulse response"""
        self.figure2.clf()
        ax = self.figure2.add_subplot(1, 1, 1)
        
        t = np.arange(len(self.a))
        
        # Create stem plot
        markerline, stemlines, baseline = ax.stem(t, self.a)
        
        # Set stem formatting
        plt.setp(markerline, 'color', 'blue', 'markersize', 4)
        plt.setp(stemlines, 'color', 'blue', 'linewidth', 2)
        
        # Highlight the current step during animation
        if highlight_step is not None and highlight_step < len(self.a):
            ax.stem([highlight_step], [self.a[highlight_step]], linefmt='r-', markerfmt='ro', basefmt=' ')
            ax.plot(highlight_step, self.a[highlight_step], 'ro', markersize=8, zorder=10)
        
        # Set axis limits
        ax.set_xlim(-0.5, len(self.a) - 0.5)
        
        # Calculate y limits with some margin
        y_max = max(self.a) * 1.1 if max(self.a) > 0 else 1
        y_min = min(self.a) * 1.1 if min(self.a) < 0 else -0.1
        ax.set_ylim(y_min, y_max)
        
        # Set integer ticks for x-axis
        x_ticks = np.arange(0, len(self.a), max(1, len(self.a)//8))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(tick)) for tick in x_ticks])
        
        # # Set y-axis ticks with 1 decimal place
        # y_range = y_max - y_min
        # y_interval = y_range / 5  # Show about 5 ticks
        # y_start = y_min
        # y_end = y_max
        # y_ticks = np.linspace(y_start, y_end, 6)  # 6 points for 5 intervals
        # ax.set_yticks(y_ticks)
        # ax.set_yticklabels([f"{tick:.1f}" for tick in y_ticks])
        
        ax.set_title('Impulse Response a(t)', fontsize=12, pad=5)
        # ax.set_xlabel('Time Step')
        # ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        
        # Adjust subplot margins to give more space for y-axis labels
        self.figure2.subplots_adjust(left=0.2)
        
        self.canvas2.draw()

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
        """Prepare animation frames for visualizing the impulse response"""
        if self.is_animating:
            self.stop_animation()
            
        # Build lw11 matrix from coefficients
        lw11 = np.array([[self.lw11_coeff[0], self.lw11_coeff[1]], 
                        [self.lw11_coeff[2], 0]]).transpose()
        
        # Generate animation frames by processing each input step by step
        self.animation_frames = []

        # print(self.iw11, lw11, self.b1, self.lw21, self.b2, self.a_0, self.p)

        net = state_space(self.iw11, lw11, self.b1, self.lw21, self.b2, self.a_0)
        
        # First frame: initial condition (index 0, not animated)
        initial_output = np.matmul(self.lw21, net.a_0) + self.b2
        frame = {
            'step': 0,  # Initial condition at t=0
            'input_value': 0,
            'output_value': initial_output,
            'state': net.a_0.copy()
        }
        self.animation_frames.append(frame)
        
        # Subsequent frames: process each input (animate indices 1-8)
        for i in range(self.n):
            output = net.step(self.p[i])
            frame = {
                'step': i + 1,  # Animation steps are 1-8 (table indices)
                'input_value': self.p[i],
                'output_value': output,
                'state': net.a_0.copy()
            }
            self.animation_frames.append(frame)
        
        self.current_step = 0  # Start animation from step 0 (include initial condition)
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
        
        # Update table to show progress
        self.update_animation_table(frame['step'])
        
        # Update plots with current state highlighting
        current_a = [self.animation_frames[j]['output_value'] for j in range(min(self.current_step + 1, len(self.animation_frames)))]
        current_a += [0] * (len(self.a) - len(current_a))  # Pad with zeros
        
        # Update plots with highlighting
        self.plot_poles()
        # Highlight current step (0-8)
        self.plot_impulse_response(highlight_step=frame['step'])
        
        self.current_step += 1
        
    def update_animation_table(self, current_step):
        """Update table during animation"""
        self.table.itemChanged.disconnect()

        for i in range(self.n_outputs):
            # Show values progressively from index 0 to current step
            if i <= current_step:
                output_val = self.animation_frames[i]['output_value']
                item_a = QTableWidgetItem(f"{output_val:.2f}")
            else:
                item_a = QTableWidgetItem("")

            item_a.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_a.setFlags(item_a.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_a.setBackground(QtGui.QColor(240, 240, 240))
            self.table.setItem(1, i, item_a)

        self.table.itemChanged.connect(self.on_table_item_changed)
        
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
