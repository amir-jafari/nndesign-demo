from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH
from nndesigndemos.book2.chapter11.smoothing_response.SequenceUtilities import input_output


class SequenceSmoothingResponse(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(SequenceSmoothingResponse, self).__init__(w_ratio, h_ratio, dpi, main_menu=2)

        self.fill_chapter(f"Sequence Smoothing Response", 11, "Change input sequence by\nentering values in input\nfields.\n\n"
                                                        "Click [Update] to apply\nyour changes.\n\n"
                                                        "Click [Set Default] to\nrestore original values.",
                          PACKAGE_PATH + "Logo/Logo_Ch_11.svg", PACKAGE_PATH + "Figures/nndeep11_IIR_1_NoEq.svg", 2,
                          icon_move_left=80, icon_move_up=0, description_coords=(535, 130, 450, 220))

        self.make_plot(1, (10, 460, 250, 220))
        self.make_plot(2, (260, 460, 250, 220))

        self.p_str = ['0', '1', '2', '3', '2', '1', '0', '0', '0']
        self.iw_str = ['0', '0.5']
        self.lw_str = ['0.5']
        self.b_str = '0'

        self.p = np.array(self.p_str, dtype=int)
        self.iw = np.array(self.iw_str, dtype=float)
        self.lw = np.array(self.lw_str, dtype=float)
        self.b = float(self.b_str)

        self.n = len(self.p_str)

        self.initialize_table()
        
        # Weight sliders (only iw1 and lw visible, others kept for calculations)
        slider_y_start = 360
        slider_spacing = 60
        self.make_slider("iw1_slider", QtCore.Qt.Orientation.Horizontal, (-100, 100), QtWidgets.QSlider.TickPosition.TicksBelow, 10, 
                        int(self.iw[1] * 100), (self.x_chapter_usual, slider_y_start, self.w_chapter_slider, 50), 
                        self.slider_update, "label_iw1", "iw1: 0.50", (self.x_chapter_usual+20, slider_y_start-25, self.w_chapter_slider, 50))
        
        # Layer weight slider
        self.make_slider("lw_slider", QtCore.Qt.Orientation.Horizontal, (-100, 100), QtWidgets.QSlider.TickPosition.TicksBelow, 10, 
                        int(self.lw[0] * 100), (self.x_chapter_usual, slider_y_start + slider_spacing, self.w_chapter_slider, 50), 
                        self.slider_update, "label_lw", "lw: 0.50", (self.x_chapter_usual+20, slider_y_start + slider_spacing-25, self.w_chapter_slider, 50))
        
        # Update button
        self.make_button("update_button", "Update", (self.x_chapter_button, 500, self.w_chapter_button, self.h_chapter_button), self.update_values)
        
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
        """Update weights from slider values and refresh display"""
        iw1_val = self.get_slider_value_and_update(self.iw1_slider, self.label_iw1, 1/100, 2) 
        lw_val = self.get_slider_value_and_update(self.lw_slider, self.label_lw, 1/100, 2)
        
        # Keep original values for iw0 and b; update only iw1 and lw
        self.iw = np.array([self.iw[0], iw1_val], dtype=float)
        self.lw = np.array([lw_val], dtype=float)
        # self.b remains unchanged
        self.update_values()

    def initialize_table(self):
        """Create and setup the sequence table"""
        self.table = QTableWidget(4, self.n, self)
        self.table.setGeometry(20, 290, 480, 180)
        
        self.table.setVerticalHeaderLabels(['p(t)', 'a(t)', 'p_TDL', 'a_TDL'])
        
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
            item_p = QTableWidgetItem(str(self.p[i]))
            item_p.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # Center text horizontally
            self.table.setItem(0, i, item_p)
            
            item_a = QTableWidgetItem(f"{self.a[i]:.2f}")
            item_a.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # Center text horizontally
            item_a.setFlags(item_a.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)  # Make read-only
            item_a.setBackground(QtGui.QColor(240, 240, 240))  # Light gray background
            self.table.setItem(1, i, item_a)
        
            p_tdl_text = "\n".join([f"{val:.2f}" for val in self.p_tdl_history[i]])
            item_p_tdl = QTableWidgetItem(p_tdl_text)
            item_p_tdl.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # Center text horizontally
            item_p_tdl.setFlags(item_p_tdl.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)  # Make read-only
            item_p_tdl.setBackground(QtGui.QColor(240, 240, 240))  # Light gray background
            self.table.setItem(2, i, item_p_tdl)
            
            a_tdl_text = f"{self.a_tdl_history[i][0]:.2f}" if len(self.a_tdl_history[i]) > 0 else "0.00"
            item_a_tdl = QTableWidgetItem(a_tdl_text)
            item_a_tdl.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # Center text horizontally
            item_a_tdl.setFlags(item_a_tdl.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)  # Make read-only
            item_a_tdl.setBackground(QtGui.QColor(240, 240, 240))  # Light gray background
            self.table.setItem(3, i, item_a_tdl)

    def set_default_values(self):
        # Reset P values
        for i in range(self.n):
            item_p = QTableWidgetItem(self.p_str[i])
            item_p.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)  # Center text horizontally
            self.table.setItem(0, i, item_p)
        
        self.iw = np.array(self.iw_str, dtype=float)
        self.lw = np.array(self.lw_str, dtype=float)
        self.b = float(self.b_str)

        self.iw1_slider.setValue(int(self.iw[1] * 100))
        self.lw_slider.setValue(int(self.lw[0] * 100))
        
        # Update labels to show values (only visible sliders)
        self.label_iw1.setText(f"iw1: {float(self.iw[1]):.2f}")
        self.label_lw.setText(f"lw: {float(self.lw[0]):.2f}")

        self.update_values()

    def update_values(self):
        try:
            for i in range(self.n):
                self.p[i] = int(self.table.item(0, i).text())
                        
            net = input_output(self.iw, self.lw, self.b)
            self.a = net.process(self.p)
            
            # Create history for TDL visualization
            net_history = input_output(self.iw, self.lw, self.b)
            self.p_tdl_history = []
            self.a_tdl_history = []
            
            for i in range(self.n):
                self.p_tdl_history.append(net_history.p_tdl.copy())
                self.a_tdl_history.append(net_history.a_tdl.copy())
                net_history.step(self.p[i])

            # print("Input:", self.p)
            # print("Output:", self.a)

            self.graph()
            self.update_table()
            
            # Restart animation if it's enabled
            if self.animation_enabled and self.checkbox_animation.checkState().value == 2:
                self.prepare_animation_frames()
            
        except ValueError:
            # Show error if parsing fails
            QtWidgets.QMessageBox.critical(self, "Input Error", 
                "Please enter valid integer values for all parameters.")

    def graph(self):
        min_value = min(min(self.p), min(self.a))
        max_value = max(max(self.p), max(self.a))
        self.plot_sequence(self.p, self.figure, self.canvas, r'Input Sequence $p(t)$', min_value, max_value)
        self.plot_sequence(self.a, self.figure2, self.canvas2, r'Output Sequence $a(t)$', min_value, max_value)

    def plot_sequence(self, array, figure, canvas, title, min_value, max_value, highlight_step=None):
        figure.clf()
        ax = figure.add_subplot(1, 1, 1)
        
        x = np.arange(len(array))
        
        # Create stem plot with blue markers and lines
        markerline, stemlines, baseline = ax.stem(x, array)
        
        # Set stem formatting (blue color)
        plt.setp(markerline, 'color', 'blue', 'markersize', 4)
        plt.setp(stemlines, 'color', 'blue', 'linewidth', 2)
        
        # Highlight the current step during animation
        if highlight_step is not None and highlight_step < len(array):
            # Highlight the current step with red color and larger marker
            ax.stem([highlight_step], [array[highlight_step]], linefmt='r-', markerfmt='ro', basefmt=' ')
            # Make the highlighted marker larger
            ax.plot(highlight_step, array[highlight_step], 'ro', markersize=8, zorder=10)
        
        # Set axis limits
        ax.set_xlim(-0.5, len(array) - 0.5)
        y_max = max_value * 1.1 if max_value > 0 else 1  # Add 10% margin on top
        y_min = min_value * 1.1 if min_value < 0 else -0.1
        ax.set_ylim(y_min, y_max)
        
        # Set integer ticks for x-axis
        x_ticks = np.arange(0, len(array), max(1, len(array)//8))  # Show at most 8 ticks
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(tick)) for tick in x_ticks])
        
        # Set integer ticks for y-axis
        y_range = y_max - y_min
        y_interval = max(1, int(y_range // 5))  # Show about 5 ticks
        y_start = int(np.floor(y_min))
        y_end = int(np.ceil(y_max))
        y_ticks = np.arange(y_start, y_end + 1, y_interval)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(int(tick)) for tick in y_ticks])
        
        # Add title
        ax.set_title(title, fontsize=12, pad=5)
        
        # Update the canvas
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
            
        # Reset a(t) and TDL to initial state (zeros)
        net = input_output(self.iw, self.lw, self.b)
        
        # Generate animation frames by processing each input step by step
        self.animation_frames = []
        for i in range(self.n):
            # Get current state before processing this input
            frame = {
                'step': i,
                'input_value': self.p[i],
                'p_tdl_state': net.p_tdl.copy(),
                'a_tdl_state': net.a_tdl.copy(),
                'output_value': net.step(self.p[i]),
                'weights': self.iw.copy()
            }
            self.animation_frames.append(frame)
        
        self.current_step = 0
        self.total_steps = len(self.animation_frames)
        self.is_animating = True
        
        # Clear the table initially
        self.clear_animation_table()
        self.start_animation()
        
    def clear_animation_table(self):
        """Clear a(t) and TDL rows for animation"""
        for i in range(self.n):
            # Clear a(t) row
            item_a = QTableWidgetItem("")
            item_a.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_a.setFlags(item_a.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_a.setBackground(QtGui.QColor(240, 240, 240))
            self.table.setItem(1, i, item_a)
            
            # Clear p_TDL row
            item_p_tdl = QTableWidgetItem("")
            item_p_tdl.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_p_tdl.setFlags(item_p_tdl.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_p_tdl.setBackground(QtGui.QColor(240, 240, 240))
            self.table.setItem(2, i, item_p_tdl)
            
            # Clear a_TDL row
            item_a_tdl = QTableWidgetItem("")
            item_a_tdl.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_a_tdl.setFlags(item_a_tdl.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_a_tdl.setBackground(QtGui.QColor(240, 240, 240))
            self.table.setItem(3, i, item_a_tdl)
        
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
        
        # Update a(t) value for current step
        item_a = QTableWidgetItem(f"{frame['output_value']:.2f}")
        item_a.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        item_a.setFlags(item_a.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        item_a.setBackground(QtGui.QColor(240, 240, 240))
        self.table.setItem(1, self.current_step, item_a)
        
        # Update p_TDL values for current step
        p_tdl_text = "\n".join([f"{val:.2f}" for val in frame['p_tdl_state']])
        item_p_tdl = QTableWidgetItem(p_tdl_text)
        item_p_tdl.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        item_p_tdl.setFlags(item_p_tdl.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        item_p_tdl.setBackground(QtGui.QColor(240, 240, 240))
        self.table.setItem(2, self.current_step, item_p_tdl)
        
        # Update a_TDL values for current step
        a_tdl_text = f"{frame['a_tdl_state'][0]:.2f}" if len(frame['a_tdl_state']) > 0 else "0.00"
        item_a_tdl = QTableWidgetItem(a_tdl_text)
        item_a_tdl.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        item_a_tdl.setFlags(item_a_tdl.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        item_a_tdl.setBackground(QtGui.QColor(240, 240, 240))
        self.table.setItem(3, self.current_step, item_a_tdl)
        
        # Update plots with current state
        current_a = [self.animation_frames[j]['output_value'] if j <= self.current_step else 0 
                    for j in range(self.n)]
        
        min_value = min(min(self.p), min(current_a))
        max_value = max(max(self.p), max(current_a))
        
        # Highlight current step in both input and output plots
        self.plot_sequence(self.p, self.figure, self.canvas, r'Input Sequence $p(t)$', min_value, max_value, highlight_step=self.current_step)
        self.plot_sequence(current_a, self.figure2, self.canvas2, r'Output Sequence $a(t)$', min_value, max_value, highlight_step=self.current_step)
        
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
