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

        self.fill_chapter(f"Impulse Response", 11, "Adjust pole locations using\nsliders or by clicking on\nthe pole plot.\n\n"
                                                   "Click [Set Default] to\nrestore original values.",
                          PACKAGE_PATH + "Logo/Logo_Ch_11.svg", PACKAGE_PATH + "Figures/nndeep11_transferfunction_simple.svg", 2,
                          icon_move_left=-15, icon_move_up=0, description_coords=(535, 130, 450, 180),
                          icon_coords=(130, 100, 250, 100), icon_rescale=True)

        self.make_plot(1, (10, 320, 250, 250))  # Pole plot
        self.make_plot(2, (260, 320, 250, 250))  # Impulse response plot

        # Labels for pole locations
        self.make_label("label_pole1", "Pole 1: ", (180, 265, 230, 20), font_size=16)
        self.make_label("label_pole2", "Pole 2: ", (180, 295, 230, 20), font_size=16)

        # Instructions below the pole plot
        self.make_label("label_instructions",
                       "Click on a complex region of the pole plot to choose a complex conjugate pair.\nClick on the real axis for real poles (requires two clicks on the real axis).\n\n"
                       "Note: Clicking the pole plot updates only α₁ and α₂ (denominator coefficients).\nβ₁ (numerator gain) remains unchanged and can only be adjusted via its slider.",
                       (30, 555, 440, 120), font_size=14)

        # Interactive pole selection variables
        self.selected_poles = []
        self.click_margin = 0.1  # Margin for considering a click on the real axis
        self.temp_pole_marker = None  # Marker for first real pole click

        # Default values
        self.p_str = ['1', '0', '0', '0', '0', '0', '0', '0']

        # Numerator coefficient (beta_1) and denominator coefficients (alpha_1, alpha_2)
        self.beta1_default = '1'        # β₁: numerator gain coefficient
        self.alpha1_default = '1'       # α₁: first denominator coefficient
        self.alpha2_default = '-0.24'   # α₂: second denominator coefficient

        self.p = np.array(self.p_str, dtype=int)

        # Store coefficients as [beta_1, alpha_1, alpha_2] for convenience
        self.beta1 = float(self.beta1_default)
        self.alpha1 = float(self.alpha1_default)
        self.alpha2 = float(self.alpha2_default)
        
        # iw11 will be set based on beta_1 (lw11_coeff[0])
        # Will be updated in update_values()
        self.lw21 = np.array([1, 0])
        self.b1 = np.zeros(2)
        self.b2 = 0
        self.a_0 = [0, 0]

        self.n = len(self.p_str)
        self.n_outputs = self.n + 1  # Network outputs include initial condition

        self.initialize_table()
        
        # Three coefficient sliders for lw11
        slider_spacing = 70
        # slider_y_start = 390
        # self.x_chapter_usual = 520
        slider_y_start = 300
        self.x_chapter_usual = 520

        self.make_slider("coeff1_slider", QtCore.Qt.Orientation.Horizontal, (-100, 100), QtWidgets.QSlider.TickPosition.TicksBelow, 10,
                        int(self.beta1 * 100), (self.x_chapter_usual, slider_y_start, self.w_chapter_slider, 50),
                        self.slider_update, "label_coeff1", f"β₁: {self.beta1:.2f}", (self.x_chapter_usual+20, slider_y_start-25, self.w_chapter_slider, 50))

        self.make_slider("coeff2_slider", QtCore.Qt.Orientation.Horizontal, (-100, 100), QtWidgets.QSlider.TickPosition.TicksBelow, 10,
                        int(self.alpha1 * 100), (self.x_chapter_usual, slider_y_start + slider_spacing, self.w_chapter_slider, 50),
                        self.slider_update, "label_coeff2", f"α₁: {self.alpha1:.2f}", (self.x_chapter_usual+20, slider_y_start + slider_spacing-25, self.w_chapter_slider, 50))

        self.make_slider("coeff3_slider", QtCore.Qt.Orientation.Horizontal, (-100, 100), QtWidgets.QSlider.TickPosition.TicksBelow, 10,
                        int(self.alpha2 * 100), (self.x_chapter_usual, slider_y_start + 2*slider_spacing, self.w_chapter_slider, 50),
                        self.slider_update, "label_coeff3", f"α₂: {self.alpha2:.2f}", (self.x_chapter_usual+20, slider_y_start + 2*slider_spacing-25, self.w_chapter_slider, 50))
        
        # Set Default button
        self.make_button("default_button", "Set Default",
                        (self.x_chapter_usual+5, slider_y_start + 220, self.w_chapter_button, self.h_chapter_button),
                        self.set_default_values)

        self.set_default_values()

        # Connect mouse click event to pole plot
        self.canvas.mpl_connect('button_press_event', self.on_pole_click)
    
    def slider_update(self):
        """Update coefficients from slider values and refresh display"""
        self.beta1 = self.get_slider_value_and_update(self.coeff1_slider, self.label_coeff1, 1/100, 2)
        self.alpha1 = self.get_slider_value_and_update(self.coeff2_slider, self.label_coeff2, 1/100, 2)
        self.alpha2 = self.get_slider_value_and_update(self.coeff3_slider, self.label_coeff3, 1/100, 2)

        self.update_values()

    def initialize_table(self):
        """Create and setup the sequence table"""
        self.table = QTableWidget(1, self.n_outputs, self)
        self.table.setGeometry(20, 190, 480, 63)

        self.table.setVerticalHeaderLabels(['a(t)'])

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

        self.table.show()

    def update_table(self):
        """Update table with current sequence data"""
        for i in range(self.n_outputs):
            item_a = QTableWidgetItem(f"{self.a[i]:.2f}")
            item_a.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_a.setFlags(item_a.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item_a.setBackground(QtGui.QColor(240, 240, 240))
            self.table.setItem(0, i, item_a)

    def set_default_values(self):
        """Reset all values to defaults"""
        # Reset P values
        self.p = np.array(self.p_str, dtype=int)

        # Reset coefficients to defaults
        self.beta1 = float(self.beta1_default)
        self.alpha1 = float(self.alpha1_default)
        self.alpha2 = float(self.alpha2_default)

        # Temporarily disconnect sliders to prevent premature updates
        self.coeff1_slider.valueChanged.disconnect()
        self.coeff2_slider.valueChanged.disconnect()
        self.coeff3_slider.valueChanged.disconnect()

        self.coeff1_slider.setValue(int(self.beta1 * 100))
        self.coeff2_slider.setValue(int(self.alpha1 * 100))
        self.coeff3_slider.setValue(int(self.alpha2 * 100))

        # Update labels
        self.label_coeff1.setText(f"β₁: {self.beta1:.2f}")
        self.label_coeff2.setText(f"α₁: {self.alpha1:.2f}")
        self.label_coeff3.setText(f"α₂: {self.alpha2:.2f}")

        # Reconnect sliders
        self.coeff1_slider.valueChanged.connect(self.slider_update)
        self.coeff2_slider.valueChanged.connect(self.slider_update)
        self.coeff3_slider.valueChanged.connect(self.slider_update)

        self.update_values()

    def update_values(self):
        """Update network calculations and plots"""
        try:
            # Build iw11 with beta_1 (numerator coefficient)
            iw11 = np.array([self.beta1, 0])

            # Build lw11 matrix from alpha coefficients (Eq. 11.40)
            # First column contains alpha terms, second column is [1, 0]
            lw11 = np.array([[self.alpha1, self.alpha2],
                            [1, 0]]).transpose()

            # print(iw11, lw11, self.b1, self.lw21, self.b2, self.a_0, self.p)
            # Create network and process sequence
            net = state_space(iw11, lw11, self.b1, self.lw21, self.b2, self.a_0)
            self.a = net.process(self.p)

            self.graph()
            self.update_table()

        except Exception as e:
            # Handle any calculation errors
            print(f"Error in calculations: {e}")

    def graph(self):
        """Update both plots"""
        self.plot_poles()
        self.plot_impulse_response()

    def plot_poles(self):
        """Plot poles in the complex plane"""
        self.figure.clf()
        ax = self.figure.add_subplot(1, 1, 1)

        # Plot unit circle for reference
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, linewidth=1)

        # Plot axes
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3)

        # If we're in the middle of selecting real poles, only show the temporary marker
        if len(self.selected_poles) == 1:
            self.temp_pole_marker, = ax.plot(self.selected_poles[0], 0, 'rx', markersize=8, markeredgewidth=2)
            # Update labels to show we're waiting for second pole
            self.label_pole1.setText(f"Pole 1: {self.selected_poles[0]:.4f}")
            self.label_pole2.setText("Pole 2: Click to select")
        else:
            # Calculate poles from denominator coefficients
            # State-space characteristic equation: z^2 - alpha1*z - alpha2 = 0
            # Beta_1 is in the numerator, not denominator
            denominator = [1, -self.alpha1, -self.alpha2]
            poles = np.roots(denominator)

            # Update pole location labels
            self.update_pole_labels(poles)

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

    def plot_impulse_response(self):
        """Plot impulse response"""
        self.figure2.clf()
        ax = self.figure2.add_subplot(1, 1, 1)

        t = np.arange(len(self.a))

        # Create stem plot
        markerline, stemlines, baseline = ax.stem(t, self.a)

        # Set stem formatting
        plt.setp(markerline, 'color', 'blue', 'markersize', 4)
        plt.setp(stemlines, 'color', 'blue', 'linewidth', 2)

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

        ax.set_title('Impulse Response a(t)', fontsize=12, pad=5)
        # ax.set_xlabel('Time Step')
        # ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)

        # Adjust subplot margins to give more space for y-axis labels
        self.figure2.subplots_adjust(left=0.2)

        self.canvas2.draw()

    def update_pole_labels(self, poles):
        """Update the pole location labels with numerical values"""
        if len(poles) >= 2:
            pole1, pole2 = poles[0], poles[1]

            # Format pole 1
            if abs(pole1.imag) < 1e-10:  # Real pole
                pole1_str = f"Pole 1: {pole1.real:.4f}"
            else:
                pole1_str = f"Pole 1: {pole1.real:.4f} {'+' if pole1.imag >= 0 else ''}{pole1.imag:.4f}j"

            # Format pole 2
            if abs(pole2.imag) < 1e-10:  # Real pole
                pole2_str = f"Pole 2: {pole2.real:.4f}"
            else:
                pole2_str = f"Pole 2: {pole2.real:.4f} {'+' if pole2.imag >= 0 else ''}{pole2.imag:.4f}j"

            self.label_pole1.setText(pole1_str)
            self.label_pole2.setText(pole2_str)

    def on_pole_click(self, event):
        """Handle mouse clicks on the pole plot for interactive pole selection"""
        # Only process clicks on the first plot (pole plot)
        if event.inaxes != self.figure.gca():
            return

        if event.xdata is None or event.ydata is None:
            return

        # Check if click is within the unit circle region
        if abs(event.xdata) > 1.2 or abs(event.ydata) > 1.2:
            return

        # Determine if click is on or near the real axis
        if abs(event.ydata) < self.click_margin:
            # Real pole - add it to selected poles
            self.selected_poles.append(event.xdata)

            # If we have one real pole, show it
            if len(self.selected_poles) == 1:
                self.graph()  # Redraw to show the first pole marker
            # If we have two real poles, update coefficients
            elif len(self.selected_poles) == 2:
                poles = np.array(self.selected_poles)
                self.update_coefficients_from_poles(poles)
                self.selected_poles = []
        else:
            # Complex pole - automatically create conjugate pair
            # Clear any previously selected real pole
            self.selected_poles = []
            pole = complex(event.xdata, event.ydata)
            poles = np.array([pole, np.conj(pole)])
            self.update_coefficients_from_poles(poles)

    def update_coefficients_from_poles(self, poles):
        """Update the coefficient sliders based on selected poles"""
        # Get denominator coefficients from poles using np.poly
        # np.poly gives (z-p1)(z-p2) = z^2 - (p1+p2)*z + p1*p2 = [1, -(p1+p2), p1*p2]
        # For state-space form z^2 - alpha1*z - alpha2 = 0, we need:
        # alpha1 = p1+p2 = -denominator[1]
        # alpha2 = -p1*p2 = -denominator[2]
        denominator = np.poly(poles).real  # Take real part to handle numerical errors

        # Update only alpha1 and alpha2, keep beta_1 unchanged
        if len(denominator) == 3:
            # Keep beta_1 unchanged
            # Update alpha_1 and alpha_2 to match state-space characteristic equation
            self.alpha1 = -denominator[1]
            self.alpha2 = -denominator[2]

            # Temporarily disconnect sliders to prevent recursive updates
            self.coeff2_slider.valueChanged.disconnect()
            self.coeff3_slider.valueChanged.disconnect()

            # Update slider values for alpha1 and alpha2 (clamp to slider range -1 to 1)
            self.coeff2_slider.setValue(int(np.clip(self.alpha1, -1, 1) * 100))
            self.coeff3_slider.setValue(int(np.clip(self.alpha2, -1, 1) * 100))

            # Update labels (beta_1 unchanged)
            self.label_coeff2.setText(f"α₁: {self.alpha1:.2f}")
            self.label_coeff3.setText(f"α₂: {self.alpha2:.2f}")

            # Reconnect sliders
            self.coeff2_slider.valueChanged.connect(self.slider_update)
            self.coeff3_slider.valueChanged.connect(self.slider_update)

            # Update the display
            self.update_values()
