"""
Self-Attention 3D Surface Demo
===============================

Two 3D surface plots show how the two elements of the selected attention
output (o(1) or o(2)) vary as a function of p(1), while p(2) is held
fixed and controlled by dragging its handle.
"""

from PyQt6 import QtWidgets, QtCore
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH


class SelfAttention3D(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(SelfAttention3D, self).__init__(w_ratio, h_ratio, dpi, main_menu=2)

        self.fill_chapter("Self-Attention 3D", 13,
                          "\nTwo 3D surfaces show how\no(n) varies as a function\nof p(1).\n\n"
                          "Drag p(2) in the 2D panel.\n\n"
                          "Select o(1) or o(2) with\nthe radio buttons.",
                          PACKAGE_PATH + "Chapters/2_D/Logo_Ch_2.svg", None, 2,
                          icon_move_left=120, description_coords=(535, 105, 450, 250))

        # State
        self.p2 = np.array([0.5, 1.0])
        self.dragging = False
        self.output_choice = 0  # 0 -> o(1), 1 -> o(2)
        self.GRID_N = 30
        self.GRID_LIM = 2.0

        gx = np.linspace(-self.GRID_LIM, self.GRID_LIM, self.GRID_N)
        gy = np.linspace(-self.GRID_LIM, self.GRID_LIM, self.GRID_N)
        self.GX, self.GY = np.meshgrid(gx, gy)

        # Two 3D plots for the surfaces
        self.make_plot(1, (2, 115, 265, 265))
        self.make_plot(2, (258, 115, 265, 265))

        # Small 2D plot for dragging p(2)
        self.make_plot(3, (10, 380, 330, 270))
        self.figure3.set_tight_layout(True)

        # Radio buttons to select o(1) or o(2)
        self.make_label("label_output", "Select output:", (380, 410, 150, 30))
        self.radio_o1 = QtWidgets.QRadioButton("o(1)", self)
        self.radio_o1.setGeometry(380 * self.w_ratio, 440 * self.h_ratio, 80 * self.w_ratio, 30 * self.h_ratio)
        self.radio_o1.setChecked(True)
        self.radio_o1.toggled.connect(self.on_radio)

        self.radio_o2 = QtWidgets.QRadioButton("o(2)", self)
        self.radio_o2.setGeometry(380 * self.w_ratio, 470 * self.h_ratio, 80 * self.w_ratio, 30 * self.h_ratio)
        self.radio_o2.toggled.connect(self.on_radio)

        # Sliders for W matrices
        self.make_slider("slider_wq", QtCore.Qt.Orientation.Horizontal, (-180, 180),
                         QtWidgets.QSlider.TickPosition.TicksBelow, 30, 0,
                         (self.x_chapter_usual, 380, self.w_chapter_slider, 50), self.on_slider,
                         "label_wq", "W^Q angle: 0", (self.x_chapter_usual + 20, 355, 200, 50))

        self.make_slider("slider_wk", QtCore.Qt.Orientation.Horizontal, (-180, 180),
                         QtWidgets.QSlider.TickPosition.TicksBelow, 30, 0,
                         (self.x_chapter_usual, 450, self.w_chapter_slider, 50), self.on_slider,
                         "label_wk", "W^K angle: 0", (self.x_chapter_usual + 20, 425, 200, 50))

        self.make_slider("slider_wv", QtCore.Qt.Orientation.Horizontal, (-180, 180),
                         QtWidgets.QSlider.TickPosition.TicksBelow, 30, 0,
                         (self.x_chapter_usual, 520, self.w_chapter_slider, 50), self.on_slider,
                         "label_wv", "W^V angle: 0", (self.x_chapter_usual + 20, 495, 200, 50))

        # Reset button
        self.make_button("btn_reset", "Reset", (self.x_chapter_button, 580, self.w_chapter_button, self.h_chapter_button),
                         self.on_reset)

        # Connect mouse events for p(2) dragging
        self.canvas3.mpl_connect('button_press_event', self.on_press)
        self.canvas3.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas3.mpl_connect('button_release_event', self.on_release)

        self.redraw()

    @staticmethod
    def softmax_columns(M):
        """Softmax over each column of M."""
        e = np.exp(M - M.max(axis=0, keepdims=True))
        return e / e.sum(axis=0, keepdims=True)

    @staticmethod
    def rot(deg):
        """2x2 rotation matrix for angle in degrees."""
        r = np.radians(deg)
        c, s = np.cos(r), np.sin(r)
        return np.array([[c, -s], [s, c]])

    def self_attention(self, P, WQ, WK, WV):
        """Compute self-attention on input matrix P (2xT)."""
        AQ = WQ @ P
        AK = WK @ P
        AV = WV @ P
        SK = AK.shape[0]
        NA = (AK.T @ AQ) / np.sqrt(SK)
        AA = self.softmax_columns(NA)
        AO = AV @ AA
        return dict(AQ=AQ, AK=AK, AV=AV, NA=NA, AA=AA, AO=AO)

    def compute_surface(self, grid_x, grid_y, p2, WQ, WK, WV, output_idx):
        """
        Sweep p(1) over the grid with p(2) held fixed.
        Return two surfaces for the two elements of o(output_idx+1).
        """
        rows, cols = grid_x.shape
        S0 = np.zeros((rows, cols))
        S1 = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                p1 = np.array([grid_x[i, j], grid_y[i, j]])
                P = np.column_stack([p1, p2])
                res = self.self_attention(P, WQ, WK, WV)
                out = res['AO'][:, output_idx]
                S0[i, j] = out[0]
                S1[i, j] = out[1]

        return S0, S1

    def redraw(self):
        """Recompute surfaces and redraw everything."""
        WQ = self.rot(self.slider_wq.value())
        WK = self.rot(self.slider_wk.value())
        WV = self.rot(self.slider_wv.value())

        oi = self.output_choice
        out_label = f'o({oi+1})'

        S0, S1 = self.compute_surface(self.GX, self.GY, self.p2, WQ, WK, WV, oi)

        # Left 3D: output element [0]
        self.figure.clf()
        ax3d_0 = self.figure.add_subplot(111, projection='3d')
        ax3d_0.plot_surface(self.GX, self.GY, S0, cmap='coolwarm', alpha=0.85, edgecolor='none')
        ax3d_0.set_xlabel('p(1)[0]', fontsize=8)
        ax3d_0.set_ylabel('p(1)[1]', fontsize=8)
        ax3d_0.set_zlabel(f'{out_label}[0]', fontsize=8)
        ax3d_0.set_title(f'{out_label}[0]', fontsize=10, fontweight='bold')
        self.figure.subplots_adjust(left=-0.1, right=0.9)
        self.canvas.draw()

        # Right 3D: output element [1]
        self.figure2.clf()
        ax3d_1 = self.figure2.add_subplot(111, projection='3d')
        ax3d_1.plot_surface(self.GX, self.GY, S1, cmap='coolwarm', alpha=0.85, edgecolor='none')
        ax3d_1.set_xlabel('p(1)[0]', fontsize=8)
        ax3d_1.set_ylabel('p(1)[1]', fontsize=8)
        ax3d_1.set_zlabel(f'{out_label}[1]', fontsize=8)
        ax3d_1.set_title(f'{out_label}[1]', fontsize=10, fontweight='bold')
        self.figure2.subplots_adjust(left=-0.1, right=0.9)
        self.canvas2.draw()

        # Small 2D axis: draggable p(2)
        self.figure3.clf()
        ax2d = self.figure3.add_subplot(111)
        lim = self.GRID_LIM
        ax2d.set_xlim(-lim, lim)
        ax2d.set_ylim(-lim, lim)
        ax2d.set_aspect('equal')
        ax2d.grid(True, alpha=0.3)
        ax2d.axhline(0, color='k', linewidth=0.5)
        ax2d.axvline(0, color='k', linewidth=0.5)
        ax2d.set_title('Drag to move p(2)', fontsize=10)
        ax2d.set_xlabel('p(2)[0]', fontsize=9)
        ax2d.set_ylabel('p(2)[1]', fontsize=9)

        # Arrow + handle
        ax2d.annotate('', xy=self.p2, xytext=[0, 0],
                      arrowprops=dict(arrowstyle='->', color='#b03020',
                                      lw=2.5, mutation_scale=15,
                                      shrinkA=0, shrinkB=0))
        ax2d.plot(self.p2[0], self.p2[1], 'o', color='#b03020', markersize=12,
                  markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        ax2d.text(self.p2[0] + 0.12, self.p2[1] + 0.12,
                  f'p(2)=[{self.p2[0]:.2f}, {self.p2[1]:.2f}]',
                  fontsize=8, color='#b03020', fontweight='bold')

        self.canvas3.draw()

    def on_radio(self):
        if self.radio_o1.isChecked():
            self.output_choice = 0
        else:
            self.output_choice = 1
        self.redraw()

    def on_slider(self):
        self.label_wq.setText(f"W^Q angle: {self.slider_wq.value()}")
        self.label_wk.setText(f"W^K angle: {self.slider_wk.value()}")
        self.label_wv.setText(f"W^V angle: {self.slider_wv.value()}")
        self.redraw()

    def on_reset(self):
        self.p2[:] = [0.5, 1.0]
        self.output_choice = 0
        self.radio_o1.setChecked(True)
        self.slider_wq.setValue(0)
        self.slider_wk.setValue(0)
        self.slider_wv.setValue(0)
        self.redraw()

    def on_press(self, event):
        """Start dragging if click is near p(2) tip."""
        if event.inaxes is None:
            return
        click = np.array([event.xdata, event.ydata])
        ax = self.figure3.axes[0] if self.figure3.axes else None
        if ax is None:
            return
        xlim = ax.get_xlim()
        grab = 0.12 * (xlim[1] - xlim[0])
        if np.linalg.norm(click - self.p2) < grab:
            self.dragging = True

    def on_motion(self, event):
        """Update p(2) to follow the mouse."""
        if not self.dragging or event.inaxes is None:
            return
        self.p2[:] = np.clip([event.xdata, event.ydata], -self.GRID_LIM, self.GRID_LIM)
        self.redraw()

    def on_release(self, event):
        """Stop dragging."""
        self.dragging = False
