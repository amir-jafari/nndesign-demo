"""
Self-Attention Demo with 2D Embedding Vectors
==============================================

Interactive visualization of the self-attention operation from Chapter 13.
Two input tokens are represented as 2D vectors. The query, key, and value
weight matrices (W^Q, W^K, W^V) can be adjusted with sliders.

Click and drag the tips of p(1) or p(2) to move the input vectors.
"""

from PyQt6 import QtWidgets, QtCore
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH


class SelfAttention2D(NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(SelfAttention2D, self).__init__(w_ratio, h_ratio, dpi, main_menu=2)

        self.fill_chapter("Self-Attention 2D", 13,
                          "\nClick and drag the tips of\np(1) or p(2) to move the\ninput vectors.\n\n"
                          "Adjust W^Q, W^K, W^V\nrotation angles with sliders.\n\n"
                          "The dashed arrows show\nattention output vectors.",
                          PACKAGE_PATH + "Chapters/2_D/Logo_Ch_2.svg", None, 2,
                          icon_move_left=120, description_coords=(535, 105, 450, 250))

        # State
        self.p1 = np.array([1.0, 0.3])
        self.p2 = np.array([0.3, 1.0])
        self.dragging = None
        self.GRAB_RADIUS = 0.15

        # Main vector plot
        self.make_plot(1, (15, 130, 500, 400))
        self.figure.set_tight_layout(True)

        # Text panel for numerical results
        self.make_plot(2, (15, 530, 500, 120))
        self.figure2.set_tight_layout(True)

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

        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

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

    def draw_vector(self, ax, origin, vec, color, label, linestyle='-', linewidth=2.2, alpha=1.0):
        """Draw a vector as an arrow from origin."""
        ax.annotate('', xy=origin + vec, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=linewidth, linestyle=linestyle,
                                    mutation_scale=15, shrinkA=0, shrinkB=0,
                                    alpha=alpha))
        offset = vec / (np.linalg.norm(vec) + 1e-9) * 0.12
        ax.text(origin[0] + vec[0] + offset[0],
                origin[1] + vec[1] + offset[1],
                label, fontsize=11, fontweight='bold', color=color,
                ha='center', va='center', alpha=alpha)

    def draw_handle(self, ax, pos, color):
        """Draw a draggable handle (circle) at a vector tip."""
        ax.plot(pos[0], pos[1], 'o', color=color, markersize=10,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)

    def redraw(self):
        """Recompute attention and redraw everything."""
        P = np.column_stack([self.p1, self.p2])

        WQ = self.rot(self.slider_wq.value())
        WK = self.rot(self.slider_wk.value())
        WV = self.rot(self.slider_wv.value())

        res = self.self_attention(P, WQ, WK, WV)
        AO = res['AO']
        AA = res['AA']
        o1, o2 = AO[:, 0], AO[:, 1]

        # Vector plot
        self.figure.clf()
        ax = self.figure.add_subplot(111)

        lim = max(np.max(np.abs(P)), np.max(np.abs(AO)), 1.2) + 0.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_xlabel('Dimension 1', fontsize=11)
        ax.set_ylabel('Dimension 2', fontsize=11)
        ax.set_title('Drag circles to move p(1) and p(2)', fontsize=12)

        origin = np.array([0.0, 0.0])

        # Input vectors (solid)
        self.draw_vector(ax, origin, self.p1, '#2060b0', 'p(1)', linewidth=2.5)
        self.draw_vector(ax, origin, self.p2, '#b03020', 'p(2)', linewidth=2.5)

        # Draggable handles at the tips
        self.draw_handle(ax, self.p1, '#2060b0')
        self.draw_handle(ax, self.p2, '#b03020')

        # Output vectors (dashed)
        self.draw_vector(ax, origin, o1, '#2060b0', 'o(1)', linestyle='--', linewidth=2.0, alpha=0.75)
        self.draw_vector(ax, origin, o2, '#b03020', 'o(2)', linestyle='--', linewidth=2.0, alpha=0.75)

        # Annotation near each output tip showing the weight decomposition
        for (out_vec, col_idx) in [(o1, 0), (o2, 1)]:
            w1, w2 = AA[0, col_idx], AA[1, col_idx]
            color = '#2060b0' if col_idx == 0 else '#b03020'
            ax.text(out_vec[0] - 0.05, out_vec[1] - 0.18,
                    f'{w1:.2f}*p(1) + {w2:.2f}*p(2)',
                    fontsize=8, color=color, alpha=0.7,
                    ha='center', bbox=dict(boxstyle='round,pad=0.2',
                                           facecolor='white', alpha=0.8))

        self.canvas.draw()

        # Text panel
        self.figure2.clf()
        ax_text = self.figure2.add_subplot(111)
        ax_text.axis('off')

        def fmt_vec(v):
            return f'[{v[0]:6.3f}, {v[1]:6.3f}]'

        lines = []
        lines.append(f'Inputs:  p(1)={fmt_vec(self.p1)}  p(2)={fmt_vec(self.p2)}')
        lines.append(f'Outputs: o(1)={fmt_vec(o1)}  o(2)={fmt_vec(o2)}')
        lines.append(f'Attention Weights: A^A = [[{AA[0,0]:.3f}, {AA[0,1]:.3f}], [{AA[1,0]:.3f}, {AA[1,1]:.3f}]]')

        text = '\n'.join(lines)
        ax_text.text(0.02, 0.5, text, transform=ax_text.transAxes,
                     fontsize=9, fontfamily='monospace',
                     verticalalignment='center')

        self.canvas2.draw()

    def on_slider(self):
        self.label_wq.setText(f"W^Q angle: {self.slider_wq.value()}")
        self.label_wk.setText(f"W^K angle: {self.slider_wk.value()}")
        self.label_wv.setText(f"W^V angle: {self.slider_wv.value()}")
        self.redraw()

    def on_reset(self):
        self.p1[:] = [1.0, 0.3]
        self.p2[:] = [0.3, 1.0]
        self.slider_wq.setValue(0)
        self.slider_wk.setValue(0)
        self.slider_wv.setValue(0)
        self.redraw()

    def on_press(self, event):
        """Start dragging if click is near a vector tip."""
        if event.inaxes is None:
            return
        click = np.array([event.xdata, event.ydata])

        d1 = np.linalg.norm(click - self.p1)
        d2 = np.linalg.norm(click - self.p2)

        ax = self.figure.axes[0] if self.figure.axes else None
        if ax is None:
            return
        xlim = ax.get_xlim()
        grab = self.GRAB_RADIUS * (xlim[1] - xlim[0])

        if d1 < grab and d1 <= d2:
            self.dragging = 'p1'
        elif d2 < grab:
            self.dragging = 'p2'
        else:
            self.dragging = None

    def on_motion(self, event):
        """Update the dragged vector tip to follow the mouse."""
        if self.dragging is None or event.inaxes is None:
            return
        new_pos = np.array([event.xdata, event.ydata])
        if self.dragging == 'p1':
            self.p1[:] = new_pos
        else:
            self.p2[:] = new_pos
        self.redraw()

    def on_release(self, event):
        """Stop dragging."""
        self.dragging = None
