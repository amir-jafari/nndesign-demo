"""
Chapter 7 — Forward Propagation demos: shared math and helpers.

Pure functions and small UI helpers reused by both the LFFN and GLDDN
demos. Kept separate so the two demo windows stay small and don't duplicate
the numerics.

  LFFN  : Layered Feed-Forward Network   (static, no delays)
  GLDDN : General LDDN                    (arbitrary cross-layer connections)

Weight functions  : dotprod · dist · negdist · normprod
Net-input funcs   : netsum  · netprod
Activations       : Sigmoid · Tanh · Linear · ReLU · Hard Limit · Softplus
"""

import numpy as np

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QSlider, QFrame,
    QCheckBox, QPushButton,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


# ══════════════════════════════════════════════════════════════
# Activation functions
# ══════════════════════════════════════════════════════════════
def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

ACT_FUNCS = {
    'Sigmoid':    _sig,
    'Tanh':       np.tanh,
    'Linear':     lambda x: np.asarray(x, float),
    'ReLU':       lambda x: np.maximum(0.0, x),
    'Hard Limit': lambda x: (np.asarray(x, float) >= 0).astype(float),
    'Softplus':   lambda x: np.log1p(np.exp(np.clip(x, -500, 80))),
}
ACT_NAMES = list(ACT_FUNCS.keys())

# ══════════════════════════════════════════════════════════════
# Weight functions   wf(W, p)  →  z   (S-vector)
#
#   W : S × R  weight matrix
#   p : R-vector  (previous layer activation or input)
#   z : S-vector  (output before net-input function)
# ══════════════════════════════════════════════════════════════
def _wf_dotprod(W, p):
    """zᵢ = Wᵢ · p   (standard weighted sum per neuron)"""
    return W @ np.asarray(p, float)

def _wf_dist(W, p):
    """zᵢ = ‖Wᵢ − p‖₂   (Euclidean distance, e.g. RBF networks)"""
    p = np.asarray(p, float)
    return np.array([np.linalg.norm(W[i] - p) for i in range(W.shape[0])])

def _wf_negdist(W, p):
    """zᵢ = −‖Wᵢ − p‖₂   (negative distance)"""
    return -_wf_dist(W, p)

def _wf_normprod(W, p):
    """zᵢ = (Wᵢ · p) / (‖Wᵢ‖ · ‖p‖)   (cosine similarity)"""
    p = np.asarray(p, float)
    pn = np.linalg.norm(p) + 1e-12
    return np.array(
        [(W[i] @ p) / (np.linalg.norm(W[i]) * pn + 1e-12)
         for i in range(W.shape[0])]
    )

WF_FN = {
    'dotprod':  _wf_dotprod,
    'dist':     _wf_dist,
    'negdist':  _wf_negdist,
    'normprod': _wf_normprod,
}
WF_FORMULA = {
    'dotprod':  r'$z_i = W_i \cdot a^{m-1}$',
    'dist':     r'$z_i = \|W_i - a^{m-1}\|$',
    'negdist':  r'$z_i = -\|W_i - a^{m-1}\|$',
    'normprod': r'$z_i = \frac{W_i \cdot a^{m-1}}{\|W_i\|\,\|a^{m-1}\|}$',
}
WF_NAMES = list(WF_FN.keys())

# ══════════════════════════════════════════════════════════════
# Net-input functions   nif(z, b)  →  n   (S-vector)
# ══════════════════════════════════════════════════════════════
def _nif_netsum(z, b):
    """n = z + b   (standard; b is additive bias, default 0)"""
    return np.asarray(z, float) + np.asarray(b, float)

def _nif_netprod(z, b):
    """n = z ⊙ b   (element-wise product; b is scale, default 1)"""
    return np.asarray(z, float) * np.asarray(b, float)

NIF_FN = {
    'netsum':  _nif_netsum,
    'netprod': _nif_netprod,
}
NIF_FORMULA = {
    'netsum':  r'$n = z + b$',
    'netprod': r'$n = z \odot b$',
}
NIF_NAMES = list(NIF_FN.keys())

# ══════════════════════════════════════════════════════════════
# Colours
# ══════════════════════════════════════════════════════════════
LAYER_COLORS = ['#FFB347', '#87CEEB', '#90EE90', '#DDA0DD', '#F08080', '#B0E0E6', '#FAFAD2']
SIG_COLORS   = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# ══════════════════════════════════════════════════════════════
# Layout (base 700 × 680 window coordinates)
#
# These demos need a wider control panel than the standard split allows, so
# the vertical divider is moved left of the default 520. The header band of
# the right panel puts the description (left) and chapter logo (right) on one
# row, leaving the rest of the panel height to the controls.
# ══════════════════════════════════════════════════════════════
DIVIDER_X  = 400                     # x of the blue vertical line
PLOT_RECT  = (5, 135, 391, 535)      # figure, up to (not touching) the divider
PANEL_RECT = (403, 135, 294, 505)    # control panel, below the header band

# Control-panel fonts. Group boxes propagate their font to every child widget,
# so setting the group font sizes all the labels/combos/sliders inside it.
def _grp_font():
    return QFont('Arial', 12, QFont.Weight.Bold)


# ══════════════════════════════════════════════════════════════
# LFFN  —  Layered Feed-Forward Network
#
#   a^0 = p
#   For m = 1 … M:
#       z^m  = wf^m  ( W^m,  a^{m-1} )    weight function
#       n^m  = nif^m ( z^m,  b^m     )    net-input function
#       a^m  = f^m   ( n^m            )    activation
# ══════════════════════════════════════════════════════════════
class LFFN:
    def __init__(self, layer_sizes, act_names, wf_names, nif_names):
        self.layer_sizes = layer_sizes
        self.act_names   = act_names
        self.wf_names    = wf_names
        self.nif_names   = nif_names
        self.W = []; self.b = []
        self.a = []; self.z_vals = []; self.n_vals = []

    def init_weights(self, R, seed=42):
        np.random.seed(seed)
        sizes = [R] + self.layer_sizes
        self.W, self.b = [], []
        for i in range(1, len(sizes)):
            S_out, S_in = sizes[i], sizes[i - 1]
            wf = self.wf_names[i - 1]
            if wf in ('dist', 'negdist'):
                W = np.random.randn(S_out, S_in) * 0.5
            elif wf == 'normprod':
                W = np.random.randn(S_out, S_in)
                W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
            else:
                W = np.random.randn(S_out, S_in) * np.sqrt(2.0 / max(S_in, 1))
            self.W.append(W)
            # netprod uses multiplicative bias (default 1), netsum additive (default 0)
            self.b.append(
                np.ones(S_out) if self.nif_names[i - 1] == 'netprod'
                else np.zeros(S_out)
            )

    def forward(self, p):
        self.a      = [np.asarray(p, float).ravel()]
        self.z_vals = [None]
        self.n_vals = [None]
        for m in range(len(self.W)):
            z = WF_FN[self.wf_names[m]](self.W[m], self.a[-1])
            n = NIF_FN[self.nif_names[m]](z, self.b[m])
            a = np.asarray(ACT_FUNCS[self.act_names[m]](n), float).ravel()
            self.z_vals.append(z)
            self.n_vals.append(n)
            self.a.append(a)
        return self.a[-1]


# ══════════════════════════════════════════════════════════════
# Scalar helpers (for the dynamic GLDDN network with scalar taps)
# ══════════════════════════════════════════════════════════════
def _scalar_wf(wf_name, w, x):
    """Apply weight function for a single scalar (w, x) pair."""
    if   wf_name == 'dotprod':  return w * x
    elif wf_name == 'dist':     return abs(w - x)
    elif wf_name == 'negdist':  return -abs(w - x)
    elif wf_name == 'normprod':
        denom = (abs(w) * abs(x)) + 1e-12
        return (w * x) / denom
    return w * x

def _scalar_nif(nif_name, z, b):
    """Apply net-input function for scalar z, b."""
    if   nif_name == 'netsum':  return z + b
    elif nif_name == 'netprod': return z * b
    return z + b


# ══════════════════════════════════════════════════════════════
# Shared Qt helpers
# ══════════════════════════════════════════════════════════════
def _separator():
    """Thin horizontal line widget."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line

def _make_slider(lo, hi, val, scale=100):
    sl = QSlider(Qt.Orientation.Horizontal)
    sl.setRange(int(lo * scale), int(hi * scale))
    sl.setValue(int(val * scale))
    return sl

def _slider_row(label, lo, hi, val, scale, parent_vl, label_width=120, tooltip=None):
    """Add a labeled slider row to parent_vl; returns (slider, val_label).

    label_width lets narrow (half-width) panels use a compact label, and
    tooltip shows the full description on hover of the label/slider.
    """
    rw = QWidget()
    rh = QHBoxLayout(rw)
    rh.setContentsMargins(0, 1, 0, 1)
    lbl = QLabel(label)
    lbl.setFixedWidth(label_width)
    sl  = _make_slider(lo, hi, val, scale)
    vl  = QLabel(f'{val:.2f}')
    vl.setFixedWidth(46)
    if tooltip:
        lbl.setToolTip(tooltip); sl.setToolTip(tooltip)
    rh.addWidget(lbl); rh.addWidget(sl); rh.addWidget(vl)
    parent_vl.addWidget(rw)
    return sl, vl

def _signal_panel(parent_vl):
    """Input signal group; returns dict of widgets."""
    grp = QGroupBox("Input Signal  p(t)")
    grp.setFont(_grp_font())
    vl  = QVBoxLayout(grp)
    vl.setSpacing(2)

    def _row(lbl_txt, widget):
        rw = QWidget(); rh = QHBoxLayout(rw); rh.setContentsMargins(0, 0, 0, 0)
        rh.addWidget(QLabel(lbl_txt)); rh.addWidget(widget)
        vl.addWidget(rw); return widget

    sig_cb = QComboBox()
    sig_cb.addItems(['Square Wave', 'Sine Wave', 'Step', 'Ramp', 'Random'])
    _row("Signal type:", sig_cb)
    amp_sp  = QDoubleSpinBox(); amp_sp.setRange(0.1, 5.0); amp_sp.setValue(1.0); amp_sp.setSingleStep(0.1)
    per_sp  = QDoubleSpinBox(); per_sp.setRange(2.0, 100.0); per_sp.setValue(10.0); per_sp.setSingleStep(1.0)
    T_sp    = QSpinBox(); T_sp.setRange(20, 400); T_sp.setValue(80)
    _row("Amplitude:", amp_sp)
    _row("Period (steps):", per_sp)
    _row("Time steps T:", T_sp)
    parent_vl.addWidget(grp)
    return dict(sig=sig_cb, amp=amp_sp, period=per_sp, T=T_sp)

def _build_signal(widgets):
    T   = widgets['T'].value()
    A   = widgets['amp'].value()
    per = widgets['period'].value()
    t   = np.arange(T)
    sig = widgets['sig'].currentText()
    if   sig == 'Square Wave': p = A * np.where((t % per) < per / 2, 1.0, -1.0)
    elif sig == 'Sine Wave':   p = A * np.sin(2 * np.pi * t / per)
    elif sig == 'Step':        p = A * (t >= T // 4).astype(float)
    elif sig == 'Ramp':        p = A * np.clip(t / max(T // 2, 1), 0.0, 1.0)
    else:
        np.random.seed(0); p = A * np.random.randn(T)
    return t, p


def _place_header(demo, description):
    """Lay out the control-panel header band: description text on the left and
    the chapter logo on the right, sharing one row at the top of the panel."""
    demo.make_label("label_desc", description, (425, 30, 148, 100))
    demo.icon1.setGeometry(int(575 * demo.w_ratio), int(30 * demo.h_ratio),
                           int(120 * demo.w_ratio), int(100 * demo.h_ratio))


def _anim_interval_ms(slider_value):
    """Map the speed slider (100–1000) to a frame interval in ms.

    The 0.7 factor speeds up the whole range: 700 ms (slowest) → 385 ms at the
    centred default → 70 ms (fastest)."""
    return int(0.7 * (1100 - slider_value))


def _animation_group(mixin):
    """Build the shared Animation control group (checkbox / speed / play-pause).

    Assigns ``checkbox_animation``, ``slider_animation`` and ``btn_play_pause``
    on ``mixin`` and wires them to the mixin's handlers. Returns the QGroupBox.
    """
    grp = QGroupBox("Animation")
    grp.setFont(_grp_font())
    vl = QVBoxLayout(grp)
    vl.setSpacing(2)

    mixin.checkbox_animation = QCheckBox("Enable step-by-step animation")
    mixin.checkbox_animation.setChecked(False)
    mixin.checkbox_animation.stateChanged.connect(mixin.toggle_animation)
    vl.addWidget(mixin.checkbox_animation)

    sr = QWidget(); srh = QHBoxLayout(sr); srh.setContentsMargins(0, 0, 0, 0)
    srh.addWidget(QLabel("Speed:"))
    mixin.slider_animation = QSlider(Qt.Orientation.Horizontal)
    mixin.slider_animation.setRange(100, 1000)
    mixin.slider_animation.setValue(550)   # centred; interval via _anim_interval_ms
    mixin.slider_animation.setToolTip("Slide right to speed up the animation")
    mixin.slider_animation.valueChanged.connect(mixin.change_animation_speed)
    srh.addWidget(mixin.slider_animation)
    vl.addWidget(sr)

    mixin.btn_play_pause = QPushButton("Pause")
    mixin.btn_play_pause.setToolTip("Pause / resume a running animation")
    mixin.btn_play_pause.clicked.connect(mixin.toggle_play_pause)
    vl.addWidget(mixin.btn_play_pause)
    return grp


# ══════════════════════════════════════════════════════════════
# Step-by-step drawing animation
#
#   The numerical result is always computed instantly by ._compute(); only the
#   *drawing* is animated (mirrors book2/chapter8/Convolution_networks_ani.py).
#
#   Subclasses implement four hooks:
#       _compute()             → result object (or None to skip drawing)
#       _total_frames(result)  → number of animation steps
#       _draw_static(result)   → draw the full result at once
#       _draw_frame(result, k) → draw the result revealed up to step k
#
#   Requires the widgets checkbox_animation / slider_animation / btn_play_pause
#   (created via _animation_group above).
# ══════════════════════════════════════════════════════════════
class Ch7AnimationMixin:
    def init_animation_state(self):
        self.animation_speed = _anim_interval_ms(550)   # matches the slider default
        self.is_animating    = False
        self.animation_paused = False
        self.animation       = None         # QTimer (auto-stopped by NNDLayout.closeEvent)
        self.current_step    = 0
        self.total_steps     = 0
        self._anim_result    = None

    @property
    def animation_enabled(self):
        return self.checkbox_animation.checkState().value == 2

    # ── main entry: recompute instantly, then draw (instant or animated) ──
    def _run(self):
        result = self._compute()
        if result is None:
            return
        self._anim_result = result
        if self.animation_enabled:
            self._start_animation(result)
        else:
            self._stop_animation()
            self._draw_static(result)

    def _start_animation(self, result):
        self._stop_animation()
        self.total_steps  = self._total_frames(result)
        self.current_step = 0
        self.is_animating = True
        self.animation_paused = False
        self.btn_play_pause.setText("Pause")
        self.animation = QtCore.QTimer()
        self.animation.timeout.connect(self._animate_next_step)
        self.animation.start(self.animation_speed)

    def _animate_next_step(self):
        if self.current_step >= self.total_steps:
            if self.animation is not None:
                self.animation.stop()
            self.is_animating = False
            self._draw_static(self._anim_result)   # clean final frame
            return
        self._draw_frame(self._anim_result, self.current_step)
        self.current_step += 1

    def _stop_animation(self):
        if self.animation is not None:
            self.animation.stop()
            self.animation = None
        self.is_animating = False
        self.animation_paused = False

    # ── animation control widgets ──
    def toggle_animation(self):
        if self.animation_enabled:
            if self._anim_result is not None:
                self._start_animation(self._anim_result)
            else:
                self._run()
        else:
            self._stop_animation()
            if self._anim_result is not None:
                self._draw_static(self._anim_result)

    def change_animation_speed(self):
        self.animation_speed = _anim_interval_ms(self.slider_animation.value())
        if self.animation is not None:
            self.animation.setInterval(self.animation_speed)

    def toggle_play_pause(self):
        if not self.is_animating:
            # nothing running: (re)start from the top if animation is enabled
            if self.animation_enabled and self._anim_result is not None:
                self._start_animation(self._anim_result)
            return
        if self.animation_paused:
            self.animation_paused = False
            self.btn_play_pause.setText("Pause")
            if self.animation is not None:
                self.animation.start(self.animation_speed)
        else:
            self.animation_paused = True
            self.btn_play_pause.setText("Play")
            if self.animation is not None:
                self.animation.stop()

    # ── live-update handlers ──
    def _on_change(self):
        """Discrete control changed → instant redraw, or (re)start animation."""
        self._run()

    def _on_slider_drag(self):
        # While dragging: update instantly only when NOT animating (avoids the
        # animation restarting from step 0 on every tick).
        if not self.animation_enabled:
            self._run()

    def _on_slider_release(self):
        # On release: kick off the step-by-step animation once.
        if self.animation_enabled:
            self._run()
