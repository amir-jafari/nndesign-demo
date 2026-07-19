"""
Chapter 7 — General Notation: Deep Learning
Forward Propagation Demo  ·  LFFN | LDDN | GLDDN

Networks
--------
  LFFN  : Layered Feed-Forward Network   (static, no delays)
  LDDN  : Layered Delayed Dynamic Network (tapped delay lines)
  GLDDN : General LDDN                    (arbitrary cross-layer connections)

Weight functions  : dotprod · dist · negdist · normprod
Net-input funcs   : netsum  · netprod
Activations       : Sigmoid · Tanh · Linear · ReLU · Hard Limit · Softplus

Usage: python ch7_fp_demo.py
"""

import sys
import os

# macOS: ensure Qt can find its platform plugin (cocoa) when installed via pip
if sys.platform == 'darwin':
    try:
        import PyQt6 as _pyqt6
        _plugins = os.path.join(os.path.dirname(_pyqt6.__file__), 'Qt6', 'plugins', 'platforms')
        os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', _plugins)
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QPushButton, QGroupBox, QScrollArea, QSplitter, QSlider,
    QSizePolicy,
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
# Shared helpers
# ══════════════════════════════════════════════════════════════
def _make_slider(lo, hi, val, scale=100):
    sl = QSlider(Qt.Orientation.Horizontal)
    sl.setRange(int(lo * scale), int(hi * scale))
    sl.setValue(int(val * scale))
    return sl

def _slider_row(label, lo, hi, val, scale, parent_vl):
    """Add a labeled slider row to parent_vl; returns (slider, val_label)."""
    rw = QWidget()
    rh = QHBoxLayout(rw)
    rh.setContentsMargins(0, 2, 0, 2)
    lbl = QLabel(label)
    lbl.setFixedWidth(120)
    sl  = _make_slider(lo, hi, val, scale)
    vl  = QLabel(f'{val:.2f}')
    vl.setFixedWidth(38)
    rh.addWidget(lbl); rh.addWidget(sl); rh.addWidget(vl)
    parent_vl.addWidget(rw)
    return sl, vl

def _signal_panel(parent_vl):
    """Input signal group; returns dict of widgets."""
    grp = QGroupBox("Input Signal  p(t)")
    grp.setFont(QFont('Arial', 9, QFont.Weight.Bold))
    vl  = QVBoxLayout(grp)

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


# ══════════════════════════════════════════════════════════════
# Tab 1 — LFFN   (Layered Feed-Forward Network)
# ══════════════════════════════════════════════════════════════
class LFFNTab(QWidget):
    def __init__(self):
        super().__init__()
        self._layer_rows  = []   # [(neurons_spin, act_cb, wf_cb, nif_cb), …]
        self._input_spins = []
        self._build_ui()

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left panel ───────────────────────────────────────
        left = QWidget(); left.setFixedWidth(340)
        lv   = QVBoxLayout(left); lv.setSpacing(8)

        # Network shape
        shape_grp  = QGroupBox("Network Shape")
        shape_grp.setFont(QFont('Arial', 9, QFont.Weight.Bold))
        shape_vl   = QVBoxLayout(shape_grp)
        def _spin_row(label, lo, hi, val, vl):
            rw = QWidget(); rh = QHBoxLayout(rw); rh.setContentsMargins(0, 0, 0, 0)
            rh.addWidget(QLabel(label))
            sp = QSpinBox(); sp.setRange(lo, hi); sp.setValue(val)
            rh.addWidget(sp); vl.addWidget(rw); return sp
        self.m_spin = _spin_row("Layers  M :", 1, 6, 2, shape_vl)
        self.r_spin = _spin_row("Inputs  R :", 1, 8, 2, shape_vl)
        lv.addWidget(shape_grp)

        # Per-layer settings
        layer_grp = QGroupBox("Per-layer: S^m | f^m | wf^m | nif^m")
        layer_grp.setFont(QFont('Arial', 9, QFont.Weight.Bold))
        layer_grp_vl = QVBoxLayout(layer_grp)
        self.layer_scroll = QScrollArea()
        self.layer_scroll.setWidgetResizable(True)
        self.layer_scroll.setFixedHeight(200)
        self._rebuild_layers(self.m_spin.value())
        layer_grp_vl.addWidget(self.layer_scroll)
        lv.addWidget(layer_grp)

        # Input vector
        inp_grp = QGroupBox("Input vector  p")
        inp_grp.setFont(QFont('Arial', 9, QFont.Weight.Bold))
        inp_grp_vl = QVBoxLayout(inp_grp)
        self.inp_scroll = QScrollArea()
        self.inp_scroll.setWidgetResizable(True)
        self.inp_scroll.setFixedHeight(130)
        self._rebuild_inputs(self.r_spin.value())
        inp_grp_vl.addWidget(self.inp_scroll)
        lv.addWidget(inp_grp)

        # Seed
        seed_grp = QGroupBox("Weight Init")
        seed_vl  = QHBoxLayout(seed_grp)
        seed_vl.addWidget(QLabel("Random seed:"))
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 9999); self.seed_spin.setValue(42)
        seed_vl.addWidget(self.seed_spin)
        lv.addWidget(seed_grp)

        btn = QPushButton("▶  Run Forward Pass")
        btn.setStyleSheet(
            "QPushButton{background:#2E86AB;color:white;font-weight:bold;padding:7px;border-radius:4px;}"
            "QPushButton:hover{background:#1a6a8a;}")
        lv.addWidget(btn)
        lv.addStretch()

        # ── Right: canvas ────────────────────────────────────
        right = QWidget()
        rv    = QVBoxLayout(right)
        self.fig  = Figure(facecolor='#F8F8FF')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        rv.addWidget(self.canvas)

        splitter.addWidget(left); splitter.addWidget(right)
        splitter.setSizes([340, 700])
        QHBoxLayout(self).addWidget(splitter)

        # Connections
        self.m_spin.valueChanged.connect(lambda v: (self._rebuild_layers(v), self._run()))
        self.r_spin.valueChanged.connect(lambda v: (self._rebuild_inputs(v), self._run()))
        btn.clicked.connect(self._run)
        self.seed_spin.valueChanged.connect(lambda _: self._run())
        self._run()

    # ── Per-layer widget ─────────────────────────────────────
    def _rebuild_layers(self, M):
        container = QWidget()
        vl = QVBoxLayout(container); vl.setSpacing(3)
        self._layer_rows = []
        def_n   = [4, 3, 4, 2, 3, 2]
        def_act = ['Sigmoid', 'Tanh', 'ReLU', 'Linear', 'Sigmoid', 'Sigmoid']
        def_wf  = ['dotprod'] * 6
        def_nif = ['netsum'] * 6
        for m in range(M):
            rw = QWidget(); rh = QHBoxLayout(rw); rh.setContentsMargins(2, 1, 2, 1)
            rh.addWidget(QLabel(f"L{m+1}:").setFixedWidth and QLabel(f"L{m+1}:"))
            n_sp = QSpinBox(); n_sp.setRange(1, 16); n_sp.setValue(def_n[m]); n_sp.setFixedWidth(46)
            n_sp.setToolTip(f"S^{m+1} — neurons in layer {m+1}")
            act_cb = QComboBox(); act_cb.addItems(ACT_NAMES); act_cb.setCurrentText(def_act[m])
            act_cb.setToolTip("Activation  f^m"); act_cb.setFixedWidth(84)
            wf_cb  = QComboBox(); wf_cb.addItems(WF_NAMES);  wf_cb.setCurrentText(def_wf[m])
            wf_cb.setToolTip("Weight function  wf^m"); wf_cb.setFixedWidth(76)
            nif_cb = QComboBox(); nif_cb.addItems(NIF_NAMES); nif_cb.setCurrentText(def_nif[m])
            nif_cb.setToolTip("Net-input function  nif^m"); nif_cb.setFixedWidth(68)
            for w in (n_sp, act_cb, wf_cb, nif_cb): rh.addWidget(w)
            vl.addWidget(rw)
            self._layer_rows.append((n_sp, act_cb, wf_cb, nif_cb))
            for w in (n_sp, act_cb, wf_cb, nif_cb):
                (w.valueChanged if isinstance(w, QSpinBox) else w.currentIndexChanged).connect(
                    lambda _: self._run())
        vl.addStretch()
        self.layer_scroll.setWidget(container)

    def _rebuild_inputs(self, R):
        container = QWidget()
        vl = QVBoxLayout(container); vl.setSpacing(3)
        self._input_spins = []
        defaults = [0.5, 0.8, -0.3, 1.0, 0.2, -0.6, 0.9, -0.1]
        for r in range(R):
            rw = QWidget(); rh = QHBoxLayout(rw); rh.setContentsMargins(2, 1, 2, 1)
            lbl = QLabel(f"p[{r+1}] ="); lbl.setFixedWidth(50)
            sp = QDoubleSpinBox(); sp.setRange(-5.0, 5.0); sp.setSingleStep(0.1)
            sp.setValue(defaults[r]); sp.setDecimals(2)
            sp.valueChanged.connect(lambda _: self._run())
            rh.addWidget(lbl); rh.addWidget(sp)
            vl.addWidget(rw); self._input_spins.append(sp)
        vl.addStretch()
        self.inp_scroll.setWidget(container)

    # ── Forward pass ─────────────────────────────────────────
    def _run(self):
        M = self.m_spin.value(); R = self.r_spin.value()
        if len(self._layer_rows) != M or len(self._input_spins) != R:
            return
        layer_sizes = [row[0].value()             for row in self._layer_rows]
        act_names   = [row[1].currentText()        for row in self._layer_rows]
        wf_names    = [row[2].currentText()        for row in self._layer_rows]
        nif_names   = [row[3].currentText()        for row in self._layer_rows]
        p = np.array([sp.value() for sp in self._input_spins])

        net = LFFN(layer_sizes, act_names, wf_names, nif_names)
        net.init_weights(R, seed=self.seed_spin.value())
        net.forward(p)
        self._plot(net)

    def _plot(self, net):
        self.fig.clear()
        M      = len(net.W)
        n_cols = M + 1
        axs    = self.fig.subplots(1, n_cols, gridspec_kw={'wspace': 0.5})
        if n_cols == 1: axs = [axs]

        for m, ax in enumerate(axs):
            a   = net.a[m]
            clr = LAYER_COLORS[m % len(LAYER_COLORS)]
            y   = np.arange(len(a))
            ax.barh(y, a, color=clr, edgecolor='#334455', alpha=0.88, height=0.6)
            ax.axvline(0, color='gray', lw=0.7)
            for j, v in enumerate(a):
                ax.text(v + 0.03 * np.sign(v + 1e-9), j,
                        f'{v:.2f}', va='center', fontsize=5.5, color='#222')
            ax.set_yticks(y)
            ax.set_yticklabels([f'u{j+1}' for j in range(len(a))], fontsize=5.5)
            ax.tick_params(labelsize=6)
            lim = max(np.max(np.abs(a)) * 1.3, 1.1)
            ax.set_xlim(-lim, lim)
            if m == 0:
                ax.set_title(f'$a^0 = p$\nR={len(a)}', fontsize=7.5, color='#003366')
            else:
                wf_lbl  = net.wf_names[m-1]
                nif_lbl = net.nif_names[m-1]
                fn_lbl  = net.act_names[m-1]
                ax.set_title(f'$a^{m}$  S={len(a)}\n'
                             f'{wf_lbl} | {nif_lbl}\n{fn_lbl}',
                             fontsize=6.5, color='#003366')

        # formula string for suptitle
        wf_set  = set(net.wf_names);  wf_str  = ','.join(sorted(wf_set))
        nif_set = set(net.nif_names); nif_str = ','.join(sorted(nif_set))
        title = (f'LFFN Forward Pass   '
                 + '→'.join(f'S{m+1}={s}' for m, s in enumerate(net.layer_sizes))
                 + f'   wf: {wf_str}   nif: {nif_str}')
        self.fig.suptitle(title, fontsize=8.5, color='darkred', y=1.01)
        self.canvas.draw()


# ══════════════════════════════════════════════════════════════
# Tab 2 — LDDN   (Layered Delayed Dynamic Network)
#
#   Single scalar layer with tapped delay lines:
#
#     z(t) = wf( IW(0)·p(t), IW(1)·p(t-1), IW(2)·p(t-2),
#                LW(1)·a(t-1), LW(2)·a(t-2) )
#     n(t) = nif( z(t), b )
#     a(t) = f( n(t) )
#
#   The weight function is applied to the *combined* scalar z,
#   where for dist/negdist the distance is between the weight
#   scalar and the delayed signal scalar at each tap.
# ══════════════════════════════════════════════════════════════
class LDDNTab(QWidget):
    def __init__(self):
        super().__init__()
        self._iw_sliders = []  # (slider, val_lbl, scale) per input delay tap
        self._lw_sliders = []  # (slider, val_lbl, scale) per layer delay tap
        self._build_ui()
        self._run()

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget(); left.setFixedWidth(330)
        lv   = QVBoxLayout(left); lv.setSpacing(6)

        # Signal
        self.sig_w = _signal_panel(lv)

        # Delays
        delay_grp = QGroupBox("Delay Structure")
        delay_grp.setFont(QFont('Arial', 9, QFont.Weight.Bold))
        dv = QVBoxLayout(delay_grp)
        def _drow(lbl, lo, hi, val):
            rw = QWidget(); rh = QHBoxLayout(rw); rh.setContentsMargins(0,0,0,0)
            rh.addWidget(QLabel(lbl))
            sp = QSpinBox(); sp.setRange(lo, hi); sp.setValue(val)
            rh.addWidget(sp); dv.addWidget(rw); return sp
        self.du_spin = _drow("Input delay taps  D_u:", 1, 4, 2)
        self.dl_spin = _drow("Layer delay taps  D_l:", 0, 4, 1)
        lv.addWidget(delay_grp)

        # Weight sliders (rebuilt when delays change)
        self.wt_grp = QGroupBox("Weights  IW(d) / LW(d) / bias b")
        self.wt_grp.setFont(QFont('Arial', 9, QFont.Weight.Bold))
        self.wt_vl = QVBoxLayout(self.wt_grp)
        lv.addWidget(self.wt_grp)
        self._rebuild_sliders()

        # Activation / weight func / net-input func
        func_grp = QGroupBox("Functions")
        func_grp.setFont(QFont('Arial', 9, QFont.Weight.Bold))
        fv = QVBoxLayout(func_grp)
        def _combo_row(lbl, items, default, vl):
            rw = QWidget(); rh = QHBoxLayout(rw); rh.setContentsMargins(0,0,0,0)
            rh.addWidget(QLabel(lbl))
            cb = QComboBox(); cb.addItems(items); cb.setCurrentText(default)
            rh.addWidget(cb); vl.addWidget(rw); return cb
        self.act_cb = _combo_row("Activation  f(n):", ACT_NAMES,  'Sigmoid', fv)
        self.wf_cb  = _combo_row("Weight func  wf:", WF_NAMES,   'dotprod', fv)
        self.nif_cb = _combo_row("Net-input  nif:", NIF_NAMES,   'netsum',  fv)
        lv.addWidget(func_grp)

        btn = QPushButton("▶  Run Simulation")
        btn.setStyleSheet(
            "QPushButton{background:#E07000;color:white;font-weight:bold;padding:7px;border-radius:4px;}"
            "QPushButton:hover{background:#b05500;}")
        lv.addWidget(btn)
        lv.addStretch()

        # Right canvas
        right = QWidget(); rv = QVBoxLayout(right)
        self.fig    = Figure(facecolor='#F8F8FF')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        rv.addWidget(self.canvas)
        splitter.addWidget(left); splitter.addWidget(right)
        splitter.setSizes([330, 700])
        QHBoxLayout(self).addWidget(splitter)

        # Connections
        btn.clicked.connect(self._run)
        for w in (self.act_cb, self.wf_cb, self.nif_cb): w.currentIndexChanged.connect(lambda _: self._run())
        for w in self.sig_w.values():
            (w.valueChanged if hasattr(w, 'valueChanged') else w.currentIndexChanged).connect(lambda _: self._run())
        self.du_spin.valueChanged.connect(lambda _: (self._rebuild_sliders(), self._run()))
        self.dl_spin.valueChanged.connect(lambda _: (self._rebuild_sliders(), self._run()))

    def _rebuild_sliders(self):
        # Clear existing
        while self.wt_vl.count():
            item = self.wt_vl.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self._iw_sliders = []
        self._lw_sliders = []
        D_u = self.du_spin.value()
        D_l = self.dl_spin.value()
        iw_defaults = [1.2, 0.4, -0.3, 0.2]
        lw_defaults = [0.55, -0.2, 0.1, 0.15]
        for d in range(D_u + 1):
            sl, vl = _slider_row(f"IW({d})  p(t−{d}):", -3.0, 3.0,
                                 iw_defaults[d % len(iw_defaults)], 100, self.wt_vl)
            sl.valueChanged.connect(lambda v, lb=vl: lb.setText(f'{v/100:.2f}'))
            sl.valueChanged.connect(lambda _: self._run())
            self._iw_sliders.append((sl, vl))
        for d in range(1, D_l + 1):
            sl, vl = _slider_row(f"LW({d})  a(t−{d}):", -1.5, 1.5,
                                 lw_defaults[(d-1) % len(lw_defaults)], 100, self.wt_vl)
            sl.valueChanged.connect(lambda v, lb=vl: lb.setText(f'{v/100:.2f}'))
            sl.valueChanged.connect(lambda _: self._run())
            self._lw_sliders.append((sl, vl))
        # Bias
        self.b_sl, self.b_vl = _slider_row("Bias  b:", -3.0, 3.0, 0.0, 100, self.wt_vl)
        self.b_sl.valueChanged.connect(lambda v: self.b_vl.setText(f'{v/100:.2f}'))
        self.b_sl.valueChanged.connect(lambda _: self._run())

    def _get_weights(self):
        IW = [sl.value() / 100 for sl, _ in self._iw_sliders]
        LW = [sl.value() / 100 for sl, _ in self._lw_sliders]
        b  = self.b_sl.value() / 100
        return IW, LW, b

    def _run(self):
        t, p = _build_signal(self.sig_w)
        IW, LW, b = self._get_weights()
        wf_name  = self.wf_cb.currentText()
        nif_name = self.nif_cb.currentText()
        act_name = self.act_cb.currentText()

        T = len(p)
        n_hist = np.zeros(T)
        a_hist = np.zeros(T)
        max_delay = max(len(IW), len(LW) + 1, 2)
        p_buf = np.zeros(max_delay)   # circular-ish buffer
        a_buf = np.zeros(max_delay)
        act_fn = ACT_FUNCS[act_name]

        for i in range(T):
            # Compute weighted contributions per tap using selected weight function
            contributions = []
            for d, iw in enumerate(IW):
                p_d = p_buf[-(d + 1)] if d < len(p_buf) else 0.0
                contributions.append(_scalar_wf(wf_name, iw, p_d))
            for d, lw in enumerate(LW):
                a_d = a_buf[-(d + 1)] if d < len(a_buf) else 0.0
                contributions.append(_scalar_wf(wf_name, lw, a_d))

            z = sum(contributions) if contributions else 0.0
            n = _scalar_nif(nif_name, z, b)
            a_t = float(act_fn(np.array([n]))[0])
            n_hist[i] = n
            a_hist[i] = a_t
            p_buf = np.roll(p_buf, -1); p_buf[-1] = p[i]
            a_buf = np.roll(a_buf, -1); a_buf[-1] = a_t

        self._plot(t, p, n_hist, a_hist, IW, LW, b, wf_name, nif_name, act_name)

    def _plot(self, t, p, n_hist, a_hist, IW, LW, b, wf_name, nif_name, act_name):
        self.fig.clear()
        gs  = self.fig.add_gridspec(3, 1, hspace=0.55)
        ax1 = self.fig.add_subplot(gs[0])
        ax2 = self.fig.add_subplot(gs[1])
        ax3 = self.fig.add_subplot(gs[2])

        ax1.plot(t, p, color=SIG_COLORS[0], lw=1.8, label='p(t)  input')
        ax1.axhline(0, color='gray', lw=0.5)
        ax1.set_title('Input Signal  $p(t)$', fontsize=8.5, color='navy')
        ax1.set_ylabel('$p(t)$', fontsize=7); ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.2); ax1.tick_params(labelsize=7)

        iw_str = '  '.join(f'IW({d})={v:.2f}' for d, v in enumerate(IW))
        lw_str = '  '.join(f'LW({d+1})={v:.2f}' for d, v in enumerate(LW))
        info   = f'wf={wf_name}   nif={nif_name}   f={act_name}\n{iw_str}   {lw_str}   b={b:.2f}'
        ax2.plot(t, n_hist, color=SIG_COLORS[2], lw=1.8, label='n(t)  net input')
        ax2.axhline(0, color='gray', lw=0.5)
        ax2.set_title('Net Input  $n(t)$', fontsize=8.5, color='darkgreen')
        ax2.set_ylabel('$n(t)$', fontsize=7); ax2.legend(fontsize=7)
        ax2.text(0.01, 0.97, info, transform=ax2.transAxes, fontsize=6.5, va='top',
                 bbox=dict(fc='lightyellow', ec='darkorange', alpha=0.9, pad=2))
        ax2.grid(True, alpha=0.2); ax2.tick_params(labelsize=7)

        pn = p / (np.max(np.abs(p)) + 1e-9)
        ax3.plot(t, pn,    color=SIG_COLORS[0], lw=1.4, ls='--', alpha=0.6, label='p(t) [norm]')
        ax3.plot(t, a_hist, color=SIG_COLORS[3], lw=2.0,          label='a(t)  output')
        ax3.fill_between(t, a_hist, alpha=0.08, color=SIG_COLORS[3])
        ax3.axhline(0, color='gray', lw=0.5)
        ax3.set_title('Layer Output  $a(t) = f(n(t))$', fontsize=8.5, color='darkred')
        ax3.set_xlabel('Time step  t', fontsize=7)
        ax3.set_ylabel('$a(t)$', fontsize=7); ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.2); ax3.tick_params(labelsize=7)

        self.canvas.draw()


# ══════════════════════════════════════════════════════════════
# Tab 3 — GLDDN  (General Layered Delayed Dynamic Network)
#
#   Two-layer network with configurable cross-connections:
#
#   Layer 1:  z1(t) = wf1[ IW11(0)·p(t) + IW11(1)·p(t-1) + LW11(1)·a1(t-1) ]
#             n1(t) = nif1( z1(t), b1 )
#             a1(t) = f1( n1(t) )
#
#   Layer 2:  z2(t) = wf2[ LW21(0)·a1(t) + LW22(1)·a2(t-1) ]
#             n2(t) = nif2( z2(t), b2 )
#             a2(t) = f2( n2(t) )
#
#   The "General" part: any layer can connect to any other layer
#   with any delay. Here layers 1 & 2 have cross connections.
# ══════════════════════════════════════════════════════════════
class GLDDNTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()
        self._run()

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget(); left.setFixedWidth(360)
        lv   = QVBoxLayout(left); lv.setSpacing(6)

        # Signal
        self.sig_w = _signal_panel(lv)

        # ── Layer 1 weights ──────────────────────────────────
        l1_grp = QGroupBox("Layer 1 Weights")
        l1_grp.setFont(QFont('Arial', 9, QFont.Weight.Bold))
        l1_vl  = QVBoxLayout(l1_grp)

        w_defs1 = [('IW¹(0)  p(t)',      -3.0, 3.0, 1.2),
                   ('IW¹(1)  p(t−1)',    -3.0, 3.0, 0.4),
                   ('LW¹¹(1) a¹(t−1)',  -1.5, 1.5, 0.55),
                   ('b¹  (bias)',         -3.0, 3.0, 0.0)]
        self.sl1 = []
        for label, lo, hi, val in w_defs1:
            sl, vl = _slider_row(label, lo, hi, val, 100, l1_vl)
            sl.valueChanged.connect(lambda v, lb=vl: lb.setText(f'{v/100:.2f}'))
            sl.valueChanged.connect(lambda _: self._run())
            self.sl1.append(sl)
        lv.addWidget(l1_grp)

        # ── Layer 2 weights ──────────────────────────────────
        l2_grp = QGroupBox("Layer 2 Weights")
        l2_grp.setFont(QFont('Arial', 9, QFont.Weight.Bold))
        l2_vl  = QVBoxLayout(l2_grp)

        w_defs2 = [('LW²¹(0)  a¹(t)',     -3.0, 3.0, 1.0),
                   ('LW²¹(1)  a¹(t−1)',   -3.0, 3.0, 0.2),
                   ('LW²²(1)  a²(t−1)',   -1.5, 1.5, 0.3),
                   ('b²  (bias)',           -3.0, 3.0, 0.0)]
        self.sl2 = []
        for label, lo, hi, val in w_defs2:
            sl, vl = _slider_row(label, lo, hi, val, 100, l2_vl)
            sl.valueChanged.connect(lambda v, lb=vl: lb.setText(f'{v/100:.2f}'))
            sl.valueChanged.connect(lambda _: self._run())
            self.sl2.append(sl)
        lv.addWidget(l2_grp)

        # ── Functions per layer ──────────────────────────────
        fn_grp = QGroupBox("Functions  (per layer)")
        fn_grp.setFont(QFont('Arial', 9, QFont.Weight.Bold))
        fn_vl  = QVBoxLayout(fn_grp)

        def _cb_row(lbl, items, default, vl):
            rw = QWidget(); rh = QHBoxLayout(rw); rh.setContentsMargins(0,0,0,0)
            rh.addWidget(QLabel(lbl)); cb = QComboBox(); cb.addItems(items); cb.setCurrentText(default)
            rh.addWidget(cb); vl.addWidget(rw); return cb

        self.act1_cb = _cb_row("f¹   activation:", ACT_NAMES, 'Sigmoid', fn_vl)
        self.wf1_cb  = _cb_row("wf¹  weight fn:", WF_NAMES,   'dotprod', fn_vl)
        self.nif1_cb = _cb_row("nif¹ net-input:", NIF_NAMES,  'netsum',  fn_vl)
        fn_vl.addWidget(_separator())
        self.act2_cb = _cb_row("f²   activation:", ACT_NAMES, 'Tanh',   fn_vl)
        self.wf2_cb  = _cb_row("wf²  weight fn:", WF_NAMES,   'dotprod', fn_vl)
        self.nif2_cb = _cb_row("nif² net-input:", NIF_NAMES,  'netsum',  fn_vl)
        lv.addWidget(fn_grp)

        btn = QPushButton("▶  Run Simulation")
        btn.setStyleSheet(
            "QPushButton{background:#5B7C99;color:white;font-weight:bold;padding:7px;border-radius:4px;}"
            "QPushButton:hover{background:#3d5a73;}")
        lv.addWidget(btn)
        lv.addStretch()

        # Right canvas
        right = QWidget(); rv = QVBoxLayout(right)
        self.fig    = Figure(facecolor='#F8F8FF')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        rv.addWidget(self.canvas)
        splitter.addWidget(left); splitter.addWidget(right)
        splitter.setSizes([360, 700])
        QHBoxLayout(self).addWidget(splitter)

        # Connections
        btn.clicked.connect(self._run)
        for w in (self.act1_cb, self.wf1_cb, self.nif1_cb,
                  self.act2_cb, self.wf2_cb, self.nif2_cb):
            w.currentIndexChanged.connect(lambda _: self._run())
        for w in self.sig_w.values():
            (w.valueChanged if hasattr(w, 'valueChanged') else w.currentIndexChanged).connect(lambda _: self._run())

    def _run(self):
        t, p = _build_signal(self.sig_w)
        T    = len(p)

        IW1_0  = self.sl1[0].value() / 100
        IW1_1  = self.sl1[1].value() / 100
        LW11_1 = self.sl1[2].value() / 100
        b1     = self.sl1[3].value() / 100

        LW21_0 = self.sl2[0].value() / 100
        LW21_1 = self.sl2[1].value() / 100
        LW22_1 = self.sl2[2].value() / 100
        b2     = self.sl2[3].value() / 100

        wf1  = self.wf1_cb.currentText();  nif1 = self.nif1_cb.currentText()
        act1 = self.act1_cb.currentText()
        wf2  = self.wf2_cb.currentText();  nif2 = self.nif2_cb.currentText()
        act2 = self.act2_cb.currentText()
        act1_fn = ACT_FUNCS[act1];         act2_fn = ACT_FUNCS[act2]

        n1_h = np.zeros(T); a1_h = np.zeros(T)
        n2_h = np.zeros(T); a2_h = np.zeros(T)
        p_prev = 0.0; a1_prev = 0.0; a2_prev = 0.0; a1_cur = 0.0

        for i in range(T):
            # Layer 1: receives p(t), p(t-1), a1(t-1)
            z1 = (_scalar_wf(wf1, IW1_0,  p[i])
                + _scalar_wf(wf1, IW1_1,  p_prev)
                + _scalar_wf(wf1, LW11_1, a1_prev))
            n1 = _scalar_nif(nif1, z1, b1)
            a1_cur = float(act1_fn(np.array([n1]))[0])

            # Layer 2: receives a1(t), a1(t-1), a2(t-1)
            z2 = (_scalar_wf(wf2, LW21_0, a1_cur)
                + _scalar_wf(wf2, LW21_1, a1_prev)
                + _scalar_wf(wf2, LW22_1, a2_prev))
            n2 = _scalar_nif(nif2, z2, b2)
            a2_cur = float(act2_fn(np.array([n2]))[0])

            n1_h[i] = n1; a1_h[i] = a1_cur
            n2_h[i] = n2; a2_h[i] = a2_cur
            p_prev  = p[i]; a1_prev = a1_cur; a2_prev = a2_cur

        self._plot(t, p, n1_h, a1_h, n2_h, a2_h,
                   wf1, nif1, act1, wf2, nif2, act2)

    def _plot(self, t, p, n1, a1, n2, a2, wf1, nif1, act1, wf2, nif2, act2):
        self.fig.clear()
        gs   = self.fig.add_gridspec(3, 2, hspace=0.6, wspace=0.35)
        ax_p  = self.fig.add_subplot(gs[0, :])
        ax_n1 = self.fig.add_subplot(gs[1, 0])
        ax_a1 = self.fig.add_subplot(gs[1, 1])
        ax_n2 = self.fig.add_subplot(gs[2, 0])
        ax_a2 = self.fig.add_subplot(gs[2, 1])

        ax_p.plot(t, p, color=SIG_COLORS[0], lw=1.8)
        ax_p.axhline(0, color='gray', lw=0.4)
        ax_p.set_title('Input  $p(t)$', fontsize=8); ax_p.set_ylabel('p(t)', fontsize=7)
        ax_p.grid(True, alpha=0.2); ax_p.tick_params(labelsize=7)

        ax_n1.plot(t, n1, color=SIG_COLORS[2], lw=1.6)
        ax_n1.axhline(0, color='gray', lw=0.4)
        ax_n1.set_title(f'Layer 1 net: $n^1(t)$\nwf={wf1}  nif={nif1}', fontsize=7.5, color='darkgreen')
        ax_n1.set_ylabel('$n^1$', fontsize=7); ax_n1.grid(True, alpha=0.2); ax_n1.tick_params(labelsize=7)

        ax_a1.plot(t, a1, color=SIG_COLORS[1], lw=2.0)
        ax_a1.fill_between(t, a1, alpha=0.1, color=SIG_COLORS[1])
        ax_a1.axhline(0, color='gray', lw=0.4)
        ax_a1.set_title(f'Layer 1 out: $a^1(t)$  [{act1}]', fontsize=7.5, color='#7f5200')
        ax_a1.set_ylabel('$a^1$', fontsize=7); ax_a1.grid(True, alpha=0.2); ax_a1.tick_params(labelsize=7)

        ax_n2.plot(t, n2, color=SIG_COLORS[4], lw=1.6)
        ax_n2.axhline(0, color='gray', lw=0.4)
        ax_n2.set_title(f'Layer 2 net: $n^2(t)$\nwf={wf2}  nif={nif2}', fontsize=7.5, color='purple')
        ax_n2.set_ylabel('$n^2$', fontsize=7); ax_n2.set_xlabel('t', fontsize=7)
        ax_n2.grid(True, alpha=0.2); ax_n2.tick_params(labelsize=7)

        ax_a2.plot(t, a2, color=SIG_COLORS[3], lw=2.0)
        ax_a2.fill_between(t, a2, alpha=0.1, color=SIG_COLORS[3])
        ax_a2.axhline(0, color='gray', lw=0.4)
        ax_a2.set_title(f'Layer 2 out: $a^2(t)$  [{act2}]', fontsize=7.5, color='darkred')
        ax_a2.set_ylabel('$a^2$', fontsize=7); ax_a2.set_xlabel('t', fontsize=7)
        ax_a2.grid(True, alpha=0.2); ax_a2.tick_params(labelsize=7)

        self.fig.suptitle(
            'GLDDN Forward Pass — 2-Layer General Dynamic Network',
            fontsize=9, color='#222', y=1.01)
        self.canvas.draw()


# ══════════════════════════════════════════════════════════════
# Scalar helpers (for dynamic tabs with scalar weight taps)
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

def _separator():
    """Thin horizontal line widget."""
    from PyQt6.QtWidgets import QFrame
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


# ══════════════════════════════════════════════════════════════
# Main Window
# ══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Ch. 7 — Forward Propagation Demo: LFFN | LDDN | GLDDN')
        self.resize(1100, 720)

        tabs = QTabWidget()
        tabs.setFont(QFont('Arial', 9))
        tabs.addTab(LFFNTab(),  '  LFFN — Feed-Forward  ')
        tabs.addTab(LDDNTab(),  '  LDDN — Dynamic (delays)  ')
        tabs.addTab(GLDDNTab(), '  GLDDN — General Dynamic  ')
        self.setCentralWidget(tabs)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec())