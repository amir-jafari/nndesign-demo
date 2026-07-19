"""
Chapter 7 — Forward Propagation: LFFN (Layered Feed-Forward Network)

Static, delay-free network:

    a^0 = p
    For m = 1 … M:
        z^m = wf^m ( W^m , a^{m-1} )
        n^m = nif^m( z^m , b^m     )
        a^m = f^m  ( n^m           )

The plot updates live as you change the network shape, per-layer functions or
the input vector. Tick "Enable step-by-step animation" to watch the signal
propagate through the layers one at a time instead of appearing instantly.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QScrollArea,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH
from nndesigndemos.book2.chapter7.ch7_utils import (
    ACT_NAMES, WF_NAMES, NIF_NAMES, LAYER_COLORS, LFFN,
    Ch7AnimationMixin, _animation_group, _place_header,
)


class LFFNForwardPropagation(Ch7AnimationMixin, NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(LFFNForwardPropagation, self).__init__(
            w_ratio, h_ratio, dpi, main_menu=2, overwrite_size=(1090, 740))

        self.fill_chapter(
            "LFFN", 7,
            "\nLayered Feed-Forward\nNetwork.",
            PACKAGE_PATH + "Logo/book2_logos/7.svg", None, 2, show_info=False)

        # Header band: description (left) + chapter logo (right)
        _place_header(self, "Forward propagation through\na static layered\nfeed-forward network.")

        self._layer_rows  = []   # [(neurons_spin, act_cb, wf_cb, nif_cb), …]
        self._input_spins = []

        self.init_animation_state()

        # ── Figure (left region) ─────────────────────────────
        self.make_plot(1, (15, 150, 500, 555))

        # ── Control panel (right region) ─────────────────────
        self._build_panel()

        # Initial forward pass (instant, since animation starts disabled)
        self._run()

    # ─────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────
    def _build_panel(self):
        panel = QWidget()
        lv = QVBoxLayout(panel); lv.setSpacing(8)
        lv.setContentsMargins(4, 4, 4, 4)

        # Network shape
        shape_grp = QGroupBox("Network Shape")
        shape_grp.setFont(QFont('Arial', 9, QFont.Weight.Bold))
        shape_vl  = QVBoxLayout(shape_grp)

        def _spin_row(label, lo, hi, val, vl):
            rw = QWidget(); rh = QHBoxLayout(rw); rh.setContentsMargins(0, 0, 0, 0)
            rh.addWidget(QLabel(label))
            sp = QSpinBox(); sp.setRange(lo, hi); sp.setValue(val)
            rh.addWidget(sp); vl.addWidget(rw); return sp

        self.m_spin = _spin_row("Layers  M :", 1, 6, 2, shape_vl)
        self.r_spin = _spin_row("Inputs  R :", 1, 8, 2, shape_vl)
        lv.addWidget(shape_grp)

        # Per-layer settings (no scrollbars; M ≤ 6 rows fit directly)
        layer_grp = QGroupBox("Per-layer: S^m | f^m | wf^m | nif^m")
        layer_grp.setFont(QFont('Arial', 9, QFont.Weight.Bold))
        self.layer_grp_vl = QVBoxLayout(layer_grp)
        self._layer_container = None
        self._rebuild_layers(self.m_spin.value())
        lv.addWidget(layer_grp)

        # Input vector — 2-column grid (max 4 rows for R ≤ 8), no scrollbar
        inp_grp = QGroupBox("Input vector  p")
        inp_grp.setFont(QFont('Arial', 9, QFont.Weight.Bold))
        self.inp_grp_vl = QVBoxLayout(inp_grp)
        self._inp_container = None
        self._rebuild_inputs(self.r_spin.value())
        lv.addWidget(inp_grp)

        # Seed
        seed_grp = QGroupBox("Weight Init")
        seed_vl  = QHBoxLayout(seed_grp)
        seed_vl.addWidget(QLabel("Random seed:"))
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 9999); self.seed_spin.setValue(42)
        self.seed_spin.valueChanged.connect(self._on_change)
        seed_vl.addWidget(self.seed_spin)
        lv.addWidget(seed_grp)

        # Animation controls
        lv.addWidget(_animation_group(self))
        lv.addStretch()

        # Live update: changing the shape rebuilds the per-layer / input rows,
        # then redraws (instantly, or as a fresh animation).
        self.m_spin.valueChanged.connect(lambda v: (self._rebuild_layers(v), self._on_change()))
        self.r_spin.valueChanged.connect(lambda v: (self._rebuild_inputs(v), self._on_change()))

        # Wrap in a scroll area. Keep the vertical bar always visible (so it
        # doesn't pop in/out) and never show a horizontal bar (panel is wide
        # enough for its content).
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(panel)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.set_layout((525, 160, 520, 555), scroll)

    def _rebuild_layers(self, M):
        # Replace the previous rows (if any)
        if self._layer_container is not None:
            self.layer_grp_vl.removeWidget(self._layer_container)
            self._layer_container.setParent(None)   # detach immediately (no ghost)
            self._layer_container.deleteLater()
        container = QWidget()
        vl = QVBoxLayout(container); vl.setSpacing(3)
        self._layer_rows = []
        def_n   = [4, 3, 4, 2, 3, 2]
        def_act = ['Sigmoid', 'Tanh', 'ReLU', 'Linear', 'Sigmoid', 'Sigmoid']
        def_wf  = ['dotprod'] * 6
        def_nif = ['netsum'] * 6
        for m in range(M):
            rw = QWidget(); rh = QHBoxLayout(rw); rh.setContentsMargins(2, 1, 2, 1)
            lbl = QLabel(f"L{m+1}:"); lbl.setFixedWidth(30)
            rh.addWidget(lbl)
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
            # Live update on any per-layer change (connected after defaults are set)
            n_sp.valueChanged.connect(self._on_change)
            act_cb.currentIndexChanged.connect(self._on_change)
            wf_cb.currentIndexChanged.connect(self._on_change)
            nif_cb.currentIndexChanged.connect(self._on_change)
        self._layer_container = container
        self.layer_grp_vl.addWidget(container)

    def _rebuild_inputs(self, R):
        # Replace the previous grid (if any)
        if self._inp_container is not None:
            self.inp_grp_vl.removeWidget(self._inp_container)
            self._inp_container.setParent(None)   # detach immediately (no ghost)
            self._inp_container.deleteLater()
        container = QWidget()
        grid = QGridLayout(container); grid.setSpacing(3); grid.setContentsMargins(2, 1, 2, 1)
        self._input_spins = []
        defaults = [0.5, 0.8, -0.3, 1.0, 0.2, -0.6, 0.9, -0.1]
        for r in range(R):
            cell = QWidget(); ch = QHBoxLayout(cell); ch.setContentsMargins(2, 1, 2, 1)
            lbl = QLabel(f"p[{r+1}] ="); lbl.setFixedWidth(46)
            sp = QDoubleSpinBox(); sp.setRange(-5.0, 5.0); sp.setSingleStep(0.1)
            sp.setValue(defaults[r]); sp.setDecimals(2)
            sp.valueChanged.connect(self._on_change)
            ch.addWidget(lbl); ch.addWidget(sp)
            grid.addWidget(cell, r // 2, r % 2)   # 2 columns → half width
            self._input_spins.append(sp)
        self._inp_container = container
        self.inp_grp_vl.addWidget(container)

    # ─────────────────────────────────────────────────────────
    # Animation hooks
    # ─────────────────────────────────────────────────────────
    def _compute(self):
        M = self.m_spin.value(); R = self.r_spin.value()
        if len(self._layer_rows) != M or len(self._input_spins) != R:
            return None
        layer_sizes = [row[0].value()      for row in self._layer_rows]
        act_names   = [row[1].currentText() for row in self._layer_rows]
        wf_names    = [row[2].currentText() for row in self._layer_rows]
        nif_names   = [row[3].currentText() for row in self._layer_rows]
        p = np.array([sp.value() for sp in self._input_spins])

        net = LFFN(layer_sizes, act_names, wf_names, nif_names)
        net.init_weights(R, seed=self.seed_spin.value())
        net.forward(p)
        return net

    def _total_frames(self, net):
        return len(net.a)          # M + 1  (input layer + M layers)

    def _draw_static(self, net):
        self._draw(net, upto=len(net.a) - 1, highlight=None)

    def _draw_frame(self, net, k):
        self._draw(net, upto=k, highlight=k)

    # ─────────────────────────────────────────────────────────
    # Drawing
    # ─────────────────────────────────────────────────────────
    def _draw(self, net, upto, highlight):
        self.figure.clear()
        M      = len(net.W)
        n_cols = M + 1
        axs    = self.figure.subplots(1, n_cols, gridspec_kw={'wspace': 0.5})
        if n_cols == 1: axs = [axs]
        # Reserve room at the top so the suptitle is fully visible.
        self.figure.subplots_adjust(top=0.80, bottom=0.08, left=0.08, right=0.97)

        for m, ax in enumerate(axs):
            a   = net.a[m]
            y   = np.arange(len(a))
            revealed = (m <= upto)
            if revealed:
                clr = LAYER_COLORS[m % len(LAYER_COLORS)]
                ax.barh(y, a, color=clr, edgecolor='#334455', alpha=0.88, height=0.6)
                for j, v in enumerate(a):
                    ax.text(v + 0.03 * np.sign(v + 1e-9), j,
                            f'{v:.2f}', va='center', fontsize=5.5, color='#222')
                lim = max(np.max(np.abs(a)) * 1.3, 1.1)
            else:
                # not yet reached: faint placeholder so the axis doesn't jump
                ax.barh(y, np.zeros_like(a), color='#eeeeee', edgecolor='#dddddd', height=0.6)
                lim = 1.1
            ax.axvline(0, color='gray', lw=0.7)
            ax.set_yticks(y)
            ax.set_yticklabels([f'u{j+1}' for j in range(len(a))], fontsize=5.5,
                               color='#222' if revealed else '#bbb')
            ax.tick_params(labelsize=6)
            ax.set_xlim(-lim, lim)

            title_color = '#003366' if revealed else '#bbbbbb'
            if m == 0:
                ax.set_title(f'$a^0 = p$\nR={len(a)}', fontsize=7.5, color=title_color)
            else:
                wf_lbl  = net.wf_names[m-1]
                nif_lbl = net.nif_names[m-1]
                fn_lbl  = net.act_names[m-1]
                ax.set_title(f'$a^{m}$  S={len(a)}\n{wf_lbl} | {nif_lbl}\n{fn_lbl}',
                             fontsize=6.5, color=title_color)

            if m == highlight:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#d62728'); spine.set_linewidth(2.2)

        wf_set  = set(net.wf_names);  wf_str  = ','.join(sorted(wf_set))
        nif_set = set(net.nif_names); nif_str = ','.join(sorted(nif_set))
        title = ('LFFN Forward Pass   '
                 + '→'.join(f'S{m+1}={s}' for m, s in enumerate(net.layer_sizes))
                 + f'   wf: {wf_str}   nif: {nif_str}')
        self.figure.suptitle(title, fontsize=8.5, color='darkred', y=0.97)
        self.canvas.draw()
