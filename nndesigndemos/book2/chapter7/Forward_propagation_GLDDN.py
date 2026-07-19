"""
Chapter 7 — Forward Propagation: GLDDN (General Layered Delayed Dynamic Network)

Two-layer network with configurable cross-connections and tapped delays:

  Layer 1:  z1(t) = wf1[ IW11(0)·p(t) + IW11(1)·p(t-1) + LW11(1)·a1(t-1) ]
            a1(t) = f1( nif1( z1(t), b1 ) )

  Layer 2:  z2(t) = wf2[ LW21(0)·a1(t) + LW21(1)·a1(t-1) + LW22(1)·a2(t-1) ]
            a2(t) = f2( nif2( z2(t), b2 ) )

The plot updates live as you change the signal, weights or per-layer functions.
Tick "Enable step-by-step animation" to watch the signals get drawn through
time (with a moving cursor) instead of appearing instantly.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QGroupBox, QScrollArea,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH
from nndesigndemos.book2.chapter7.ch7_utils import (
    ACT_NAMES, WF_NAMES, NIF_NAMES, ACT_FUNCS, SIG_COLORS,
    _scalar_wf, _scalar_nif, _separator, _slider_row, _signal_panel, _build_signal,
    Ch7AnimationMixin, _animation_group, _place_header,
)


class GLDDNForwardPropagation(Ch7AnimationMixin, NNDLayout):
    def __init__(self, w_ratio, h_ratio, dpi):
        super(GLDDNForwardPropagation, self).__init__(
            w_ratio, h_ratio, dpi, main_menu=2, overwrite_size=(1090, 740))

        self.fill_chapter(
            "GLDDN", 7,
            "\nGeneral Layered Delayed\nDynamic Network.",
            PACKAGE_PATH + "Logo/book2_logos/7.svg", None, 2, show_info=False)

        # Header band: description (left) + chapter logo (right)
        _place_header(self, "Forward propagation through\na dynamic network with\ndelays and feedback.")

        self.init_animation_state()
        self._stride = 1

        # ── Figure (left region) ─────────────────────────────
        self.make_plot(1, (15, 150, 500, 555))

        # ── Control panel (right region) ─────────────────────
        self._build_panel()

        # Initial simulation (instant, since animation starts disabled)
        self._run()

    # ─────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────
    def _build_panel(self):
        panel = QWidget()
        lv = QVBoxLayout(panel); lv.setSpacing(6)
        lv.setContentsMargins(4, 4, 4, 4)

        grp_font = QFont('Arial', 9, QFont.Weight.Bold)

        # Input signal
        self.sig_w = _signal_panel(lv)
        for w in self.sig_w.values():
            (w.valueChanged if hasattr(w, 'valueChanged') else w.currentIndexChanged).connect(self._on_change)

        # Layer 1 & Layer 2 weights — side by side, each half width
        weights_row = QWidget()
        wr = QHBoxLayout(weights_row); wr.setContentsMargins(0, 0, 0, 0); wr.setSpacing(6)

        # (short label, full tooltip, lo, hi, default)
        l1_grp = QGroupBox("Layer 1 Weights"); l1_grp.setFont(grp_font)
        l1_vl  = QVBoxLayout(l1_grp)
        w_defs1 = [('IW¹(0)',  'IW¹(0)  p(t)',      -3.0, 3.0, 1.2),
                   ('IW¹(1)',  'IW¹(1)  p(t−1)',    -3.0, 3.0, 0.4),
                   ('LW¹¹(1)', 'LW¹¹(1)  a¹(t−1)',  -1.5, 1.5, 0.55),
                   ('b¹',      'b¹  (bias)',        -3.0, 3.0, 0.0)]
        self.sl1 = []
        for label, tip, lo, hi, val in w_defs1:
            sl, vlab = _slider_row(label, lo, hi, val, 100, l1_vl, label_width=62, tooltip=tip)
            self._wire_weight_slider(sl, vlab)
            self.sl1.append(sl)
        wr.addWidget(l1_grp)

        l2_grp = QGroupBox("Layer 2 Weights"); l2_grp.setFont(grp_font)
        l2_vl  = QVBoxLayout(l2_grp)
        w_defs2 = [('LW²¹(0)', 'LW²¹(0)  a¹(t)',    -3.0, 3.0, 1.0),
                   ('LW²¹(1)', 'LW²¹(1)  a¹(t−1)',  -3.0, 3.0, 0.2),
                   ('LW²²(1)', 'LW²²(1)  a²(t−1)',  -1.5, 1.5, 0.3),
                   ('b²',      'b²  (bias)',        -3.0, 3.0, 0.0)]
        self.sl2 = []
        for label, tip, lo, hi, val in w_defs2:
            sl, vlab = _slider_row(label, lo, hi, val, 100, l2_vl, label_width=62, tooltip=tip)
            self._wire_weight_slider(sl, vlab)
            self.sl2.append(sl)
        wr.addWidget(l2_grp)

        lv.addWidget(weights_row)

        # Functions per layer
        fn_grp = QGroupBox("Functions  (per layer)")
        fn_grp.setFont(grp_font)
        fn_vl  = QVBoxLayout(fn_grp)

        def _cb_row(lbl, items, default, vl):
            rw = QWidget(); rh = QHBoxLayout(rw); rh.setContentsMargins(0, 0, 0, 0)
            rh.addWidget(QLabel(lbl)); cb = QComboBox(); cb.addItems(items); cb.setCurrentText(default)
            cb.currentIndexChanged.connect(self._on_change)
            rh.addWidget(cb); vl.addWidget(rw); return cb

        self.act1_cb = _cb_row("f¹   activation:", ACT_NAMES, 'Sigmoid', fn_vl)
        self.wf1_cb  = _cb_row("wf¹  weight fn:", WF_NAMES,   'dotprod', fn_vl)
        self.nif1_cb = _cb_row("nif¹ net-input:", NIF_NAMES,  'netsum',  fn_vl)
        fn_vl.addWidget(_separator())
        self.act2_cb = _cb_row("f²   activation:", ACT_NAMES, 'Tanh',    fn_vl)
        self.wf2_cb  = _cb_row("wf²  weight fn:", WF_NAMES,   'dotprod', fn_vl)
        self.nif2_cb = _cb_row("nif² net-input:", NIF_NAMES,  'netsum',  fn_vl)
        lv.addWidget(fn_grp)

        # Animation controls
        lv.addWidget(_animation_group(self))
        lv.addStretch()

        # Wrap in a scroll area. Keep the vertical bar always visible (so it
        # doesn't pop in/out) and never show a horizontal bar (panel is wide
        # enough for its content).
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(panel)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.set_layout((525, 160, 520, 555), scroll)

    def _wire_weight_slider(self, sl, vlab):
        """Weight slider: always update its value label; instant redraw while
        dragging when NOT animating; kick off the animation on release when it
        IS animating (see Ch7AnimationMixin._on_slider_drag / _on_slider_release)."""
        sl.valueChanged.connect(lambda v, lb=vlab: lb.setText(f'{v/100:.2f}'))
        sl.valueChanged.connect(self._on_slider_drag)
        sl.sliderReleased.connect(self._on_slider_release)

    # ─────────────────────────────────────────────────────────
    # Animation hooks
    # ─────────────────────────────────────────────────────────
    def _compute(self):
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
        p_prev = 0.0; a1_prev = 0.0; a2_prev = 0.0

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

        return dict(t=t, p=p, n1=n1_h, a1=a1_h, n2=n2_h, a2=a2_h,
                    wf1=wf1, nif1=nif1, act1=act1, wf2=wf2, nif2=nif2, act2=act2)

    def _total_frames(self, res):
        T = len(res['t'])
        self._stride = max(1, T // 100)          # cap at ~100 frames
        return (T + self._stride - 1) // self._stride

    def _draw_static(self, res):
        self._draw(res, upto=len(res['t']) - 1, cursor=False)

    def _draw_frame(self, res, k):
        T   = len(res['t'])
        idx = min(T - 1, (k + 1) * self._stride - 1)
        self._draw(res, upto=idx, cursor=True)

    # ─────────────────────────────────────────────────────────
    # Drawing
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _ylim(*arrays):
        lo = min(float(np.min(a)) for a in arrays)
        hi = max(float(np.max(a)) for a in arrays)
        if hi - lo < 1e-6:
            lo, hi = lo - 1.0, hi + 1.0
        pad = 0.1 * (hi - lo)
        return lo - pad, hi + pad

    def _draw(self, res, upto, cursor):
        t  = res['t']; T = len(t)
        p  = res['p']; n1 = res['n1']; a1 = res['a1']; n2 = res['n2']; a2 = res['a2']
        s  = slice(0, upto + 1)

        self.figure.clear()
        # top< 1 reserves room so the suptitle is fully visible.
        gs   = self.figure.add_gridspec(3, 2, hspace=0.6, wspace=0.35,
                                        top=0.90, bottom=0.06, left=0.10, right=0.97)
        ax_p  = self.figure.add_subplot(gs[0, :])
        ax_n1 = self.figure.add_subplot(gs[1, 0])
        ax_a1 = self.figure.add_subplot(gs[1, 1])
        ax_n2 = self.figure.add_subplot(gs[2, 0])
        ax_a2 = self.figure.add_subplot(gs[2, 1])

        # Fixed axis limits from the FULL result so the view doesn't jump as
        # the animation progresses.
        xlim = (0, max(T - 1, 1))

        def _panel(ax, series, color, title, tcolor, ylabel, xlabel=None, fill=False):
            ax.plot(t[s], series[s], color=color, lw=1.8)
            if fill:
                ax.fill_between(t[s], series[s], alpha=0.1, color=color)
            ax.axhline(0, color='gray', lw=0.4)
            ax.set_xlim(*xlim)
            ax.set_ylim(*self._ylim(series))
            ax.set_title(title, fontsize=7.5 if tcolor != 'navy' else 8, color=tcolor)
            ax.set_ylabel(ylabel, fontsize=7)
            if xlabel: ax.set_xlabel(xlabel, fontsize=7)
            ax.grid(True, alpha=0.2); ax.tick_params(labelsize=7)
            if cursor and upto < T - 1:
                ax.axvline(t[upto], color='#888', lw=0.9, ls='--', alpha=0.8)

        _panel(ax_p,  p,  SIG_COLORS[0], 'Input  $p(t)$', 'navy', 'p(t)')
        _panel(ax_n1, n1, SIG_COLORS[2], f"Layer 1 net: $n^1(t)$\nwf={res['wf1']}  nif={res['nif1']}",
               'darkgreen', '$n^1$')
        _panel(ax_a1, a1, SIG_COLORS[1], f"Layer 1 out: $a^1(t)$  [{res['act1']}]",
               '#7f5200', '$a^1$', fill=True)
        _panel(ax_n2, n2, SIG_COLORS[4], f"Layer 2 net: $n^2(t)$\nwf={res['wf2']}  nif={res['nif2']}",
               'purple', '$n^2$', xlabel='t')
        _panel(ax_a2, a2, SIG_COLORS[3], f"Layer 2 out: $a^2(t)$  [{res['act2']}]",
               'darkred', '$a^2$', xlabel='t', fill=True)

        self.figure.suptitle(
            'GLDDN Forward Pass — 2-Layer General Dynamic Network',
            fontsize=9, color='#222', y=0.975)
        self.canvas.draw()
