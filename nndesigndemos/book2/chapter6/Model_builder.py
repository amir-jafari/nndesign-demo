"""
Chapter 6 - TensorFlow Introduction: Model Builder
==================================================
"Constructing the Model"

The chapter builds a 2-10-2 network three different ways:
    1) the Sequential class
    2) the Functional API
    3) the Model subclass
...and they all produce *the same network*.

Here you design the network with the sliders (R = 2, S1 = 10, S2 = 5 ...).
The demo then

    * draws the architecture in the ABBREVIATED MATRIX NOTATION of the
      book -- the Figure 2.7 style block diagram, with the weight matrix
      W^m, the summing junction, the bias b^m and the activation block
      f^m drawn separately. Every array carries its REAL dimensions, so
      W2 of a 2-10-5 network is labelled 5 x 10, not S^2 x S^1, and the
      numbers follow the sliders as you move them,
    * counts the parameters of the whole network, W^m by W^m and
      b^m by b^m, as well as the Keras `model.summary()` table,
    * writes out the Keras code for all three styles at once (and the
      PyTorch equivalent of Chapter 10), and
    * pushes an input vector through the layers so you can watch the
      activations light up.

The three operations of a layer, exactly as the chapter lists them:
    1) the weight operation        z^m = W^m a^(m-1)
    2) the net input operation     n^m = z^m + b^m
    3) the activation function     a^m = f^m(n^m)
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path
from PyQt6 import QtCore, QtWidgets

from nndesigndemos.book2.panel_utils import PanelDemo, mono_font, ui_font, FS

ORD = ["First", "Second", "Third", "Fourth", "Fifth"]
DIAGRAM_RECT = (5, 100, 419, 365)
CODE_RECT = (5, 470, 419, 200)
TORCH_ACT = {"tanh": "Tanh", "relu": "ReLU", "sigmoid": "Sigmoid", "linear": "Identity"}


# ===========================================================================
#  Drawing helpers for the book's abbreviated notation (Figures 2.6 / 2.7)
# ===========================================================================
def brace(ax, x0, x1, y, h=0.045, up=True, lw=1.1):
    """The curly brace the book puts over (and under) each layer."""
    s = 1.0 if up else -1.0
    d = min(0.055, (x1 - x0) / 5.0)          # how far the tips curl over
    xm = 0.5 * (x0 + x1)
    v = [(x0, y),
         (x0, y + s * h), (x0 + d, y + s * h),          # left tip curls over
         (xm - d, y + s * h),                            # straight run
         (xm, y + s * h), (xm, y + s * (h + 0.5 * h)),   # the centre point
         (xm, y + s * h), (xm + d, y + s * h),
         (x1 - d, y + s * h),
         (x1, y + s * h), (x1, y)]
    c = [Path.MOVETO,
         Path.CURVE3, Path.CURVE3,
         Path.LINETO,
         Path.CURVE3, Path.CURVE3,
         Path.CURVE3, Path.CURVE3,
         Path.LINETO,
         Path.CURVE3, Path.CURVE3]
    ax.add_patch(PathPatch(Path(v, c), fc="none", ec="k", lw=lw, zorder=2))


def arrow(ax, x0, y0, x1, y1, lw=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=11, color="k", lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=3))


def box(ax, x, y, w, h, label, dim, fs, fc="white"):
    """A labelled block (W, b or f) with its dimensions written underneath."""
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="k", lw=1.3, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs + 2, zorder=4)
    if dim:
        ax.text(x + w / 2, y - 0.012, dim, ha="center", va="top",
                fontsize=fs - 2, style="italic", zorder=4)


def signal(ax, x, y, name, dim, fs):
    """
    The name of a signal (p, z, n, a) with its dimensions underneath,
    written just above the arrow that carries it. `y` is the height of
    the arrow at that x.
    """
    ax.text(x, y + 0.052, name, ha="center", va="bottom",
            fontsize=fs + 1, zorder=4)
    ax.text(x, y + 0.016, dim, ha="center", va="bottom",
            fontsize=fs - 2.5, style="italic", zorder=4)


# ---------------------------------------------------------------- activations
ACT = {
    "tanh":    lambda n: np.tanh(n),
    "relu":    lambda n: np.maximum(0.0, n),
    "sigmoid": lambda n: 1.0 / (1.0 + np.exp(-n)),
    "linear":  lambda n: n,
    "softmax": lambda n: np.exp(n - n.max()) / np.exp(n - n.max()).sum(),
}


# ===========================================================================
class ModelBuilder(PanelDemo):
    title = "Model Builder"
    chapter = 6
    description = ("Build a neural network\nwith the sliders and see\nits diagram, parameter\n"
                   "count, model.summary()\nand the Keras / PyTorch\ncode that creates it.")

    def __init__(self, w_ratio, h_ratio, dpi):
        super(ModelBuilder, self).__init__(w_ratio, h_ratio, dpi)

        # ---------------- figure + code tabs (left) ----------------
        self.make_figure(DIAGRAM_RECT)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.tabBar().setFont(ui_font(self, 9))
        self.code_views = {}
        for name in ("1. Sequential class", "2. Functional API", "3. Model subclass",
                     "PyTorch Sequential method"):
            te = self.text_box(8)
            self.code_views[name] = te
            self.tabs.addTab(te, name)
        self.place(CODE_RECT, self.tabs)

        # ---------------- controls (right) ----------------
        panel = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(panel)
        lv.setContentsMargins(4, 0, 4, 0)
        lv.setSpacing(4)

        gbox, gl = self.group("Architecture")
        self.s_in = self.slider("inputs  R", 1, 8, 2, self.rebuild)
        self.s_nlayers = self.slider("number of hidden layers", 1, 3, 1, self.rebuild)
        self.s_h1 = self.slider("hidden neurons  S<sup>1</sup>", 1, 20, 10, self.rebuild)
        self.s_h2 = self.slider("hidden neurons  S<sup>2</sup>", 1, 20, 8, self.rebuild)
        self.s_h3 = self.slider("hidden neurons  S<sup>3</sup>", 1, 20, 6, self.rebuild)
        self.s_out = self.slider("output neurons  S<sup>2</sup>", 1, 8, 2, self.rebuild)
        for s in (self.s_in, self.s_nlayers, self.s_h1, self.s_h2, self.s_h3, self.s_out):
            gl.addWidget(s)
        self.c_hidden_act = self.combo(["tanh", "relu", "sigmoid", "linear"], self.rebuild)
        self.c_out_act = self.combo(["softmax", "sigmoid", "linear", "tanh"], self.rebuild)
        for text, cb in (("hidden-layer activation", self.c_hidden_act),
                         ("output-layer activation", self.c_out_act)):
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(self.label(text))
            row.addWidget(cb, 1)
            gl.addLayout(row)
        self.chk_bias = self.checkbox("layers use a bias  (use_bias=True)", True, self.rebuild)
        gl.addWidget(self.chk_bias)
        self.btn_reinit = self.button("Re-initialize the weights", self.reinit)
        gl.addWidget(self.btn_reinit)
        lv.addWidget(gbox)

        self.gin, self.gin_lay = self.group("Input vector  p")
        self.in_sliders = []          # created dynamically for the input vector
        lv.addWidget(self.gin)

        pbox, pl = self.group("Number of parameters in the network")
        self.lbl_params = self.label("", pt=13, bold=True)
        pl.addWidget(self.lbl_params)
        self.params_detail = QtWidgets.QLabel()
        self.params_detail.setFont(mono_font(self, 8))
        self.params_detail.setTextFormat(QtCore.Qt.TextFormat.RichText)
        pl.addWidget(self.params_detail)
        lv.addWidget(pbox)

        lv.addWidget(self.label("model.summary()", bold=True))
        self.summary = self.text_box(9, min_height=70)
        lv.addWidget(self.summary, 1)
        self.make_panel(panel)

        self.seed = 1
        self.rebuild()

    # ------------------------------------------------------------------
    def reinit(self):
        self.seed += 1
        self.rebuild()

    def sizes(self):
        n = self.s_nlayers.value()
        hs = [self.s_h1.value(), self.s_h2.value(), self.s_h3.value()][:n]
        return [self.s_in.value()] + hs + [self.s_out.value()]

    def rebuild(self):
        n = self.s_nlayers.value()
        self.s_h2.setVisible(n >= 2)
        self.s_h3.setVisible(n >= 3)
        # the output layer is layer M, so its symbol follows the layer count
        self.s_out.set_name("output neurons  S<sup>{}</sup>".format(n + 1))

        sizes = self.sizes()

        # ---- rebuild the input-vector sliders if the input size changed ----
        if len(self.in_sliders) != sizes[0]:
            while self.gin_lay.count():
                it = self.gin_lay.takeAt(0)
                if it.widget():
                    it.widget().deleteLater()
            self.in_sliders = []
            for k in range(sizes[0]):
                s = self.slider("p[{}]".format(k), -2.0, 2.0, 1.0 if k == 0 else 0.0,
                                self.forward_and_draw, decimals=2)
                self.gin_lay.addWidget(s)
                self.in_sliders.append(s)

        # ---- weights ----
        rng = np.random.default_rng(self.seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            # Glorot-uniform, the Keras default for Dense
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(rng.uniform(-lim, lim, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

        self._write_params(sizes)
        self._write_summary(sizes)
        self._write_code(sizes)
        self.forward_and_draw()

    # ------------------------------------------------------------------
    def n_params(self, sizes):
        """(per-layer counts, total).  W is S^m x S^(m-1), b is S^m x 1."""
        use_b = self.chk_bias.isChecked()
        per = []
        for i in range(len(sizes) - 1):
            w = sizes[i] * sizes[i + 1]
            b = sizes[i + 1] if use_b else 0
            per.append((w, b))
        return per, sum(w + b for w, b in per)

    def _write_params(self, sizes):
        per, total = self.n_params(sizes)
        self.lbl_params.setText("{:,} parameters".format(total))
        rows = ["<pre style='margin:0'>"]
        for i, (w, b) in enumerate(per):
            m = i + 1
            rows.append("W{} {:>3d} x{:>3d} = {:>5d}".format(m, sizes[i + 1], sizes[i], w)
                        + ("   b{} {:>3d} = {:>4d}".format(m, sizes[i + 1], b) if b else "   (no bias)"))
        rows.append("{:<19s}= {:>5d}".format("total", total))
        rows.append("</pre>")
        self.params_detail.setText("\n".join(rows))

    # ------------------------------------------------------------------
    def forward(self):
        p = np.array([s.value() for s in self.in_sliders], dtype=float)
        acts = [p]
        nets = []
        h_act = ACT[self.c_hidden_act.currentText()]
        o_act = ACT[self.c_out_act.currentText()]
        a = p
        for i in range(len(self.W)):
            n = a @ self.W[i] + (self.b[i] if self.chk_bias.isChecked() else 0.0)
            f = o_act if i == len(self.W) - 1 else h_act
            a = f(n)
            nets.append(n)
            acts.append(a)
        return acts, nets

    def forward_and_draw(self):
        acts, nets = self.forward()
        sizes = self.sizes()
        self.figure.clear()
        gs = self.figure.add_gridspec(2, max(1, len(acts)), height_ratios=[1.62, 1])

        # ------- architecture in the book's abbreviated notation -------
        ax = self.figure.add_subplot(gs[0, :])
        self._draw_book_diagram(ax, sizes)

        # ---------------- activations per layer ----------------
        nplot = len(acts)
        for li in range(nplot):
            ax2 = self.figure.add_subplot(gs[1, li])
            v = acts[li]
            ax2.bar(np.arange(len(v)), v,
                    color=["#c0392b" if x < 0 else "#2471a3" for x in v])
            ax2.axhline(0, color="k", lw=0.6)
            ax2.set_title("p" if li == 0 else "a$^{}$".format(li), fontsize=FS, fontweight="bold")
            ax2.tick_params(labelsize=FS)
        out = acts[-1]
        self.figure.supxlabel(
            "a$^{}$ = {}".format(nplot - 1, np.array2string(out, precision=3, suppress_small=True))
            + ("   sum={:.3f}".format(out.sum()) if self.c_out_act.currentText() == "softmax" else ""),
            fontsize=FS)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _draw_book_diagram(self, ax, sizes):
        """
        Figure 2.7 of the Multilayer Networks chapter, drawn for whatever
        architecture the sliders currently describe.

              p -> [W^m] -> z^m -\\
                                 (+) -> n^m -> [f^m] -> a^m
              1 -> [b^m] -------/
        """
        M = len(sizes) - 1                       # number of LAYERS (not counting input)
        ax.set_axis_off()
        ax.set_xlim(0.03, 0.42 + M * 1.0 + 0.10)
        ax.set_ylim(-0.36, 1.16)
        fs = float(np.clip(10.0 - 1.0 * M, 6.0, 8.5))

        acts_name = [self.c_hidden_act.currentText()] * (M - 1) + \
                    [self.c_out_act.currentText()]
        use_b = self.chk_bias.isChecked()

        # ---------------- the input ----------------
        ax.add_patch(Rectangle((0.10, 0.20), 0.045, 0.62, fc="k", ec="k", zorder=3))
        brace(ax, 0.045, 0.21, 0.86, up=True)
        ax.text(0.128, 0.985, "Input", ha="center", fontsize=fs, zorder=4)
        brace(ax, 0.045, 0.21, 0.062, up=False)
        ax.text(0.128, 0.098, "$R$ = {}".format(sizes[0]), ha="center", va="bottom",
                fontsize=fs, style="italic")

        # ---------------- one block per layer ----------------
        for m in range(M):
            xo = 0.42 + m * 1.0                  # left edge of this layer
            R_in = sizes[m]                      # S^(m-1), or R for the first
            S = sizes[m + 1]                     # S^m
            sup = "{}".format(m + 1)

            # W, b, the summing junction and f -- dimensions as real numbers
            box(ax, xo + 0.06, 0.56, 0.17, 0.15, "${{W}}^{}$".format(sup),
                "{} x {}".format(S, R_in), fs)
            if use_b:
                box(ax, xo + 0.06, 0.15, 0.17, 0.15, "${{b}}^{}$".format(sup),
                    "{} x 1".format(S), fs)
                ax.text(xo - 0.055, 0.225, "1", ha="center", va="center", fontsize=fs)
                arrow(ax, xo - 0.02, 0.225, xo + 0.055, 0.225)
                arrow(ax, xo + 0.23, 0.225, xo + 0.395, 0.375)
                ax.plot([xo + 0.44], [0.42], marker="o", ms=11 + 2 * (fs - 5.5),
                        mfc="white", mec="k", mew=1.3, zorder=3, clip_on=False)
                ax.text(xo + 0.44, 0.42, "+", ha="center", va="center",
                        fontsize=fs + 2, zorder=4)
            box(ax, xo + 0.63, 0.19, 0.15, 0.52,
                "${{f}}^{}$".format(sup), None, fs + 2)
            ax.text(xo + 0.705, 0.098, "$S^{}$ = {}".format(sup, S), ha="center",
                    va="bottom", fontsize=fs, style="italic")

            # the signals
            if m == 0:                                  # p comes from the input bar
                arrow(ax, 0.15, 0.635, xo + 0.055, 0.635)
                signal(ax, 0.29, 0.635, "${p}$", "{} x 1".format(R_in), fs)
            else:                                       # a^(m-1) from the layer before
                arrow(ax, xo - 0.22, 0.42, xo + 0.055, 0.635)
                signal(ax, xo - 0.10, 0.514, "${{a}}^{}$".format(m), "{} x 1".format(R_in), fs)
            if use_b:
                arrow(ax, xo + 0.23, 0.635, xo + 0.40, 0.47)
                signal(ax, xo + 0.335, 0.533, "${{z}}^{}$".format(sup),
                       "{} x 1".format(S), fs)
                arrow(ax, xo + 0.48, 0.42, xo + 0.625, 0.42)
                signal(ax, xo + 0.555, 0.420, "${{n}}^{}$".format(sup),
                       "{} x 1".format(S), fs)
            else:                              # no bias: z^m IS the net input
                arrow(ax, xo + 0.23, 0.635, xo + 0.625, 0.45)
                signal(ax, xo + 0.43, 0.541,
                       "${{z}}^{0} = {{n}}^{0}$".format(sup), "{} x 1".format(S), fs)

            # brace, layer name and the layer equation
            brace(ax, xo + 0.02, xo + 0.82, 0.86, up=True)
            ax.text(xo + 0.42, 0.985,
                    "{} Layer".format(ORD[m] if m < len(ORD) else str(m + 1)),
                    ha="center", fontsize=fs, zorder=4)
            brace(ax, xo + 0.02, xo + 0.82, 0.062, up=False)
            prev = "{p}" if m == 0 else "{{a}}^{}".format(m)
            eq = ("${{a}}^{0} = {{f}}^{0}({{W}}^{0}{1}".format(sup, prev)
                  + (" + {{b}}^{})$".format(sup) if use_b else ")$"))
            ax.text(xo + 0.42, -0.045, eq, ha="center", va="top", fontsize=fs)
            ax.text(xo + 0.42, -0.135, "$f^{}$ = {}".format(sup, acts_name[m]),
                    ha="center", va="top", fontsize=fs - 1.5, color="#1f4e9c")

        # the last output arrow
        xe = 0.42 + (M - 1) * 1.0
        arrow(ax, xe + 0.78, 0.42, xe + 0.98, 0.42)
        signal(ax, xe + 0.92, 0.420, "${{a}}^{}$".format(M), "{} x 1".format(sizes[-1]), fs)

        # the whole network, composed
        comp = "{p}"
        for m in range(M):
            comp = ("{{f}}^{0}({{W}}^{0}".format(m + 1) + comp
                    + (" + {{b}}^{})".format(m + 1) if use_b else ")"))
        ax.text(0.5 * (0.03 + 0.42 + M * 1.0 + 0.10), -0.245, "${{a}}^{} = {}$".format(M, comp),
                ha="center", va="top", fontsize=fs)
        _, total = self.n_params(sizes)
        ax.set_title("Network in abbreviated matrix notation   " + "-".join(map(str, sizes))
                     + "      {:,} parameters".format(total), fontsize=FS)

    # ------------------------------------------------------------------
    def _write_summary(self, sizes):
        use_b = self.chk_bias.isChecked()
        rows, total = [], 0
        for i in range(len(sizes) - 1):
            pcount = sizes[i] * sizes[i + 1] + (sizes[i + 1] if use_b else 0)
            total += pcount
            act = (self.c_out_act.currentText() if i == len(sizes) - 2
                   else self.c_hidden_act.currentText())
            rows.append(("dense_{} ({})".format(i, act), "(None, {})".format(sizes[i + 1]), pcount,
                         "W{}x{}".format(sizes[i], sizes[i + 1]) + (" + b{}".format(sizes[i + 1]) if use_b else "")))
        w = 18
        txt = ["Model: \"sequential\"",
               "_" * 40,
               "{:<{w}}{:<14}{:>8}".format("Layer (type)", "Output Shape", "Param #", w=w),
               "=" * 40]
        for name, shp, pc, expl in rows:
            txt.append("{:<{w}}{:<14}{:>8}".format(name, shp, pc, w=w))
            txt.append("  {}".format(expl))
        txt += ["=" * 40,
                "Total params: {:,}".format(total),
                "Trainable params: {:,}".format(total),
                "_" * 40,
                "",
                "Every Dense layer computes",
                "    n = a_prev @ W + b",
                "    a = f(n)",
                "W has shape (inputs, neurons),",
                "b has shape (neurons,).",
                "That is where the parameter count",
                "comes from."]
        self.summary.setPlainText("\n".join(txt))

    def _write_code(self, sizes):
        ha, oa = self.c_hidden_act.currentText(), self.c_out_act.currentText()
        use_b = self.chk_bias.isChecked()
        bias = "" if use_b else ", use_bias=False"
        tbias = "" if use_b else ", bias=False"
        hid = sizes[1:-1]
        S = sizes[-1]
        R = sizes[0]

        seq = ["from tensorflow.keras import models, layers", "",
               "model = models.Sequential()"]
        for k, h in enumerate(hid):
            if k == 0:
                seq.append("model.add(layers.Dense({}, activation='{}',"
                           " input_shape=({},){}))".format(h, ha, R, bias))
            else:
                seq.append("model.add(layers.Dense({}, activation='{}'{}))".format(h, ha, bias))
        seq.append("model.add(layers.Dense({}, activation='{}'{}))".format(S, oa, bias))
        seq += ["", "model.summary()",
                "",
                "# Only the FIRST layer needs input_shape. Every later layer",
                "# works out its input size from the layer before it."]

        fun = ["from tensorflow.keras import models, layers", "",
               "p = layers.Input(shape=({},))".format(R)]
        prev = "p"
        for k, h in enumerate(hid):
            fun.append("a{} = layers.Dense({}, activation='{}'{})({})".format(k + 1, h, ha, bias, prev))
            prev = "a{}".format(k + 1)
        fun.append("out = layers.Dense({}, activation='{}'{})({})".format(S, oa, bias, prev))
        fun += ["model = models.Model(inputs=p, outputs=out)", "",
                "# Layers are called like functions. This is the style you need",
                "# for networks that branch or merge (not just a straight line)."]

        sub = ["from tensorflow.keras import models, layers", "",
               "class MyNet(models.Model):", "    def __init__(self):",
               "        super().__init__()"]
        for k, h in enumerate(hid):
            sub.append("        self.dense{} = layers.Dense({}, activation='{}'{})".format(k + 1, h, ha, bias))
        sub.append("        self.out = layers.Dense({}, activation='{}'{})".format(S, oa, bias))
        sub += ["", "    def call(self, inputs):", "        x = inputs"]
        for k in range(len(hid)):
            sub.append("        x = self.dense{}(x)".format(k + 1))
        sub += ["        return self.out(x)", "",
                "model = MyNet()", "",
                "# The input size is not fixed until the first fit()/call().",
                "# Most flexible: `call` is ordinary Python, so you can put",
                "# if-statements and loops in the forward pass."]

        pt = ["import torch", "",
              "model = torch.nn.Sequential("]
        prev_sz = R
        for h in hid:
            pt.append("    torch.nn.Linear({}, {}{}),".format(prev_sz, h, tbias))
            pt.append("    torch.nn.{}(),".format(TORCH_ACT[ha]))
            prev_sz = h
        pt.append("    torch.nn.Linear({}, {}{}))".format(prev_sz, S, tbias))
        pt += ["", "# Two differences from Keras (Ch.10):",
               "#  * Linear, not Dense -- and you give BOTH sizes, every layer.",
               "#  * No softmax on the last layer: torch.nn.CrossEntropyLoss",
               "#    applies softmax itself. See the softmax demo.",
               "", "print(model)"]

        for name, code in zip(self.code_views,
                              ["\n".join(seq), "\n".join(fun), "\n".join(sub), "\n".join(pt)]):
            self.code_views[name].setPlainText(code)
