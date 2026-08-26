"""
Chapter 10 - PyTorch Introduction: Training Loop
================================================
The training loop of the chapter, one line at a time.

    optimizer.zero_grad()
    output = model(p)
    loss = loss_fn(output, t)
    loss.backward()
    loss_values.append(loss.cpu().detach().numpy())
    optimizer.step()

Every line of that loop is a BUTTON. Press one and only that line runs;
the table under the figure is the state of every tensor in the loop, and
the values that the line you just pressed changed are highlighted. Nothing
else moves. That is the whole point: in PyTorch you can see which line
does what, which is why the chapter says PyTorch is easier to debug.

The model is the smallest one that still has everything in it -- a
single neuron:

        n = w*p + b
        a = f(n)
        L = (a - t)^2

so `loss.backward()` can be drawn in full. The graph shows two numbers
on every node:
    blue,  left to right : the FORWARD value
    red,   right to left : the GRADIENT  dL/d(that node)   ( = .grad )

The gradient is a VECTOR, not a number
--------------------------------------
The 3D panel draws the whole loss surface L(w, b). The red arrow lying
in the (w, b) plane is

    grad L = [ dL/dw , dL/db ] = [ w.grad , b.grad ]

with its two components drawn as dotted legs, so you can see that
backward() does not produce "the slope" -- it produces one number per
parameter, and together they point in the steepest uphill direction.
The green arrow is -grad L, which is the direction optimizer.step()
moves in. The two 2D panels are the slices you get by holding one
parameter still, which is what a single .grad value measures.

Things worth trying
-------------------
* Press `loss.backward()` twice without `optimizer.zero_grad()` in
  between. The gradient buffers do not get replaced, they get ADDED to,
  and the next `optimizer.step()` takes a step twice as big. That single
  missing line is the most common PyTorch bug.
* Press `optimizer.step()` before `loss.backward()`. The weights move by
  zero, because .grad is still zero. The order of the five steps is not
  a convention, it is a dependency.
* Move a slider after a forward pass: `output` and `loss` go stale
  (marked in the table). They are just numbers PyTorch computed earlier,
  not live formulas.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PyQt6 import QtCore, QtWidgets

from nndesigndemos.book2.panel_utils import PanelDemo, FS

LR = 0.5            # optimizer = torch.optim.SGD([w, b], lr=0.5)

# The chapter's training loop, verbatim.
STEPS = [
    ("optimizer.zero_grad()",                       "Zero the gradient"),
    ("output = model(p)",                           "Forward pass through the network"),
    ("loss = loss_fn(output, t)",                   "Calculate the loss"),
    ("loss.backward()",                             "Compute the gradient"),
    ("loss_values.append(loss.cpu().detach().numpy())", "Save the loss for plotting"),
    ("optimizer.step()",                            "Update the weights"),
]

ACTS = {
    "tanh":    (np.tanh,                        lambda n, a: 1 - a ** 2,      "1 - a^2"),
    "sigmoid": (lambda n: 1 / (1 + np.exp(-n)), lambda n, a: a * (1 - a),     "a(1 - a)"),
    "relu":    (lambda n: np.maximum(0.0, n),   lambda n, a: float(n > 0),    "1 if n>0 else 0"),
    "linear":  (lambda n: n,                    lambda n, a: 1.0,             "1"),
}

BTN_STYLE = "text-align:left; padding:3px;"


# ===========================================================================
class TrainingLoop(PanelDemo):
    title = "Training Loop"
    chapter = 10
    description = ("Run the PyTorch training\nloop one line at a time.\nEach press shows which\n"
                   "numbers that line changed\nand how the gradient\nmoves a single neuron.")

    def __init__(self, w_ratio, h_ratio, dpi):
        super(TrainingLoop, self).__init__(w_ratio, h_ratio, dpi)

        # ---------------- figure (left), what-happened text under it ----------------
        self.make_figure((5, 100, 419, 430))
        self.txt = self.text_box(8)
        self.place((5, 535, 419, 135), self.txt)

        # ---------------- controls (right) ----------------
        panel = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(panel)
        lv.setContentsMargins(4, 0, 4, 0)
        lv.setSpacing(4)

        box, bl = self.group("Leaves of the graph")
        bl.setSpacing(1)
        self.s_p = self.slider("p   (input)", -3, 3, 1.5, self.on_slider, decimals=2, fmt="{:+.2f}")
        self.s_w = self.slider("w   (weight)", -3, 3, 0.8, self.on_slider, decimals=2, fmt="{:+.2f}")
        self.s_b = self.slider("b   (bias)", -3, 3, -0.5, self.on_slider, decimals=2, fmt="{:+.2f}")
        self.s_t = self.slider("t   (target)", -1.5, 1.5, 0.9, self.on_slider, decimals=2, fmt="{:+.2f}")
        for s in (self.s_p, self.s_w, self.s_b, self.s_t):
            bl.addWidget(s)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 2, 0, 0)
        row.addWidget(self.label("activation f"))
        self.c_act = self.combo(list(ACTS), self.on_slider)
        row.addWidget(self.c_act, 1)
        bl.addLayout(row)
        lv.addWidget(box)

        # one button per line of the loop
        gsteps, vs = self.group("for epoch in range(epochs):")
        vs.setSpacing(2)
        self.step_btns = []
        for i, (code, what) in enumerate(STEPS):
            b = self.button(code, lambda _=False, k=i: self.run_step(k), pt=8, mono=True)
            b.setToolTip("{}. {}".format(i + 1, what))
            b.setStyleSheet(BTN_STYLE)
            vs.addWidget(b)
            self.step_btns.append(b)
        lv.addWidget(gsteps)

        self.btn_iter = self.button("run the whole iteration", self.run_iteration)
        self.btn_iter.setToolTip("runs the six lines in order, lighting each one up as it goes")
        self.btn_reset = self.button("reset (new optimizer, empty loss_values)", self.reset)
        lv.addWidget(self.btn_iter)
        lv.addWidget(self.btn_reset)

        # walks the loop one line at a time so the highlight is visible
        self.auto_timer = QtCore.QTimer(self)
        self.auto_timer.setInterval(650)
        self.auto_timer.timeout.connect(self._auto_tick)
        self.auto_left = 0

        lv.addWidget(self.label("state of every tensor in the loop", pt=9, bold=True))
        self.table = self.text_box(7.5, min_height=60)
        lv.addWidget(self.table, 1)
        self.make_panel(panel)

        self.reset()

    # ------------------------------------------------------------------
    #  state
    # ------------------------------------------------------------------
    def reset(self):
        if self.auto_timer.isActive():
            self._auto_stop()
        self.last_run = None          # which line ran most recently
        self.w_grad = None            # .grad is None until the first backward()
        self.b_grad = None
        self.output = None            # what `model(p)` returned, last time it ran
        self.loss = None              # what `loss_fn(output, t)` returned
        self.loss_values = []
        self.stage = 0                # the line that comes next
        self.stale = False            # a slider moved after the forward pass
        self.msg = "press the lines of the loop, top to bottom"
        self.changed = set()
        self.redraw()

    def on_slider(self, *_):
        # p, w, b, t are the leaves. Changing one does NOT change `output`
        # or `loss`: those are numbers computed by an earlier forward pass.
        if self.output is not None:
            self.stale = True
        self.changed = set()
        self.msg = ("a leaf changed. `output` and `loss` still hold the values "
                    "from the last forward pass -- run it again")
        self.redraw()

    # ------------------------------------------------------------------
    def surface(self, w, b):
        """L(w, b) for whole arrays of w and b at once -- the loss surface."""
        p, t = self.s_p.value(), self.s_t.value()
        f = ACTS[self.c_act.currentText()][0]
        return (f(w * p + b) - t) ** 2

    def compute(self, w=None, b=None):
        """The forward and backward maths, done by hand."""
        p = self.s_p.value()
        w = self.s_w.value() if w is None else w
        b = self.s_b.value() if b is None else b
        t = self.s_t.value()
        f, df, _ = ACTS[self.c_act.currentText()]
        n = w * p + b
        a = float(f(n))
        L = (a - t) ** 2
        dL_da = 2 * (a - t)
        da_dn = float(df(n, a))
        dL_dn = dL_da * da_dn
        return dict(p=p, w=w, b=b, t=t, n=n, a=a, L=L,
                    dL_da=dL_da, da_dn=da_dn, dL_dn=dL_dn,
                    dL_dw=dL_dn * p, dL_db=dL_dn * 1.0, dL_dp=dL_dn * w)

    # ------------------------------------------------------------------
    #  the six lines of the loop
    # ------------------------------------------------------------------
    def run_step(self, k):
        c = self.compute()
        self.changed = set()

        if k == 0:                                   # optimizer.zero_grad()
            had = (self.w_grad not in (None, 0.0)) or (self.b_grad not in (None, 0.0))
            self.w_grad = self.b_grad = 0.0
            self.changed = {"w.grad", "b.grad"}
            self.msg = ("w.grad and b.grad set to 0."
                        if had else
                        "w.grad and b.grad set to 0 (they were already empty).")

        elif k == 1:                                 # output = model(p)
            self.output = c["a"]
            self.stale = False
            self.changed = {"output"}
            self.msg = ("n = w*p + b = {:+.4f},  output = f(n) = {:+.4f}\n"
                        "no gradient exists yet -- the forward pass only "
                        "RECORDS the graph.".format(c["n"], c["a"]))

        elif k == 2:                                 # loss = loss_fn(output, t)
            if self.output is None:
                self.msg = "there is no `output` yet. Run `output = model(p)` first."
            else:
                self.loss = (self.output - c["t"]) ** 2
                self.changed = {"loss"}
                self.msg = ("loss = (output - t)^2 = ({:+.4f} - {:+.4f})^2 = {:.6f}\n"
                            "one scalar -- that is what backward() needs."
                            .format(self.output, c["t"], self.loss))

        elif k == 3:                                 # loss.backward()
            if self.loss is None:
                self.msg = "there is no `loss` to differentiate yet."
            else:
                onto = bool(self.w_grad) or bool(self.b_grad)   # was it already full?
                self.w_grad = (0.0 if self.w_grad is None else self.w_grad) + c["dL_dw"]
                self.b_grad = (0.0 if self.b_grad is None else self.b_grad) + c["dL_db"]
                self.changed = {"w.grad", "b.grad"}
                self.msg = ("the chain rule filled in the leaves:\n"
                            "  w.grad += dL/dw = {:+.6f}\n"
                            "  b.grad += dL/db = {:+.6f}".format(c["dL_dw"], c["dL_db"]))
                if onto:
                    self.msg += ("\nnote the +=. Gradients ACCUMULATE. Without "
                                 "zero_grad() they pile up.")

        elif k == 4:                                 # loss_values.append(...)
            if self.loss is None:
                self.msg = "there is no `loss` to append yet."
            else:
                self.loss_values.append(self.loss)
                self.changed = {"loss_values"}
                self.msg = ("loss_values now has {} entries.\n"
                            ".cpu() moves it off the GPU, .detach() drops the "
                            "graph so the whole history is not kept alive.".format(len(self.loss_values)))

        else:                                        # optimizer.step()
            gw = self.w_grad or 0.0
            gb = self.b_grad or 0.0
            lr = LR
            self.s_w.set_value(float(np.clip(c["w"] - lr * gw, -3, 3)))
            self.s_b.set_value(float(np.clip(c["b"] - lr * gb, -3, 3)))
            self.changed = {"w", "b"}
            if gw == 0.0 and gb == 0.0:
                self.msg = ("nothing moved: .grad is zero. step() only applies "
                            "the gradient that backward() already computed.")
            else:
                self.msg = ("w <- w - lr * w.grad = {:+.4f} - {:.2f} * {:+.4f} = {:+.4f}\n"
                            "b <- b - lr * b.grad = {:+.4f} - {:.2f} * {:+.4f} = {:+.4f}"
                            .format(c["w"], lr, gw, self.s_w.value(), c["b"], lr, gb, self.s_b.value()))
                self.stale = self.output is not None

        self.last_run = k
        self.stage = (k + 1) % len(STEPS)
        self.redraw()

    # ------------------------------------------------------------------
    def run_iteration(self):
        """
        Walk the six lines in order, on a timer, so each button lights up
        as its line runs. Press again to stop part way through.
        """
        if self.auto_timer.isActive():
            self._auto_stop()
            return
        self.auto_left = len(STEPS)
        self.btn_iter.setText("stop")
        self._auto_tick()                     # run the first line immediately
        self.auto_timer.start()

    def _auto_tick(self):
        if self.auto_left <= 0:
            self._auto_stop()
            return
        self.run_step(self.stage)             # whichever line comes next
        self.auto_left -= 1
        if self.auto_left <= 0:
            self._auto_stop()

    def _auto_stop(self):
        self.auto_timer.stop()
        self.auto_left = 0
        self.btn_iter.setText("run the whole iteration")

    # ------------------------------------------------------------------
    def redraw(self):
        c = self.compute()

        # green = the line that just ran, yellow = the line that comes next
        for i, b in enumerate(self.step_btns):
            style = BTN_STYLE
            if i == self.last_run:
                style += "background:#b7e1a1; font-weight:bold;"
            elif i == self.stage:
                style += "background:#ffe58a; font-weight:bold;"
            b.setStyleSheet(style)

        self.figure.clear()
        gs = self.figure.add_gridspec(4, 2, height_ratios=[1.5, 0.63, 0.63, 0.63], width_ratios=[1.15, 1])

        # ============ the computational graph ============
        ax = self.figure.add_subplot(gs[0, :])
        ax.set_axis_off()
        ax.set_xlim(0, 12.3)
        ax.set_ylim(-0.2, 5.1)

        def node(x, y, label, fwd, grad, col, bw=1.1):
            """A box wide enough for its own label -- bw is the box width."""
            ax.add_patch(FancyBboxPatch((x - bw / 2, y - .34), bw, .68,
                                        boxstyle="round,pad=0.06",
                                        fc=col, ec="k", lw=1.1, zorder=2))
            ax.text(x, y + .13, label, ha="center", va="center",
                    fontsize=8, weight="bold", zorder=3)
            ax.text(x, y - .17, "{:+.4f}".format(fwd) if fwd is not None else "None",
                    ha="center", va="center", fontsize=6.5, color="#0b3d91", zorder=3)
            if grad is not None:
                ax.text(x, y - .66, ".grad = {}".format(grad), ha="center",
                        fontsize=6, color="#c0392b", weight="bold")

        def arrow(x1, y1, x2, y2):
            """Forward (blue) and backward (red, dashed) along the same edge."""
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                         arrowstyle="-|>", mutation_scale=10,
                                         color="#0b3d91", lw=1.2, zorder=1))
            ax.add_patch(FancyArrowPatch((x2, y2 - .16), (x1, y1 - .16),
                                         arrowstyle="-|>", mutation_scale=8,
                                         color="#c0392b", lw=1.0, ls="--", zorder=1))

        def dlabel(x, y, lab):
            """A derivative, on a white patch so it never sits across a line."""
            ax.text(x, y, lab, ha="center", va="center", fontsize=6,
                    color="#c0392b", zorder=6,
                    bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.95))

        def gtxt(g):
            return "None" if g is None else "{:+.4f}".format(g)

        leaf, mid, out = "#dbe9f7", "#e8e8e8", "#f7dbdb"
        hot = "#ffe58a"
        node(1.05, 4.30, "p", c["p"], None, leaf)
        node(1.05, 2.50, "w", c["w"], gtxt(self.w_grad),
             hot if "w" in self.changed else leaf)
        node(1.05, 0.70, "b", c["b"], gtxt(self.b_grad),
             hot if "b" in self.changed else leaf)
        node(5.20, 2.50, "n = w*p+b", c["n"], None, mid, bw=2.1)
        node(8.30, 2.50, "output = f(n)", self.output, None,
             hot if "output" in self.changed else mid, bw=2.4)
        node(11.25, 2.50, "loss", self.loss, None,
             hot if "loss" in self.changed else out, bw=1.4)
        node(8.30, 0.70, "t", c["t"], None, leaf)

        arrow(1.60, 4.30, 4.15, 2.85)
        arrow(1.60, 2.50, 4.15, 2.50)
        arrow(1.60, 0.70, 4.15, 2.15)
        arrow(6.25, 2.50, 7.10, 2.50)
        arrow(9.50, 2.50, 10.55, 2.50)
        ax.annotate("", xy=(8.30, 2.16), xytext=(8.30, 1.04),
                    arrowprops=dict(arrowstyle="-|>", color="#0b3d91", lw=1.1))

        # the three left-hand derivatives sit on their own edge, well apart
        dlabel(2.85, 3.58, "dn/dp = w = {:+.2f}".format(c["w"]))
        dlabel(2.85, 2.50, "dn/dw = p = {:+.2f}".format(c["p"]))
        dlabel(2.85, 1.42, "dn/db = 1")
        # these two labels are wider than the gap, so they go above the boxes
        dlabel(6.70, 3.55, "da/dn = {} = {:+.3f}".format(ACTS[self.c_act.currentText()][2], c["da_dn"]))
        dlabel(10.0, 3.55, "dL/da = 2(a-t) = {:+.3f}".format(c["dL_da"]))

        ax.text(0.15, 4.85, "FORWARD  ->", color="#0b3d91", fontsize=7, weight="bold")
        ax.text(10.55, 4.85, "<-  BACKWARD", color="#c0392b", fontsize=7, weight="bold")
        ax.set_title("next line:  {}".format(STEPS[self.stage][0]), fontsize=FS,
                     color="#8a6d00", fontweight="bold")
        if self.stale:
            ax.text(7.7, -0.12, "output / loss are STALE: a leaf moved after the forward pass",
                    color="#c0392b", fontsize=6.5, ha="center", weight="bold")

        # ============ the surface L(w, b), and the gradient VECTOR ========
        gw, gb = c["dL_dw"], c["dL_db"]
        gnorm = float(np.hypot(gw, gb))

        ax1 = self.figure.add_subplot(gs[1:, 0], projection="3d")
        gwv, gbv = np.meshgrid(np.linspace(-3, 3, 44), np.linspace(-3, 3, 44))
        Lg = self.surface(gwv, gbv)
        ax1.plot_surface(gwv, gbv, Lg, cmap="viridis", alpha=.55,
                         linewidth=0, antialiased=True, rstride=1, cstride=1)
        z0 = float(Lg.min() - 0.22 * (Lg.max() - Lg.min() + 1e-9))
        ax1.contour(gwv, gbv, Lg, levels=10, zdir="z", offset=z0,
                    cmap="viridis", linewidths=.7)
        # where we are now, and a dropline down to the (w, b) plane
        ax1.plot([c["w"]], [c["b"]], [c["L"]], "o", color="#c0392b", ms=5, zorder=5)
        ax1.plot([c["w"], c["w"]], [c["b"], c["b"]], [z0, c["L"]], ":", color="#444", lw=1.0)
        ax1.plot([c["w"]], [c["b"]], [z0], "o", color="#c0392b", ms=3)
        # the gradient itself: one arrow in the (w, b) plane, two components
        if gnorm > 1e-9:
            s = 1.5 / gnorm                       # drawn at a readable length
            ax1.quiver(c["w"], c["b"], z0, gw * s, gb * s, 0, color="#c0392b",
                       lw=1.8, arrow_length_ratio=.22, zorder=6)
            ax1.quiver(c["w"], c["b"], z0, -gw * s, -gb * s, 0, color="#137333",
                       lw=1.8, arrow_length_ratio=.22, zorder=6)
            # and its two components, so the vector is visibly [dL/dw, dL/db]
            ax1.plot([c["w"], c["w"] + gw * s], [c["b"], c["b"]], [z0, z0],
                     color="#c0392b", lw=0.9, ls=":")
            ax1.plot([c["w"] + gw * s, c["w"] + gw * s],
                     [c["b"], c["b"] + gb * s], [z0, z0],
                     color="#c0392b", lw=0.9, ls=":")
        ax1.set_xlabel("w", fontsize=FS, labelpad=-10)
        ax1.set_ylabel("b", fontsize=FS, labelpad=-10)
        ax1.set_zlabel("L", fontsize=FS, labelpad=-10)
        ax1.tick_params(labelsize=FS, pad=-3)
        ax1.set_box_aspect((1, 1, 0.72), zoom=1.1)
        ax1.set_title("the gradient is a VECTOR\non the surface L(w, b)\n"
                      "grad L = [{:+.3f}, {:+.3f}]     |grad L| = {:.3f}".format(gw, gb, gnorm), fontsize=FS)
        ax1.text2D(-0.04, 0.98, "red   = grad L, uphill\ngreen = -grad L, where step() goes",
                   transform=ax1.transAxes, fontsize=FS - 1, va="top", color="#444")
        ax1.view_init(elev=30, azim=-128)

        # ============ L as a function of w ============
        ax2 = self.figure.add_subplot(gs[1, 1])
        ws = np.linspace(-3, 3, 400)
        ax2.plot(ws, self.surface(ws, c["b"]), color="#333", lw=1.4)
        ax2.plot(c["w"], c["L"], "o", ms=6, color="#c0392b", zorder=4)
        if self.w_grad:
            dw = 0.9
            ax2.plot([c["w"] - dw, c["w"] + dw],
                     [c["L"] - dw * self.w_grad, c["L"] + dw * self.w_grad],
                     color="#c0392b", lw=1.6, ls="--",
                     label="slope = w.grad = {:+.4f}".format(self.w_grad))
            w_new = np.clip(c["w"] - LR * self.w_grad, -3, 3)
            ax2.annotate("", xy=(w_new, self.compute(w=w_new)["L"]),
                         xytext=(c["w"], c["L"]),
                         arrowprops=dict(arrowstyle="-|>", color="#137333", lw=1.8))
            ax2.text(w_new, self.compute(w=w_new)["L"], "  step()", color="#137333", fontsize=6.5)
            ax2.legend(fontsize=FS)
        ax2.set_xlabel("w", fontsize=FS)
        ax2.set_ylabel("L", fontsize=FS)
        ax2.set_title("slice along w  (b held still)", fontsize=FS)
        ax2.tick_params(labelsize=FS)
        ax2.grid(alpha=.3)

        # ============ L as a function of b ============
        ax3 = self.figure.add_subplot(gs[2, 1])
        bs = np.linspace(-3, 3, 400)
        ax3.plot(bs, self.surface(c["w"], bs), color="#333", lw=1.4)
        ax3.plot(c["b"], c["L"], "o", ms=6, color="#c0392b", zorder=4)
        if self.b_grad:
            db = 0.9
            ax3.plot([c["b"] - db, c["b"] + db],
                     [c["L"] - db * self.b_grad, c["L"] + db * self.b_grad],
                     color="#c0392b", lw=1.6, ls="--",
                     label="slope = b.grad = {:+.4f}".format(self.b_grad))
            ax3.legend(fontsize=FS)
        ax3.set_xlabel("b", fontsize=FS)
        ax3.set_ylabel("L", fontsize=FS)
        ax3.set_title("slice along b  (w held still)", fontsize=FS)
        ax3.tick_params(labelsize=FS)
        ax3.grid(alpha=.3)

        # ============ loss_values ============
        ax4 = self.figure.add_subplot(gs[3, 1])
        if self.loss_values:
            ax4.plot(np.arange(1, len(self.loss_values) + 1), self.loss_values,
                     "o-", color="#1f4e9c", ms=3)
        else:
            ax4.text(.5, .5, "loss_values is empty", transform=ax4.transAxes,
                     ha="center", fontsize=FS, color="#888")
        ax4.set_xlabel("append number", fontsize=FS)
        ax4.set_ylabel("loss", fontsize=FS)
        ax4.set_title("loss_values  (what you plot later)", fontsize=FS)
        ax4.tick_params(labelsize=FS)
        ax4.grid(alpha=.3)

        self.canvas.draw_idle()
        self._write_table(c)
        self._write_text(c)

    # ------------------------------------------------------------------
    def _write_table(self, c):
        """Every tensor in the loop; the ones the last line touched are lit."""
        lv = ("[]" if not self.loss_values else
              "[" + ", ".join("{:.3f}".format(v) for v in self.loss_values[-3:]) + "]"
              + ("   ({} total)".format(len(self.loss_values)) if len(self.loss_values) > 3 else ""))
        rows = [
            ("p",           "{:+.4f}".format(c["p"]),       "input, a leaf"),
            ("t",           "{:+.4f}".format(c["t"]),       "target, a leaf"),
            ("w",           "{:+.4f}".format(c["w"]),       "parameter"),
            ("b",           "{:+.4f}".format(c["b"]),       "parameter"),
            ("output",      "None" if self.output is None else "{:+.4f}".format(self.output),
             "set by model(p)" + ("   STALE" if self.stale else "")),
            ("loss",        "None" if self.loss is None else "{:.6f}".format(self.loss),
             "set by loss_fn"),
            ("w.grad",      "None" if self.w_grad is None else "{:+.6f}".format(self.w_grad),
             "set by backward()"),
            ("b.grad",      "None" if self.b_grad is None else "{:+.6f}".format(self.b_grad),
             "set by backward()"),
            ("loss_values", lv, "python list"),
        ]
        html = ["<pre style='font-family:Menlo,Consolas,monospace;margin:0'>",
                "<b>{:<12s} {:<10s} </b>".format("name", "value")]
        for name, val, note in rows:
            line = "{:<12s} {:<10s} <span style='color:#888'>{}</span>".format(name, val, note)
            if name in self.changed:
                html.append("<span style='background:#ffe58a'>{}  &lt;-- changed</span>".format(line))
            else:
                html.append("<span style='color:#444'>{}</span>".format(line))
        html.append("</pre>")
        self.table.setHtml("\n".join(html))

    def _write_text(self, c):
        h = 1e-5
        num_w = (self.compute(w=c["w"] + h)["L"] - self.compute(w=c["w"] - h)["L"]) / (2 * h)
        num_b = (self.compute(b=c["b"] + h)["L"] - self.compute(b=c["b"] - h)["L"]) / (2 * h)
        act = self.c_act.currentText()
        self.txt.setPlainText(
            "what the line you just pressed did\n"
            "----------------------------------\n{}\n\n".format(self.msg) +
            "chain rule for the CURRENT values of the leaves\n"
            "----------------------------------------------\n"
            "dL/da = 2(a - t)      = {:+.6f}\n".format(c["dL_da"]) +
            "da/dn = {:<14s}= {:+.6f}\n".format(ACTS[act][2], c["da_dn"]) +
            "dL/dn = dL/da * da/dn = {:+.6f}\n".format(c["dL_dn"]) +
            "dL/dw = dL/dn * p     = {:+.6f}   (numerical check {:+.6f})\n".format(c["dL_dw"], num_w) +
            "dL/db = dL/dn * 1     = {:+.6f}   (numerical check {:+.6f})\n\n".format(c["dL_db"], num_b) +
            "the same model in PyTorch\n"
            "-------------------------\n"
            "w = torch.tensor({:.2f}, requires_grad=True)\n".format(c["w"]) +
            "b = torch.tensor({:.2f}, requires_grad=True)\n".format(c["b"]) +
            "model     = lambda p: " + ("w*p + b\n" if act == "linear" else "torch.{}(w*p + b)\n".format(act)) +
            "loss_fn   = lambda out, t: (out - t)**2\n"
            "optimizer = torch.optim.SGD([w, b], lr={})".format(LR))
