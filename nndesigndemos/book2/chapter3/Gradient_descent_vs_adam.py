"""
Chapter 3 - Multilayer Network Training: Gradient Descent vs Adam
=================================================================
"Optimizing Performance - Gradient Descent" and
"Variations on Gradient Descent - Adam"

Two copies of the SAME network, started from the SAME initial weights x(0)
and trained on the SAME data. What differs is the update rule -- and its
learning rate, because the two algorithms do not want the same one:

    Gradient descent          (Eq. 3.14)
        x(k+1) = x(k) - a * grad F(x(k))

    Adam                      (Eq. 3.29 - 3.33)
        m(k+1) = B1*m(k) + (1 - B1)*g(k)
        v(k+1) = B2*v(k) + (1 - B2)*g(k) o g(k)
        mhat   = m(k+1) / (1 - B1^(k+1))
        vhat   = v(k+1) / (1 - B2^(k+1))
        x(k+1) = x(k) - a * mhat / (sqrt(vhat) + e)

The performance index F(x) is cross entropy with a softmax last layer, so the
network output a = softmax(model(p)) is the probability of each class. The
network, the performance index and both optimizers are written out in NumPy
below.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from PyQt6 import QtWidgets

from nndesigndemos.book2.panel_utils import PanelDemo, PLOT_FULL, FS, plain_log_ticks

METHODS = ["Gradient Descent", "Adam"]
COLOR = {"Gradient Descent": "#c0392b", "Adam": "#1f4e9c"}


# ===========================================================================
#  A multilayer network, written out.  R - S1 - S2, tanh then softmax.
# ===========================================================================
class Network:
    def __init__(self, S1=10, seed=0):
        rng = np.random.default_rng(seed)
        k1 = 1 / np.sqrt(2)
        k2 = 1 / np.sqrt(S1)
        # x = [W1, b1, W2, b2]  -- the parameter vector of Chapter 3
        self.x = [rng.uniform(-k1, k1, (2, S1)), rng.uniform(-k1, k1, S1),
                  rng.uniform(-k2, k2, (S1, 2)), rng.uniform(-k2, k2, 2)]

    def net_out(self, p):
        """n2, the net input of the last layer (the logits)."""
        W1, b1, W2, b2 = self.x
        return np.tanh(p @ W1 + b1) @ W2 + b2

    @staticmethod
    def softmax(n):
        z = n - n.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def a(self, p):
        """a = softmax(model(p)) -- the class probabilities."""
        return self.softmax(self.net_out(p))

    def F(self, p, t):
        """Cross-entropy performance index."""
        a = self.a(p)
        return float(-np.log(a[np.arange(len(t)), t] + 1e-300).mean())

    def grad(self, p, t):
        """grad F(x), by backpropagation."""
        W1, b1, W2, b2 = self.x
        n1 = p @ W1 + b1
        a1 = np.tanh(n1)
        a2 = self.softmax(a1 @ W2 + b2)
        Q = len(t)
        s2 = a2.copy()                      # dF/dn2 = a - t for softmax + CE
        s2[np.arange(Q), t] -= 1.0
        s2 /= Q
        s1 = (s2 @ W2.T) * (1 - a1 ** 2)
        return [p.T @ s1, s1.sum(0), a1.T @ s2, s2.sum(0)]


class GradientDescentRule:
    """x(k+1) = x(k) - a grad F(x(k))        Eq. 3.14"""

    def __init__(self, x, alpha):
        self.alpha = alpha

    def step(self, x, g):
        for xi, gi in zip(x, g):
            xi -= self.alpha * gi


class AdamRule:
    """Eq. 3.29 - 3.33, with the bias correction of Eq. 3.31 - 3.32."""

    def __init__(self, x, alpha, b1=0.9, b2=0.999, eps=1e-8):
        self.alpha, self.b1, self.b2, self.eps = alpha, b1, b2, eps
        self.m = [np.zeros_like(xi) for xi in x]
        self.v = [np.zeros_like(xi) for xi in x]
        self.k = 0

    def step(self, x, g):
        self.k += 1
        for i, (xi, gi) in enumerate(zip(x, g)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * gi
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * gi * gi
            mhat = self.m[i] / (1 - self.b1 ** self.k)
            vhat = self.v[i] / (1 - self.b2 ** self.k)
            xi -= self.alpha * mhat / (np.sqrt(vhat) + self.eps)


# ===========================================================================
class GradientDescentVsAdam(PanelDemo):
    title = "Gradient Descent vs Adam"
    chapter = 3
    description = ("Two identical networks\nlearn the same task, one\nusing gradient descent,\n"
                   "one using Adam. Press\nTrain to see which one\nreduces F(x) faster.")

    def __init__(self, w_ratio, h_ratio, dpi):
        super(GradientDescentVsAdam, self).__init__(w_ratio, h_ratio, dpi)

        self.p = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
        self.t = np.array([0, 1, 1, 0])

        # ---------------- figure (left) ----------------
        self.make_figure(PLOT_FULL)

        # ---------------- controls (right) ----------------
        panel = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(panel)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(4)

        box, bl = self.group("Network and algorithm")
        self.s_h = self.slider("neurons in the first layer  S1", 1, 30, 10, self.reset)
        self.s_lr_gd = self.slider("learning rate  a  --  Gradient Descent", -4, 0, 0.1, self.reset, log=True)
        self.s_lr_adam = self.slider("learning rate  a  --  Adam", -4, 0, 0.01, self.reset, log=True)
        self.s_b1 = self.slider("Adam  B1", 0.0, 0.999, 0.9, self.reset, decimals=3)
        self.s_b2 = self.slider("Adam  B2", 0.9, 0.99999, 0.999, self.reset, decimals=5)
        self.s_iter = self.slider("iterations per press of Train", 1, 500, 100)
        for w_ in (self.s_h, self.s_lr_gd, self.s_lr_adam, self.s_b1, self.s_b2, self.s_iter):
            bl.addWidget(w_)
        lv.addWidget(box)

        self.btn_train = self.button("Train", self.train, bold=True)
        self.btn_reset = self.button("reset (same starting weights)", self.reset)
        lv.addWidget(self.btn_train)
        lv.addWidget(self.btn_reset)

        self.info = self.text_box(9, min_height=80)
        lv.addWidget(self.info, 1)
        self.make_panel(panel)

        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        S1 = self.s_h.value()
        self.alpha = {"Gradient Descent": self.s_lr_gd.value(), "Adam": self.s_lr_adam.value()}
        self.net = {m: Network(S1, seed=0) for m in METHODS}     # identical x(0)
        self.opt = {"Gradient Descent": GradientDescentRule(self.net["Gradient Descent"].x,
                                                            self.alpha["Gradient Descent"]),
                    "Adam": AdamRule(self.net["Adam"].x, self.alpha["Adam"],
                                     self.s_b1.value(), self.s_b2.value())}
        self.hist = {m: [self.net[m].F(self.p, self.t)] for m in METHODS}
        self.iteration = 0
        self.redraw()

    def train(self):
        n = self.s_iter.value()
        for _ in range(n):
            for m in METHODS:
                net = self.net[m]
                g = net.grad(self.p, self.t)
                self.opt[m].step(net.x, g)
                self.hist[m].append(net.F(self.p, self.t))
        self.iteration += n
        self.redraw()

    # ------------------------------------------------------------------
    def redraw(self):
        fs = FS
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 2, height_ratios=[1, 1.15])

        # ---------- figure 1: both performance curves ----------
        ax = self.figure.add_subplot(gs[0, :])
        for m in METHODS:
            h = np.array(self.hist[m], float)
            h = np.where(np.isfinite(h), h, np.nan)
            ax.semilogy(np.arange(len(h)), np.clip(h, 1e-12, None), color=COLOR[m], lw=1.6,
                        label="{}   (a = {:.5g})".format(m, self.alpha[m]))
        ax.axhline(np.log(2), color="#888", ls="--", lw=1, label="ln 2 = guessing")
        ax.set_xlabel("iteration  k", fontsize=fs)
        ax.set_ylabel("F(x)", fontsize=fs)
        ax.set_title("Performance index F(x)  --  cross entropy", fontsize=fs, fontweight="bold")
        ax.legend(fontsize=fs)
        ax.grid(alpha=.3, which="both")
        ax.tick_params(labelsize=fs)
        plain_log_ticks(ax.yaxis)

        # ---------- figures 2 and 3: softmax(model(p)) ----------
        gx, gy = np.meshgrid(np.linspace(-.4, 1.4, 80), np.linspace(-.4, 1.4, 80))
        grid = np.stack([gx.ravel(), gy.ravel()], 1)
        axes = []
        for j, m in enumerate(METHODS):
            ax2 = self.figure.add_subplot(gs[1, j])
            a = self.net[m].a(grid)[:, 1].reshape(gx.shape)
            cf = ax2.contourf(gx, gy, a, levels=np.linspace(0, 1, 21), cmap="RdBu_r", vmin=0, vmax=1)
            ax2.contour(gx, gy, a, levels=[0.5], colors="k", linewidths=1.3)
            for cls, mk, col in ((0, "o", "#1a3fa0"), (1, "s", "#a01a1a")):
                sel = self.t == cls
                ax2.scatter(self.p[sel, 0], self.p[sel, 1], marker=mk, s=70, c=col,
                            edgecolors="w", linewidths=1.4, zorder=3)
            out = self.net[m].a(self.p)
            ok = int((out.argmax(1) == self.t).sum())
            ax2.set_title("softmax(model(p))  --  {},\na = {:.5g}\nF(x) = {:.5f}      {}/4 correct".format(
                m, self.alpha[m], self.hist[m][-1], ok), fontsize=fs, color=COLOR[m])
            ax2.set_xlabel("p1", fontsize=fs)
            ax2.set_ylabel("p2", fontsize=fs)
            ax2.tick_params(labelsize=fs)
            axes.append(ax2)
        # one colour scale for both panels (they share vmin=0, vmax=1)
        cbar = self.figure.colorbar(cf, ax=axes, fraction=.046, shrink=0.8, ticks=[0, .5, 1])
        cbar.ax.tick_params(labelsize=fs)
        cbar.set_label("a = P(class 1)", fontsize=fs)
        self.canvas.draw_idle()

        # ---------- the numbers ----------
        lines = ["iteration k = {}".format(self.iteration), "",
                 "{:<17s}{:>8s}{:>11s}{:>9s}".format("", "a", "F(x)", "correct")]
        for m in METHODS:
            out = self.net[m].a(self.p)
            lines.append("{:<17s}{:>8.5g}{:>11.6f}{:>7d}/4".format(
                m, self.alpha[m], self.hist[m][-1], int((out.argmax(1) == self.t).sum())))
        a_gd, a_ad = self.alpha["Gradient Descent"], self.alpha["Adam"]
        b1, b2 = self.s_b1.value(), self.s_b2.value()
        lines += ["",
                  "Gradient Descent          (Eq. 3.14)",
                  "  x(k+1) = x(k) - {:.5g} * grad F(x(k))".format(a_gd),
                  "",
                  "Adam                      (Eq. 3.29 - 3.33)",
                  "  m(k+1) = {:.3f}*m(k) + {:.3f}*g(k)".format(b1, 1 - b1),
                  "  v(k+1) = {:.5f}*v(k) + {:.5f}*g(k) o g(k)".format(b2, 1 - b2),
                  "  mhat   = m(k+1) / (1 - B1^(k+1))",
                  "  vhat   = v(k+1) / (1 - B2^(k+1))",
                  "  x(k+1) = x(k) - {:.5g} * mhat / (sqrt(vhat) + e)".format(a_ad),
                  "",
                  "Each algorithm has its own learning rate, so",
                  "each can be tuned to its own best value. Set",
                  "them equal and the only difference left",
                  "between the two figures below is the update",
                  "rule; both networks always start from the",
                  "same x(0).",
                  "",
                  "Kingma and Ba suggest a = 0.001 for Adam.",
                  "Gradient descent normally needs a much larger",
                  "a than that, and goes unstable long before",
                  "Adam does."]
        self.info.setPlainText("\n".join(lines))
