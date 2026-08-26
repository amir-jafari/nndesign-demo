"""
Chapter 6 - TensorFlow Introduction: XOR Training
=================================================
"Training the Network"   --  model.compile(...) then model.fit(p, t, epochs, batch_size)

This is the chapter's XOR example, but nothing is hidden: the whole
2-10-2 network, the softmax, the cross-entropy loss and the optimizers
are written out in NumPy below, in about 80 lines.

Things worth trying
-------------------
* Set the hidden layer to 1 neuron. XOR becomes impossible: the loss
  flattens out around 0.48 and accuracy stops at 3/4 no matter how long
  you train or which seed you use. One hidden tanh neuron can only cut
  the plane with a single line, and no single line separates XOR.
* Turn the learning rate up to ~1.0 with plain SGD and watch it blow up.
* Compare SGD / momentum / Adam at the SAME learning rate.
* Switch to the noisy 200-point problem and change batch_size: notice
  the ITERATION counter grows much faster than the EPOCH counter.
* Press "Re-initialize" a few times: XOR sometimes gets stuck in a bad
  local minimum. That is a real property of the problem, not a bug.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from nndesigndemos.book2.panel_utils import PanelDemo, mono_font, FS, plain_log_ticks

PLOT_RECT = (5, 100, 419, 390)
TABLE_RECT = (5, 495, 419, 175)
DATASETS = ["XOR  (the 4 chapter points)", "AND", "OR", "noisy XOR  (200 points)", "two rings"]
KERAS_OPT = {"adam": "Adam", "sgd": "SGD", "momentum": "SGD", "rmsprop": "RMSprop"}


# ===========================================================================
#  The network -- plain NumPy. This is what Keras builds for you.
# ===========================================================================
def to_categorical(t, n_classes):
    """Same as tensorflow.keras.utils.to_categorical: one-hot encoding."""
    out = np.zeros((len(t), n_classes))
    out[np.arange(len(t)), t.astype(int)] = 1.0
    return out


def softmax(n):
    e = np.exp(n - n.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


ACTS = {
    "tanh":    (lambda n: np.tanh(n),                 lambda a: 1 - a ** 2),
    "relu":    (lambda n: np.maximum(0, n),           lambda a: (a > 0).astype(float)),
    "sigmoid": (lambda n: 1 / (1 + np.exp(-n)),       lambda a: a * (1 - a)),
}


class Net:
    """A 2-H-2 network:  p -> Dense(H, f) -> Dense(2, softmax)"""

    def __init__(self, R, H, S, act="tanh", seed=0):
        rng = np.random.default_rng(seed)
        l1 = np.sqrt(6 / (R + H)); l2 = np.sqrt(6 / (H + S))
        self.W1 = rng.uniform(-l1, l1, (R, H)); self.b1 = np.zeros(H)
        self.W2 = rng.uniform(-l2, l2, (H, S)); self.b2 = np.zeros(S)
        self.f, self.df = ACTS[act]

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, p):
        self.n1 = p @ self.W1 + self.b1
        self.a1 = self.f(self.n1)
        self.n2 = self.a1 @ self.W2 + self.b2
        self.a2 = softmax(self.n2)          # 'softmax' output layer
        return self.a2

    def backward(self, p, t_onehot, loss_kind):
        Q = len(p)
        a = self.a2
        if loss_kind == "categorical_crossentropy":
            # the famous simplification: d(CE)/d(logits) = (a - t)
            dn2 = (a - t_onehot) / Q
        else:                                # mean squared error on the softmax outputs
            dLda = 2 * (a - t_onehot) / (Q * a.shape[1])
            dn2 = a * (dLda - (dLda * a).sum(axis=1, keepdims=True))
        gW2 = self.a1.T @ dn2
        gb2 = dn2.sum(0)
        da1 = dn2 @ self.W2.T
        dn1 = da1 * self.df(self.a1)
        gW1 = p.T @ dn1
        gb1 = dn1.sum(0)
        return [gW1, gb1, gW2, gb2]

    @staticmethod
    def loss(a, t_onehot, kind):
        if kind == "categorical_crossentropy":
            return float(-(t_onehot * np.log(np.clip(a, 1e-12, 1))).sum(1).mean())
        return float(((a - t_onehot) ** 2).mean())


class Optimizer:
    """SGD / SGD+momentum / RMSprop / Adam -- the `optimizer=` of model.compile."""

    def __init__(self, kind, lr, params, mom=0.9, b1=0.9, b2=0.999, eps=1e-7):
        self.kind, self.lr, self.mom = kind, lr, mom
        self.b1, self.b2, self.eps = b1, b2, eps
        self.v = [np.zeros_like(p) for p in params]
        self.s = [np.zeros_like(p) for p in params]
        self.k = 0

    def step(self, params, grads):
        self.k += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            if self.kind == "sgd":
                p -= self.lr * g
            elif self.kind == "momentum":
                self.v[i] = self.mom * self.v[i] - self.lr * g
                p += self.v[i]
            elif self.kind == "rmsprop":
                self.s[i] = 0.9 * self.s[i] + 0.1 * g * g
                p -= self.lr * g / (np.sqrt(self.s[i]) + self.eps)
            else:                                            # adam
                self.v[i] = self.b1 * self.v[i] + (1 - self.b1) * g
                self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * g * g
                vh = self.v[i] / (1 - self.b1 ** self.k)
                sh = self.s[i] / (1 - self.b2 ** self.k)
                p -= self.lr * vh / (np.sqrt(sh) + self.eps)


# ===========================================================================
#  Datasets
# ===========================================================================
def make_data(name):
    if name == "XOR  (the 4 chapter points)":
        p = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
        t = np.array([0, 1, 1, 0])
    elif name == "AND":
        p = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
        t = np.array([0, 0, 0, 1])
    elif name == "OR":
        p = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
        t = np.array([0, 1, 1, 1])
    elif name == "noisy XOR  (200 points)":
        rng = np.random.default_rng(3)
        centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
        lab = np.array([0, 1, 1, 0])
        idx = rng.integers(0, 4, 200)
        p = centers[idx] + rng.normal(0, 0.16, (200, 2))
        t = lab[idx]
    else:                                         # two rings
        rng = np.random.default_rng(5)
        n = 200
        r = np.concatenate([rng.uniform(0.0, 0.30, n // 2),
                            rng.uniform(0.45, 0.65, n // 2)])
        th = rng.uniform(0, 2 * np.pi, n)
        p = np.stack([0.5 + r * np.cos(th), 0.5 + r * np.sin(th)], 1)
        t = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    return p, t


# ===========================================================================
class XORTraining(PanelDemo):
    title = "XOR Training"
    chapter = 6
    description = ("Train a small network to\nsolve XOR. Choose the\noptimizer, loss, learning\n"
                   "rate and batch size, then\nwatch the loss curve and\ndecision boundary evolve.")

    def __init__(self, w_ratio, h_ratio, dpi):
        super(XORTraining, self).__init__(w_ratio, h_ratio, dpi)

        # ---------------- figure + prediction table (left) ----------------
        self.make_figure(PLOT_RECT)

        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(2)
        bl.addWidget(self.label("model.predict(p)  -- the softmax output for each training input", pt=9, bold=True))
        self.table = QtWidgets.QTableWidget()
        self.table.setFont(mono_font(self, 8))
        self.table.horizontalHeader().setFont(mono_font(self, 8))
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(int(self.table.fontMetrics().height() * 1.6))
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        bl.addWidget(self.table, 1)
        self.place(TABLE_RECT, bottom)

        # ---------------- controls (right) ----------------
        panel = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(panel)
        lv.setContentsMargins(4, 0, 4, 0)
        lv.setSpacing(4)

        box, gl = self.group("Data")
        self.c_data = self.combo(DATASETS, self.new_data)
        gl.addWidget(self.c_data)
        lv.addWidget(box)

        box, gl = self.group("Model")
        self.s_hidden = self.slider("hidden neurons  H", 1, 40, 10, self.reinit)
        self.c_act = self.combo(["tanh", "relu", "sigmoid"], self.reinit)
        self.s_seed = self.slider("weight init seed", 0, 50, 0, self.reinit)
        self.btn_init = self.button("Re-initialize weights", self.reinit)
        gl.addWidget(self.s_hidden)
        gl.addLayout(self.row("hidden activation", self.c_act))
        gl.addWidget(self.s_seed)
        gl.addWidget(self.btn_init)
        lv.addWidget(box)

        box, gl = self.group("model.compile(...)")
        self.c_opt = self.combo(["adam", "sgd", "momentum", "rmsprop"], self.new_optimizer)
        self.c_loss = self.combo(["categorical_crossentropy", "mean_squared_error"], self.reset_hist)
        self.s_lr = self.slider("learning rate", -4, 0.3, 0.01, self.new_optimizer, log=True)
        gl.addLayout(self.row("optimizer", self.c_opt))
        gl.addLayout(self.row("loss", self.c_loss))
        gl.addWidget(self.s_lr)
        lv.addWidget(box)

        box, gl = self.group("model.fit(...)")
        self.s_bs = self.slider("batch_size", 1, 4, 4, self.redraw)
        self.s_epochs = self.slider("epochs to run", 100, 20000, 5000, self.redraw)
        self.s_speed = self.slider("animation: epochs / frame", 1, 200, 25)
        self.btn_fit = self.button("model.fit()   [start]", self.toggle, bold=True)
        self.btn_step = self.button("run 1 epoch", lambda: self.run_epochs(1))
        self.btn_reset = self.button("reset history", self.reset_hist)
        for w_ in (self.s_bs, self.s_epochs, self.s_speed, self.btn_fit):
            gl.addWidget(w_)
        hb = QtWidgets.QHBoxLayout()
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(4)
        hb.addWidget(self.btn_step)
        hb.addWidget(self.btn_reset)
        gl.addLayout(hb)
        lv.addWidget(box)

        self.readout = self.text_box(9, min_height=60)
        lv.addWidget(self.readout, 1)
        self.panel_scroll = self.make_panel(panel)

        # ---------------- state / wiring ----------------
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.running = False

        self.new_data()

    def row(self, text, widget):
        """A compact 'label: widget' line for the combos."""
        hb = QtWidgets.QHBoxLayout()
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(4)
        hb.addWidget(self.label(text))
        hb.addWidget(widget, 1)
        return hb

    # ------------------------------------------------------------------
    def new_data(self):
        self.p, self.t = make_data(self.c_data.currentText())
        self.t_oh = to_categorical(self.t, 2)
        Q = len(self.p)
        self.s_bs.set_range(1, Q, min(self.s_bs.value(), Q) or Q)
        self.reinit()

    def reinit(self):
        was = self.running
        self.stop()
        self.net = Net(2, self.s_hidden.value(), 2,
                       act=self.c_act.currentText(), seed=self.s_seed.value())
        self.new_optimizer()
        self.reset_hist()
        if was:
            self.toggle()

    def new_optimizer(self):
        self.opt = Optimizer(self.c_opt.currentText(), self.s_lr.value(),
                             self.net.params())

    def reset_hist(self):
        self.hist = []
        self.epoch = 0
        self.iteration = 0
        self.redraw()

    # ------------------------------------------------------------------
    def toggle(self):
        if self.running:
            self.stop()
        else:
            self.running = True
            self.btn_fit.setText("model.fit()   [pause]")
            self.timer.start(16)

    def stop(self):
        self.running = False
        self.timer.stop()
        self.btn_fit.setText("model.fit()   [start]")

    def _tick(self):
        # keep one animation frame cheap, whatever the batch size is
        per_epoch = int(np.ceil(len(self.p) / max(1, min(self.s_bs.value(), len(self.p)))))
        n = max(1, min(self.s_speed.value(), 2000 // per_epoch or 1))
        self.run_epochs(n, draw=True)
        if self.epoch >= self.s_epochs.value():
            self.stop()

    def run_epochs(self, n, draw=True):
        p, t_oh = self.p, self.t_oh
        Q = len(p)
        bs = max(1, min(self.s_bs.value(), Q))
        kind = self.c_loss.currentText()
        rng = np.random.default_rng(self.epoch + 12345)
        for _ in range(n):
            order = rng.permutation(Q)
            ep_loss = 0.0
            nb = 0
            for k in range(0, Q, bs):                 # <-- one ITERATION per minibatch
                idx = order[k:k + bs]
                pb, tb = p[idx], t_oh[idx]
                a = self.net.forward(pb)
                ep_loss += self.net.loss(a, tb, kind) * len(idx)
                grads = self.net.backward(pb, tb, kind)
                self.opt.step(self.net.params(), grads)
                self.iteration += 1
                nb += 1
            self.epoch += 1
            self.hist.append(ep_loss / Q)
            if not np.isfinite(self.hist[-1]):
                self.stop()
                break
        if draw:
            self.redraw()

    # ------------------------------------------------------------------
    def redraw(self):
        fs = FS
        p, t = self.p, self.t
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 1, height_ratios=[1, 1.2])

        # ---- loss curve ----
        ax = self.figure.add_subplot(gs[0, 0])
        if self.hist:
            h = np.array(self.hist)
            ax.semilogy(np.arange(1, len(h) + 1), np.clip(h, 1e-12, None), color="#1f4e9c", lw=1.4)
            plain_log_ticks(ax.yaxis)
        ax.set_title("Training Loss     (history.history['loss'])", fontsize=fs, fontweight="bold")
        ax.set_xlabel("Epochs", fontsize=fs)
        ax.set_ylabel(self.c_loss.currentText(), fontsize=fs)
        ax.grid(alpha=.3, which="both")
        ax.tick_params(labelsize=fs)
        if self.c_loss.currentText() == "categorical_crossentropy":
            ax.axhline(np.log(2), color="crimson", ls="--", lw=1, label="ln 2 = pure guessing")
            ax.legend(fontsize=fs)

        # ---- decision boundary ----
        ax2 = self.figure.add_subplot(gs[1, 0])
        lo = p.min(0) - 0.35
        hi = p.max(0) + 0.35
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 90),
                             np.linspace(lo[1], hi[1], 90))
        grid = np.stack([gx.ravel(), gy.ravel()], 1)
        prob = self.net.forward(grid)[:, 1].reshape(gx.shape)
        ax2.contourf(gx, gy, prob, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=.75)
        ax2.contour(gx, gy, prob, levels=[0.5], colors="k", linewidths=2)
        for cls, mk, col in ((0, "o", "#1a3fa0"), (1, "s", "#a01a1a")):
            m = t == cls
            ax2.scatter(p[m, 0], p[m, 1], marker=mk, s=40, c=col, edgecolors="w", linewidths=1.0,
                        label="class {}".format(cls), zorder=3)
        ax2.set_title("What the network has learned\n(colour = P(class 1), black line = decision boundary)",
                      fontsize=fs)
        ax2.tick_params(labelsize=fs)
        ax2.legend(fontsize=fs, loc="upper right")
        self.canvas.draw_idle()

        # ---- prediction table ----
        a = self.net.forward(p)
        nshow = min(len(p), 12)
        self.table.setRowCount(nshow)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["p (input)", "t (class)", "t one-hot", "predict -> softmax",
                                              "arg max", "correct?"])
        for i in range(nshow):
            vals = ["[{:.2f}, {:.2f}]".format(p[i, 0], p[i, 1]),
                    str(int(t[i])),
                    "[{:.0f}, {:.0f}]".format(self.t_oh[i, 0], self.t_oh[i, 1]),
                    "[{:.4f}, {:.4f}]".format(a[i, 0], a[i, 1]),
                    str(int(np.argmax(a[i]))),
                    "yes" if np.argmax(a[i]) == t[i] else "NO"]
            for j, v in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(v)
                if j == 5:
                    item.setForeground(QtGui.QColor("#137333" if v == "yes" else "#c5221f"))
                self.table.setItem(i, j, item)
        self.table.resizeColumnsToContents()

        # ---- text readout ----
        acc = float((np.argmax(a, 1) == t).mean())
        cur = self.hist[-1] if self.hist else float("nan")
        Q, bs = len(p), max(1, min(self.s_bs.value(), len(p)))
        per_epoch = int(np.ceil(Q / bs))
        opt, lr = self.c_opt.currentText(), self.s_lr.value()
        # the original readout, re-wrapped so every line fits the narrow panel
        lines = ["epoch      = {:>7d}   / {}".format(self.epoch, self.s_epochs.value()),
                 "iteration  = {:>7d}   ({} per epoch:".format(self.iteration, per_epoch),
                 "                 ceil({}/{}))".format(Q, bs),
                 "loss       = {:.6g}".format(cur),
                 "accuracy   = {:.1f} %   ({}/{})".format(acc * 100, int(acc * Q), Q),
                 "",
                 "the equivalent Keras program",
                 "----------------------------",
                 "model = models.Sequential([",
                 "  layers.Dense({}, activation='{}',".format(self.s_hidden.value(), self.c_act.currentText()),
                 "               input_shape=(2,)),",
                 "  layers.Dense(2, activation='softmax')])",
                 "model.compile(optimizer=keras.optimizers.",
                 "    {}({:.4g}{}),".format(KERAS_OPT[opt], lr, ", momentum=0.9" if opt == "momentum" else ""),
                 "    loss='{}')".format(self.c_loss.currentText()),
                 "history = model.fit(p, t, epochs={},".format(self.s_epochs.value()),
                 "                    batch_size={})".format(bs)]
        self.readout.setPlainText("\n".join(lines))
