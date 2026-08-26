"""
Chapter 6 - TensorFlow Introduction: Dataset Pipeline
=====================================================
"Advanced Data Loading"  --  tf.data.Dataset

The chapter builds a pipeline out of SampleDF.csv:

    dataset = Dataset.from_tensor_slices((P, T))
    dataset = dataset.shuffle(Q)
    dataset = dataset.batch(5)
    for feat, targ in dataset: ...

A tf.data pipeline is just a chain of operations. This demo lets you
assemble the chain and then walk through the batches one at a time.

    Q                the number of samples taken from SampleDF.csv
    sequence order   .shuffle(Q) -- draw the samples in a random order
                     instead of the order they sit in the file
    batch_size       how many samples ride together in one minibatch

What to watch
-------------
* Untick "sequence order" and the middle plot becomes a straight
  diagonal: the model sees row 0, then row 1, then row 2 ... Tick it and
  the diagonal turns into noise -- that is the whole point of shuffling.
* The bottom plot is the target the model actually receives, in the
  order it receives it. Unshuffled data often arrives sorted, and the
  network then spends a whole epoch seeing only one part of the range.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib.patches
import numpy as np
from PyQt6 import QtCore, QtWidgets

from nndesigndemos.book2.panel_utils import PanelDemo, PLOT_TOP, TEXT_BOTTOM, load_sample_df, FS


# ===========================================================================
class DatasetPipeline(PanelDemo):
    title = "Dataset Pipeline"
    chapter = 6
    description = ("See how a tf.data pipeline\nturns a CSV file into\nshuffled minibatches. Step\n"
                   "through them to see which\nrows the network gets,\nand in which order.")

    def __init__(self, w_ratio, h_ratio, dpi):
        super(DatasetPipeline, self).__init__(w_ratio, h_ratio, dpi)

        self.P, self.T, msg = load_sample_df()
        self.N = len(self.P)
        # the wording of the original demo, from the message of load_sample_df
        if msg.startswith("SampleDF.csv:"):
            self.src_msg = "read {} rows from SampleDF.csv".format(self.N)
        else:
            exc = msg[msg.find("(") + 1:msg.find(")")]
            self.src_msg = "SampleDF.csv not found ({}); using fake data".format(exc)

        # ---------------- figure (left, top) ----------------
        self.make_figure(PLOT_TOP)

        # ---------------- the chapter's print loop (left, bottom) ----------------
        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(2)
        bl.addWidget(self.label("for feat, targ in dataset: print(...)", pt=9, bold=True))
        self.out = self.text_box(8)
        bl.addWidget(self.out, 1)
        self.place(TEXT_BOTTOM, bottom)

        # ---------------- controls (right) ----------------
        panel = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(panel)
        lv.setContentsMargins(4, 0, 4, 0)
        lv.setSpacing(4)

        box, gl = self.group("Source:  Dataset.from_tensor_slices((P, T))")
        gl.addWidget(self.label("P = FVC   (input feature)\nT = Percent (target)\n"
                                "Q = how many samples are taken from the file", wrap=True))
        self.s_n = self.slider("Q", 8, self.N, min(40, self.N), self.rebuild)
        gl.addWidget(self.s_n)
        lv.addWidget(box)

        box, gl = self.group("Pipeline (top to bottom)")
        self.chk_shuf = self.checkbox("sequence order   .shuffle(Q)", True, self.rebuild)
        self.s_bs = self.slider(".batch(batch_size)", 1, 32, 5, self.rebuild)
        gl.addWidget(self.chk_shuf)
        gl.addWidget(self.s_bs)
        lv.addWidget(box)

        box, gl = self.group("Iterate")
        self.btn_prev = self.button("<-  previous batch", lambda: self.move(-1))
        self.btn_next = self.button("next batch  ->", lambda: self.move(+1))
        self.btn_play = self.button("auto-play", self.toggle_play)
        self.btn_reshuffle = self.button("re-shuffle (new random seed)", self._reshuffle)
        for w_ in (self.btn_prev, self.btn_next, self.btn_play, self.btn_reshuffle):
            gl.addWidget(w_)
        lv.addWidget(box)

        self.counters = self.text_box(9, min_height=80)
        lv.addWidget(self.counters, 1)
        self.make_panel(panel)

        # ---------------- wiring ----------------
        self.seed = 0
        self.cur = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(lambda: self.move(+1))
        self.playing = False

        self.rebuild()

    # ------------------------------------------------------------------
    def _reshuffle(self):
        self.seed += 1
        self.rebuild()

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.setText("stop" if self.playing else "auto-play")
        if self.playing:
            self.timer.start(500)
        else:
            self.timer.stop()

    def move(self, d):
        if self.batches:
            self.cur = (self.cur + d) % len(self.batches)
        self.redraw()

    # ------------------------------------------------------------------
    def buffer_shuffle(self, items, buf, rng):
        """
        tf.data's shuffle: fill a buffer of `buf` elements, then repeatedly
        emit a RANDOM element from the buffer and refill from the stream.
        buf=1 => no shuffling at all.  buf>=len => a real shuffle.
        """
        out, buffer_, it = [], [], iter(items)
        for _ in range(buf):
            try:
                buffer_.append(next(it))
            except StopIteration:
                break
        while buffer_:
            j = rng.integers(len(buffer_))
            out.append(buffer_.pop(int(j)))
            try:
                buffer_.append(next(it))
            except StopIteration:
                pass
        return out

    def rebuild(self):
        n = min(self.s_n.value(), self.N)          # n is Q
        self.s_bs.set_max(max(1, min(32, n)))

        self.Pp, self.Tp = self.P[:n].copy(), self.T[:n].copy()
        rng = np.random.default_rng(self.seed * 1000 + 1)

        idx = list(range(n))
        bs = max(1, self.s_bs.value())

        # the shuffle buffer is the whole dataset:  .shuffle(Q)
        if self.chk_shuf.isChecked():
            idx = self.buffer_shuffle(idx, n, rng)

        self.batches = [idx[k:k + bs] for k in range(0, n, bs)]
        self.cur = min(self.cur, max(0, len(self.batches) - 1))
        self.redraw()

    # ------------------------------------------------------------------
    def redraw(self):
        fs = FS
        n = len(self.Pp)
        bs = max(1, self.s_bs.value())
        nb = len(self.batches)
        self.figure.clear()
        gs = self.figure.add_gridspec(3, 1, height_ratios=[0.6, 1, 1])

        # ---------- 1. batch-membership strip ----------
        ax = self.figure.add_subplot(gs[0])
        strip = np.full(n, np.nan)
        for bi, b in enumerate(self.batches):
            for s in b:
                strip[s] = bi
        ax.imshow(strip[None, :], aspect="auto", cmap="tab20", interpolation="nearest",
                  extent=(-0.5, n - 0.5, 0, 1))
        for s in self.batches[self.cur] if nb else []:
            ax.add_patch(matplotlib.patches.Rectangle((s - .5, 0), 1, 1, fill=False, ec="k", lw=2))
        ax.set_yticks([])
        ax.set_xlabel("original row number in the CSV", fontsize=fs)
        ax.set_title("Which batch does each row end up in?\n"
                     "(colour = batch number, black = the batch you are looking at)", fontsize=fs)
        ax.tick_params(labelsize=fs)

        # ---------- 2. order of arrival ----------
        ax2 = self.figure.add_subplot(gs[1])
        order = [s for b in self.batches for s in b]
        ax2.plot(order, ".-", lw=.7, ms=4, color="#555")
        if nb:
            start = sum(len(b) for b in self.batches[:self.cur])
            sel = self.batches[self.cur]
            ax2.plot(range(start, start + len(sel)), sel, "o", ms=8, color="crimson")
        ax2.set_xlabel("position in the stream the model sees", fontsize=fs)
        ax2.set_ylabel("CSV row", fontsize=fs)
        ax2.set_title("A diagonal line means 'not shuffled'.\n"
                      "With the buffer set to Q every row can travel anywhere.", fontsize=fs)
        ax2.grid(alpha=.3)
        ax2.tick_params(labelsize=fs)

        # ---------- 3. the targets, in the order the model gets them ----------
        ax3 = self.figure.add_subplot(gs[2])
        ax3.plot(np.arange(len(order)), self.Tp[order], ".-", lw=.7, ms=5, color="#888", label="whole dataset")
        if nb:
            start = sum(len(b) for b in self.batches[:self.cur])
            sel = self.batches[self.cur]
            ax3.plot(range(start, start + len(sel)), self.Tp[sel], "o", ms=8, c="crimson",
                     markeredgecolor="k", zorder=3, label="batch {}".format(self.cur))
        ax3.set_xlabel("position in the stream the model sees", fontsize=fs)
        ax3.set_ylabel("Percent", fontsize=fs)
        ax3.set_title("The target values in the order they reach the network", fontsize=fs)
        ax3.legend(fontsize=fs, loc="upper right")
        ax3.grid(alpha=.3)
        ax3.tick_params(labelsize=fs)
        self.canvas.draw_idle()

        # ---------- text: the chapter's print loop ----------
        lines = []
        for bi in range(nb):
            b = self.batches[bi]
            feat = np.array2string(self.Pp[b], precision=1, max_line_width=1000, separator=" ",
                                   suppress_small=True)
            targ = np.array2string(self.Tp[b], precision=3, max_line_width=1000, separator=" ",
                                   suppress_small=True)
            mark = "  <== you are here" if bi == self.cur else ""
            lines.append("Features: {}, Target: {}{}".format(feat, targ, mark))
        self.out.setPlainText("\n".join(lines) if lines else "(no batches)")

        # ---------- counters ----------
        steps = ["Dataset.from_tensor_slices((P, T))"]
        if self.chk_shuf.isChecked():
            steps.append(".shuffle({})          # buffer_size = Q".format(n))
        steps.append(".batch({}))".format(bs))
        code = "dataset = (\n    " + "\n    ".join(steps)

        short = "   (last one is short)" if nb and len(self.batches[-1]) != bs else ""
        self.counters.setPlainText(
            "{}\n\n".format(self.src_msg)
            + "Q             = {}\n".format(n)
            + "batch_size    = {}\n".format(bs)
            + "batches       = {}{}\n".format(nb, short)
            + "=> iterations = {}\n\n".format(nb)
            + code)
