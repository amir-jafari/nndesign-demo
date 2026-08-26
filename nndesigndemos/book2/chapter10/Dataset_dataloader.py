"""
Chapter 10 - PyTorch Introduction: Dataset & DataLoader
=======================================================
"Advanced Data Loading"  --  Dataset  vs  DataLoader

The chapter draws a sharp line between the two:

    Dataset     says WHAT the data are.
                __len__()      -> how many samples exist
                __getitem__(i) -> give me sample i (one input, one target)

    DataLoader  says HOW the data are fed to training.
                batching, shuffling, workers, drop_last

This demo implements the chapter's SimpleDataset over SampleDF.csv in
plain NumPy and then shows both objects working:

  * the grid shows which batch each sample lands in, one ROW PER EPOCH,
    so you can see that shuffle=True reshuffles at the START of every
    epoch (a fresh grid row), while shuffle=False repeats the same rows.
  * drop_last decides what happens to the short final batch.
  * the counter shows how often __getitem__ is called, which is why
    heavy preprocessing belongs in __init__.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib.patches
import numpy as np
from PyQt6 import QtCore, QtWidgets

from nndesigndemos.book2.panel_utils import PanelDemo, PLOT_TOP, TEXT_BOTTOM, load_sample_df, FS


# ===========================================================================
#  torch.utils.data.Dataset, written out
# ===========================================================================
class SimpleDataset:
    """
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, P, T):
            self.P = torch.tensor(P, dtype=torch.float32)
            self.T = torch.tensor(T, dtype=torch.float32)
        def __len__(self):
            return len(self.P)
        def __getitem__(self, idx):
            return self.P[idx], self.T[idx]
    """

    def __init__(self, P, T):
        self.P = np.asarray(P, dtype=np.float32)
        self.T = np.asarray(T, dtype=np.float32)
        self.n_getitem_calls = 0

    def __len__(self):
        return len(self.P)

    def __getitem__(self, idx):
        self.n_getitem_calls += 1
        return self.P[idx], self.T[idx]


class DataLoader:
    """
    torch.utils.data.DataLoader(dataset, batch_size=..., shuffle=..., drop_last=...)
    An ITERABLE. Each pass over it is one epoch, and it re-shuffles each time.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False, seed=0):
        self.ds, self.bs = dataset, batch_size
        self.shuffle, self.drop_last, self.seed = shuffle, drop_last, seed

    def __len__(self):
        n = len(self.ds)
        return n // self.bs if self.drop_last else int(np.ceil(n / self.bs))

    def index_batches(self, epoch=0):
        idx = np.arange(len(self.ds))
        if self.shuffle:
            np.random.default_rng(self.seed * 977 + epoch).shuffle(idx)
        out = [idx[k:k + self.bs] for k in range(0, len(idx), self.bs)]
        if self.drop_last and out and len(out[-1]) < self.bs:
            out = out[:-1]
        return out

    def epoch(self, e=0):
        """Yield (features, targets) exactly the way DataLoader does:
        by calling dataset[i] for every i in the batch and stacking."""
        for b in self.index_batches(e):
            feats = np.stack([self.ds[i][0] for i in b])
            targs = np.stack([self.ds[i][1] for i in b])
            yield b, feats, targs


# ===========================================================================
class DatasetDataLoader(PanelDemo):
    title = "Dataset & DataLoader"
    chapter = 10
    description = ("Watch a DataLoader feed\na Dataset to training:\nset batch_size, shuffle\n"
                   "and drop_last, and step\nthrough the epochs\nbatch by batch.")

    def __init__(self, w_ratio, h_ratio, dpi):
        super(DatasetDataLoader, self).__init__(w_ratio, h_ratio, dpi)

        self.P, self.T, self.msg = load_sample_df()
        self.N = len(self.P)

        # ---------------- figure + print loop (left) ----------------
        self.make_figure(PLOT_TOP)

        bottom = QtWidgets.QWidget()
        bv = QtWidgets.QVBoxLayout(bottom)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.setSpacing(2)
        bv.addWidget(self.label("<b>for feat, targ in loader:</b>   -- the batches of the current epoch", pt=9))
        self.out = self.text_box(8)
        bv.addWidget(self.out, 1)
        self.place(TEXT_BOTTOM, bottom)

        # ---------------- controls (right) ----------------
        panel = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(panel)
        lv.setContentsMargins(4, 0, 4, 0)
        lv.setSpacing(4)

        box, bl = self.group("Dataset  --  what the data are")
        self.s_n = self.slider("len(dataset)  -- rows used", 4, self.N, 24, self.rebuild)
        bl.addWidget(self.s_n)
        lv.addWidget(box)

        box, bl = self.group("DataLoader  --  how it is fed")
        self.s_bs = self.slider("batch_size", 1, 32, 4, self.rebuild)
        self.chk_shuf = self.checkbox("shuffle=True", True, self.rebuild)
        self.chk_drop = self.checkbox("drop_last=True", False, self.rebuild)
        self.s_ep = self.slider("epochs to show", 1, 8, 4, self.rebuild)
        self.btn_seed = self.button("new shuffle seed", self.new_seed)
        bl.addWidget(self.s_bs)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.chk_shuf)
        row.addWidget(self.chk_drop)
        bl.addLayout(row)
        bl.addWidget(self.s_ep)
        bl.addWidget(self.btn_seed)
        lv.addWidget(box)

        box, bl = self.group("the training loop")
        self.s_cur_ep = self.slider("current epoch", 0, 3, 0, self.redraw)
        self.s_cur_b = self.slider("current batch", 0, 5, 0, self.redraw)
        self.btn_next = self.button("next batch  ->", self.next_batch)
        self.btn_play = self.button("auto-play the training loop", self.toggle_play)
        bl.addWidget(self.s_cur_ep)
        bl.addWidget(self.s_cur_b)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.btn_next)
        row.addWidget(self.btn_play, 1)
        bl.addLayout(row)
        lv.addWidget(box)

        self.info = self.text_box(9, min_height=70)
        lv.addWidget(self.info, 1)
        self.make_panel(panel)

        self.seed = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_batch)
        self.playing = False

        self.rebuild()

    # ------------------------------------------------------------------
    def new_seed(self):
        self.seed += 1
        self.rebuild()

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.setText("stop" if self.playing else "auto-play the training loop")
        if self.playing:
            self.timer.start(450)
        else:
            self.timer.stop()

    def next_batch(self):
        nb = len(self.loader)
        b = self.s_cur_b.value() + 1
        e = self.s_cur_ep.value()
        if b >= nb:                                  # end of epoch -> next epoch
            b = 0
            e = (e + 1) % self.s_ep.value()
        self.s_cur_b.set_value(b)
        self.s_cur_ep.set_value(e)
        self.redraw()

    # ------------------------------------------------------------------
    def rebuild(self):
        n = min(self.s_n.value(), self.N)
        self.ds = SimpleDataset(self.P[:n], self.T[:n])
        self.s_bs.set_max(min(32, n))
        bs = max(1, min(self.s_bs.value(), n))
        self.loader = DataLoader(self.ds, batch_size=bs,
                                 shuffle=self.chk_shuf.isChecked(),
                                 drop_last=self.chk_drop.isChecked(), seed=self.seed)
        self.s_cur_ep.set_max(self.s_ep.value() - 1)
        self.s_cur_b.set_max(max(0, len(self.loader) - 1))
        self.redraw()

    # ------------------------------------------------------------------
    def redraw(self):
        fs = FS
        n = len(self.ds)
        nb = len(self.loader)
        ne = self.s_ep.value()
        ep = min(self.s_cur_ep.value(), ne - 1)
        bi = min(self.s_cur_b.value(), max(0, nb - 1))

        self.figure.clear()
        gs = self.figure.add_gridspec(2, 2, height_ratios=[1.15, 1])

        # ---------- batch assignment, one row per epoch ----------
        ax = self.figure.add_subplot(gs[0, :])
        grid = np.full((ne, n), np.nan)
        for e in range(ne):
            for k, b in enumerate(self.loader.index_batches(e)):
                grid[e, b] = k
        ax.imshow(grid, aspect="auto", cmap="tab20", interpolation="nearest",
                  extent=(-0.5, n - 0.5, ne - 0.5, -0.5))
        # mark dropped samples
        drop = np.isnan(grid)
        ys, xs = np.where(drop)
        ax.scatter(xs, ys, marker="x", s=30, c="k", label="dropped (drop_last)")
        # highlight the current batch
        cur = self.loader.index_batches(ep)[bi] if nb else []
        for s in cur:
            ax.add_patch(matplotlib.patches.Rectangle(
                (s - .5, ep - .5), 1, 1, fill=False, ec="k", lw=2, zorder=4))
        ax.set_yticks(range(ne), ["epoch {}".format(e) for e in range(ne)])
        ax.set_xlabel("dataset index  (the argument of __getitem__)", fontsize=fs)
        ax.set_title("Which batch does each sample land in?  colour = batch number.\n"
                     "shuffle=True gives every epoch a different arrangement.",
                     fontsize=fs)
        ax.tick_params(labelsize=fs)
        if drop.any():
            ax.legend(fontsize=fs, loc="upper right")

        # ---------- the whole dataset ----------
        ax2 = self.figure.add_subplot(gs[1, 0])
        ax2.scatter(self.ds.P, self.ds.T, s=14, c="#c9c9c9", label="dataset")
        if len(cur):
            ax2.scatter(self.ds.P[cur], self.ds.T[cur], s=50, c="crimson",
                        edgecolors="k", zorder=3, label="batch {} of epoch {}".format(bi, ep))
        ax2.set_xlabel("P = FVC", fontsize=fs)
        ax2.set_ylabel("T = Percent", fontsize=fs)
        ax2.legend(fontsize=fs)
        ax2.grid(alpha=.3)
        ax2.tick_params(labelsize=fs)

        # ---------- batch sizes ----------
        ax3 = self.figure.add_subplot(gs[1, 1])
        sizes = [len(b) for b in self.loader.index_batches(ep)]
        cols = ["#c0392b" if k == bi else "#4a7ebb" for k in range(len(sizes))]
        ax3.bar(range(len(sizes)), sizes, color=cols)
        ax3.axhline(self.loader.bs, color="k", ls="--", lw=1, label="batch_size")
        ax3.set_xlabel("batch number within the epoch", fontsize=fs)
        ax3.set_ylabel("samples in batch", fontsize=fs)
        ax3.set_title("len(loader) = {} batches per epoch".format(nb), fontsize=fs)
        ax3.legend(fontsize=fs)
        ax3.tick_params(labelsize=fs)
        self.canvas.draw_idle()

        # ---------- the print loop ----------
        lines = []
        for b, feats, targs in self.loader.epoch(ep):
            mark = "   <== current" if len(b) and len(cur) and b[0] == cur[0] else ""
            lines.append("Features: {}, Target: {}{}".format(
                np.array2string(feats, precision=0, max_line_width=400, separator=" "),
                np.array2string(targs, precision=2, max_line_width=400, separator=" "),
                mark))
            if len(lines) >= 8 and nb > 8:
                lines.append("... ({} more batches)".format(nb - 8))
                break
        self.out.setPlainText("\n".join(lines))

        # ---------- info ----------
        shuf, drop_last = self.chk_shuf.isChecked(), self.chk_drop.isChecked()
        seen = sum(sizes)
        lines = [self.msg, "",
                 "dataset = SimpleDataset(P, T)",
                 "loader  = DataLoader(dataset,",
                 "            batch_size={},".format(self.loader.bs),
                 "            shuffle={}, drop_last={})".format(shuf, drop_last),
                 "",
                 "len(dataset)  = {:<4d} # __len__".format(n),
                 "len(loader)   = {:<4d} # batches per epoch".format(nb),
                 "              = {}({}/{})".format("floor" if drop_last else "ceil", n, self.loader.bs),
                 "iterations    = {} x {} epochs = {}".format(nb, ne, nb * ne),
                 "samples seen per epoch = {}".format(seen)]
        if seen != n:
            lines.append("   <-- some are dropped!")
        lines += ["",
                  "__getitem__ has been called {} times".format(self.ds.n_getitem_calls),
                  "   (DataLoader calls it once per sample",
                  "    per epoch -- that is why heavy work",
                  "    belongs in __init__, not in",
                  "    __getitem__, unless you want it done",
                  "    every epoch)"]
        self.info.setPlainText("\n".join(lines))
