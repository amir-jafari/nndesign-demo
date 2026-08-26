"""
Chapter 6 - TensorFlow Introduction: Tensor Shapes
==================================================
"Loading the Data"  --  What does each axis of a tensor mean?

The chapter says:
    * vector data  ->  2D tensor (samples, features)
    * time series  ->  3D tensor (samples, timesteps, features)
    * images       ->  4D tensor (samples, height, width, channels)
    * video        ->  5D tensor (samples, timesteps, height, width, channels)

This demo lets you *build* one of those tensors with sliders and then
point at a single element with the index sliders, so you can see exactly
which number `x[i, j, k, ...]` picks out.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib.patches
import numpy as np
from PyQt6 import QtWidgets

from nndesigndemos.book2.panel_utils import PanelDemo, PLOT_FULL, FS


# ===========================================================================
#  Fake data with some structure, so the pictures look like *something*.
# ===========================================================================
def smooth_field(h, w, rng, n_waves=6):
    """A low-frequency random image, so pictures look like *something*."""
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    img = np.zeros((h, w))
    for _ in range(n_waves):
        fx, fy = rng.uniform(0.5, 3.0, 2)
        ph = rng.uniform(0, 2 * np.pi)
        img += rng.normal() * np.sin(2 * np.pi * (fx * xx + fy * yy) + ph)
    img -= img.min()
    return img / (img.max() + 1e-12)


def colour_image(h, w, c, rng):
    """
    A fake picture. The channels share one underlying scene and differ only
    a little -- exactly like the R, G, B planes of a real photograph, where
    the three channels are highly correlated rather than independent noise.
    """
    base = smooth_field(h, w, rng)
    return np.clip(np.stack([0.75 * base + 0.25 * smooth_field(h, w, rng, 3)
                             for _ in range(c)], -1), 0, 1)


# ===========================================================================
KINDS = [
    ("Vector data  (samples, features)",
     "2D", ["samples", "features"]),
    ("Time series  (samples, timesteps, features)",
     "3D", ["samples", "timesteps", "features"]),
    ("Images  (samples, height, width, channels)",
     "4D", ["samples", "height", "width", "channels"]),
    ("Video  (samples, timesteps, height, width, channels)",
     "5D", ["samples", "timesteps", "height", "width", "channels"]),
]


def index_expr(idx):
    """The source text of x[idx], e.g. (0, :, :, 2) -> 'x[0, :, :, 2]'."""
    return "x[" + ", ".join(":" if isinstance(k, slice) else str(k) for k in idx) + "]"


# ===========================================================================
class TensorShapes(PanelDemo):
    title = "Tensor Shapes"
    chapter = 6
    description = ("Build a tensor of vector,\ntime series, image or\nvideo data, then move the\n"
                   "index sliders to see what\neach axis means and which\nslice x[i, j, ...] picks.")

    def __init__(self, w_ratio, h_ratio, dpi):
        super(TensorShapes, self).__init__(w_ratio, h_ratio, dpi)

        self.rng = np.random.default_rng(0)
        self.data = None

        # ---------------- figure (left) ----------------
        self.make_figure(PLOT_FULL)

        # ---------------- controls (right) ----------------
        panel = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(panel)
        lv.setContentsMargins(4, 0, 4, 0)
        lv.setSpacing(4)

        lv.addWidget(self.label("1. What kind of data?", bold=True))
        self.kind = self.combo([k[0] for k in KINDS], pt=9)
        self.kind.setCurrentIndex(2)          # start on images: most visual
        lv.addWidget(self.kind)

        box, bl = self.group("2. Size of each axis")
        bl.setSpacing(1)
        self.s_samples = self.slider("samples  (Q)", 1, 64, 8, self.rebuild)
        self.s_features = self.slider("features  (R)", 1, 32, 6, self.rebuild)
        self.s_steps = self.slider("timesteps", 1, 24, 8, self.rebuild)
        self.s_h = self.slider("height", 4, 64, 24, self.rebuild)
        self.s_w = self.slider("width", 4, 64, 24, self.rebuild)
        self.s_c = self.slider("channels", 1, 4, 3, self.rebuild)
        self.size_sliders = (self.s_samples, self.s_features, self.s_steps, self.s_h, self.s_w, self.s_c)
        for s in self.size_sliders:
            bl.addWidget(s)
        lv.addWidget(box)

        box, bl = self.group("3. Point at one slice")
        bl.setSpacing(1)
        self.i_sample = self.slider("index: sample  i", 0, 7, 0, self.redraw)
        self.i_step = self.slider("index: timestep  j", 0, 7, 0, self.redraw)
        self.i_chan = self.slider("index: channel", 0, 2, 0, self.redraw)
        for s in (self.i_sample, self.i_step, self.i_chan):
            bl.addWidget(s)
        lv.addWidget(box)

        self.btn_new = self.button("Re-roll the random data", self.reroll)
        lv.addWidget(self.btn_new)

        self.info = self.text_box(9, min_height=60)
        lv.addWidget(self.info, 1)
        self.make_panel(panel)

        self.kind.currentIndexChanged.connect(self.rebuild)
        self.rebuild()

    # ------------------------------------------------------------------
    def reroll(self):
        self.rng = np.random.default_rng(self.rng.integers(1 << 30))
        self.rebuild()

    def ndim(self):
        return KINDS[self.kind.currentIndex()][1]

    def axis_names(self):
        return KINDS[self.kind.currentIndex()][2]

    def shape(self):
        Q = self.s_samples.value()
        nd = self.ndim()
        if nd == "2D":
            return (Q, self.s_features.value())
        if nd == "3D":
            return (Q, self.s_steps.value(), self.s_features.value())
        if nd == "4D":
            return (Q, self.s_h.value(), self.s_w.value(), self.s_c.value())
        return (Q, self.s_steps.value(), self.s_h.value(), self.s_w.value(), self.s_c.value())

    def rebuild(self):
        """Regenerate the tensor and show/hide the sliders that apply."""
        nd = self.ndim()
        self.s_features.setVisible(nd in ("2D", "3D"))
        self.s_steps.setVisible(nd in ("3D", "5D"))
        self.s_h.setVisible(nd in ("4D", "5D"))
        self.s_w.setVisible(nd in ("4D", "5D"))
        self.s_c.setVisible(nd in ("4D", "5D"))
        self.i_step.setVisible(nd in ("3D", "5D"))
        self.i_chan.setVisible(nd in ("4D", "5D"))

        shp = self.shape()
        rng = np.random.default_rng(self.rng.integers(1 << 30))

        if nd == "2D":
            Q, R = shp
            # cluster-ish feature vectors so the picture has structure
            self.data = (rng.normal(size=(Q, R)) * 0.4
                         + np.sin(np.linspace(0, 3, R))[None, :]
                         + rng.normal(size=(Q, 1)))
        elif nd == "3D":
            Q, T, R = shp
            t = np.linspace(0, 4 * np.pi, T)[None, :, None]
            f = rng.uniform(0.4, 1.6, (Q, 1, R))
            ph = rng.uniform(0, 2 * np.pi, (Q, 1, R))
            self.data = np.sin(f * t + ph) + 0.08 * rng.normal(size=(Q, T, R))
        elif nd == "4D":
            Q, H, W, C = shp
            self.data = np.stack([colour_image(H, W, C, rng) for _ in range(Q)])
        else:
            Q, T, H, W, C = shp
            base = [colour_image(H, W, C, rng) for _ in range(Q)]
            # a "video": each frame is the base image drifted sideways
            self.data = np.stack(
                [np.stack([np.roll(b, 2 * j, axis=1) for j in range(T)])
                 for b in base])

        self.i_sample.set_max(shp[0] - 1)
        if nd in ("3D", "5D"):
            self.i_step.set_max(shp[1] - 1)
        if nd in ("4D", "5D"):
            self.i_chan.set_max(shp[-1] - 1)
        self.redraw()

    # ------------------------------------------------------------------
    @staticmethod
    def _rgb(img):
        """An (H, W, C) picture as something imshow can show in colour."""
        return np.clip(img[:, :, :3], 0, 1) if img.shape[-1] >= 3 else np.repeat(img[:, :, :1], 3, -1)

    @staticmethod
    def _grid(ax, frames, ncols, highlight):
        """Pictures tiled in rows of ``ncols``, numbered, one outlined in red."""
        h, w = frames[0].shape[:2]
        n = len(frames)
        ncols = min(ncols, n)
        nrows = -(-n // ncols)
        tiles = np.ones((nrows * h, ncols * w, 3))
        for k, f in enumerate(frames):
            r, c = divmod(k, ncols)
            tiles[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
        ax.imshow(tiles)
        for k in range(n):
            r, c = divmod(k, ncols)
            ax.text(c * w - .2, r * h - .2, str(k), fontsize=7, color="w", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.15", fc="k", alpha=.6, lw=0))
        for c in range(1, ncols):
            ax.axvline(c * w - .5, color="w", lw=1.5)
        for r in range(1, nrows):
            ax.axhline(r * h - .5, color="w", lw=1.5)
        if highlight < n:
            r, c = divmod(highlight, ncols)
            ax.add_patch(matplotlib.patches.Rectangle(
                (c * w - .5, r * h - .5), w, h, fill=False, ec="red", lw=2.5))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ("left", "right", "top", "bottom"):
            ax.spines[sp].set_visible(False)

    @staticmethod
    def _grid_ratio(n, h, w):
        """(rows, columns, height / width) of the thumbnail grid for n pictures of h x w."""
        ncols = min(4, n)
        nrows = -(-n // ncols)
        return nrows, ncols, nrows * h / (ncols * w)

    def redraw(self):
        fs = FS
        nd = self.ndim()
        x = self.data
        shp = x.shape
        i = min(self.i_sample.value(), shp[0] - 1)
        self.figure.clear()
        _ = slice(None)

        if nd == "2D":
            gs = self.figure.add_gridspec(2, 1, height_ratios=[1.4, 1])
            ax = self.figure.add_subplot(gs[0])
            ax.imshow(x, aspect="auto", cmap="viridis")
            ax.axhline(i - .5, color="r", lw=1.5)
            ax.axhline(i + .5, color="r", lw=1.5)
            ax.set_xlabel("axis 1 = features  (R)", fontsize=fs)
            ax.set_ylabel("axis 0 = samples  (Q)", fontsize=fs)
            ax.set_title("The whole tensor. One ROW = one sample = one input vector", fontsize=fs + 1)
            ax.tick_params(labelsize=fs)
            ax2 = self.figure.add_subplot(gs[1])
            ax2.bar(np.arange(shp[1]), x[i], color="#3b7dd8")
            ax2.set_title("x[{}]  ->  the feature vector of sample {}   shape {}".format(i, i, x[i].shape),
                          fontsize=fs + 1)
            ax2.set_xlabel("feature number", fontsize=fs)
            ax2.tick_params(labelsize=fs)
            idx = (i, _)

        elif nd == "3D":
            j = min(self.i_step.value(), shp[1] - 1)
            gs = self.figure.add_gridspec(2, 1, height_ratios=[1.4, 1])
            ax = self.figure.add_subplot(gs[0])
            for r in range(shp[2]):
                ax.plot(x[i, :, r], marker="o", ms=3, lw=1.2, label="feature {}".format(r))
            ax.axvline(j, color="r", ls="--", lw=1.5)
            ax.set_title("x[{}]  ->  sample {} is a whole time series, shape {}".format(i, i, x[i].shape),
                         fontsize=fs + 1)
            ax.set_xlabel("axis 1 = timesteps", fontsize=fs)
            if shp[2] <= 8:
                ax.legend(fontsize=fs, ncol=2)
            ax.tick_params(labelsize=fs)
            ax2 = self.figure.add_subplot(gs[1])
            ax2.bar(np.arange(shp[2]), x[i, j], color="#d8733b")
            ax2.set_title("x[{}, {}]  ->  the features at one instant, shape {}".format(i, j, x[i, j].shape),
                          fontsize=fs + 1)
            ax2.set_xlabel("axis 2 = features", fontsize=fs)
            ax2.tick_params(labelsize=fs)
            idx = (i, j, _)

        elif nd == "4D":
            c = min(self.i_chan.value(), shp[-1] - 1)
            n_show = min(shp[0], 12)
            H, W = shp[1], shp[2]
            nrows, ncols, grid_ratio = self._grid_ratio(n_show, H, W)
            # row heights follow the pictures: two side-by-side images, then the thumbnail grid
            gs = self.figure.add_gridspec(2, 2, height_ratios=[0.5 * H / W + 0.15, grid_ratio + 0.1])
            ax = self.figure.add_subplot(gs[0, 0])
            ax.imshow(self._rgb(x[i]) if shp[-1] >= 3 else x[i, :, :, 0], cmap="gray")
            ax.set_title("x[{}]  ->  image {},\nshape {}".format(i, i, x[i].shape), fontsize=fs + 1)
            ax.set_xlabel("axis 2 = width", fontsize=fs)
            ax.set_ylabel("axis 1 = height", fontsize=fs)
            ax.tick_params(labelsize=fs)
            ax2 = self.figure.add_subplot(gs[0, 1])
            ax2.imshow(x[i, :, :, c], cmap="gray")
            ax2.set_title("x[{}, :, :, {}]  ->  one channel,\nshape {}".format(i, c, x[i, :, :, c].shape),
                          fontsize=fs + 1)
            ax2.tick_params(labelsize=fs)
            ax3 = self.figure.add_subplot(gs[1, :])
            self._grid(ax3, [self._rgb(x[k]) for k in range(n_show)], ncols, i)
            ax3.set_title("axis 0 = samples  (first {} of {})".format(n_show, shp[0]), fontsize=fs + 1)
            idx = (i, _, _, c)

        else:  # 5D
            j = min(self.i_step.value(), shp[1] - 1)
            c = min(self.i_chan.value(), shp[-1] - 1)
            n_show = min(shp[1], 12)
            H, W = shp[2], shp[3]
            nrows, ncols, grid_ratio = self._grid_ratio(n_show, H, W)
            gs = self.figure.add_gridspec(2, 2, height_ratios=[grid_ratio + 0.1, 0.5 * H / W + 0.15])
            ax = self.figure.add_subplot(gs[0, :])
            self._grid(ax, [self._rgb(x[i, k]) for k in range(n_show)], ncols, j)
            ax.set_title("x[{}]  ->  video {}: {} frames (axis 1 = timesteps)".format(i, i, shp[1]),
                         fontsize=fs + 1)
            ax2 = self.figure.add_subplot(gs[1, 0])
            ax2.imshow(self._rgb(x[i, j]) if shp[-1] >= 3 else x[i, j, :, :, 0], cmap="gray")
            ax2.set_title("x[{}, {}]  ->  one frame,\nshape {}".format(i, j, x[i, j].shape), fontsize=fs + 1)
            ax2.tick_params(labelsize=fs)
            ax3 = self.figure.add_subplot(gs[1, 1])
            ax3.imshow(x[i, j, :, :, c], cmap="gray")
            ax3.set_title("x[{}, {}, :, :, {}]  ->  one channel\nof one frame".format(i, j, c), fontsize=fs + 1)
            ax3.tick_params(labelsize=fs)
            idx = (i, j, _, _, c)

        self.canvas.draw_idle()
        self._write_info(idx)

    # ------------------------------------------------------------------
    def _write_info(self, idx):
        shp = self.data.shape
        n = int(np.prod(shp))
        lines = [
            "x.shape  = {}".format(shp),
            "x.ndim   = {}".format(self.data.ndim),
            "x.size   = {:,} numbers".format(n),
            "float32  = {:,.1f} KB   ({:,.2f} MB)".format(n * 4 / 1024, n * 4 / 1024 ** 2),
            "",
            "axis meaning",
            "------------",
        ]
        for k, (nm, sz) in enumerate(zip(self.axis_names(), shp)):
            lines.append("  axis {}  {:<10s} = {}".format(k, nm, sz))
        lines += [
            "",
            "you are looking at   {}".format(index_expr(idx)),
            "  -> shape {}".format(self.data[idx].shape),
            "",
            "In Keras this array is handed to",
            "the model as-is:",
            "    model.fit(x, t, epochs=...,",
            "              batch_size=...)",
            "The FIRST axis is always 'samples'.",
            "Keras slices minibatches out of",
            "that axis for you.",
        ]
        self.info.setPlainText("\n".join(lines))
