from PyQt6 import QtWidgets, QtCore
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from mpl_toolkits.mplot3d import art3d

from nndesigndemos.nndesign_layout import NNDLayout
from nndesigndemos.get_package_path import PACKAGE_PATH


class SurfaceLine3D(art3d.Line3D):
    """A Line3D lying on the error surface that is hidden where the surface occludes it.

    mplot3d has no real hidden-line removal: an artist is either drawn over the
    surface or under it as a whole. The vertices of this line live on the
    surface, so before every draw the owner demo works out, for the current
    viewing direction, which vertices are on the visible side of the surface;
    the hidden ones are blanked (NaN), which breaks the line / hides the marker.
    """

    def set_surface_data(self, owner, xs, ys, zs):
        self._owner = owner
        self._surface_verts = (np.asarray(xs, float), np.asarray(ys, float), np.asarray(zs, float))
        self.set_data_3d(*self._surface_verts)

    def draw(self, renderer):
        xs, ys, zs = self._surface_verts
        if len(xs):
            visible = self._owner.visible_from_view(xs, ys, zs)
            xs, ys, zs = (np.where(visible, v, np.nan) for v in (xs, ys, zs))
        self.set_data_3d(xs, ys, zs)
        super().draw(renderer)


class GradientDescentBase(NNDLayout):
    """Shared implementation of the chapter 3 gradient descent demos.

    A single-neuron ADALINE least-squares problem (4 patterns, 2 weights) gives
    the quadratic error surface F(w) = 1/2 w'Aw + b'w + c that both demos
    display: as a 3D surface (top) and as a contour plot (bottom). Clicking on
    the contour plot picks the initial weights; the descent trajectory is then
    animated on both plots at once — a ball rolling on the surface leaving a
    trajectory line, and the same path on the contours.

    Subclasses define the update rule via `gradient_step` and the demo title.
    """

    title = ""
    max_epoch = 100
    animation_interval = 70    # ms per step
    convergence_tol = 1e-3     # stop the run once a step is shorter than this
    domain = (-3.0, 0.0)       # the surface / contours are drawn on domain x domain
    diverge_margin = 1.0       # trajectory leaving domain by more than this = diverged
    initial_view = (48, -60)   # (elev, azim) of the 3D plot when the demo opens: seen from above
    ball_lift = 1.5            # how far (in F units, ~ the marker radius) the ball sits above the surface

    def __init__(self, w_ratio, h_ratio, dpi):
        super(GradientDescentBase, self).__init__(w_ratio, h_ratio, dpi, main_menu=2)

        self.fill_chapter(self.title, 3, "\nClick anywhere on the\ngraph to start an initial guess."
                                         "\nThen the steepest descent\ntrajectory will be shown.\n\n"
                                         "Modify the learning rate\nby moving the slide bar.\n\n"
                                         "Experiment with different\ninitial guesses and\nlearning rates.",
                          PACKAGE_PATH + "Chapters/3_D/Logo_Ch_3.svg", None, 2,
                          icon_move_left=120, description_coords=(535, 105, 450, 250))

        # --- Problem definition: F(x) = 1/2 x'Ax + b'x + c (sum squared error) ---
        self.hh = np.array([[-1, 2, 0, -1], [2, -1, -1, 0]], dtype=float)   # input patterns (columns)
        self.t = np.array([-1, -1, 1, 1], dtype=float)                       # targets
        self.A = 2 * self.hh @ self.hh.T
        self.b = -2 * self.hh @ self.t
        self.c = float(self.t @ self.t)

        lo, hi = self.domain
        grid = np.linspace(lo, hi, 61)
        X1, X2 = np.meshgrid(grid, grid)
        F = self.error(X1, X2)

        # --- Plots (built once; only the trajectory artists change afterwards) ---
        self.make_plot(1, (100, 100, 290, 290))
        self.make_plot(2, (100, 390, 290, 290))
        self.figure.set_tight_layout(True)
        self.figure2.set_tight_layout(True)

        # computed_zorder=False: draw in zorder order, so the ball and its path
        # (zorder 5/6) are drawn over the surface (zorder 1) instead of being
        # hidden by mplot3d's depth sorting. Which parts of them are actually
        # visible from the current viewpoint is handled by SurfaceLine3D.
        self.ax3d = self.figure.add_subplot(projection='3d', computed_zorder=False)
        self.ax3d.plot_surface(X1, X2, F, cmap='coolwarm', alpha=0.85, edgecolor='none', zorder=1)
        self.ax3d.set_xlim(lo, hi)
        self.ax3d.set_ylim(lo, hi)
        self.ax3d.set_zlim(0, float(F.max()))
        self.ax3d.view_init(elev=self.initial_view[0], azim=self.initial_view[1])
        self.ax3d.set_xlabel(r'$w_{1,1}$', fontsize=8)
        self.ax3d.set_ylabel(r'$w_{1,2}$', fontsize=8)
        self.ax3d.set_zlabel('F', fontsize=8)
        self.ax3d.set_title('Error Surface', fontsize=10, fontweight='bold')
        self.path3d = SurfaceLine3D([], [], [], color='k', linewidth=1.5, zorder=5)
        self.ball3d = SurfaceLine3D([], [], [], linestyle='none', marker='o', color='red',
                                    markersize=7, markeredgecolor='k', zorder=6)
        for line in (self.path3d, self.ball3d):
            line.set_surface_data(self, [], [], [])
            self.ax3d.add_line(line)

        self.ax2d = self.figure2.add_subplot(111)
        self.ax2d.contour(X1, X2, F)
        self.ax2d.set_xlim(lo, hi)
        self.ax2d.set_ylim(lo, hi)
        self.ax2d.set_xlabel(r'$\mathrm{w}_{1,1}$')
        self.ax2d.set_ylabel(r'$\mathrm{w}_{1,2}$')
        self.setup_contour_axes()
        self.start2d, = self.ax2d.plot([], [], 'o', fillstyle='none', markersize=11, color='k')
        self.path2d, = self.ax2d.plot([], [], '.-', color='b', linewidth=1, markersize=3)
        self.ball2d, = self.ax2d.plot([], [], 'o', color='red', markersize=7, markeredgecolor='k')

        self.animated_artists = {self.canvas: (self.path3d, self.ball3d),
                                 self.canvas2: (self.start2d, self.path2d, self.ball2d)}
        self.backgrounds = {}
        self.canvas.mpl_connect('draw_event', self.on_draw)
        self.canvas2.mpl_connect('draw_event', self.on_draw)
        self.canvas2.mpl_connect('button_press_event', self.on_mouseclick)

        # --- Learning rate slider ---
        self.make_slider("slider_lr", QtCore.Qt.Orientation.Horizontal, (0, 15), QtWidgets.QSlider.TickPosition.TicksBelow, 1, 1,
                         (self.x_chapter_usual, 360, self.w_chapter_slider, 50), self.slider, "label_lr",
                         "Learning rate: 0.01", (self.x_chapter_usual + 40, 360 - 25, self.w_chapter_slider, 50))
        self.lr = float(self.slider_lr.value() / 100)
        # A run (and possibly a warning popup) starts only once the slider is
        # released / left alone for a moment, not on every tick of a drag.
        self.slider_lr.sliderReleased.connect(self.slider)
        self.lr_timer = QtCore.QTimer(self)
        self.lr_timer.setSingleShot(True)
        self.lr_timer.setInterval(150)
        self.lr_timer.timeout.connect(self.on_lr_settled)

        # --- Animation state ---
        self.ani_timer = QtCore.QTimer(self)
        self.ani_timer.setInterval(self.animation_interval)
        self.ani_timer.timeout.connect(self.on_frame)
        self.trajectory = np.zeros((1, 2))
        self.trajectory_z = np.zeros(1)
        self.frame = 0

        self.start = (-1.0, -2.95)
        self.canvas.draw()
        self.canvas2.draw()
        self.start_run()

    # ------------------------------------------------------------------ hooks
    def gradient_step(self, x):
        """Return the update direction (negative SSE gradient, or a sample of it) at weights x."""
        raise NotImplementedError

    def setup_contour_axes(self):
        """Optional per-demo tweaks of the contour axes."""
        pass

    # ------------------------------------------------------------------ math
    def error(self, x1, x2):
        """Sum squared error F evaluated at (x1, x2); works element-wise on arrays."""
        A, b, c = self.A, self.b, self.c
        return (A[0, 0] * x1 ** 2 + (A[0, 1] + A[1, 0]) * x1 * x2 + A[1, 1] * x2 ** 2) / 2 \
            + b[0] * x1 + b[1] * x2 + c

    def compute_trajectory(self, start):
        """Run the descent from `start`; returns the visited points, (n+1) x 2."""
        x = np.array(start, dtype=float)
        points = [x.copy()]
        for _ in range(self.max_epoch):
            step = self.lr * self.gradient_step(x)
            x = x + step
            points.append(x.copy())
            if not np.all(np.isfinite(x)) or np.linalg.norm(step) < self.convergence_tol:
                break
        return np.array(points)

    def diverges(self, points):
        lo, hi = self.domain[0] - self.diverge_margin, self.domain[1] + self.diverge_margin
        return (not np.all(np.isfinite(points))) or np.any(points < lo) or np.any(points > hi)

    def eye_position(self):
        """Viewpoint of the 3D axes in data coordinates.

        Mirrors Axes3D.get_proj: the eye sits `dist` box-units away from the box
        centre along the (elev, azim) direction, in the normalised world box
        whose sides are the box aspect. Roll spins around the view axis and does
        not move the eye, so it is irrelevant here.
        """
        ax = self.ax3d
        elev, azim = np.deg2rad(ax.elev), np.deg2rad(ax.azim)
        direction = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
        aspect = np.asarray(getattr(ax, "_box_aspect", (1.15, 1.15, 0.86)), dtype=float)
        eye_world = 0.5 * aspect + getattr(ax, "_dist", 10) * direction
        lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
        return lims[:, 0] + eye_world / aspect * (lims[:, 1] - lims[:, 0])

    def visible_from_view(self, xs, ys, zs):
        """Which points (xs, ys, zs) lying on the surface can be seen from the current viewpoint.

        The ball / line sits a hair (`ball_lift`) above the surface z = F(x1, x2).
        F is a convex quadratic, so along the segment from such a point P
        towards the eye E the height above the surface is the concave parabola
        g(s) = lift + beta s - alpha s^2 (beta: slope out of the tangent plane,
        alpha = 1/2 d'Ad >= 0), which has exactly one positive root: where the
        line of sight pierces the (infinite) surface. P is hidden iff that
        piercing point lies on the plotted surface patch, i.e. inside the
        domain and between P and the eye. Seen from below, the root is right
        next to P (hidden); seen from above it is far away or outside the
        domain (visible); a point at the rim can peek over the edge.
        """
        pts = np.column_stack([xs, ys])
        d = self.eye_position()[None, :] - np.column_stack([pts, zs])
        d_xy = d[:, :2]
        grad = pts @ self.A + self.b                       # gradient of F at each point (A symmetric)
        beta = d[:, 2] - np.sum(grad * d_xy, axis=1)
        alpha = 0.5 * np.sum((d_xy @ self.A) * d_xy, axis=1)
        lo, hi = self.domain
        with np.errstate(divide='ignore', invalid='ignore'):
            root = (beta + np.sqrt(beta ** 2 + 4 * alpha * self.ball_lift)) / (2 * alpha)
            root_linear = np.where(beta < 0, -self.ball_lift / beta, np.inf)   # alpha == 0: line of sight along z
            s_cross = np.where(alpha > 0, root, root_linear)
            crossing = pts + s_cross[:, None] * d_xy
            occluded = (s_cross <= 1) & np.all((crossing >= lo) & (crossing <= hi), axis=1)
        return ~occluded

    # ------------------------------------------------------------------ events
    def slider(self):
        self.lr = float(self.slider_lr.value() / 100)
        self.label_lr.setText("Learning rate: {}".format(self.lr))
        self.lr_timer.start()

    def on_lr_settled(self):
        if self.slider_lr.isSliderDown():
            return  # still dragging; sliderReleased will bring us back here
        self.start_run()

    def on_mouseclick(self, event):
        if event.xdata is None or event.ydata is None:
            return
        self.start = (float(event.xdata), float(event.ydata))
        self.start_run()

    def on_draw(self, event):
        # Whenever a canvas is fully redrawn (start of a run, 3D view rotated,
        # resize, ...) cache the freshly drawn background so blitting stays
        # correct. During a run the animated artists are excluded from the draw.
        canvas = event.canvas
        self.backgrounds[canvas] = canvas.copy_from_bbox(canvas.figure.bbox)

    # ------------------------------------------------------------------ animation
    def start_run(self):
        self.ani_timer.stop()
        self.trajectory = self.compute_trajectory(self.start)
        self.trajectory_z = self.error(self.trajectory[:, 0], self.trajectory[:, 1])
        self.frame = 0
        self.set_frame_data(0)

        if self.diverges(self.trajectory):
            # Nothing sensible to draw: show only the initial guess and warn.
            self.set_animated(False)
            self.canvas.draw()
            self.canvas2.draw()
            # Headline in setText, details in setInformativeText: that is the
            # layout macOS expects (bold headline + regular body), and no manual
            # line breaks so the text wraps to the box instead of leaving
            # orphaned words.
            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msg.setWindowTitle("Learning rate too large")
            msg.setText("Learning rate too large")
            msg.setInformativeText("With learning rate {} the trajectory diverges and leaves the "
                                   "plotted error surface, so it cannot be shown. "
                                   "Please reduce the learning rate.".format(self.lr))
            msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            msg.exec()
            return

        # Full draw without the animated artists -> on_draw caches the
        # backgrounds; from now on only the trajectory artists are blitted.
        self.set_animated(True)
        self.canvas.draw()
        self.canvas2.draw()
        self.blit()
        self.ani_timer.start()

    def stop_animation(self):
        self.ani_timer.stop()
        # Make the finished trajectory part of the normal draw so it survives
        # later full redraws (3D view rotation, resize).
        self.set_animated(False)
        self.canvas.draw()
        self.canvas2.draw()

    def set_animated(self, flag):
        for artists in self.animated_artists.values():
            for artist in artists:
                artist.set_animated(flag)

    def on_frame(self):
        if self.frame >= len(self.trajectory) - 1:
            self.stop_animation()
            return
        self.frame += 1
        self.set_frame_data(self.frame)
        self.blit()

    def set_frame_data(self, i):
        pts, zs = self.trajectory[:i + 1], self.trajectory_z[:i + 1]
        self.start2d.set_data([pts[0, 0]], [pts[0, 1]])
        self.path2d.set_data(pts[:, 0], pts[:, 1])
        self.ball2d.set_data([pts[-1, 0]], [pts[-1, 1]])
        self.path3d.set_surface_data(self, pts[:, 0], pts[:, 1], zs)
        self.ball3d.set_surface_data(self, [pts[-1, 0]], [pts[-1, 1]], [zs[-1]])

    def blit(self):
        for canvas, artists in self.animated_artists.items():
            background = self.backgrounds.get(canvas)
            if background is None:
                canvas.draw()
                continue
            canvas.restore_region(background)
            for artist in artists:
                artist.axes.draw_artist(artist)
            canvas.blit(canvas.figure.bbox)
