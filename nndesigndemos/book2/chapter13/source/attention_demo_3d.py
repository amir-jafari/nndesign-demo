"""
Self-Attention 3D Surface Demo
===============================

Two 3D surface plots show how the two elements of the selected attention
output (o(1) or o(2)) vary as a function of p(1), while p(2) is held
fixed and controlled by dragging its handle.

  Left  surface:  o(n)[0]  vs  p(1)[0], p(1)[1]
  Right surface:  o(n)[1]  vs  p(1)[0], p(1)[1]

Use the radio button to switch between o(1) and o(2).  Because both
outputs are plotted against the *same* variable p(1), the surfaces
are genuinely different: o(1) shows how a token's own output depends
on its input, while o(2) shows how one token's output is affected by
the *other* token's input.

Equations (from Chapter 13):
  A^Q = W^Q P,  A^K = W^K P,  A^V = W^V P
  N^A = (1/sqrt(S^K)) [A^K]^T A^Q
  A^A = softmax(N^A, columns)
  A^O = A^V A^A
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D

# ── Attention computation ────────────────────────────────────────────

def softmax_columns(M):
    """Softmax over each column of M."""
    e = np.exp(M - M.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)

def self_attention(P, WQ, WK, WV):
    AQ = WQ @ P
    AK = WK @ P
    AV = WV @ P
    SK = AK.shape[0]
    NA = (AK.T @ AQ) / np.sqrt(SK)
    AA = softmax_columns(NA)
    AO = AV @ AA
    return dict(AQ=AQ, AK=AK, AV=AV, NA=NA, AA=AA, AO=AO)

def compute_surface(grid_x, grid_y, p2, WQ, WK, WV, output_idx):
    """
    Sweep p(1) over the grid with p(2) held fixed.
    Return two surfaces for the two elements of o(output_idx+1).

    output_idx: 0 → return o(1)[0] and o(1)[1]
                1 → return o(2)[0] and o(2)[1]
    """
    rows, cols = grid_x.shape
    S0 = np.zeros((rows, cols))
    S1 = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            p1 = np.array([grid_x[i, j], grid_y[i, j]])
            P = np.column_stack([p1, p2])
            res = self_attention(P, WQ, WK, WV)
            out = res['AO'][:, output_idx]
            S0[i, j] = out[0]
            S1[i, j] = out[1]

    return S0, S1

# ── Helper: rotation matrix ─────────────────────────────────────────

def rot(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s], [s, c]])

# ── State ────────────────────────────────────────────────────────────

p2 = np.array([0.5, 1.0])
dragging = False
output_choice = 0                 # 0 → show o(1), 1 → show o(2)
GRID_N = 40
GRID_LIM = 2.0

gx = np.linspace(-GRID_LIM, GRID_LIM, GRID_N)
gy = np.linspace(-GRID_LIM, GRID_LIM, GRID_N)
GX, GY = np.meshgrid(gx, gy)

# ── Figure layout ────────────────────────────────────────────────────

fig = plt.figure(figsize=(15, 8))
title_text = fig.suptitle("", fontsize=14, fontweight='bold')

# Two 3D axes for the surfaces
ax3d_0 = fig.add_subplot(1, 3, 1, projection='3d')
ax3d_1 = fig.add_subplot(1, 3, 2, projection='3d')

# Small 2D axis for dragging p(2)  (right side)
ax2d = fig.add_axes([0.72, 0.42, 0.24, 0.42])

# Radio buttons to select o(1) or o(2)
ax_radio = fig.add_axes([0.74, 0.24, 0.10, 0.12])
radio = RadioButtons(ax_radio, ['o(1)', 'o(2)'], active=0)
for label in radio.labels:
    label.set_fontsize(12)
    label.set_fontweight('bold')

# W matrix sliders at bottom
ax_wq = fig.add_axes([0.10, 0.10, 0.35, 0.025])
ax_wk = fig.add_axes([0.10, 0.06, 0.35, 0.025])
ax_wv = fig.add_axes([0.10, 0.02, 0.35, 0.025])

s_wq = Slider(ax_wq, '$W^Q$ angle', -180, 180, valinit=0, color='#7ab87a')
s_wk = Slider(ax_wk, '$W^K$ angle', -180, 180, valinit=0, color='#7ab87a')
s_wv = Slider(ax_wv, '$W^V$ angle', -180, 180, valinit=0, color='#7ab87a')

# Reset button
ax_reset = fig.add_axes([0.55, 0.02, 0.10, 0.04])
btn_reset = Button(ax_reset, 'Reset', color='#d0d8e8', hovercolor='#a0b0c8')

# ── Drawing ──────────────────────────────────────────────────────────

def redraw():
    WQ = rot(s_wq.val)
    WK = rot(s_wk.val)
    WV = rot(s_wv.val)

    oi = output_choice
    out_label = f'o({oi+1})'

    title_text.set_text(
        f"Self-Attention:  {out_label}  vs  p(1)    "
        f"(drag p(2) in right panel)")

    S0, S1 = compute_surface(GX, GY, p2, WQ, WK, WV, oi)

    # ── Left 3D: output element [0] ─────────────────────────────
    ax3d_0.clear()
    ax3d_0.plot_surface(GX, GY, S0, cmap='coolwarm', alpha=0.85,
                        edgecolor='none')
    ax3d_0.set_xlabel('p(1)[0]', fontsize=10)
    ax3d_0.set_ylabel('p(1)[1]', fontsize=10)
    ax3d_0.set_zlabel(f'{out_label}[0]', fontsize=10)
    ax3d_0.set_title(f'{out_label}[0]', fontsize=12, fontweight='bold')

    # ── Right 3D: output element [1] ────────────────────────────
    ax3d_1.clear()
    ax3d_1.plot_surface(GX, GY, S1, cmap='coolwarm', alpha=0.85,
                        edgecolor='none')
    ax3d_1.set_xlabel('p(1)[0]', fontsize=10)
    ax3d_1.set_ylabel('p(1)[1]', fontsize=10)
    ax3d_1.set_zlabel(f'{out_label}[1]', fontsize=10)
    ax3d_1.set_title(f'{out_label}[1]', fontsize=12, fontweight='bold')

    # ── Small 2D axis: draggable p(2) ───────────────────────────
    ax2d.clear()
    lim = GRID_LIM
    ax2d.set_xlim(-lim, lim)
    ax2d.set_ylim(-lim, lim)
    ax2d.set_aspect('equal')
    ax2d.grid(True, alpha=0.3)
    ax2d.axhline(0, color='k', linewidth=0.5)
    ax2d.axvline(0, color='k', linewidth=0.5)
    ax2d.set_title('Drag to move p(2)', fontsize=11)
    ax2d.set_xlabel('p(2)[0]', fontsize=10)
    ax2d.set_ylabel('p(2)[1]', fontsize=10)

    # Arrow + handle
    ax2d.annotate('', xy=p2, xytext=[0, 0],
                  arrowprops=dict(arrowstyle='->', color='#b03020',
                                  lw=2.5, mutation_scale=15,
                                  shrinkA=0, shrinkB=0))
    ax2d.plot(p2[0], p2[1], 'o', color='#b03020', markersize=12,
              markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    ax2d.text(p2[0] + 0.12, p2[1] + 0.12,
              f'p(2)=[{p2[0]:.2f}, {p2[1]:.2f}]',
              fontsize=9, color='#b03020', fontweight='bold')

    fig.canvas.draw_idle()

# ── Radio button callback ────────────────────────────────────────────

def on_radio(label):
    global output_choice
    output_choice = 0 if label == 'o(1)' else 1
    redraw()

radio.on_clicked(on_radio)

# ── Mouse events for dragging p(2) ───────────────────────────────────

def on_press(event):
    global dragging
    if event.inaxes is not ax2d:
        return
    click = np.array([event.xdata, event.ydata])
    xlim = ax2d.get_xlim()
    grab = 0.12 * (xlim[1] - xlim[0])
    if np.linalg.norm(click - p2) < grab:
        dragging = True

def on_motion(event):
    if not dragging or event.inaxes is not ax2d:
        return
    p2[:] = np.clip([event.xdata, event.ydata], -GRID_LIM, GRID_LIM)
    redraw()

def on_release(event):
    global dragging
    dragging = False

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

# ── Slider callbacks ─────────────────────────────────────────────────

for s in [s_wq, s_wk, s_wv]:
    s.on_changed(lambda val: redraw())

def reset(event):
    global output_choice
    p2[:] = [0.5, 1.0]
    output_choice = 0
    radio.set_active(0)
    for s in [s_wq, s_wk, s_wv]:
        s.reset()
    redraw()

btn_reset.on_clicked(reset)

# ── Go ───────────────────────────────────────────────────────────────

redraw()
plt.show()
