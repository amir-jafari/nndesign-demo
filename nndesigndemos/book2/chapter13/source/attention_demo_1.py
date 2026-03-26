"""
Self-Attention Demo with 2D Embedding Vectors
==============================================

Interactive visualization of the self-attention operation from Chapter 13.
Two input tokens are represented as 2D vectors. The query, key, and value
weight matrices (W^Q, W^K, W^V) can be adjusted with sliders.

Click and drag the tips of p(1) or p(2) to move the input vectors.

With identity weight matrices, Q=K=V=P, so the attention output is a
weighted combination of the input vectors, where the weights come from
the softmax of the (scaled) dot products between the inputs.

Equations (from Chapter 13):
  A^Q = W^Q P,  A^K = W^K P,  A^V = W^V P
  N^A = (1/sqrt(S^K)) [A^K]^T A^Q
  A^A = softmax(N^A, columns)
  A^O = A^V A^A
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ── Attention computation ────────────────────────────────────────────

def softmax_columns(M):
    """Softmax over each column of M."""
    e = np.exp(M - M.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)

def self_attention(P, WQ, WK, WV):
    """
    Compute self-attention on input matrix P (2xT).
    Returns dict with all intermediate and final results.
    """
    AQ = WQ @ P
    AK = WK @ P
    AV = WV @ P
    SK = AK.shape[0]
    NA = (AK.T @ AQ) / np.sqrt(SK)
    AA = softmax_columns(NA)
    AO = AV @ AA
    return dict(AQ=AQ, AK=AK, AV=AV, NA=NA, AA=AA, AO=AO)

# ── State ────────────────────────────────────────────────────────────

# Two input tokens as 2D vectors (mutable state)
p1 = np.array([1.0, 0.3])
p2 = np.array([0.3, 1.0])

# Drag state
dragging = None        # None, 'p1', or 'p2'
GRAB_RADIUS = 0.15     # data-units; how close click must be to a tip

# ── Helper: rotation matrix from angle in degrees ────────────────────

def rot(deg):
    """2x2 rotation matrix for angle in degrees."""
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s], [s, c]])

# ── Figure layout ────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 8))
fig.suptitle("Self-Attention Demo  (2D Embeddings, 2 Tokens)",
             fontsize=14, fontweight='bold')

# Main vector plot — takes most of the figure
ax = fig.add_axes([0.08, 0.22, 0.55, 0.70])

# Text panel for numerical results
ax_text = fig.add_axes([0.67, 0.22, 0.30, 0.70])
ax_text.axis('off')

# ── Sliders for W matrices only ──────────────────────────────────────

slider_color = '#d0d8e8'

ax_wq = fig.add_axes([0.12, 0.12, 0.35, 0.025])
ax_wk = fig.add_axes([0.12, 0.08, 0.35, 0.025])
ax_wv = fig.add_axes([0.12, 0.04, 0.35, 0.025])

s_wq = Slider(ax_wq, '$W^Q$ angle', -180, 180, valinit=0, color='#7ab87a')
s_wk = Slider(ax_wk, '$W^K$ angle', -180, 180, valinit=0, color='#7ab87a')
s_wv = Slider(ax_wv, '$W^V$ angle', -180, 180, valinit=0, color='#7ab87a')

# Reset button
ax_reset = fig.add_axes([0.60, 0.04, 0.12, 0.04])
btn_reset = Button(ax_reset, 'Reset', color=slider_color, hovercolor='#a0b0c8')

# ── Drawing ──────────────────────────────────────────────────────────

def draw_vector(ax, origin, vec, color, label, linestyle='-', linewidth=2.2,
                alpha=1.0):
    """Draw a vector as an arrow from origin."""
    ax.annotate('', xy=origin + vec, xytext=origin,
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=linewidth, linestyle=linestyle,
                                mutation_scale=15, shrinkA=0, shrinkB=0,
                                alpha=alpha))
    # Label at the tip
    offset = vec / (np.linalg.norm(vec) + 1e-9) * 0.12
    ax.text(origin[0] + vec[0] + offset[0],
            origin[1] + vec[1] + offset[1],
            label, fontsize=11, fontweight='bold', color=color,
            ha='center', va='center', alpha=alpha)

def draw_handle(ax, pos, color):
    """Draw a draggable handle (circle) at a vector tip."""
    ax.plot(pos[0], pos[1], 'o', color=color, markersize=10,
            markeredgecolor='white', markeredgewidth=1.5, zorder=5)

def redraw():
    """Recompute attention and redraw everything."""
    P = np.column_stack([p1, p2])

    WQ = rot(s_wq.val)
    WK = rot(s_wk.val)
    WV = rot(s_wv.val)

    res = self_attention(P, WQ, WK, WV)
    AO = res['AO']
    AA = res['AA']
    o1, o2 = AO[:, 0], AO[:, 1]

    # ── Vector plot ──────────────────────────────────────────────
    ax.clear()
    lim = max(np.max(np.abs(P)), np.max(np.abs(AO)), 1.2) + 0.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Dimension 1', fontsize=11)
    ax.set_ylabel('Dimension 2', fontsize=11)
    ax.set_title('Drag the circles to move p(1) and p(2)', fontsize=12)

    origin = np.array([0.0, 0.0])

    # Input vectors (solid)
    draw_vector(ax, origin, p1, '#2060b0', 'p(1)', linewidth=2.5)
    draw_vector(ax, origin, p2, '#b03020', 'p(2)', linewidth=2.5)

    # Draggable handles at the tips
    draw_handle(ax, p1, '#2060b0')
    draw_handle(ax, p2, '#b03020')

    # Output vectors (dashed)
    draw_vector(ax, origin, o1, '#2060b0', 'o(1)', linestyle='--',
                linewidth=2.0, alpha=0.75)
    draw_vector(ax, origin, o2, '#b03020', 'o(2)', linestyle='--',
                linewidth=2.0, alpha=0.75)

    # Annotation near each output tip showing the weight decomposition
    for (out_vec, col_idx) in [(o1, 0), (o2, 1)]:
        w1, w2 = AA[0, col_idx], AA[1, col_idx]
        color = '#2060b0' if col_idx == 0 else '#b03020'
        ax.text(out_vec[0] - 0.05, out_vec[1] - 0.18,
                f'{w1:.2f}·p(1) + {w2:.2f}·p(2)',
                fontsize=8, color=color, alpha=0.7,
                ha='center', bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor='white', alpha=0.8))

    # ── Text panel ───────────────────────────────────────────────
    ax_text.clear()
    ax_text.axis('off')

    def fmt_vec(v):
        return f'[{v[0]:6.3f}, {v[1]:6.3f}]'

    def fmt_mat(M, name, indent='  '):
        r0 = f'[{M[0,0]:6.3f}  {M[0,1]:6.3f}]'
        r1 = f'[{M[1,0]:6.3f}  {M[1,1]:6.3f}]'
        return f'{name} = {r0}\n{indent}    {r1}'

    lines = []
    lines.append('━━━ Inputs ━━━')
    lines.append(f'  p(1) = {fmt_vec(p1)}')
    lines.append(f'  p(2) = {fmt_vec(p2)}')
    lines.append('')
    lines.append('━━━ Weight Matrices ━━━')
    lines.append(fmt_mat(WQ, 'W^Q'))
    lines.append(fmt_mat(WK, 'W^K'))
    lines.append(fmt_mat(WV, 'W^V'))
    lines.append('')
    lines.append('━━━ Queries, Keys, Values ━━━')
    lines.append(f'  q(1) = {fmt_vec(res["AQ"][:,0])}')
    lines.append(f'  q(2) = {fmt_vec(res["AQ"][:,1])}')
    lines.append(f'  k(1) = {fmt_vec(res["AK"][:,0])}')
    lines.append(f'  k(2) = {fmt_vec(res["AK"][:,1])}')
    lines.append(f'  v(1) = {fmt_vec(res["AV"][:,0])}')
    lines.append(f'  v(2) = {fmt_vec(res["AV"][:,1])}')
    lines.append('')
    lines.append('━━━ Scaled Dot Products (N^A) ━━━')
    lines.append(fmt_mat(res['NA'], 'N^A'))
    lines.append('')
    lines.append('━━━ Attention Weights (A^A) ━━━')
    lines.append(fmt_mat(AA, 'A^A'))
    lines.append('')
    lines.append('━━━ Outputs ━━━')
    lines.append(f'  o(1) = {AA[0,0]:.2f}·v(1) + {AA[1,0]:.2f}·v(2)')
    lines.append(f'       = {fmt_vec(o1)}')
    lines.append(f'  o(2) = {AA[0,1]:.2f}·v(1) + {AA[1,1]:.2f}·v(2)')
    lines.append(f'       = {fmt_vec(o2)}')

    text = '\n'.join(lines)
    ax_text.text(0.0, 1.0, text, transform=ax_text.transAxes,
                 fontsize=9, fontfamily='monospace',
                 verticalalignment='top')

    fig.canvas.draw_idle()

# ── Mouse event handlers for click-and-drag ──────────────────────────

def on_press(event):
    """Start dragging if click is near a vector tip."""
    global dragging
    if event.inaxes is not ax:
        return
    click = np.array([event.xdata, event.ydata])

    # Check distance to each vector tip; pick the closer one
    d1 = np.linalg.norm(click - p1)
    d2 = np.linalg.norm(click - p2)

    # Scale grab radius with the current axis limits so it feels
    # consistent regardless of zoom level
    xlim = ax.get_xlim()
    grab = GRAB_RADIUS * (xlim[1] - xlim[0])

    if d1 < grab and d1 <= d2:
        dragging = 'p1'
    elif d2 < grab:
        dragging = 'p2'
    else:
        dragging = None

def on_motion(event):
    """Update the dragged vector tip to follow the mouse."""
    global p1, p2
    if dragging is None or event.inaxes is not ax:
        return
    new_pos = np.array([event.xdata, event.ydata])
    if dragging == 'p1':
        p1[:] = new_pos
    else:
        p2[:] = new_pos
    redraw()

def on_release(event):
    """Stop dragging."""
    global dragging
    dragging = None

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

# ── Slider callbacks ─────────────────────────────────────────────────

for s in [s_wq, s_wk, s_wv]:
    s.on_changed(lambda val: redraw())

def reset(event):
    global p1, p2
    p1[:] = [1.0, 0.3]
    p2[:] = [0.3, 1.0]
    for s in [s_wq, s_wk, s_wv]:
        s.reset()
    redraw()

btn_reset.on_clicked(reset)

# ── Initial draw and show ────────────────────────────────────────────

redraw()
plt.show()
