# =============================================================================
# graph_tool.py — AUTHORITATIVE SOURCE FOR ALL GRAPH STYLING AND BEHAVIOR
# =============================================================================
#
# DO NOT modify, refactor, restyle, simplify, optimize, or rewrite anything
# in this file. All styling decisions have been finalized here.
#
# TO USE THIS FILE FOR GRAPH GENERATION:
#   1. Copy this file in full as generate_graphs.py
#   2. Append graph-generation blocks below the final section marker
#   3. Never insert code into the middle of this file
#   4. Never override styling in graph-generation blocks
#
# FAIL THE TASK IF:
#   - Any existing line above the append section is modified
#   - Any graph is generated without using functions from this file
#   - Styling is manually overridden outside graph-generation blocks
#   - HTML, CSS, MathJax, or layout in the resource file is changed
#
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _exit_arrows(ax, f, fprime, color, xmin, xmax, ymin, ymax):
    x = np.linspace(xmin, xmax, 2000)
    y = f(x)
    exit_points = []
    yl = f(xmin)
    if ymin <= yl <= ymax:
        exit_points.append((xmin, yl, -1, 'side'))
    yr = f(xmax)
    if ymin <= yr <= ymax:
        exit_points.append((xmax, yr, 1, 'side'))
    for edge_y, direction in [(ymin, -1), (ymax, 1)]:
        vals = y - edge_y
        for idx in np.where(np.diff(np.sign(vals)))[0]:
            xr = np.interp(0, [vals[idx], vals[idx+1]], [x[idx], x[idx+1]])
            exit_points.append((xr, edge_y, direction, 'topbot'))
    for (xe, ye, direction, edge) in exit_points:
        slope = fprime(xe)
        if abs(slope) < 0.001:
            continue
        if edge == 'topbot':
            dy_dir = float(direction)
            dx_dir = dy_dir / slope
        else:
            dx_dir = float(direction)
            dy_dir = slope * dx_dir
        mag = np.sqrt(dx_dir**2 + dy_dir**2)
        dx_dir /= mag
        dy_dir /= mag
        L = 0.45
        ax.annotate('', xy=(xe + dx_dir*L, ye + dy_dir*L),
                        xytext=(xe, ye),
                    arrowprops=dict(arrowstyle='-|>', color=color,
                                    lw=1.5, mutation_scale=12))


def _find_key_points(functions, xmin, xmax, ymin, ymax):
    key_points = []
    x_check = np.linspace(xmin, xmax, 2000)
    if len(functions) >= 2:
        for i in range(len(functions)):
            for j in range(i+1, len(functions)):
                try:
                    diff = functions[i]['expr'](x_check) - functions[j]['expr'](x_check)
                    for idx in np.where(np.diff(np.sign(diff)))[0]:
                        xr = np.interp(0, [diff[idx], diff[idx+1]],
                                          [x_check[idx], x_check[idx+1]])
                        yr = functions[i]['expr'](np.array([xr]))[0]
                        if ymin <= yr <= ymax:
                            key_points.append((xr, yr))
                except:
                    pass
    for fn in functions:
        try:
            dy = fn['deriv'](x_check)
            for idx in np.where(np.diff(np.sign(dy)))[0]:
                xv = np.interp(0, [dy[idx], dy[idx+1]],
                                  [x_check[idx], x_check[idx+1]])
                yv = fn['expr'](np.array([xv]))[0]
                if ymin <= yv <= ymax:
                    key_points.append((xv, yv))
        except:
            pass
    return key_points


def _score_corner(cx, cy, functions, key_points):
    score = 0
    for fn in functions:
        try:
            fy = fn['expr'](np.array([cx]))[0]
            score += abs(fy - cy) if np.isfinite(fy) else 20
        except:
            score += 20
    for (kx, ky) in key_points:
        dist = np.sqrt((cx - kx)**2 + (cy - ky)**2)
        if dist < 3:
            score -= (3 - dist) * 50
    return score


def _draw_legend(ax, functions, key_points, xmin, xmax, ymin, ymax):
    if len(functions) <= 1:
        return
    handles = [mpatches.Patch(color=fn['color'], label=fn['label'])
               for fn in functions if fn.get('label')]
    if not handles:
        return
    span_x = xmax - xmin
    span_y = ymax - ymin
    corners = {
        'upper right': (xmin + span_x*0.7, ymin + span_y*0.8),
        'upper left':  (xmin + span_x*0.2, ymin + span_y*0.8),
        'lower right': (xmin + span_x*0.7, ymin + span_y*0.15),
        'lower left':  (xmin + span_x*0.2, ymin + span_y*0.15),
    }
    best = max(corners, key=lambda c: _score_corner(*corners[c], functions, key_points))
    ax.legend(
        handles=handles,
        prop={'family': 'Times New Roman', 'size': 11, 'weight': 'bold'},
        loc=best,
        framealpha=0.9,
        edgecolor='#aaaaaa',
        handlelength=1.2,
        borderpad=0.5,
        labelspacing=0.3,
    )


def _nice_grid_step(data_range, max_lines=20):
    raw = data_range / max_lines
    mag = 10 ** np.floor(np.log10(max(raw, 1e-9)))
    for nice in [1, 2, 2.5, 5, 10]:
        step = nice * mag
        if data_range / step <= max_lines:
            return step
    return mag * 10


def _label_every(n_lines):
    if n_lines <= 10:
        return 2
    if n_lines <= 15:
        return 3
    return 5


def _fmt(v):
    return str(int(v)) if v == int(v) else f'{v:g}'


def _draw_axes_standard(ax, xmin, xmax, ymin, ymax):
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_linewidth(1.8)
        ax.spines[s].set_color('#222222')
    # set_bounds stops the spine at ±10.5 — xlim is set to ±10.5 in the
    # calling function so the spine exactly reaches the plot edge.
    # Arrow tips go to ±10.6 with annotation_clip=False so they render
    # outside the plot boundary and the spine cannot reach them.
    ax.spines['left'].set_bounds(ymin - 0.5, ymax + 0.5)
    ax.spines['bottom'].set_bounds(xmin - 0.5, xmax + 0.5)
    tri = dict(arrowstyle='-|>', color='#222222', lw=1.8, mutation_scale=14)
    ax.annotate('', xy=(xmax+0.6, 0), xytext=(xmax, 0),
                arrowprops=tri, annotation_clip=False)
    ax.annotate('', xy=(xmin-0.6, 0), xytext=(xmin, 0),
                arrowprops=tri, annotation_clip=False)
    ax.annotate('', xy=(0, ymax+0.6), xytext=(0, ymax),
                arrowprops=tri, annotation_clip=False)
    ax.annotate('', xy=(0, ymin-0.6), xytext=(0, ymin),
                arrowprops=tri, annotation_clip=False)
    ax.text(xmax+0.35, 0.4, 'x', fontsize=14, fontweight='bold',
            fontfamily='Times New Roman', ha='center', va='bottom')
    ax.text(0.35, ymax+0.35, 'y', fontsize=14, fontweight='bold',
            fontfamily='Times New Roman', ha='left', va='center')


# ─────────────────────────────────────────────────────────────────────────────
# TYPE 1 — STANDARD COORDINATE PLANE  (-10 to 10)
# Use for: algebra, parent functions, transformations, intersections.
#
# functions : list of dicts
#   'expr'  : lambda x: ...       the function
#   'deriv' : lambda x: ...       its derivative
#   'color' : 'steelblue'         steelblue | firebrick | darkorange
#   'label' : 'f'                 legend label — omit or None for 1 function
#
# title: LaTeX supported via r'$...$'  e.g.  r'$f(x)=x^2$'
# ─────────────────────────────────────────────────────────────────────────────

def make_standard_graph(ax, functions, title=''):
    XMIN, XMAX, YMIN, YMAX = -10, 10, -10, 10

    for fn in functions:
        f      = fn['expr']
        fprime = fn['deriv']
        color  = fn['color']
        label  = fn.get('label', None)
        x = np.linspace(XMIN, XMAX, 2000)
        y = f(x)
        mask = (y >= YMIN) & (y <= YMAX)
        segments = np.split(np.where(mask)[0],
                            np.where(np.diff(np.where(mask)[0]) > 5)[0] + 1)
        for seg in segments:
            if len(seg) > 1:
                ax.plot(x[seg], y[seg], color=color, linewidth=2, label=label)
                label = None
        _exit_arrows(ax, f, fprime, color, XMIN, XMAX, YMIN, YMAX)

    key_points = _find_key_points(functions, XMIN, XMAX, YMIN, YMAX)
    _draw_legend(ax, functions, key_points, XMIN, XMAX, YMIN, YMAX)

    ax.set_xlim(XMIN - 0.5, XMAX + 0.5)
    ax.set_ylim(YMIN - 0.5, YMAX + 0.5)
    ax.set_xticks(np.arange(XMIN, XMAX+1, 1), minor=True)
    ax.set_yticks(np.arange(YMIN, YMAX+1, 1), minor=True)
    ax.grid(True, which='minor', color='#aaaaaa', linewidth=0.6)
    ax.set_xticks([-10, -5, 5, 10])
    ax.set_yticks([-10, -5, 5, 10])
    ax.set_xticklabels(['-10','-5','5','10'],
                       fontfamily='Times New Roman', fontsize=11)
    ax.set_yticklabels(['-10','-5','5','10'],
                       fontfamily='Times New Roman', fontsize=11)
    ax.grid(True, which='major', color='#aaaaaa', linewidth=0.6)
    ax.tick_params(which='major', length=5, width=1.2, color='#222222')
    ax.tick_params(which='minor', length=2, width=0.8, color='#555555')
    _draw_axes_standard(ax, XMIN, XMAX, YMIN, YMAX)
    ax.set_title(title, fontfamily='Times New Roman', fontsize=12, pad=8)


# ─────────────────────────────────────────────────────────────────────────────
# TYPE 2 — CONTEXT / MODELING GRAPH
# Custom axis ranges for real-world problems. Q1 only (xmin/ymin typically 0).
# Max 20 grid lines per axis. Labels every 2nd-5th line automatically.
# Same grid darkness and weight as Type 1.
#
# functions : same structure as Type 1
# xmin/xmax/ymin/ymax : axis bounds
# xlabel/ylabel : axis label strings  e.g. 'Time (s)', 'Height (m)'
# title: LaTeX supported via r'$...$'
# ─────────────────────────────────────────────────────────────────────────────

def make_context_graph(ax, functions,
                       xmin, xmax, ymin, ymax,
                       xlabel='x', ylabel='y', title=''):
    x_range = xmax - xmin
    y_range = ymax - ymin

    x_step = _nice_grid_step(x_range)
    y_step = _nice_grid_step(y_range)
    x_ticks = np.arange(xmin, xmax + x_step*0.01, x_step)
    y_ticks = np.arange(ymin, ymax + y_step*0.01, y_step)
    if len(x_ticks) > 20:
        x_step = _nice_grid_step(x_range, max_lines=10)
        x_ticks = np.arange(xmin, xmax + x_step*0.01, x_step)
    if len(y_ticks) > 20:
        y_step = _nice_grid_step(y_range, max_lines=10)
        y_ticks = np.arange(ymin, ymax + y_step*0.01, y_step)

    x_label_every = _label_every(len(x_ticks))
    y_label_every = _label_every(len(y_ticks))

    # ── axis headroom: space for arrowhead past the last gridline ──────────
    # Spine extends 0.2 grid units past xmax/ymax; arrow tip is 0.3 past that,
    # so the spine always ends inside the arrowhead with room to spare.
    # The left/bottom margin is exactly zero — no gap before the first gridline.
    # xlim/ylim stop at xmax + stub — spine ends here.
    # Arrow tip extends x_arrow_pad further using clip_on=False so it
    # pokes out past the plot edge and the spine never reaches it.
    x_stub      = x_step * 0.35
    y_stub      = y_step * 0.35
    x_arrow_pad = x_step * 0.25
    y_arrow_pad = y_step * 0.25
    ax.set_xlim(xmin, xmax + x_stub)
    ax.set_ylim(ymin, ymax + y_stub)

    # ── gridlines clipped exactly to [xmin,xmax] × [ymin,ymax] ────────────
    # Using ax.plot instead of axvline/axhline avoids the infinite-line
    # bleed that created the apparent offset at the axes.
    for xt in x_ticks:
        ax.plot([xt, xt], [ymin, ymax], color='#aaaaaa', linewidth=0.6,
                zorder=0, clip_on=True)
    for yt in y_ticks:
        ax.plot([xmin, xmax], [yt, yt], color='#aaaaaa', linewidth=0.6,
                zorder=0, clip_on=True)

    # ── function curves + exit arrows ─────────────────────────────────────
    # Arrow length scaled to the shorter of the two grid steps so it looks
    # consistent regardless of axis range.
    arrow_L = min(x_step, y_step) * 0.45

    for fn in functions:
        f      = fn['expr']
        fprime = fn['deriv']
        color  = fn['color']
        label  = fn.get('label', None)
        x = np.linspace(xmin, xmax, 2000)
        y = f(x)
        mask = (y >= ymin) & (y <= ymax)
        segments = np.split(np.where(mask)[0],
                            np.where(np.diff(np.where(mask)[0]) > 5)[0] + 1)
        for seg in segments:
            if len(seg) > 1:
                ax.plot(x[seg], y[seg], color=color, linewidth=2, label=label)
                label = None

        # right-edge exit arrow (function leaves through xmax)
        yr_val = f(xmax)
        if ymin <= yr_val <= ymax:
            slope = fprime(xmax)
            dx_d, dy_d = 1.0, float(slope)
            mag = np.sqrt(dx_d**2 + dy_d**2)
            if mag > 0.001:
                dx_d /= mag; dy_d /= mag
                ax.annotate('', xy=(xmax + dx_d*arrow_L, yr_val + dy_d*arrow_L),
                                xytext=(xmax, yr_val),
                            arrowprops=dict(arrowstyle='-|>', color=color,
                                            lw=1.5, mutation_scale=14))

        # top/bottom exit arrows (function crosses ymax or ymin)
        x_arr = np.linspace(xmin, xmax, 2000)
        y_arr = f(x_arr)
        for edge_y, direction in [(ymin, -1), (ymax, 1)]:
            vals = y_arr - edge_y
            for idx in np.where(np.diff(np.sign(vals)))[0]:
                xr = np.interp(0, [vals[idx], vals[idx+1]],
                                  [x_arr[idx], x_arr[idx+1]])
                if xr <= xmin + 0.01:
                    continue
                slope = fprime(xr)
                if abs(slope) < 0.001:
                    continue
                dy_d = float(direction)
                dx_d = dy_d / slope
                mag = np.sqrt(dx_d**2 + dy_d**2)
                dx_d /= mag; dy_d /= mag
                ax.annotate('', xy=(xr + dx_d*arrow_L, edge_y + dy_d*arrow_L),
                                xytext=(xr, edge_y),
                            arrowprops=dict(arrowstyle='-|>', color=color,
                                            lw=1.5, mutation_scale=14))

    key_points = _find_key_points(functions, xmin, xmax, ymin, ymax)
    _draw_legend(ax, functions, key_points, xmin, xmax, ymin, ymax)

    # ── tick labels ────────────────────────────────────────────────────────
    x_labeled = [t for i, t in enumerate(x_ticks) if i % x_label_every == 0]
    y_labeled = [t for i, t in enumerate(y_ticks) if i % y_label_every == 0]
    ax.set_xticks(x_labeled)
    ax.set_yticks(y_labeled)
    ax.set_xticklabels([_fmt(t) for t in x_labeled],
                       fontfamily='Times New Roman', fontsize=11)
    ax.set_yticklabels([_fmt(t) for t in y_labeled],
                       fontfamily='Times New Roman', fontsize=11)
    ax.tick_params(which='major', length=4, width=1.0, color='#444444')

    # ── spines: lighter weight, Q1 only ───────────────────────────────────
    ax.spines['left'].set_linewidth(1.8)
    ax.spines['left'].set_color('#222222')
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['bottom'].set_color('#222222')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # ── axis arrowheads ────────────────────────────────────────────────────
    # Tail at xmax/ymax (last gridline), tip beyond xlim/ylim edge.
    # clip_on=False lets the arrowhead render outside the plot area so
    # the spine (which stops at xlim) never reaches the tip.
    tri = dict(arrowstyle='-|>', color='#222222', lw=1.8, mutation_scale=14)
    ax.annotate('', xy=(xmax + x_stub + x_arrow_pad, ymin),
                    xytext=(xmax, ymin),
                arrowprops=tri,
                annotation_clip=False)
    ax.annotate('', xy=(xmin, ymax + y_stub + y_arrow_pad),
                    xytext=(xmin, ymax),
                arrowprops=tri,
                annotation_clip=False)

    ax.set_xlabel(xlabel, fontfamily='Times New Roman', fontsize=13,
                  fontweight='bold', labelpad=6)
    ax.set_ylabel(ylabel, fontfamily='Times New Roman', fontsize=13,
                  fontweight='bold', labelpad=6, rotation=90)
    ax.set_title(title, fontfamily='Times New Roman', fontsize=12, pad=8)


# ─────────────────────────────────────────────────────────────────────────────
# TYPE 3 — NUMBER LINE
# Use for: inequalities, absolute value solutions, domain, range.
#
# intervals : list of dicts
#   'start'      : left endpoint value
#   'end'        : right endpoint value (same as start for rays)
#   'start_open' : True = open circle, False = closed dot
#   'end_open'   : True = open circle, False = closed dot
#   'direction'  : None | 'left' | 'right'  for rays to infinity
#   'color'      : line color (default 'steelblue')
#
# make_number_line_blank() — blank number line with student writing space above
#   label: LaTeX string shown at top  e.g. r'$x > 3$'
# ─────────────────────────────────────────────────────────────────────────────

def _draw_number_line_base(ax, xmin, xmax):
    x_range = xmax - xmin
    margin  = x_range * 0.07
    ticks   = np.arange(xmin, xmax + 0.5)   # one tick per integer, no duplicates
    line_y  = 0.0
    tick_h  = 0.12
    lw_line = 1.8
    lw_tick = 1.2

    right_tip = xmax + margin * 0.95
    left_tip  = xmin - margin * 0.95

    ax.annotate('', xy=(right_tip, line_y), xytext=(left_tip, line_y),
                arrowprops=dict(arrowstyle='->', color='black',
                                lw=lw_line, mutation_scale=14,
                                shrinkA=0, shrinkB=0))
    ax.annotate('', xy=(left_tip, line_y), xytext=(right_tip, line_y),
                arrowprops=dict(arrowstyle='->', color='black',
                                lw=lw_line, mutation_scale=14,
                                shrinkA=0, shrinkB=0))

    for t in ticks:
        ax.plot([t, t], [line_y - tick_h, line_y + tick_h],
                color='black', linewidth=lw_tick, zorder=3)

    for t in ticks:
        tv = int(round(t))
        if tv == 0 or tv % 2 == 0:
            ax.text(t, line_y - tick_h - 0.10, str(tv),
                    ha='center', va='top',
                    fontfamily='Times New Roman', fontsize=11, color='black')

    return margin, right_tip, left_tip


def _nl_dot(ax, x, y, open_dot, color, size=8):
    if open_dot:
        ax.plot(x, y, 'o', markersize=size, markerfacecolor='white',
                markeredgecolor=color, markeredgewidth=2.5, zorder=6,
                clip_on=False)
    else:
        ax.plot(x, y, 'o', markersize=size, markerfacecolor=color,
                markeredgecolor=color, markeredgewidth=2, zorder=6,
                clip_on=False)


def make_number_line(ax, intervals, xmin=-10, xmax=10):
    x_range = xmax - xmin
    line_y  = 0.0
    seg_y   = 0.55
    lw_seg  = 1.8

    ax.set_xlim(xmin - x_range*0.07, xmax + x_range*0.07)
    ax.set_ylim(-0.5, 0.9)
    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_facecolor('white')

    margin, right_tip, left_tip = _draw_number_line_base(ax, xmin, xmax)

    for iv in intervals:
        color      = iv.get('color', 'steelblue')
        start      = iv['start']
        end        = iv['end']
        direction  = iv.get('direction', None)
        start_open = iv.get('start_open', False)
        end_open   = iv.get('end_open', False)

        if direction == 'right':
            _nl_dot(ax, start, line_y, start_open, color)
            ax.plot([start, start], [line_y + 0.06, seg_y],
                    color=color, linewidth=lw_seg, zorder=2,
                    solid_capstyle='butt')
            ax.annotate('', xy=(right_tip * 0.98, seg_y),
                            xytext=(start, seg_y),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=lw_seg, mutation_scale=14,
                                        shrinkA=0, shrinkB=0))

        elif direction == 'left':
            _nl_dot(ax, start, line_y, start_open, color)
            ax.plot([start, start], [line_y + 0.06, seg_y],
                    color=color, linewidth=lw_seg, zorder=2,
                    solid_capstyle='butt')
            ax.annotate('', xy=(left_tip * 0.98, seg_y),
                            xytext=(start, seg_y),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=lw_seg, mutation_scale=14,
                                        shrinkA=0, shrinkB=0))

        else:
            _nl_dot(ax, start, line_y, start_open, color)
            _nl_dot(ax, end,   line_y, end_open,   color)
            ax.plot([start, start], [line_y + 0.06, seg_y],
                    color=color, linewidth=lw_seg, zorder=2,
                    solid_capstyle='butt')
            ax.plot([end, end], [line_y + 0.06, seg_y],
                    color=color, linewidth=lw_seg, zorder=2,
                    solid_capstyle='butt')
            ax.plot([start, end], [seg_y, seg_y],
                    color=color, linewidth=lw_seg,
                    solid_capstyle='butt', zorder=2)


def make_number_line_blank(ax, label=None, xmin=-10, xmax=10):
    x_range = xmax - xmin
    line_y  = 0.0

    ax.set_xlim(xmin - x_range*0.07, xmax + x_range*0.07)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_facecolor('white')

    _draw_number_line_base(ax, xmin, xmax)

    if label:
        ax.text(0, 2.2, label, ha='center', va='top',
                fontfamily='Times New Roman', fontsize=12, color='black')


# ─────────────────────────────────────────────────────────────────────────────
# SAVE — 504 x 504 px (3.5 in at 144 dpi)
# Number lines save at 504 x 180 px (3.5 x 1.25 in at 144 dpi)
# ─────────────────────────────────────────────────────────────────────────────

def save_graph(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=144, bbox_inches='tight')
    print(f'Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# PASTE YOUR GRAPH CODE BELOW THIS LINE
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TYPE 4 — 2x2 GRID
# Four standard coordinate planes in one 504x504 figure.
# Use for: comparing parent functions, transformations, multiple examples.
#
# functions_list : list of 4 function lists, same format as make_standard_graph
# titles         : list of 4 LaTeX title strings e.g. r'$f(x)=x^2$'
# ─────────────────────────────────────────────────────────────────────────────

def _exit_arrows_small(ax, f, fprime, color, xmin, xmax, ymin, ymax):
    x = np.linspace(xmin, xmax, 2000)
    y = f(x)
    exit_points = []
    yl = f(xmin)
    if ymin <= yl <= ymax:
        exit_points.append((xmin, yl, -1, 'side'))
    yr = f(xmax)
    if ymin <= yr <= ymax:
        exit_points.append((xmax, yr, 1, 'side'))
    for edge_y, direction in [(ymin, -1), (ymax, 1)]:
        vals = y - edge_y
        for idx in np.where(np.diff(np.sign(vals)))[0]:
            xr = np.interp(0, [vals[idx], vals[idx+1]], [x[idx], x[idx+1]])
            exit_points.append((xr, edge_y, direction, 'topbot'))
    for (xe, ye, direction, edge) in exit_points:
        slope = fprime(xe)
        if abs(slope) < 0.001:
            continue
        if edge == 'topbot':
            dy_dir = float(direction)
            dx_dir = dy_dir / slope
        else:
            dx_dir = float(direction)
            dy_dir = slope * dx_dir
        mag = np.sqrt(dx_dir**2 + dy_dir**2)
        dx_dir /= mag
        dy_dir /= mag
        L = 0.45
        ax.annotate('', xy=(xe + dx_dir*L, ye + dy_dir*L),
                        xytext=(xe, ye),
                    arrowprops=dict(arrowstyle='-|>', color=color,
                                    lw=1.0, mutation_scale=7))


def make_2x2_grid(functions_list, titles=None):
    if titles is None:
        titles = ['', '', '', '']

    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.5))
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    XMIN, XMAX, YMIN, YMAX = -10, 10, -10, 10

    for ax, functions, title in zip(axes.flatten(), functions_list, titles):

        for fn in functions:
            f      = fn['expr']
            fprime = fn['deriv']
            color  = fn['color']
            label  = fn.get('label', None)
            x = np.linspace(XMIN, XMAX, 2000)
            y = f(x)
            mask = (y >= YMIN) & (y <= YMAX)
            segments = np.split(np.where(mask)[0],
                                np.where(np.diff(np.where(mask)[0]) > 5)[0] + 1)
            for seg in segments:
                if len(seg) > 1:
                    ax.plot(x[seg], y[seg], color=color, linewidth=1.2,
                            label=label)
                    label = None
            _exit_arrows_small(ax, f, fprime, color, XMIN, XMAX, YMIN, YMAX)

        ax.set_xlim(XMIN - 0.5, XMAX + 0.5)
        ax.set_ylim(YMIN - 0.5, YMAX + 0.5)

        ax.set_xticks(np.arange(XMIN, XMAX+1, 1), minor=True)
        ax.set_yticks(np.arange(YMIN, YMAX+1, 1), minor=True)
        ax.grid(True, which='minor', color='#aaaaaa', linewidth=0.6)

        ax.set_xticks([-10, -5, 5, 10])
        ax.set_yticks([-10, -5, 5, 10])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which='major', color='#aaaaaa', linewidth=0.6)
        ax.tick_params(which='major', length=3.5, width=0.9, color='#222222')
        ax.tick_params(which='minor', length=1.0, width=0.4, color='#555555')

        tick_fs = 8.5
        offset  = 1.0
        for val in [-10, -5, 5, 10]:
            lbl = str(val)
            ax.text(val, -offset, lbl, ha='center', va='top',
                    fontsize=tick_fs, fontfamily='Times New Roman',
                    color='black', clip_on=False)
            ax.text(-offset, val, lbl, ha='right', va='center',
                    fontsize=tick_fs, fontfamily='Times New Roman',
                    color='black', clip_on=False)

        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for s in ['left', 'bottom']:
            ax.spines[s].set_linewidth(1.4)
            ax.spines[s].set_color('#222222')
        ax.spines['left'].set_bounds(YMIN - 0.5, YMAX + 0.5)
        ax.spines['bottom'].set_bounds(XMIN - 0.5, XMAX + 0.5)

        tri = dict(arrowstyle='-|>', color='#222222', lw=1.4, mutation_scale=8)
        ax.annotate('', xy=(XMAX+0.6, 0), xytext=(XMAX, 0), arrowprops=tri, annotation_clip=False)
        ax.annotate('', xy=(XMIN-0.6, 0), xytext=(XMIN, 0), arrowprops=tri, annotation_clip=False)
        ax.annotate('', xy=(0, YMAX+0.6), xytext=(0, YMAX), arrowprops=tri, annotation_clip=False)
        ax.annotate('', xy=(0, YMIN-0.6), xytext=(0, YMIN), arrowprops=tri, annotation_clip=False)

        ax.text(XMAX + 0.2, 0.9, 'x', fontsize=10, fontweight='bold',
                fontfamily='Times New Roman', ha='center', va='bottom')
        ax.text(0.6, YMAX + 0.2, 'y', fontsize=10, fontweight='bold',
                fontfamily='Times New Roman', ha='left', va='center')

        if title:
            ax.set_title(title, fontsize=11, pad=4)

    return fig

# ─────────────────────────────────────────────────────────────────────────────
# TYPE 9 — 2x1 GRID
# Two coordinate planes side by side, each 504x504px (3.5in).
# Use for: before/after, compare/contrast, function and transformation pairs.
#
# functions_list : list of 2 function lists, same format as make_standard_graph
# titles         : list of 2 LaTeX title strings e.g. r'$f(x)=x^2$'
# ─────────────────────────────────────────────────────────────────────────────

def make_2x1_grid(functions_list, titles=None):
    if titles is None:
        titles = ['', '']

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    fig.subplots_adjust(wspace=0.3)

    XMIN, XMAX, YMIN, YMAX = -10, 10, -10, 10

    for ax, functions, title in zip(axes, functions_list, titles):

        for fn in functions:
            f      = fn['expr']
            fprime = fn['deriv']
            color  = fn['color']
            label  = fn.get('label', None)
            x = np.linspace(XMIN, XMAX, 2000)
            y = f(x)
            mask = (y >= YMIN) & (y <= YMAX)
            segments = np.split(np.where(mask)[0],
                                np.where(np.diff(np.where(mask)[0]) > 5)[0] + 1)
            for seg in segments:
                if len(seg) > 1:
                    ax.plot(x[seg], y[seg], color=color, linewidth=1.5,
                            label=label)
                    label = None
            _exit_arrows_small(ax, f, fprime, color, XMIN, XMAX, YMIN, YMAX)

        ax.set_xlim(XMIN - 0.5, XMAX + 0.5)
        ax.set_ylim(YMIN - 0.5, YMAX + 0.5)

        ax.set_xticks(np.arange(XMIN, XMAX+1, 1), minor=True)
        ax.set_yticks(np.arange(YMIN, YMAX+1, 1), minor=True)
        ax.grid(True, which='minor', color='#aaaaaa', linewidth=0.6)

        ax.set_xticks([-10, -5, 5, 10])
        ax.set_yticks([-10, -5, 5, 10])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which='major', color='#aaaaaa', linewidth=0.6)
        ax.tick_params(which='major', length=3.5, width=0.9, color='#222222')
        ax.tick_params(which='minor', length=1.0, width=0.4, color='#555555')

        tick_fs = 8.5
        offset  = 1.0
        for val in [-10, -5, 5, 10]:
            lbl = str(val)
            ax.text(val, -offset, lbl, ha='center', va='top',
                    fontsize=tick_fs, fontfamily='Times New Roman',
                    color='black', clip_on=False)
            ax.text(-offset, val, lbl, ha='right', va='center',
                    fontsize=tick_fs, fontfamily='Times New Roman',
                    color='black', clip_on=False)

        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for s in ['left', 'bottom']:
            ax.spines[s].set_linewidth(1.4)
            ax.spines[s].set_color('#222222')
        ax.spines['left'].set_bounds(YMIN - 0.5, YMAX + 0.5)
        ax.spines['bottom'].set_bounds(XMIN - 0.5, XMAX + 0.5)

        tri = dict(arrowstyle='-|>', color='#222222', lw=1.4, mutation_scale=8)
        ax.annotate('', xy=(XMAX+0.6, 0), xytext=(XMAX, 0), arrowprops=tri, annotation_clip=False)
        ax.annotate('', xy=(XMIN-0.6, 0), xytext=(XMIN, 0), arrowprops=tri, annotation_clip=False)
        ax.annotate('', xy=(0, YMAX+0.6), xytext=(0, YMAX), arrowprops=tri, annotation_clip=False)
        ax.annotate('', xy=(0, YMIN-0.6), xytext=(0, YMIN), arrowprops=tri, annotation_clip=False)

        ax.text(XMAX + 0.2, 0.9, 'x', fontsize=10, fontweight='bold',
                fontfamily='Times New Roman', ha='center', va='bottom')
        ax.text(0.6, YMAX + 0.2, 'y', fontsize=10, fontweight='bold',
                fontfamily='Times New Roman', ha='left', va='center')

        if title:
            ax.set_title(title, fontsize=11, pad=4)

    return fig

# ─────────────────────────────────────────────────────────────────────────────
# TYPE 5 — 3x1 GRID
# Three coordinate planes side by side, each ~300px wide.
# Use for: comparing transformations, showing shift/stretch/reflect series.
#
# functions_list : list of 3 function lists, same format as make_standard_graph
# titles         : list of 3 LaTeX title strings e.g. r'$f(x)=x^2$'
# ─────────────────────────────────────────────────────────────────────────────

def make_3x1_grid(functions_list, titles=None):
    if titles is None:
        titles = ['', '', '']

    fig, axes = plt.subplots(1, 3, figsize=(6.25, 2.08))
    fig.subplots_adjust(wspace=0.3)

    XMIN, XMAX, YMIN, YMAX = -10, 10, -10, 10

    for ax, functions, title in zip(axes, functions_list, titles):

        for fn in functions:
            f      = fn['expr']
            fprime = fn['deriv']
            color  = fn['color']
            label  = fn.get('label', None)
            x = np.linspace(XMIN, XMAX, 2000)
            y = f(x)
            mask = (y >= YMIN) & (y <= YMAX)
            segments = np.split(np.where(mask)[0],
                                np.where(np.diff(np.where(mask)[0]) > 5)[0] + 1)
            for seg in segments:
                if len(seg) > 1:
                    ax.plot(x[seg], y[seg], color=color, linewidth=1.2,
                            label=label)
                    label = None
            _exit_arrows_small(ax, f, fprime, color, XMIN, XMAX, YMIN, YMAX)

        ax.set_xlim(XMIN - 0.5, XMAX + 0.5)
        ax.set_ylim(YMIN - 0.5, YMAX + 0.5)

        ax.set_xticks(np.arange(XMIN, XMAX+1, 1), minor=True)
        ax.set_yticks(np.arange(YMIN, YMAX+1, 1), minor=True)
        ax.grid(True, which='minor', color='#aaaaaa', linewidth=0.6)

        ax.set_xticks([-10, -5, 5, 10])
        ax.set_yticks([-10, -5, 5, 10])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which='major', color='#aaaaaa', linewidth=0.6)
        ax.tick_params(which='major', length=3.5, width=0.9, color='#222222')
        ax.tick_params(which='minor', length=1.0, width=0.4, color='#555555')

        tick_fs = 8.5
        offset  = 1.0
        for val in [-10, -5, 5, 10]:
            lbl = str(val)
            ax.text(val, -offset, lbl, ha='center', va='top',
                    fontsize=tick_fs, fontfamily='Times New Roman',
                    color='black', clip_on=False)
            ax.text(-offset, val, lbl, ha='right', va='center',
                    fontsize=tick_fs, fontfamily='Times New Roman',
                    color='black', clip_on=False)

        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for s in ['left', 'bottom']:
            ax.spines[s].set_linewidth(1.4)
            ax.spines[s].set_color('#222222')
        ax.spines['left'].set_bounds(YMIN - 0.5, YMAX + 0.5)
        ax.spines['bottom'].set_bounds(XMIN - 0.5, XMAX + 0.5)

        tri = dict(arrowstyle='-|>', color='#222222', lw=1.4, mutation_scale=8)
        ax.annotate('', xy=(XMAX+0.6, 0), xytext=(XMAX, 0), arrowprops=tri, annotation_clip=False)
        ax.annotate('', xy=(XMIN-0.6, 0), xytext=(XMIN, 0), arrowprops=tri, annotation_clip=False)
        ax.annotate('', xy=(0, YMAX+0.6), xytext=(0, YMAX), arrowprops=tri, annotation_clip=False)
        ax.annotate('', xy=(0, YMIN-0.6), xytext=(0, YMIN), arrowprops=tri, annotation_clip=False)

        ax.text(XMAX + 0.2, 0.9, 'x', fontsize=10, fontweight='bold',
                fontfamily='Times New Roman', ha='center', va='bottom')
        ax.text(0.6, YMAX + 0.2, 'y', fontsize=10, fontweight='bold',
                fontfamily='Times New Roman', ha='left', va='center')

        if title:
            ax.set_title(title, fontsize=11, pad=4)

    return fig

# ─────────────────────────────────────────────────────────────────────────────
# TYPE 6 — 4x1 GRID
# Four coordinate planes side by side, each ~225px wide.
# Use for: comparing four transformations or four parent functions in a row.
#
# functions_list : list of 4 function lists, same format as make_standard_graph
# titles         : list of 4 LaTeX title strings e.g. r'$f(x)=x^2$'
# ─────────────────────────────────────────────────────────────────────────────

def make_4x1_grid(functions_list, titles=None):
    if titles is None:
        titles = ['', '', '', '']

    fig, axes = plt.subplots(1, 4, figsize=(6.25, 1.56))
    fig.subplots_adjust(wspace=0.35)

    XMIN, XMAX, YMIN, YMAX = -10, 10, -10, 10

    for ax, functions, title in zip(axes, functions_list, titles):

        for fn in functions:
            f      = fn['expr']
            fprime = fn['deriv']
            color  = fn['color']
            label  = fn.get('label', None)
            x = np.linspace(XMIN, XMAX, 2000)
            y = f(x)
            mask = (y >= YMIN) & (y <= YMAX)
            segments = np.split(np.where(mask)[0],
                                np.where(np.diff(np.where(mask)[0]) > 5)[0] + 1)
            for seg in segments:
                if len(seg) > 1:
                    ax.plot(x[seg], y[seg], color=color, linewidth=1.5,
                            label=label)
                    label = None
            _exit_arrows_small(ax, f, fprime, color, XMIN, XMAX, YMIN, YMAX)

        ax.set_xlim(XMIN - 0.5, XMAX + 0.5)
        ax.set_ylim(YMIN - 0.5, YMAX + 0.5)

        ax.set_xticks(np.arange(XMIN, XMAX+1, 1), minor=True)
        ax.set_yticks(np.arange(YMIN, YMAX+1, 1), minor=True)
        ax.grid(True, which='minor', color='#aaaaaa', linewidth=0.6)

        ax.set_xticks([-10, -5, 5, 10])
        ax.set_yticks([-10, -5, 5, 10])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which='major', color='#aaaaaa', linewidth=0.6)
        ax.tick_params(which='major', length=3.5, width=0.9, color='#222222')
        ax.tick_params(which='minor', length=1.0, width=0.4, color='#555555')

        tick_fs = 8.5
        offset  = 1.0
        for val in [-10, -5, 5, 10]:
            lbl = str(val)
            ax.text(val, -offset, lbl, ha='center', va='top',
                    fontsize=tick_fs, fontfamily='Times New Roman',
                    color='black', clip_on=False)
            ax.text(-offset, val, lbl, ha='right', va='center',
                    fontsize=tick_fs, fontfamily='Times New Roman',
                    color='black', clip_on=False)

        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for s in ['left', 'bottom']:
            ax.spines[s].set_linewidth(1.4)
            ax.spines[s].set_color('#222222')
        ax.spines['left'].set_bounds(YMIN - 0.5, YMAX + 0.5)
        ax.spines['bottom'].set_bounds(XMIN - 0.5, XMAX + 0.5)

        tri = dict(arrowstyle='-|>', color='#222222', lw=1.4, mutation_scale=8)
        ax.annotate('', xy=(XMAX+0.6, 0), xytext=(XMAX, 0), arrowprops=tri, annotation_clip=False)
        ax.annotate('', xy=(XMIN-0.6, 0), xytext=(XMIN, 0), arrowprops=tri, annotation_clip=False)
        ax.annotate('', xy=(0, YMAX+0.6), xytext=(0, YMAX), arrowprops=tri, annotation_clip=False)
        ax.annotate('', xy=(0, YMIN-0.6), xytext=(0, YMIN), arrowprops=tri, annotation_clip=False)

        ax.text(XMAX + 0.2, 0.9, 'x', fontsize=10, fontweight='bold',
                fontfamily='Times New Roman', ha='center', va='bottom')
        ax.text(0.6, YMAX + 0.2, 'y', fontsize=10, fontweight='bold',
                fontfamily='Times New Roman', ha='left', va='center')

        if title:
            ax.set_title(title, fontsize=11, pad=4)

    return fig

# ─────────────────────────────────────────────────────────────────────────────
# TYPE 7 — RECTANGLE MODEL (AREA MODEL)
# Use for: multiplying polynomials, factoring, generic rectangle.
# Scales automatically to any number of rows and columns.
#
# row_labels  : list of LaTeX strings  e.g. [r'$4x$', r'$-5y$']
# col_labels  : list of LaTeX strings  e.g. [r'$2x$', r'$-3y$']
# cell_values : 2D list — use r'$...$' for LaTeX, '' for blank
#               e.g. [[r'$8x^2$', r'$-12xy$'], [r'$-10xy$', r'$15y^2$']]
# filename    : saved to ./figures
# ─────────────────────────────────────────────────────────────────────────────

def make_rectangle_model(row_labels, col_labels, cell_values,
                         filename='rectangle_model.png'):
    import matplotlib.patches as patches

    n_rows = len(row_labels)
    n_cols = len(col_labels)

    cell_w = 0.75
    cell_h = 0.50
    margin = 0.40

    fig_w = margin + n_cols * cell_w + margin
    fig_h = margin + n_rows * cell_h + margin

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    grid_x = margin
    grid_y = margin * 0.5

    for r in range(n_rows):
        for c in range(n_cols):
            cell_left   = grid_x + c * cell_w
            cell_bottom = grid_y + (n_rows - 1 - r) * cell_h
            rect = patches.Rectangle(
                (cell_left, cell_bottom), cell_w, cell_h,
                linewidth=1.2, edgecolor='black', facecolor='white'
            )
            ax.add_patch(rect)
            val = cell_values[r][c]
            if val:
                ax.text(cell_left + cell_w/2, cell_bottom + cell_h/2, val,
                        ha='center', va='center',
                        fontsize=13, fontfamily='Times New Roman')

    for c, lbl in enumerate(col_labels):
        cx = grid_x + c * cell_w + cell_w/2
        cy = grid_y + n_rows * cell_h + 0.10
        ax.text(cx, cy, lbl, ha='center', va='bottom',
                fontsize=13, fontfamily='Times New Roman')

    for r, lbl in enumerate(row_labels):
        rx = grid_x - 0.10
        ry = grid_y + (n_rows - 1 - r) * cell_h + cell_h/2
        ax.text(rx, ry, lbl, ha='right', va='center',
                fontsize=13, fontfamily='Times New Roman')

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=144, bbox_inches='tight')
    print(f'Saved: {path}')

    # ─────────────────────────────────────────────────────────────────────────────
# TYPE 8 — DIAMOND PROBLEM
# Use for: diamond problems, factoring, sum/product puzzles.
# Top = product, bottom = sum, left and right = the two factors.
# Any cell can be '' for blank — student fills it in.
#
# top    : product value  e.g. r'$-42$'  or  r'$xy$'
# left   : left factor    e.g. r'$3$'    or  r'$x$'
# right  : right factor   e.g. r'$-14$'  or  r'$y$'
# bottom : sum value      e.g. r'$-11$'  or  r'$x+y$'
# filename : saved to ./figures
# ─────────────────────────────────────────────────────────────────────────────

def make_diamond(top, left, right, bottom, filename='diamond.png'):
    import matplotlib.pyplot as plt

    size = 2.5
    fig, ax = plt.subplots(figsize=(size, size))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    r = 0.85

    top_pt    = ( 0,  r)
    right_pt  = ( r,  0)
    bottom_pt = ( 0, -r)
    left_pt   = (-r,  0)

    mid_top_right    = ((top_pt[0]+right_pt[0])/2,    (top_pt[1]+right_pt[1])/2)
    mid_right_bottom = ((right_pt[0]+bottom_pt[0])/2, (right_pt[1]+bottom_pt[1])/2)
    mid_bottom_left  = ((bottom_pt[0]+left_pt[0])/2,  (bottom_pt[1]+left_pt[1])/2)
    mid_left_top     = ((left_pt[0]+top_pt[0])/2,     (left_pt[1]+top_pt[1])/2)

    diamond = plt.Polygon(
        [top_pt, right_pt, bottom_pt, left_pt],
        closed=True, linewidth=1.4,
        edgecolor='black', facecolor='white', zorder=2
    )
    ax.add_patch(diamond)

    ax.plot([mid_left_top[0], mid_right_bottom[0]],
            [mid_left_top[1], mid_right_bottom[1]],
            color='black', linewidth=1.0, zorder=3)
    ax.plot([mid_top_right[0], mid_bottom_left[0]],
            [mid_top_right[1], mid_bottom_left[1]],
            color='black', linewidth=1.0, zorder=3)

    pad = 0.38
    positions = {
        'top':    ( 0,    pad),
        'bottom': ( 0,   -pad),
        'left':   (-pad,  0),
        'right':  ( pad,  0),
    }

    for cell, val in [('top', top), ('bottom', bottom),
                      ('left', left), ('right', right)]:
        if val:
            ax.text(*positions[cell], val,
                    ha='center', va='center',
                    fontsize=16, fontfamily='Times New Roman',
                    zorder=4)

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=144, bbox_inches='tight')
    print(f'Saved: {path}')

# ─────────────────────────────────────────────────────────────────────────────
# TYPE 10 — PIECEWISE FUNCTION
# Standard -10 to 10 plane. Each piece has its own domain; all pieces draw in steelblue.
# Dots are automatic — closed if endpoint included, open if not.
# Arrows drawn at window edges when piece extends to infinity.
#
# pieces : list of dicts, each with:
#   'expr'          : lambda x: ...    the function expression
#   'deriv'         : lambda x: ...    its derivative
#   'domain'        : (a, b)           interval for this piece
#   'include_left'  : True/False       closed dot on left endpoint
#   'include_right' : True/False       closed dot on right endpoint
#   'color'         : optional; ignored for consistency (all pieces use steelblue)
#   'arrow_left'    : True/False       exit arrow at left end
#   'arrow_right'   : True/False       exit arrow at right end
#
# title: LaTeX supported via r'$...$'
# ─────────────────────────────────────────────────────────────────────────────

def _pw_dot(ax, x, y, open_dot, color, size=7, dot_scale=0.65):
    size = size * dot_scale
    if open_dot:
        ax.plot(x, y, 'o', markersize=size, markerfacecolor='white',
                markeredgecolor=color, markeredgewidth=2, zorder=6,
                clip_on=False)
    else:
        ax.plot(x, y, 'o', markersize=size, markerfacecolor=color,
                markeredgecolor=color, markeredgewidth=2, zorder=6,
                clip_on=False)


def make_piecewise_graph(ax, pieces, title='', dot_scale=0.65):
    XMIN, XMAX, YMIN, YMAX = -10, 10, -10, 10

    for piece in pieces:
        f          = piece['expr']
        fprime     = piece['deriv']
        color      = 'steelblue'
        a, b       = piece['domain']
        inc_left   = piece.get('include_left',  True)
        inc_right  = piece.get('include_right', True)
        arrow_left  = piece.get('arrow_left',  False)
        arrow_right = piece.get('arrow_right', False)

        x_seg = np.linspace(a, b, 1000)
        y_seg = f(x_seg)
        mask  = (y_seg >= YMIN) & (y_seg <= YMAX)
        if mask.any():
            ax.plot(x_seg[mask], y_seg[mask],
                    color=color, linewidth=1.7, zorder=3)

        ya = f(a)
        if not arrow_left and YMIN <= ya <= YMAX:
            _pw_dot(ax, a, ya, open_dot=not inc_left, color=color, dot_scale=dot_scale)

        yb = f(b)
        if not arrow_right and YMIN <= yb <= YMAX:
            _pw_dot(ax, b, yb, open_dot=not inc_right, color=color, dot_scale=dot_scale)

        if arrow_left and YMIN <= ya <= YMAX:
            slope = fprime(a)
            if abs(slope) >= 0.001:
                dx_dir = -1.0
                dy_dir = slope * dx_dir
                mag = np.sqrt(dx_dir**2 + dy_dir**2)
                dx_dir /= mag; dy_dir /= mag
                ax.annotate('', xy=(a + dx_dir*0.45, ya + dy_dir*0.45),
                                xytext=(a, ya),
                            arrowprops=dict(arrowstyle='-|>', color=color,
                                            lw=1.5, mutation_scale=12))

        if arrow_right and YMIN <= yb <= YMAX:
            slope = fprime(b)
            if abs(slope) >= 0.001:
                dx_dir = 1.0
                dy_dir = slope * dx_dir
                mag = np.sqrt(dx_dir**2 + dy_dir**2)
                dx_dir /= mag; dy_dir /= mag
                ax.annotate('', xy=(b + dx_dir*0.45, yb + dy_dir*0.45),
                                xytext=(b, yb),
                            arrowprops=dict(arrowstyle='-|>', color=color,
                                            lw=1.5, mutation_scale=12))

        for edge_y, direction in [(YMIN, -1), (YMAX, 1)]:
            vals = y_seg - edge_y
            for idx in np.where(np.diff(np.sign(vals)))[0]:
                xr = np.interp(0, [vals[idx], vals[idx+1]],
                                  [x_seg[idx], x_seg[idx+1]])
                slope = fprime(xr)
                if abs(slope) < 0.001:
                    continue
                dy_dir = float(direction)
                dx_dir = dy_dir / slope
                mag = np.sqrt(dx_dir**2 + dy_dir**2)
                dx_dir /= mag; dy_dir /= mag
                ax.annotate('', xy=(xr + dx_dir*0.45, edge_y + dy_dir*0.45),
                                xytext=(xr, edge_y),
                            arrowprops=dict(arrowstyle='-|>', color=color,
                                            lw=1.5, mutation_scale=12))

    ax.set_xlim(XMIN - 0.5, XMAX + 0.5)
    ax.set_ylim(YMIN - 0.5, YMAX + 0.5)
    ax.set_xticks(np.arange(XMIN, XMAX+1, 1), minor=True)
    ax.set_yticks(np.arange(YMIN, YMAX+1, 1), minor=True)
    ax.grid(True, which='minor', color='#aaaaaa', linewidth=0.6)
    ax.set_xticks([-10, -5, 5, 10])
    ax.set_yticks([-10, -5, 5, 10])
    ax.set_xticklabels(['-10','-5','5','10'],
                       fontfamily='Times New Roman', fontsize=11)
    ax.set_yticklabels(['-10','-5','5','10'],
                       fontfamily='Times New Roman', fontsize=11)
    ax.grid(True, which='major', color='#aaaaaa', linewidth=0.6)
    ax.tick_params(which='major', length=5, width=1.2, color='#222222')
    ax.tick_params(which='minor', length=2, width=0.8, color='#555555')

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_linewidth(1.8)
        ax.spines[s].set_color('#222222')
    ax.spines['left'].set_bounds(YMIN - 0.5, YMAX + 0.5)
    ax.spines['bottom'].set_bounds(XMIN - 0.5, XMAX + 0.5)

    tri = dict(arrowstyle='-|>', color='#222222', lw=1.8, mutation_scale=14)
    ax.annotate('', xy=(XMAX+0.6, 0), xytext=(XMAX, 0), arrowprops=tri, annotation_clip=False)
    ax.annotate('', xy=(XMIN-0.6, 0), xytext=(XMIN, 0), arrowprops=tri, annotation_clip=False)
    ax.annotate('', xy=(0, YMAX+0.6), xytext=(0, YMAX), arrowprops=tri, annotation_clip=False)
    ax.annotate('', xy=(0, YMIN-0.6), xytext=(0, YMIN), arrowprops=tri, annotation_clip=False)

    ax.text(XMAX+0.35, 0.4, 'x', fontsize=14, fontweight='bold',
            fontfamily='Times New Roman', ha='center', va='bottom')
    ax.text(0.35, YMAX+0.35, 'y', fontsize=14, fontweight='bold',
            fontfamily='Times New Roman', ha='left', va='center')
    ax.set_title(title, fontfamily='Times New Roman', fontsize=12, pad=8)

    # ─────────────────────────────────────────────────────────────────────────────
# TYPE 11 — UNIT CIRCLE
# All variants save at exactly 2.5x2.5in (360x360px at 144dpi).
#
# make_unit_circle_blank()
#   Blank — circle and axes only. Student draws/writes everything.
#
# make_unit_circle_angles(ax_angle_deg=None, show_coord=True, filename=...)
#   ax_angle_deg : int — highlights that angle in blue  e.g. 120
#                  None — semi-blank, just 16 angle lines, nothing highlighted
#   show_coord   : True = coordinate label outside circle, False = no label
#
# Valid angles: 0, 30, 45, 60, 90, 120, 135, 150,
#               180, 210, 225, 240, 270, 300, 315, 330
# ─────────────────────────────────────────────────────────────────────────────

ANGLE_DATA = {
    0:   ( 1,            0           ),
    30:  ( np.sqrt(3)/2, 0.5         ),
    45:  ( np.sqrt(2)/2, np.sqrt(2)/2),
    60:  ( 0.5,          np.sqrt(3)/2),
    90:  ( 0,            1           ),
    120: (-0.5,          np.sqrt(3)/2),
    135: (-np.sqrt(2)/2, np.sqrt(2)/2),
    150: (-np.sqrt(3)/2, 0.5         ),
    180: (-1,            0           ),
    210: (-np.sqrt(3)/2,-0.5         ),
    225: (-np.sqrt(2)/2,-np.sqrt(2)/2),
    240: (-0.5,         -np.sqrt(3)/2),
    270: ( 0,           -1           ),
    300: ( 0.5,         -np.sqrt(3)/2),
    315: ( np.sqrt(2)/2,-np.sqrt(2)/2),
    330: ( np.sqrt(3)/2,-0.5         ),
}

COORD_LABELS = {
    0:   r'$(1, 0)$',
    30:  r'$\left(\frac{\sqrt{3}}{2}, \frac{1}{2}\right)$',
    45:  r'$\left(\frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{2}\right)$',
    60:  r'$\left(\frac{1}{2}, \frac{\sqrt{3}}{2}\right)$',
    90:  r'$(0, 1)$',
    120: r'$\left(-\frac{1}{2}, \frac{\sqrt{3}}{2}\right)$',
    135: r'$\left(-\frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{2}\right)$',
    150: r'$\left(-\frac{\sqrt{3}}{2}, \frac{1}{2}\right)$',
    180: r'$(-1, 0)$',
    210: r'$\left(-\frac{\sqrt{3}}{2}, -\frac{1}{2}\right)$',
    225: r'$\left(-\frac{\sqrt{2}}{2}, -\frac{\sqrt{2}}{2}\right)$',
    240: r'$\left(-\frac{1}{2}, -\frac{\sqrt{3}}{2}\right)$',
    270: r'$(0, -1)$',
    300: r'$\left(\frac{1}{2}, -\frac{\sqrt{3}}{2}\right)$',
    315: r'$\left(\frac{\sqrt{2}}{2}, -\frac{\sqrt{2}}{2}\right)$',
    330: r'$\left(\frac{\sqrt{3}}{2}, -\frac{1}{2}\right)$',
}


def _draw_unit_circle_base(ax, show_angle_lines=False):
    theta = np.linspace(0, 2*np.pi, 1000)
    ax.plot(np.cos(theta), np.sin(theta),
            color='black', linewidth=2.0, zorder=3)

    lw_ax  = 0.6
    col_ax = '#222222'
    ax.plot([-1.3, 1.3], [0, 0], color=col_ax, linewidth=lw_ax, zorder=1)
    ax.plot([0, 0], [-1.3, 1.3], color=col_ax, linewidth=lw_ax, zorder=1)

    tri = dict(arrowstyle='-|>', color=col_ax, lw=lw_ax, mutation_scale=10)
    ax.annotate('', xy=( 1.38, 0),  xytext=( 1.22, 0),  arrowprops=tri)
    ax.annotate('', xy=(-1.38, 0),  xytext=(-1.22, 0),  arrowprops=tri)
    ax.annotate('', xy=( 0,  1.38), xytext=( 0,  1.22), arrowprops=tri)
    ax.annotate('', xy=( 0, -1.38), xytext=( 0, -1.22), arrowprops=tri)

    ax.text( 1.44, 0,  'x', ha='left',   va='center', fontsize=8,
             fontfamily='Times New Roman', color=col_ax)
    ax.text( 0,  1.44, 'y', ha='center', va='bottom', fontsize=8,
             fontfamily='Times New Roman', color=col_ax)

    if show_angle_lines:
        for deg in ANGLE_DATA:
            rad = np.radians(deg)
            ax.plot([0, np.cos(rad)], [0, np.sin(rad)],
                    color='#aaaaaa', linewidth=0.6, zorder=2)

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')


def _uc_save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=144)
    print(f'Saved: {path}')


def make_unit_circle_blank(filename='unit_circle_blank.png'):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _draw_unit_circle_base(ax, show_angle_lines=False)
    _uc_save(fig, filename)


def make_unit_circle_angles(ax_angle_deg=None, show_coord=True,
                             filename='unit_circle_angles.png'):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _draw_unit_circle_base(ax, show_angle_lines=True)

    if ax_angle_deg is not None:
        blue = '#1a4f8a'
        rad  = np.radians(ax_angle_deg)
        px   = np.cos(rad)
        py   = np.sin(rad)

        ax.plot([0, px], [0, py], color=blue, linewidth=1.8, zorder=4)
        ax.plot(px, py, 'o', markersize=5, color=blue, zorder=5)

        if show_coord and ax_angle_deg in COORD_LABELS:
            lx = px * 1.12
            ly = py * 1.12
            ha = 'center'
            if px > 0.15:  ha = 'left'
            if px < -0.15: ha = 'right'
            va = 'center'
            if py > 0.15:  va = 'bottom'
            if py < -0.15: va = 'top'
            ax.text(lx, ly, COORD_LABELS[ax_angle_deg],
                    ha=ha, va=va, fontsize=12, color=blue,
                    fontfamily='Times New Roman', zorder=6)

    _uc_save(fig, filename)

    # ─────────────────────────────────────────────────────────────────────────────
# TYPE 12 — 2D INEQUALITY / SYSTEM OF INEQUALITIES
# Standard plane extended to ±11 so exit arrows clear the grid.
# Single inequality or system of 2 (overlap shown in mediumpurple).
#
# inequalities : list of dicts, each with:
#   'expr'      : lambda x: ...    boundary line function
#   'deriv'     : lambda x: ...    its derivative
#   'color'     : 'steelblue'
#   'shade'     : 'above' or 'below'
#   'inclusive' : True = solid line, False = dashed line
#
# title: LaTeX supported via r'$...$'
# ─────────────────────────────────────────────────────────────────────────────

def make_inequality_graph(ax, inequalities, title=''):
    XMIN, XMAX, YMIN, YMAX = -11, 11, -11, 11
    x = np.linspace(XMIN, XMAX, 2000)

    shade_regions = []

    for ineq in inequalities:
        f         = ineq['expr']
        fprime    = ineq['deriv']
        color     = ineq['color']
        shade     = ineq['shade']
        inclusive = ineq.get('inclusive', True)

        y = f(x)
        y_clipped = np.clip(y, YMIN, YMAX)
        shade_regions.append({'y': y_clipped, 'shade': shade, 'color': color})

        linestyle = '-' if inclusive else '--'
        mask = (y >= YMIN) & (y <= YMAX)
        segments = np.split(np.where(mask)[0],
                            np.where(np.diff(np.where(mask)[0]) > 5)[0] + 1)
        for seg in segments:
            if len(seg) > 1:
                ax.plot(x[seg], y[seg], color=color, linewidth=2,
                        linestyle=linestyle, zorder=4)

        exit_points = []
        yl = f(XMIN)
        if YMIN <= yl <= YMAX:
            exit_points.append((float(XMIN), float(yl), 'left'))
        yr = f(XMAX)
        if YMIN <= yr <= YMAX:
            exit_points.append((float(XMAX), float(yr), 'right'))
        for edge_y, edge_dir in [(YMIN, 'bottom'), (YMAX, 'top')]:
            vals = y - edge_y
            for idx in np.where(np.diff(np.sign(vals)))[0]:
                xr = float(np.interp(0, [vals[idx], vals[idx+1]],
                                        [x[idx], x[idx+1]]))
                exit_points.append((xr, float(edge_y), edge_dir))

        for (xe, ye, edge) in exit_points:
            slope = float(fprime(np.array([xe]))[0])
            if edge == 'right':
                dx_dir =  1.0
                dy_dir =  float(slope)
            elif edge == 'left':
                dx_dir = -1.0
                dy_dir = -float(slope)
            elif edge == 'top':
                dy_dir =  1.0
                dx_dir =  float(1.0 / slope) if abs(slope) > 0.001 else 0.0
            else:
                dy_dir = -1.0
                dx_dir = -float(1.0 / slope) if abs(slope) > 0.001 else 0.0
            mag = float(np.sqrt(dx_dir**2 + dy_dir**2))
            if mag < 0.001:
                continue
            dx_dir /= mag
            dy_dir /= mag
            L = 0.55
            ax.annotate('',
                        xy=(xe + dx_dir*L, ye + dy_dir*L),
                        xytext=(xe - dx_dir*0.1, ye - dy_dir*0.1),
                        arrowprops=dict(arrowstyle='-|>', color=color,
                                        lw=1.5, mutation_scale=12),
                        annotation_clip=False)

    for sr in shade_regions:
        if sr['shade'] == 'above':
            ax.fill_between(x, sr['y'], YMAX,
                            where=(sr['y'] <= YMAX),
                            color=sr['color'], alpha=0.15, zorder=2)
        elif sr['shade'] == 'below':
            ax.fill_between(x, YMIN, sr['y'],
                            where=(sr['y'] >= YMIN),
                            color=sr['color'], alpha=0.15, zorder=2)

    if len(shade_regions) == 2:
        sr0, sr1 = shade_regions[0], shade_regions[1]
        y0_top = np.full(len(x), float(YMAX)) if sr0['shade'] == 'above' else sr0['y']
        y0_bot = sr0['y'] if sr0['shade'] == 'above' else np.full(len(x), float(YMIN))
        y1_top = np.full(len(x), float(YMAX)) if sr1['shade'] == 'above' else sr1['y']
        y1_bot = sr1['y'] if sr1['shade'] == 'above' else np.full(len(x), float(YMIN))
        overlap_top = np.minimum(y0_top, y1_top)
        overlap_bot = np.maximum(y0_bot, y1_bot)
        has_overlap = overlap_top > overlap_bot
        if has_overlap.any():
            ax.fill_between(x, overlap_bot, overlap_top,
                            where=has_overlap,
                            color='mediumpurple', alpha=0.45, zorder=3)

    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-10.5, 10.5)
    ax.set_xticks(np.arange(-10, 11, 1), minor=True)
    ax.set_yticks(np.arange(-10, 11, 1), minor=True)
    ax.grid(True, which='minor', color='#aaaaaa', linewidth=0.6)
    ax.set_xticks([-10, -5, 5, 10])
    ax.set_yticks([-10, -5, 5, 10])
    ax.set_xticklabels(['-10','-5','5','10'],
                       fontfamily='Times New Roman', fontsize=11)
    ax.set_yticklabels(['-10','-5','5','10'],
                       fontfamily='Times New Roman', fontsize=11)
    ax.grid(True, which='major', color='#aaaaaa', linewidth=0.6)
    ax.tick_params(which='major', length=5, width=1.2, color='#222222')
    ax.tick_params(which='minor', length=2, width=0.8, color='#555555')

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_linewidth(1.8)
        ax.spines[s].set_color('#222222')
    ax.spines['left'].set_bounds(-10.5, 10.5)
    ax.spines['bottom'].set_bounds(-10.5, 10.5)

    tri = dict(arrowstyle='-|>', color='#222222', lw=1.8, mutation_scale=14)
    ax.annotate('', xy=( 11.6,  0), xytext=( 10,  0), arrowprops=tri, annotation_clip=False)
    ax.annotate('', xy=(-11.6,  0), xytext=(-10,  0), arrowprops=tri, annotation_clip=False)
    ax.annotate('', xy=( 0,  11.6), xytext=( 0,  10), arrowprops=tri, annotation_clip=False)
    ax.annotate('', xy=( 0, -11.6), xytext=( 0, -10), arrowprops=tri, annotation_clip=False)

    ax.text(11.8, 0.4, 'x', fontsize=14, fontweight='bold',
            fontfamily='Times New Roman', ha='center', va='bottom')
    ax.text(0.35, 11.8, 'y', fontsize=14, fontweight='bold',
            fontfamily='Times New Roman', ha='left', va='center')
    ax.set_title(title, fontfamily='Times New Roman', fontsize=12, pad=22)

    # ─────────────────────────────────────────────────────────────────────────────
# TYPE 13 — TRIG GRAPH (sine and cosine)
# Landscape figure 5x2.5in. Shows 2 full periods.
# x-axis labeled in π fractions. y-axis scales to amplitude.
#
# functions : list of dicts, each with:
#   'type'  : 'sin' or 'cos'
#   'a'     : amplitude          (default 1)
#   'b'     : frequency          (default 1)
#   'c'     : phase shift        (default 0)  f(x) = a·sin(b(x-c)) + d
#   'd'     : vertical shift     (default 0)
#   'color' : 'steelblue'
#
# title: LaTeX supported via r'$...$'
# ─────────────────────────────────────────────────────────────────────────────

def make_trig_graph(ax, functions, title=''):
    max_amp = max(abs(fn.get('a', 1)) + abs(fn.get('d', 0))
                  for fn in functions)
    y_max = max(np.ceil(max_amp * 1.3), 2)

    min_b = min(abs(fn.get('b', 1)) for fn in functions)
    period = 2 * np.pi / min_b
    x_max = period
    x = np.linspace(-x_max, x_max, 3000)

    for fn in functions:
        ftype = fn.get('type', 'sin')
        a     = fn.get('a', 1)
        b     = fn.get('b', 1)
        c     = fn.get('c', 0)
        d     = fn.get('d', 0)
        color = fn.get('color', 'steelblue')

        if ftype == 'sin':
            y = a * np.sin(b * (x - c)) + d
        else:
            y = a * np.cos(b * (x - c)) + d

        ax.plot(x, y, color=color, linewidth=2, zorder=3)

    ax.set_ylim(-y_max - 0.3, y_max + 0.3)
    ax.set_xlim(-x_max - 0.2, x_max + 0.2)

    step = np.pi / (4 * min_b)
    x_ticks = np.arange(-x_max, x_max + step*0.01, step)

    def pi_label(v):
        from fractions import Fraction
        v_pi = round(v / np.pi * 8) / 8
        if v_pi == 0:
            return '0'
        frac = Fraction(v_pi).limit_denominator(8)
        num, den = frac.numerator, frac.denominator
        if den == 1:
            if num ==  1: return r'$\pi$'
            if num == -1: return r'$-\pi$'
            return rf'${num}\pi$'
        if num ==  1: return rf'$\frac{{\pi}}{{{den}}}$'
        if num == -1: return rf'$-\frac{{\pi}}{{{den}}}$'
        if num <   0: return rf'$-\frac{{{abs(num)}\pi}}{{{den}}}$'
        return rf'$\frac{{{num}\pi}}{{{den}}}$'

    label_every = 2
    ax.set_xticks(x_ticks)
    x_labels = [pi_label(t) if i % label_every == 0 else ''
                for i, t in enumerate(x_ticks)]
    ax.set_xticklabels(x_labels, fontfamily='Times New Roman', fontsize=10)

    y_ticks = np.arange(-int(y_max), int(y_max)+1, 1)
    ax.set_yticks(y_ticks)
    y_labels = [str(int(t)) if t != 0 else '' for t in y_ticks]
    ax.set_yticklabels(y_labels, fontfamily='Times New Roman', fontsize=10)

    ax.grid(True, color='#aaaaaa', linewidth=0.6, zorder=0)
    ax.tick_params(which='major', length=4, width=1.0, color='#222222')

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_linewidth(1.8)
        ax.spines[s].set_color('#222222')
    ax.spines['left'].set_bounds(-y_max, y_max)
    ax.spines['bottom'].set_bounds(-x_max, x_max)

    tri = dict(arrowstyle='-|>', color='#222222', lw=2.0, mutation_scale=14)
    ax.annotate('', xy=( x_max+0.3, 0), xytext=( x_max, 0), arrowprops=tri, annotation_clip=False)
    ax.annotate('', xy=(-x_max-0.3, 0), xytext=(-x_max, 0), arrowprops=tri, annotation_clip=False)
    ax.annotate('', xy=(0,  y_max+0.4), xytext=(0,  y_max), arrowprops=tri, annotation_clip=False)
    ax.annotate('', xy=(0, -y_max-0.4), xytext=(0, -y_max), arrowprops=tri, annotation_clip=False)

    ax.text(x_max+0.35, 0.15, 'x', fontsize=13, fontweight='bold',
            fontfamily='Times New Roman', ha='left', va='bottom')
    ax.text(0.15, y_max+0.3, 'y', fontsize=13, fontweight='bold',
            fontfamily='Times New Roman', ha='left', va='center')

    ax.set_title(title, fontfamily='Times New Roman', fontsize=12, pad=8)

    # ─────────────────────────────────────────────────────────────────────────────
# TYPE 14 — BAR CHART
# Use for: categorical comparisons, survey results, grouped data.
# Landscape 4x3in. Horizontal grid lines only. Times New Roman throughout.
#
# categories : list of strings        e.g. ['Mon', 'Tue', 'Wed']
# values     : list of numbers        e.g. [4, 7, 3]
# colors     : single color or list   (default 'steelblue')
# ylabel     : y-axis label
# xlabel     : x-axis label
# title      : LaTeX supported via r'$...$'
#
# TYPE 15 — HISTOGRAM
# Use for: frequency distributions, data spread, statistics units.
#
# data   : list or array of raw values  e.g. [12, 15, 14, 18, ...]
# bins   : number of bins or list of bin edges (default 10)
# color  : bar color (default 'steelblue')
# ylabel : y-axis label
# xlabel : x-axis label
# title  : LaTeX supported via r'$...$'
# ─────────────────────────────────────────────────────────────────────────────

def make_bar_chart(ax, categories, values, title='',
                   colors=None, ylabel='Frequency', xlabel=''):
    if colors is None:
        colors = 'steelblue'

    n = len(categories)
    x = np.arange(n)

    ax.bar(x, values, color=colors, width=0.6,
           edgecolor='white', linewidth=0.8, zorder=3)

    ax.yaxis.grid(True, color='#aaaaaa', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    y_max = max(values) * 1.15
    ax.set_ylim(0, y_max)

    ax.set_xticks(x)
    ax.set_xticklabels(categories,
                       fontfamily='Times New Roman', fontsize=11)
    ax.set_xlim(-0.5, n - 0.5)

    ax.yaxis.set_tick_params(labelsize=11)
    for label in ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('#222222')
    ax.spines['bottom'].set_color('#222222')

    ax.set_xlabel(xlabel, fontfamily='Times New Roman',
                  fontsize=12, fontweight='bold', labelpad=6)
    ax.set_ylabel(ylabel, fontfamily='Times New Roman',
                  fontsize=12, fontweight='bold', labelpad=6)
    ax.set_title(title, fontfamily='Times New Roman', fontsize=12, pad=8)


def make_histogram(ax, data, bins=10, title='',
                   color='steelblue', ylabel='Frequency', xlabel='Value'):
    counts, edges, patches = ax.hist(data, bins=bins, color=color,
                                     edgecolor='white', linewidth=0.8,
                                     zorder=3)

    ax.yaxis.grid(True, color='#aaaaaa', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    y_max = max(counts) * 1.15
    ax.set_ylim(0, y_max)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('#222222')
    ax.spines['bottom'].set_color('#222222')

    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

    ax.set_xlabel(xlabel, fontfamily='Times New Roman',
                  fontsize=12, fontweight='bold', labelpad=6)
    ax.set_ylabel(ylabel, fontfamily='Times New Roman',
                  fontsize=12, fontweight='bold', labelpad=6)
    ax.set_title(title, fontfamily='Times New Roman', fontsize=12, pad=8)

    # ─────────────────────────────────────────────────────────────────────────────
# TYPE 16 — SCATTER PLOT
# Use for: data analysis, line of best fit, residuals, sequences/series.
# Custom axis ranges. Optional linear line of best fit with arrows.
# One dataset at a time.
#
# x_data, y_data   : lists or arrays of data points
# xmin/xmax        : x axis range
# ymin/ymax        : y axis range
# color            : point color (default 'steelblue')
# point_size       : dot size (default 25)
# line_of_best_fit : True = draw linear regression line with arrows
# xlabel/ylabel    : axis labels
# title            : LaTeX supported via r'$...$'
# ─────────────────────────────────────────────────────────────────────────────

def make_scatter_plot(ax, x_data, y_data,
                      xmin, xmax, ymin, ymax,
                      color='steelblue', point_size=25, point_scale=0.65,
                      line_of_best_fit=False,
                      xlabel='x', ylabel='y', title=''):

    x_data = np.array(x_data, dtype=float)
    y_data = np.array(y_data, dtype=float)

    ax.scatter(x_data, y_data, color=color, s=point_size * point_scale,
               zorder=4, clip_on=True)

    if line_of_best_fit:
        m, b = np.polyfit(x_data, y_data, 1)
        x_range = xmax - xmin
        x_ext = np.linspace(xmin, xmax, 1000)
        y_ext = m * x_ext + b
        mask = (y_ext >= ymin) & (y_ext <= ymax)

        if mask.any():
            ax.plot(x_ext[mask], y_ext[mask],
                    color='firebrick', linewidth=2, zorder=3)

            y_left = m * xmin + b
            if ymin <= y_left <= ymax:
                dx_dir = -1.0
                dy_dir = -float(m)
                mag = np.sqrt(dx_dir**2 + dy_dir**2)
                dx_dir /= mag; dy_dir /= mag
                L = x_range * 0.04
                ax.annotate('',
                            xy=(xmin + dx_dir*L, y_left + dy_dir*L),
                            xytext=(xmin - dx_dir*0.01, y_left - dy_dir*0.01),
                            arrowprops=dict(arrowstyle='-|>', color='firebrick',
                                            lw=1.5, mutation_scale=12),
                            annotation_clip=False)

            y_right = m * xmax + b
            if ymin <= y_right <= ymax:
                dx_dir = 1.0
                dy_dir = float(m)
                mag = np.sqrt(dx_dir**2 + dy_dir**2)
                dx_dir /= mag; dy_dir /= mag
                L = x_range * 0.04
                ax.annotate('',
                            xy=(xmax + dx_dir*L, y_right + dy_dir*L),
                            xytext=(xmax - dx_dir*0.01, y_right - dy_dir*0.01),
                            arrowprops=dict(arrowstyle='-|>', color='firebrick',
                                            lw=1.5, mutation_scale=12),
                            annotation_clip=False)

            for edge_y, direction in [(ymin, -1), (ymax, 1)]:
                vals = y_ext - edge_y
                for idx in np.where(np.diff(np.sign(vals)))[0]:
                    xr = float(np.interp(0, [vals[idx], vals[idx+1]],
                                            [x_ext[idx], x_ext[idx+1]]))
                    dy_dir = float(direction)
                    dx_dir = float(dy_dir / m) if abs(m) > 0.001 else 0.0
                    mag = np.sqrt(dx_dir**2 + dy_dir**2)
                    if mag < 0.001:
                        continue
                    dx_dir /= mag; dy_dir /= mag
                    L = x_range * 0.04
                    ax.annotate('',
                                xy=(xr + dx_dir*L, edge_y + dy_dir*L),
                                xytext=(xr - dx_dir*0.01, edge_y - dy_dir*0.01),
                                arrowprops=dict(arrowstyle='-|>', color='firebrick',
                                                lw=1.5, mutation_scale=12),
                                annotation_clip=False)

    x_range = xmax - xmin
    y_range = ymax - ymin
    mx = x_range * 0.06
    my = y_range * 0.06

    # Keep the first gridline flush with the left/bottom axes.
    # Reserve headroom only past the final gridline for arrowheads.
    ax.set_xlim(xmin, xmax + mx)
    ax.set_ylim(ymin, ymax + my)

    x_step = _nice_grid_step(x_range)
    y_step = _nice_grid_step(y_range)
    x_ticks = np.arange(xmin, xmax + x_step*0.01, x_step)
    y_ticks = np.arange(ymin, ymax + y_step*0.01, y_step)

    for xt in x_ticks:
        ax.plot([xt, xt], [ymin, ymax], color='#aaaaaa', linewidth=0.6,
                zorder=0, clip_on=True)
    for yt in y_ticks:
        ax.plot([xmin, xmax], [yt, yt], color='#aaaaaa', linewidth=0.6,
                zorder=0, clip_on=True)

    x_label_every = _label_every(len(x_ticks))
    y_label_every = _label_every(len(y_ticks))
    x_labeled = [t for i, t in enumerate(x_ticks) if i % x_label_every == 0]
    y_labeled = [t for i, t in enumerate(y_ticks) if i % y_label_every == 0]

    ax.set_xticks(x_labeled)
    ax.set_yticks(y_labeled)
    ax.set_xticklabels([_fmt(t) for t in x_labeled],
                       fontfamily='Times New Roman', fontsize=11)
    ax.set_yticklabels([_fmt(t) for t in y_labeled],
                       fontfamily='Times New Roman', fontsize=11)
    ax.tick_params(which='major', length=5, width=1.2, color='#222222')

    for s in ['left', 'bottom']:
        ax.spines[s].set_linewidth(1.8)
        ax.spines[s].set_color('#222222')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds(ymin, ymax)
    ax.spines['bottom'].set_bounds(xmin, xmax)

    tri = dict(arrowstyle='-|>', color='#222222', lw=1.8, mutation_scale=14)
    ax.annotate('', xy=(xmax + mx, ymin), xytext=(xmax, ymin),
                arrowprops=tri, annotation_clip=False)
    ax.annotate('', xy=(xmin, ymax + my), xytext=(xmin, ymax),
                arrowprops=tri, annotation_clip=False)

    ax.set_xlabel(xlabel, fontfamily='Times New Roman', fontsize=13,
                  fontweight='bold', labelpad=6)
    ax.set_ylabel(ylabel, fontfamily='Times New Roman', fontsize=13,
                  fontweight='bold', labelpad=6, rotation=90)
    ax.set_title(title, fontfamily='Times New Roman', fontsize=12, pad=8)

    # ─────────────────────────────────────────────────────────────────────────────
# TYPE 17 — HUNDRED GRID (10x10)
# Use for: shading percents, decimals, fractions visually.
# 2.5x2.5in (360x360px at 144dpi).
#
# shaded_cells : int 0-100, fills left to right top to bottom
# shaded_color : color for shaded cells (default 'steelblue')
# filename     : saved to ./figures
# ─────────────────────────────────────────────────────────────────────────────

def make_hundred_grid(shaded_cells=0, shaded_color='steelblue',
                      filename='hundred_grid.png'):
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    for row in range(10):
        for col in range(10):
            cell_num = row * 10 + col
            filled = cell_num < shaded_cells
            rect = patches.Rectangle(
                (col, 9 - row), 1, 1,
                linewidth=0.8,
                edgecolor='#555555',
                facecolor=shaded_color if filled else 'white'
            )
            ax.add_patch(rect)

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=144, bbox_inches='tight')
    print(f'Saved: {path}')

    # ─────────────────────────────────────────────────────────────────────────────
# TYPE 18 — FRACTION BAR / TAPE DIAGRAM
# Use for: fractions, ratios, part-part-whole relationships.
# Horizontal bar divided into n equal parts, some shaded left to right.
#
# n_parts      : total number of parts       e.g. 4 for fourths
# n_shaded     : parts shaded from left      (default 0 = blank)
# shaded_color : color for shaded parts      (default 'steelblue')
# label_parts  : True = label each part with fraction e.g. 1/4
# filename     : saved to ./figures
# ─────────────────────────────────────────────────────────────────────────────

def make_fraction_bar(n_parts, n_shaded=0, shaded_color='steelblue',
                      label_parts=True, filename='fraction_bar.png'):
    import matplotlib.patches as patches

    bar_w  = 4.0
    bar_h  = 0.75
    margin = 0.3

    fig_w = bar_w + margin * 2
    fig_h = bar_h + margin * 2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis('off')
    ax.set_facecolor('white')

    part_w = bar_w / n_parts

    for i in range(n_parts):
        x_left = margin + i * part_w
        filled = i < n_shaded

        rect = patches.Rectangle(
            (x_left, margin), part_w, bar_h,
            linewidth=1.2,
            edgecolor='black',
            facecolor=shaded_color if filled else 'white'
        )
        ax.add_patch(rect)

        if label_parts:
            ax.text(x_left + part_w/2, margin + bar_h/2,
                    rf'$\frac{{1}}{{{n_parts}}}$',
                    ha='center', va='center',
                    fontsize=11, fontfamily='Times New Roman',
                    color='white' if filled else 'black')

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=144, bbox_inches='tight')
    print(f'Saved: {path}')

    # ─────────────────────────────────────────────────────────────────────────────
# TYPE 19 — ALGEBRA TILES
# Portrait orientation. Positive = filled color. Negative = red fill.
# White italic LaTeX label inside each tile. Auto-wraps to rows.
#
# expression : dict — any combination of:
#   'x2'      : positive x²  (blue square)
#   'neg_x2'  : negative x²  (red square)
#   'y2'      : positive y²  (blue square)
#   'neg_y2'  : negative y²  (red square)
#   'xy'      : positive xy  (green square)
#   'neg_xy'  : negative xy  (red square)
#   'x'       : positive x   (blue tall narrow rectangle)
#   'neg_x'   : negative x   (red tall narrow rectangle)
#   'y'       : positive y   (purple tall narrow rectangle)
#   'neg_y'   : negative y   (red tall narrow rectangle)
#   'one'     : positive 1   (dark blue small square)
#   'neg_one' : negative 1   (red small square)
#
# filename : saved to ./figures
# ─────────────────────────────────────────────────────────────────────────────

def make_algebra_tiles(expression, filename='algebra_tiles.png'):
    import matplotlib.patches as patches

    C = {
        'x2':'#5bacd6','neg_x2':'#e05c5c',
        'y2':'#5bacd6','neg_y2':'#e05c5c',
        'xy':'#4dab82','neg_xy':'#e05c5c',
        'x':'#5bacd6','neg_x':'#e05c5c',
        'y':'#9b59b6','neg_y':'#e05c5c',
        'one':'#3a3ab0','neg_one':'#e05c5c',
    }
    L = {
        'x2':r'$x^2$','neg_x2':r'$-x^2$',
        'y2':r'$y^2$','neg_y2':r'$-y^2$',
        'xy':r'$x \cdot y$','neg_xy':r'$-x \cdot y$',
        'x':r'$x$','neg_x':r'$-x$',
        'y':r'$y$','neg_y':r'$-y$',
        'one':r'$1$','neg_one':r'$-1$',
    }
    D = {
        'x2':(1.0,1.0),'neg_x2':(1.0,1.0),
        'y2':(1.0,1.0),'neg_y2':(1.0,1.0),
        'xy':(0.75,0.75),'neg_xy':(0.75,0.75),
        'x':(0.28,1.0),'neg_x':(0.28,1.0),
        'y':(0.28,1.0),'neg_y':(0.28,1.0),
        'one':(0.28,0.28),'neg_one':(0.28,0.28),
    }

    order = ['x2','neg_x2','y2','neg_y2','xy','neg_xy',
             'x','neg_x','y','neg_y','one','neg_one']

    tiles = []
    for kind in order:
        for _ in range(expression.get(kind, 0)):
            tiles.append(kind)

    if not tiles:
        return

    gap       = 0.15
    pad       = 0.35
    max_row_w = 5.5

    rows    = []
    cur_row = []
    cur_w   = 0.0

    for kind in tiles:
        tw = D[kind][0]
        if cur_w + tw + gap > max_row_w and cur_row:
            rows.append(cur_row)
            cur_row = [kind]
            cur_w   = tw + gap
        else:
            cur_row.append(kind)
            cur_w += tw + gap
    if cur_row:
        rows.append(cur_row)

    row_heights = [max(D[k][1] for k in row) for row in rows]
    total_h = sum(row_heights) + gap * (len(rows)-1) + pad * 2
    total_w = max_row_w + pad * 2

    fig, ax = plt.subplots(figsize=(total_w, total_h))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis('off')

    y_cursor = total_h - pad

    for r, row in enumerate(rows):
        rh = row_heights[r]
        y_cursor -= rh
        x_cursor  = pad

        for kind in row:
            tw, th = D[kind]
            rect = patches.FancyBboxPatch(
                (x_cursor, y_cursor + (rh-th)/2), tw, th,
                boxstyle='round,pad=0.02',
                linewidth=2.0, edgecolor='white', facecolor=C[kind]
            )
            ax.add_patch(rect)
            fs = 13 if tw >= 0.7 else 9
            ax.text(x_cursor + tw/2,
                    y_cursor + (rh-th)/2 + th/2,
                    L[kind],
                    ha='center', va='center',
                    fontsize=fs, color='white',
                    fontfamily='Times New Roman', style='italic')
            x_cursor += tw + gap

        y_cursor -= gap

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=144, bbox_inches='tight')
    print(f'Saved: {path}')

# ─────────────────────────────────────────────────────────────────────────────
# PASTE YOUR GRAPH CODE BELOW THIS LINE
# ─────────────────────────────────────────────────────────────────────────────


# ============================================================================
# UNIT 7 BANK GENERATED FIGURES - APPENDED BELOW FINAL APPROVED MARKER
# ============================================================================

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.64, y0=-8.48, m=-0.15999999999999925: y0+m*(x-x0), 'deriv': lambda x, m=-0.15999999999999925: np.zeros_like(x,dtype=float)+m, 'domain': (-8.262087598107524,-7.017912401892476), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.64, y0=-4.48, m=-4.159999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-4.159999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-7.787247715415067,-7.492752284584932), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.64, y0=-0.48, m=-8.16: y0+m*(x-x0), 'deriv': lambda x, m=-8.16: np.zeros_like(x,dtype=float)+m, 'domain': (-7.716632582712098,-7.5633674172879015), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.64, y0=3.52, m=-12.16: y0+m*(x-x0), 'deriv': lambda x, m=-12.16: np.zeros_like(x,dtype=float)+m, 'domain': (-7.691634903934738,-7.588365096065261), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.64, y0=7.52, m=-16.16: y0+m*(x-x0), 'deriv': lambda x, m=-16.16: np.zeros_like(x,dtype=float)+m, 'domain': (-7.678910719650691,-7.601089280349308), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.6399999999999997, y0=-8.48, m=3.8400000000000007: y0+m*(x-x0), 'deriv': lambda x, m=3.8400000000000007: np.zeros_like(x,dtype=float)+m, 'domain': (-3.79876725722667,-3.4812327427733294), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.6399999999999997, y0=-4.48, m=-0.15999999999999925: y0+m*(x-x0), 'deriv': lambda x, m=-0.15999999999999925: np.zeros_like(x,dtype=float)+m, 'domain': (-4.262087598107524,-3.0179124018924752), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.6399999999999997, y0=-0.48, m=-4.16: y0+m*(x-x0), 'deriv': lambda x, m=-4.16: np.zeros_like(x,dtype=float)+m, 'domain': (-3.7872477154150674,-3.492752284584932), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.6399999999999997, y0=3.52, m=-8.16: y0+m*(x-x0), 'deriv': lambda x, m=-8.16: np.zeros_like(x,dtype=float)+m, 'domain': (-3.716632582712098,-3.5633674172879015), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.6399999999999997, y0=7.52, m=-12.16: y0+m*(x-x0), 'deriv': lambda x, m=-12.16: np.zeros_like(x,dtype=float)+m, 'domain': (-3.6916349039347383,-3.588365096065261), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.3600000000000001, y0=-8.48, m=7.84: y0+m*(x-x0), 'deriv': lambda x, m=7.84: np.zeros_like(x,dtype=float)+m, 'domain': (0.2802886633846188,0.4397113366153814), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.3600000000000001, y0=-4.48, m=3.8400000000000007: y0+m*(x-x0), 'deriv': lambda x, m=3.8400000000000007: np.zeros_like(x,dtype=float)+m, 'domain': (0.20123274277332964,0.5187672572266706), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.3600000000000001, y0=-0.48, m=-0.15999999999999992: y0+m*(x-x0), 'deriv': lambda x, m=-0.15999999999999992: np.zeros_like(x,dtype=float)+m, 'domain': (-0.26208759810752424,0.9820875981075244), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.3600000000000001, y0=3.52, m=-4.16: y0+m*(x-x0), 'deriv': lambda x, m=-4.16: np.zeros_like(x,dtype=float)+m, 'domain': (0.2127522845849325,0.5072477154150676), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.3600000000000001, y0=7.52, m=-8.16: y0+m*(x-x0), 'deriv': lambda x, m=-8.16: np.zeros_like(x,dtype=float)+m, 'domain': (0.2833674172879018,0.43663258271209837), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.36, y0=-8.48, m=11.84: y0+m*(x-x0), 'deriv': lambda x, m=11.84: np.zeros_like(x,dtype=float)+m, 'domain': (4.30697931338142,4.413020686618581), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.36, y0=-4.48, m=7.84: y0+m*(x-x0), 'deriv': lambda x, m=7.84: np.zeros_like(x,dtype=float)+m, 'domain': (4.280288663384619,4.439711336615382), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.36, y0=-0.48, m=3.84: y0+m*(x-x0), 'deriv': lambda x, m=3.84: np.zeros_like(x,dtype=float)+m, 'domain': (4.20123274277333,4.518767257226671), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.36, y0=3.52, m=-0.1599999999999997: y0+m*(x-x0), 'deriv': lambda x, m=-0.1599999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (3.737912401892476,4.982087598107524), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.36, y0=7.52, m=-4.159999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-4.159999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (4.212752284584933,4.507247715415068), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.36, y0=-8.48, m=15.84: y0+m*(x-x0), 'deriv': lambda x, m=15.84: np.zeros_like(x,dtype=float)+m, 'domain': (8.320306294943672,8.399693705056327), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.36, y0=-4.48, m=11.84: y0+m*(x-x0), 'deriv': lambda x, m=11.84: np.zeros_like(x,dtype=float)+m, 'domain': (8.306979313381419,8.41302068661858), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.36, y0=-0.48, m=7.84: y0+m*(x-x0), 'deriv': lambda x, m=7.84: np.zeros_like(x,dtype=float)+m, 'domain': (8.280288663384619,8.43971133661538), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.36, y0=3.52, m=3.84: y0+m*(x-x0), 'deriv': lambda x, m=3.84: np.zeros_like(x,dtype=float)+m, 'domain': (8.201232742773328,8.51876725722667), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.36, y0=7.52, m=-0.16000000000000014: y0+m*(x-x0), 'deriv': lambda x, m=-0.16000000000000014: np.zeros_like(x,dtype=float)+m, 'domain': (7.737912401892475,8.982087598107524), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d1_02_v1_fa_field_read.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.34, y0=-8.5, m=1.1600000000000001: y0+m*(x-x0), 'deriv': lambda x, m=1.1600000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-8.751351961903762,-7.928648038096238), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.34, y0=-4.5, m=-2.84: y0+m*(x-x0), 'deriv': lambda x, m=-2.84: np.zeros_like(x,dtype=float)+m, 'domain': (-8.549238825237063,-8.130761174762936), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.34, y0=-0.5, m=-6.84: y0+m*(x-x0), 'deriv': lambda x, m=-6.84: np.zeros_like(x,dtype=float)+m, 'domain': (-8.43113643322473,-8.24886356677527), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.34, y0=3.5, m=-10.84: y0+m*(x-x0), 'deriv': lambda x, m=-10.84: np.zeros_like(x,dtype=float)+m, 'domain': (-8.397872349302242,-8.282127650697758), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.34, y0=7.5, m=-14.84: y0+m*(x-x0), 'deriv': lambda x, m=-14.84: np.zeros_like(x,dtype=float)+m, 'domain': (-8.382356772332997,-8.297643227667002), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.34, y0=-8.5, m=5.16: y0+m*(x-x0), 'deriv': lambda x, m=5.16: np.zeros_like(x,dtype=float)+m, 'domain': (-4.459862874855721,-4.220137125144278), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.34, y0=-4.5, m=1.1600000000000001: y0+m*(x-x0), 'deriv': lambda x, m=1.1600000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-4.751351961903762,-3.9286480380962385), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.34, y0=-0.5, m=-2.84: y0+m*(x-x0), 'deriv': lambda x, m=-2.84: np.zeros_like(x,dtype=float)+m, 'domain': (-4.549238825237064,-4.130761174762935), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.34, y0=3.5, m=-6.84: y0+m*(x-x0), 'deriv': lambda x, m=-6.84: np.zeros_like(x,dtype=float)+m, 'domain': (-4.4311364332247285,-4.248863566775271), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.34, y0=7.5, m=-10.84: y0+m*(x-x0), 'deriv': lambda x, m=-10.84: np.zeros_like(x,dtype=float)+m, 'domain': (-4.397872349302242,-4.282127650697758), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.34, y0=-8.5, m=9.16: y0+m*(x-x0), 'deriv': lambda x, m=9.16: np.zeros_like(x,dtype=float)+m, 'domain': (-0.4083710708133372,-0.27162892918666287), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.34, y0=-4.5, m=5.16: y0+m*(x-x0), 'deriv': lambda x, m=5.16: np.zeros_like(x,dtype=float)+m, 'domain': (-0.4598628748557214,-0.22013712514427863), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.34, y0=-0.5, m=1.16: y0+m*(x-x0), 'deriv': lambda x, m=1.16: np.zeros_like(x,dtype=float)+m, 'domain': (-0.7513519619037615,0.07135196190376147), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.34, y0=3.5, m=-2.84: y0+m*(x-x0), 'deriv': lambda x, m=-2.84: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5492388252370641,-0.13076117476293592), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.34, y0=7.5, m=-6.84: y0+m*(x-x0), 'deriv': lambda x, m=-6.84: np.zeros_like(x,dtype=float)+m, 'domain': (-0.43113643322472894,-0.24886356677527113), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.66, y0=-8.5, m=13.16: y0+m*(x-x0), 'deriv': lambda x, m=13.16: np.zeros_like(x,dtype=float)+m, 'domain': (3.612265274989215,3.7077347250107855), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.66, y0=-4.5, m=9.16: y0+m*(x-x0), 'deriv': lambda x, m=9.16: np.zeros_like(x,dtype=float)+m, 'domain': (3.5916289291866628,3.7283710708133375), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.66, y0=-0.5, m=5.16: y0+m*(x-x0), 'deriv': lambda x, m=5.16: np.zeros_like(x,dtype=float)+m, 'domain': (3.5401371251442786,3.7798628748557217), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.66, y0=3.5, m=1.1600000000000001: y0+m*(x-x0), 'deriv': lambda x, m=1.1600000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (3.248648038096239,4.071351961903762), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.66, y0=7.5, m=-2.84: y0+m*(x-x0), 'deriv': lambda x, m=-2.84: np.zeros_like(x,dtype=float)+m, 'domain': (3.450761174762936,3.869238825237064), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.66, y0=-8.5, m=17.16: y0+m*(x-x0), 'deriv': lambda x, m=17.16: np.zeros_like(x,dtype=float)+m, 'domain': (7.623348893796591,7.696651106203409), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.66, y0=-4.5, m=13.16: y0+m*(x-x0), 'deriv': lambda x, m=13.16: np.zeros_like(x,dtype=float)+m, 'domain': (7.612265274989214,7.707734725010786), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.66, y0=-0.5, m=9.16: y0+m*(x-x0), 'deriv': lambda x, m=9.16: np.zeros_like(x,dtype=float)+m, 'domain': (7.591628929186663,7.7283710708133375), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.66, y0=3.5, m=5.16: y0+m*(x-x0), 'deriv': lambda x, m=5.16: np.zeros_like(x,dtype=float)+m, 'domain': (7.540137125144279,7.779862874855722), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.66, y0=7.5, m=1.1600000000000001: y0+m*(x-x0), 'deriv': lambda x, m=1.1600000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (7.248648038096238,8.071351961903762), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d1_02_v2_fa_field_read.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.65, y0=-8.19, m=0.5399999999999991: y0+m*(x-x0), 'deriv': lambda x, m=0.5399999999999991: np.zeros_like(x,dtype=float)+m, 'domain': (-9.450713911868046,-7.849286088131954), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.65, y0=-4.19, m=-3.46: y0+m*(x-x0), 'deriv': lambda x, m=-3.46: np.zeros_like(x,dtype=float)+m, 'domain': (-8.90266472780481,-8.39733527219519), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.65, y0=-0.19, m=-7.460000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-7.460000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-8.77090250586611,-8.529097494133891), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.65, y0=3.81, m=-11.46: y0+m*(x-x0), 'deriv': lambda x, m=-11.46: np.zeros_like(x,dtype=float)+m, 'domain': (-8.729106033811142,-8.570893966188859), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.65, y0=7.81, m=-15.46: y0+m*(x-x0), 'deriv': lambda x, m=-15.46: np.zeros_like(x,dtype=float)+m, 'domain': (-8.708738827912956,-8.591261172087044), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.65, y0=-8.19, m=4.539999999999999: y0+m*(x-x0), 'deriv': lambda x, m=4.539999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-4.845748269744583,-4.454251730255418), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.65, y0=-4.19, m=0.54: y0+m*(x-x0), 'deriv': lambda x, m=0.54: np.zeros_like(x,dtype=float)+m, 'domain': (-5.450713911868045,-3.8492860881319553), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.65, y0=-0.19, m=-3.46: y0+m*(x-x0), 'deriv': lambda x, m=-3.46: np.zeros_like(x,dtype=float)+m, 'domain': (-4.90266472780481,-4.397335272195191), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.65, y0=3.81, m=-7.460000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-7.460000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-4.77090250586611,-4.52909749413389), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.65, y0=7.81, m=-11.46: y0+m*(x-x0), 'deriv': lambda x, m=-11.46: np.zeros_like(x,dtype=float)+m, 'domain': (-4.729106033811143,-4.570893966188858), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.65, y0=-8.19, m=8.54: y0+m*(x-x0), 'deriv': lambda x, m=8.54: np.zeros_like(x,dtype=float)+m, 'domain': (-0.755834275162318,-0.544165724837682), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.65, y0=-4.19, m=4.540000000000001: y0+m*(x-x0), 'deriv': lambda x, m=4.540000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-0.8457482697445817,-0.45425173025541826), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.65, y0=-0.19, m=0.54: y0+m*(x-x0), 'deriv': lambda x, m=0.54: np.zeros_like(x,dtype=float)+m, 'domain': (-1.4507139118680452,0.15071391186804517), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.65, y0=3.81, m=-3.46: y0+m*(x-x0), 'deriv': lambda x, m=-3.46: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9026647278048092,-0.3973352721951909), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.65, y0=7.81, m=-7.459999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-7.459999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-0.7709025058661098,-0.5290974941338903), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.35, y0=-8.19, m=12.54: y0+m*(x-x0), 'deriv': lambda x, m=12.54: np.zeros_like(x,dtype=float)+m, 'domain': (3.2776618600394527,3.4223381399605475), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.35, y0=-4.19, m=8.540000000000001: y0+m*(x-x0), 'deriv': lambda x, m=8.540000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (3.244165724837682,3.455834275162318), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.35, y0=-0.19, m=4.54: y0+m*(x-x0), 'deriv': lambda x, m=4.54: np.zeros_like(x,dtype=float)+m, 'domain': (3.1542517302554183,3.545748269744582), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.35, y0=3.81, m=0.54: y0+m*(x-x0), 'deriv': lambda x, m=0.54: np.zeros_like(x,dtype=float)+m, 'domain': (2.549286088131955,4.150713911868046), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.35, y0=7.81, m=-3.459999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-3.459999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (3.097335272195191,3.602664727804809), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.35, y0=-8.19, m=16.54: y0+m*(x-x0), 'deriv': lambda x, m=16.54: np.zeros_like(x,dtype=float)+m, 'domain': (7.295082142596158,7.4049178574038415), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.35, y0=-4.19, m=12.54: y0+m*(x-x0), 'deriv': lambda x, m=12.54: np.zeros_like(x,dtype=float)+m, 'domain': (7.277661860039452,7.422338139960547), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.35, y0=-0.19, m=8.54: y0+m*(x-x0), 'deriv': lambda x, m=8.54: np.zeros_like(x,dtype=float)+m, 'domain': (7.244165724837681,7.455834275162318), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.35, y0=3.81, m=4.539999999999999: y0+m*(x-x0), 'deriv': lambda x, m=4.539999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (7.154251730255417,7.545748269744582), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.35, y0=7.81, m=0.54: y0+m*(x-x0), 'deriv': lambda x, m=0.54: np.zeros_like(x,dtype=float)+m, 'domain': (6.549286088131955,8.150713911868046), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d1_02_v3_fa_field_read.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.75, y0=-7.98, m=8.75: y0+m*(x-x0), 'deriv': lambda x, m=8.75: np.zeros_like(x,dtype=float)+m, 'domain': (-8.814721556961617,-8.685278443038383), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.75, y0=-3.98, m=8.75: y0+m*(x-x0), 'deriv': lambda x, m=8.75: np.zeros_like(x,dtype=float)+m, 'domain': (-8.814721556961617,-8.685278443038383), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.75, y0=0.020000000000000018, m=8.75: y0+m*(x-x0), 'deriv': lambda x, m=8.75: np.zeros_like(x,dtype=float)+m, 'domain': (-8.814721556961617,-8.685278443038383), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.75, y0=4.02, m=8.75: y0+m*(x-x0), 'deriv': lambda x, m=8.75: np.zeros_like(x,dtype=float)+m, 'domain': (-8.814721556961617,-8.685278443038383), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.75, y0=8.02, m=8.75: y0+m*(x-x0), 'deriv': lambda x, m=8.75: np.zeros_like(x,dtype=float)+m, 'domain': (-8.814721556961617,-8.685278443038383), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.75, y0=-7.98, m=4.75: y0+m*(x-x0), 'deriv': lambda x, m=4.75: np.zeros_like(x,dtype=float)+m, 'domain': (-4.86742597419841,-4.63257402580159), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.75, y0=-3.98, m=4.75: y0+m*(x-x0), 'deriv': lambda x, m=4.75: np.zeros_like(x,dtype=float)+m, 'domain': (-4.86742597419841,-4.63257402580159), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.75, y0=0.020000000000000018, m=4.75: y0+m*(x-x0), 'deriv': lambda x, m=4.75: np.zeros_like(x,dtype=float)+m, 'domain': (-4.86742597419841,-4.63257402580159), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.75, y0=4.02, m=4.75: y0+m*(x-x0), 'deriv': lambda x, m=4.75: np.zeros_like(x,dtype=float)+m, 'domain': (-4.86742597419841,-4.63257402580159), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.75, y0=8.02, m=4.75: y0+m*(x-x0), 'deriv': lambda x, m=4.75: np.zeros_like(x,dtype=float)+m, 'domain': (-4.86742597419841,-4.63257402580159), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.75, y0=-7.98, m=0.75: y0+m*(x-x0), 'deriv': lambda x, m=0.75: np.zeros_like(x,dtype=float)+m, 'domain': (-1.206,-0.29399999999999993), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.75, y0=-3.98, m=0.75: y0+m*(x-x0), 'deriv': lambda x, m=0.75: np.zeros_like(x,dtype=float)+m, 'domain': (-1.206,-0.29399999999999993), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.75, y0=0.020000000000000018, m=0.75: y0+m*(x-x0), 'deriv': lambda x, m=0.75: np.zeros_like(x,dtype=float)+m, 'domain': (-1.206,-0.29399999999999993), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.75, y0=4.02, m=0.75: y0+m*(x-x0), 'deriv': lambda x, m=0.75: np.zeros_like(x,dtype=float)+m, 'domain': (-1.206,-0.29399999999999993), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.75, y0=8.02, m=0.75: y0+m*(x-x0), 'deriv': lambda x, m=0.75: np.zeros_like(x,dtype=float)+m, 'domain': (-1.206,-0.29399999999999993), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.25, y0=-7.98, m=-3.25: y0+m*(x-x0), 'deriv': lambda x, m=-3.25: np.zeros_like(x,dtype=float)+m, 'domain': (3.082371066162612,3.417628933837388), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.25, y0=-3.98, m=-3.25: y0+m*(x-x0), 'deriv': lambda x, m=-3.25: np.zeros_like(x,dtype=float)+m, 'domain': (3.082371066162612,3.417628933837388), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.25, y0=0.020000000000000018, m=-3.25: y0+m*(x-x0), 'deriv': lambda x, m=-3.25: np.zeros_like(x,dtype=float)+m, 'domain': (3.082371066162612,3.417628933837388), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.25, y0=4.02, m=-3.25: y0+m*(x-x0), 'deriv': lambda x, m=-3.25: np.zeros_like(x,dtype=float)+m, 'domain': (3.082371066162612,3.417628933837388), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.25, y0=8.02, m=-3.25: y0+m*(x-x0), 'deriv': lambda x, m=-3.25: np.zeros_like(x,dtype=float)+m, 'domain': (3.082371066162612,3.417628933837388), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.25, y0=-7.98, m=-7.25: y0+m*(x-x0), 'deriv': lambda x, m=-7.25: np.zeros_like(x,dtype=float)+m, 'domain': (7.172116683631892,7.327883316368108), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.25, y0=-3.98, m=-7.25: y0+m*(x-x0), 'deriv': lambda x, m=-7.25: np.zeros_like(x,dtype=float)+m, 'domain': (7.172116683631892,7.327883316368108), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.25, y0=0.020000000000000018, m=-7.25: y0+m*(x-x0), 'deriv': lambda x, m=-7.25: np.zeros_like(x,dtype=float)+m, 'domain': (7.172116683631892,7.327883316368108), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.25, y0=4.02, m=-7.25: y0+m*(x-x0), 'deriv': lambda x, m=-7.25: np.zeros_like(x,dtype=float)+m, 'domain': (7.172116683631892,7.327883316368108), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.25, y0=8.02, m=-7.25: y0+m*(x-x0), 'deriv': lambda x, m=-7.25: np.zeros_like(x,dtype=float)+m, 'domain': (7.172116683631892,7.327883316368108), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d1_02_v4_fa_field_read.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.37, y0=-8.46, m=-16.830000000000002: y0+m*(x-x0), 'deriv': lambda x, m=-16.830000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (-7.406180989384755,-7.333819010615245), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.37, y0=-4.46, m=-12.83: y0+m*(x-x0), 'deriv': lambda x, m=-12.83: np.zeros_like(x,dtype=float)+m, 'domain': (-7.417401054009122,-7.322598945990878), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.37, y0=-0.45999999999999996, m=-8.83: y0+m*(x-x0), 'deriv': lambda x, m=-8.83: np.zeros_like(x,dtype=float)+m, 'domain': (-7.43864387483404,-7.30135612516596), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.37, y0=3.54, m=-4.83: y0+m*(x-x0), 'deriv': lambda x, m=-4.83: np.zeros_like(x,dtype=float)+m, 'domain': (-7.493671206698986,-7.246328793301014), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.37, y0=7.54, m=-0.8300000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-0.8300000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-7.839383552056655,-6.900616447943345), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.37, y0=-8.46, m=-12.830000000000002: y0+m*(x-x0), 'deriv': lambda x, m=-12.830000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (-3.4174010540091224,-3.322598945990878), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.37, y0=-4.46, m=-8.83: y0+m*(x-x0), 'deriv': lambda x, m=-8.83: np.zeros_like(x,dtype=float)+m, 'domain': (-3.4386438748340398,-3.3013561251659604), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.37, y0=-0.45999999999999996, m=-4.83: y0+m*(x-x0), 'deriv': lambda x, m=-4.83: np.zeros_like(x,dtype=float)+m, 'domain': (-3.4936712066989855,-3.2463287933010148), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.37, y0=3.54, m=-0.8300000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-0.8300000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-3.8393835520566553,-2.900616447943345), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.37, y0=7.54, m=3.17: y0+m*(x-x0), 'deriv': lambda x, m=3.17: np.zeros_like(x,dtype=float)+m, 'domain': (-3.553514469898554,-3.186485530101446), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6299999999999999, y0=-8.46, m=-8.830000000000002: y0+m*(x-x0), 'deriv': lambda x, m=-8.830000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (0.5613561251659601,0.6986438748340397), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6299999999999999, y0=-4.46, m=-4.83: y0+m*(x-x0), 'deriv': lambda x, m=-4.83: np.zeros_like(x,dtype=float)+m, 'domain': (0.5063287933010144,0.7536712066989854), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6299999999999999, y0=-0.45999999999999996, m=-0.8300000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-0.8300000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (0.16061644794334473,1.099383552056655), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6299999999999999, y0=3.54, m=3.17: y0+m*(x-x0), 'deriv': lambda x, m=3.17: np.zeros_like(x,dtype=float)+m, 'domain': (0.44648553010144565,0.8135144698985541), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6299999999999999, y0=7.54, m=7.17: y0+m*(x-x0), 'deriv': lambda x, m=7.17: np.zeros_like(x,dtype=float)+m, 'domain': (0.5457388634513182,0.7142611365486816), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.63, y0=-8.46, m=-4.830000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-4.830000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (4.506328793301014,4.753671206698986), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.63, y0=-4.46, m=-0.8300000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-0.8300000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (4.160616447943345,5.099383552056655), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.63, y0=-0.45999999999999996, m=3.17: y0+m*(x-x0), 'deriv': lambda x, m=3.17: np.zeros_like(x,dtype=float)+m, 'domain': (4.446485530101445,4.813514469898554), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.63, y0=3.54, m=7.17: y0+m*(x-x0), 'deriv': lambda x, m=7.17: np.zeros_like(x,dtype=float)+m, 'domain': (4.5457388634513185,4.714261136548681), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.63, y0=7.54, m=11.17: y0+m*(x-x0), 'deriv': lambda x, m=11.17: np.zeros_like(x,dtype=float)+m, 'domain': (4.575606976061249,4.684393023938751), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.629999999999999, y0=-8.46, m=-0.8300000000000018: y0+m*(x-x0), 'deriv': lambda x, m=-0.8300000000000018: np.zeros_like(x,dtype=float)+m, 'domain': (8.160616447943344,9.099383552056654), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.629999999999999, y0=-4.46, m=3.169999999999999: y0+m*(x-x0), 'deriv': lambda x, m=3.169999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (8.446485530101445,8.813514469898553), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.629999999999999, y0=-0.45999999999999996, m=7.169999999999998: y0+m*(x-x0), 'deriv': lambda x, m=7.169999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (8.545738863451318,8.71426113654868), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.629999999999999, y0=3.54, m=11.169999999999998: y0+m*(x-x0), 'deriv': lambda x, m=11.169999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (8.575606976061248,8.68439302393875), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.629999999999999, y0=7.54, m=15.169999999999998: y0+m*(x-x0), 'deriv': lambda x, m=15.169999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (8.589876139776898,8.6701238602231), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d1_02_v5_fa_field_read.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.87, y0=-8.32, m=7.87: y0+m*(x-x0), 'deriv': lambda x, m=7.87: np.zeros_like(x,dtype=float)+m, 'domain': (-7.953193857455128,-7.786806142544872), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.87, y0=-4.32, m=7.87: y0+m*(x-x0), 'deriv': lambda x, m=7.87: np.zeros_like(x,dtype=float)+m, 'domain': (-7.953193857455128,-7.786806142544872), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.87, y0=-0.31999999999999995, m=7.87: y0+m*(x-x0), 'deriv': lambda x, m=7.87: np.zeros_like(x,dtype=float)+m, 'domain': (-7.953193857455128,-7.786806142544872), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.87, y0=3.68, m=7.87: y0+m*(x-x0), 'deriv': lambda x, m=7.87: np.zeros_like(x,dtype=float)+m, 'domain': (-7.953193857455128,-7.786806142544872), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.87, y0=7.68, m=7.87: y0+m*(x-x0), 'deriv': lambda x, m=7.87: np.zeros_like(x,dtype=float)+m, 'domain': (-7.953193857455128,-7.786806142544872), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.87, y0=-8.32, m=3.87: y0+m*(x-x0), 'deriv': lambda x, m=3.87: np.zeros_like(x,dtype=float)+m, 'domain': (-4.035119238503487,-3.704880761496513), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.87, y0=-4.32, m=3.87: y0+m*(x-x0), 'deriv': lambda x, m=3.87: np.zeros_like(x,dtype=float)+m, 'domain': (-4.035119238503487,-3.704880761496513), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.87, y0=-0.31999999999999995, m=3.87: y0+m*(x-x0), 'deriv': lambda x, m=3.87: np.zeros_like(x,dtype=float)+m, 'domain': (-4.035119238503487,-3.704880761496513), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.87, y0=3.68, m=3.87: y0+m*(x-x0), 'deriv': lambda x, m=3.87: np.zeros_like(x,dtype=float)+m, 'domain': (-4.035119238503487,-3.704880761496513), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.87, y0=7.68, m=3.87: y0+m*(x-x0), 'deriv': lambda x, m=3.87: np.zeros_like(x,dtype=float)+m, 'domain': (-4.035119238503487,-3.704880761496513), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.13, y0=-8.32, m=-0.13: y0+m*(x-x0), 'deriv': lambda x, m=-0.13: np.zeros_like(x,dtype=float)+m, 'domain': (-0.524492707446537,0.784492707446537), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.13, y0=-4.32, m=-0.13: y0+m*(x-x0), 'deriv': lambda x, m=-0.13: np.zeros_like(x,dtype=float)+m, 'domain': (-0.524492707446537,0.784492707446537), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.13, y0=-0.31999999999999995, m=-0.13: y0+m*(x-x0), 'deriv': lambda x, m=-0.13: np.zeros_like(x,dtype=float)+m, 'domain': (-0.524492707446537,0.784492707446537), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.13, y0=3.68, m=-0.13: y0+m*(x-x0), 'deriv': lambda x, m=-0.13: np.zeros_like(x,dtype=float)+m, 'domain': (-0.524492707446537,0.784492707446537), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.13, y0=7.68, m=-0.13: y0+m*(x-x0), 'deriv': lambda x, m=-0.13: np.zeros_like(x,dtype=float)+m, 'domain': (-0.524492707446537,0.784492707446537), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.13, y0=-8.32, m=-4.13: y0+m*(x-x0), 'deriv': lambda x, m=-4.13: np.zeros_like(x,dtype=float)+m, 'domain': (3.9746818034788487,4.2853181965211515), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.13, y0=-4.32, m=-4.13: y0+m*(x-x0), 'deriv': lambda x, m=-4.13: np.zeros_like(x,dtype=float)+m, 'domain': (3.9746818034788487,4.2853181965211515), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.13, y0=-0.31999999999999995, m=-4.13: y0+m*(x-x0), 'deriv': lambda x, m=-4.13: np.zeros_like(x,dtype=float)+m, 'domain': (3.9746818034788487,4.2853181965211515), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.13, y0=3.68, m=-4.13: y0+m*(x-x0), 'deriv': lambda x, m=-4.13: np.zeros_like(x,dtype=float)+m, 'domain': (3.9746818034788487,4.2853181965211515), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.13, y0=7.68, m=-4.13: y0+m*(x-x0), 'deriv': lambda x, m=-4.13: np.zeros_like(x,dtype=float)+m, 'domain': (3.9746818034788487,4.2853181965211515), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.13, y0=-8.32, m=-8.13: y0+m*(x-x0), 'deriv': lambda x, m=-8.13: np.zeros_like(x,dtype=float)+m, 'domain': (8.049426411201177,8.210573588798825), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.13, y0=-4.32, m=-8.13: y0+m*(x-x0), 'deriv': lambda x, m=-8.13: np.zeros_like(x,dtype=float)+m, 'domain': (8.049426411201177,8.210573588798825), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.13, y0=-0.31999999999999995, m=-8.13: y0+m*(x-x0), 'deriv': lambda x, m=-8.13: np.zeros_like(x,dtype=float)+m, 'domain': (8.049426411201177,8.210573588798825), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.13, y0=3.68, m=-8.13: y0+m*(x-x0), 'deriv': lambda x, m=-8.13: np.zeros_like(x,dtype=float)+m, 'domain': (8.049426411201177,8.210573588798825), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.13, y0=7.68, m=-8.13: y0+m*(x-x0), 'deriv': lambda x, m=-8.13: np.zeros_like(x,dtype=float)+m, 'domain': (8.049426411201177,8.210573588798825), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d1_02_v6_fa_field_read.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.58, y0=-7.88, m=8.58: y0+m*(x-x0), 'deriv': lambda x, m=8.58: np.zeros_like(x,dtype=float)+m, 'domain': (-8.663351870538179,-8.496648129461821), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.58, y0=-3.88, m=8.58: y0+m*(x-x0), 'deriv': lambda x, m=8.58: np.zeros_like(x,dtype=float)+m, 'domain': (-8.663351870538179,-8.496648129461821), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.58, y0=0.12, m=8.58: y0+m*(x-x0), 'deriv': lambda x, m=8.58: np.zeros_like(x,dtype=float)+m, 'domain': (-8.663351870538179,-8.496648129461821), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.58, y0=4.12, m=8.58: y0+m*(x-x0), 'deriv': lambda x, m=8.58: np.zeros_like(x,dtype=float)+m, 'domain': (-8.663351870538179,-8.496648129461821), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.58, y0=8.12, m=8.58: y0+m*(x-x0), 'deriv': lambda x, m=8.58: np.zeros_like(x,dtype=float)+m, 'domain': (-8.663351870538179,-8.496648129461821), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.58, y0=-7.88, m=4.58: y0+m*(x-x0), 'deriv': lambda x, m=4.58: np.zeros_like(x,dtype=float)+m, 'domain': (-4.73358691631723,-4.42641308368277), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.58, y0=-3.88, m=4.58: y0+m*(x-x0), 'deriv': lambda x, m=4.58: np.zeros_like(x,dtype=float)+m, 'domain': (-4.73358691631723,-4.42641308368277), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.58, y0=0.12, m=4.58: y0+m*(x-x0), 'deriv': lambda x, m=4.58: np.zeros_like(x,dtype=float)+m, 'domain': (-4.73358691631723,-4.42641308368277), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.58, y0=4.12, m=4.58: y0+m*(x-x0), 'deriv': lambda x, m=4.58: np.zeros_like(x,dtype=float)+m, 'domain': (-4.73358691631723,-4.42641308368277), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.58, y0=8.12, m=4.58: y0+m*(x-x0), 'deriv': lambda x, m=4.58: np.zeros_like(x,dtype=float)+m, 'domain': (-4.73358691631723,-4.42641308368277), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.58, y0=-7.88, m=0.58: y0+m*(x-x0), 'deriv': lambda x, m=0.58: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2028224562684984,0.042822456268498454), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.58, y0=-3.88, m=0.58: y0+m*(x-x0), 'deriv': lambda x, m=0.58: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2028224562684984,0.042822456268498454), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.58, y0=0.12, m=0.58: y0+m*(x-x0), 'deriv': lambda x, m=0.58: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2028224562684984,0.042822456268498454), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.58, y0=4.12, m=0.58: y0+m*(x-x0), 'deriv': lambda x, m=0.58: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2028224562684984,0.042822456268498454), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.58, y0=8.12, m=0.58: y0+m*(x-x0), 'deriv': lambda x, m=0.58: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2028224562684984,0.042822456268498454), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.42, y0=-7.88, m=-3.42: y0+m*(x-x0), 'deriv': lambda x, m=-3.42: np.zeros_like(x,dtype=float)+m, 'domain': (3.217934486787692,3.622065513212308), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.42, y0=-3.88, m=-3.42: y0+m*(x-x0), 'deriv': lambda x, m=-3.42: np.zeros_like(x,dtype=float)+m, 'domain': (3.217934486787692,3.622065513212308), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.42, y0=0.12, m=-3.42: y0+m*(x-x0), 'deriv': lambda x, m=-3.42: np.zeros_like(x,dtype=float)+m, 'domain': (3.217934486787692,3.622065513212308), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.42, y0=4.12, m=-3.42: y0+m*(x-x0), 'deriv': lambda x, m=-3.42: np.zeros_like(x,dtype=float)+m, 'domain': (3.217934486787692,3.622065513212308), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.42, y0=8.12, m=-3.42: y0+m*(x-x0), 'deriv': lambda x, m=-3.42: np.zeros_like(x,dtype=float)+m, 'domain': (3.217934486787692,3.622065513212308), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.42, y0=-7.88, m=-7.42: y0+m*(x-x0), 'deriv': lambda x, m=-7.42: np.zeros_like(x,dtype=float)+m, 'domain': (7.323834366989816,7.516165633010184), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.42, y0=-3.88, m=-7.42: y0+m*(x-x0), 'deriv': lambda x, m=-7.42: np.zeros_like(x,dtype=float)+m, 'domain': (7.323834366989816,7.516165633010184), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.42, y0=0.12, m=-7.42: y0+m*(x-x0), 'deriv': lambda x, m=-7.42: np.zeros_like(x,dtype=float)+m, 'domain': (7.323834366989816,7.516165633010184), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.42, y0=4.12, m=-7.42: y0+m*(x-x0), 'deriv': lambda x, m=-7.42: np.zeros_like(x,dtype=float)+m, 'domain': (7.323834366989816,7.516165633010184), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.42, y0=8.12, m=-7.42: y0+m*(x-x0), 'deriv': lambda x, m=-7.42: np.zeros_like(x,dtype=float)+m, 'domain': (7.323834366989816,7.516165633010184), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d1_02_v7_fa_field_read.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.6, y0=-8.25, m=-8.6: y0+m*(x-x0), 'deriv': lambda x, m=-8.6: np.zeros_like(x,dtype=float)+m, 'domain': (-8.683160617885687,-8.516839382114313), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.6, y0=-4.25, m=-8.6: y0+m*(x-x0), 'deriv': lambda x, m=-8.6: np.zeros_like(x,dtype=float)+m, 'domain': (-8.683160617885687,-8.516839382114313), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.6, y0=-0.25, m=-8.6: y0+m*(x-x0), 'deriv': lambda x, m=-8.6: np.zeros_like(x,dtype=float)+m, 'domain': (-8.683160617885687,-8.516839382114313), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.6, y0=3.75, m=-8.6: y0+m*(x-x0), 'deriv': lambda x, m=-8.6: np.zeros_like(x,dtype=float)+m, 'domain': (-8.683160617885687,-8.516839382114313), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.6, y0=7.75, m=-8.6: y0+m*(x-x0), 'deriv': lambda x, m=-8.6: np.zeros_like(x,dtype=float)+m, 'domain': (-8.683160617885687,-8.516839382114313), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.6, y0=-8.25, m=-4.6: y0+m*(x-x0), 'deriv': lambda x, m=-4.6: np.zeros_like(x,dtype=float)+m, 'domain': (-4.752949343918351,-4.4470506560816485), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.6, y0=-4.25, m=-4.6: y0+m*(x-x0), 'deriv': lambda x, m=-4.6: np.zeros_like(x,dtype=float)+m, 'domain': (-4.752949343918351,-4.4470506560816485), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.6, y0=-0.25, m=-4.6: y0+m*(x-x0), 'deriv': lambda x, m=-4.6: np.zeros_like(x,dtype=float)+m, 'domain': (-4.752949343918351,-4.4470506560816485), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.6, y0=3.75, m=-4.6: y0+m*(x-x0), 'deriv': lambda x, m=-4.6: np.zeros_like(x,dtype=float)+m, 'domain': (-4.752949343918351,-4.4470506560816485), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.6, y0=7.75, m=-4.6: y0+m*(x-x0), 'deriv': lambda x, m=-4.6: np.zeros_like(x,dtype=float)+m, 'domain': (-4.752949343918351,-4.4470506560816485), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.6, y0=-8.25, m=-0.6: y0+m*(x-x0), 'deriv': lambda x, m=-0.6: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2173949065130318,0.017394906513031883), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.6, y0=-4.25, m=-0.6: y0+m*(x-x0), 'deriv': lambda x, m=-0.6: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2173949065130318,0.017394906513031883), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.6, y0=-0.25, m=-0.6: y0+m*(x-x0), 'deriv': lambda x, m=-0.6: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2173949065130318,0.017394906513031883), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.6, y0=3.75, m=-0.6: y0+m*(x-x0), 'deriv': lambda x, m=-0.6: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2173949065130318,0.017394906513031883), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.6, y0=7.75, m=-0.6: y0+m*(x-x0), 'deriv': lambda x, m=-0.6: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2173949065130318,0.017394906513031883), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.4, y0=-8.25, m=3.4: y0+m*(x-x0), 'deriv': lambda x, m=3.4: np.zeros_like(x,dtype=float)+m, 'domain': (3.196840246726084,3.603159753273916), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.4, y0=-4.25, m=3.4: y0+m*(x-x0), 'deriv': lambda x, m=3.4: np.zeros_like(x,dtype=float)+m, 'domain': (3.196840246726084,3.603159753273916), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.4, y0=-0.25, m=3.4: y0+m*(x-x0), 'deriv': lambda x, m=3.4: np.zeros_like(x,dtype=float)+m, 'domain': (3.196840246726084,3.603159753273916), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.4, y0=3.75, m=3.4: y0+m*(x-x0), 'deriv': lambda x, m=3.4: np.zeros_like(x,dtype=float)+m, 'domain': (3.196840246726084,3.603159753273916), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.4, y0=7.75, m=3.4: y0+m*(x-x0), 'deriv': lambda x, m=3.4: np.zeros_like(x,dtype=float)+m, 'domain': (3.196840246726084,3.603159753273916), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.4, y0=-8.25, m=7.4: y0+m*(x-x0), 'deriv': lambda x, m=7.4: np.zeros_like(x,dtype=float)+m, 'domain': (7.303579114878996,7.496420885121005), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.4, y0=-4.25, m=7.4: y0+m*(x-x0), 'deriv': lambda x, m=7.4: np.zeros_like(x,dtype=float)+m, 'domain': (7.303579114878996,7.496420885121005), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.4, y0=-0.25, m=7.4: y0+m*(x-x0), 'deriv': lambda x, m=7.4: np.zeros_like(x,dtype=float)+m, 'domain': (7.303579114878996,7.496420885121005), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.4, y0=3.75, m=7.4: y0+m*(x-x0), 'deriv': lambda x, m=7.4: np.zeros_like(x,dtype=float)+m, 'domain': (7.303579114878996,7.496420885121005), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.4, y0=7.75, m=7.4: y0+m*(x-x0), 'deriv': lambda x, m=7.4: np.zeros_like(x,dtype=float)+m, 'domain': (7.303579114878996,7.496420885121005), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d1_02_v8_fa_field_read.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.48, y0=-7.8, m=1.3199999999999994: y0+m*(x-x0), 'deriv': lambda x, m=1.3199999999999994: np.zeros_like(x,dtype=float)+m, 'domain': (-7.938931842876552,-7.021068157123449), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.48, y0=-3.8, m=-2.6800000000000006: y0+m*(x-x0), 'deriv': lambda x, m=-2.6800000000000006: np.zeros_like(x,dtype=float)+m, 'domain': (-7.745688794991663,-7.214311205008338), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.48, y0=0.20000000000000007, m=-6.680000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-6.680000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-7.592518654006676,-7.367481345993325), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.48, y0=4.2, m=-10.68: y0+m*(x-x0), 'deriv': lambda x, m=-10.68: np.zeros_like(x,dtype=float)+m, 'domain': (-7.550851145809187,-7.409148854190814), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.48, y0=8.2, m=-14.68: y0+m*(x-x0), 'deriv': lambda x, m=-14.68: np.zeros_like(x,dtype=float)+m, 'domain': (-7.531651416329513,-7.428348583670488), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.48, y0=-7.8, m=5.32: y0+m*(x-x0), 'deriv': lambda x, m=5.32: np.zeros_like(x,dtype=float)+m, 'domain': (-3.6203983474628387,-3.3396016525371612), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.48, y0=-3.8, m=1.3199999999999998: y0+m*(x-x0), 'deriv': lambda x, m=1.3199999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (-3.9389318428765514,-3.0210681571234486), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.48, y0=0.20000000000000007, m=-2.68: y0+m*(x-x0), 'deriv': lambda x, m=-2.68: np.zeros_like(x,dtype=float)+m, 'domain': (-3.745688794991663,-3.214311205008337), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.48, y0=4.2, m=-6.68: y0+m*(x-x0), 'deriv': lambda x, m=-6.68: np.zeros_like(x,dtype=float)+m, 'domain': (-3.5925186540066756,-3.3674813459933244), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.48, y0=8.2, m=-10.68: y0+m*(x-x0), 'deriv': lambda x, m=-10.68: np.zeros_like(x,dtype=float)+m, 'domain': (-3.5508511458091867,-3.4091488541908133), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.52, y0=-7.8, m=9.32: y0+m*(x-x0), 'deriv': lambda x, m=9.32: np.zeros_like(x,dtype=float)+m, 'domain': (0.43892031341276555,0.6010796865872344), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.52, y0=-3.8, m=5.32: y0+m*(x-x0), 'deriv': lambda x, m=5.32: np.zeros_like(x,dtype=float)+m, 'domain': (0.37960165253716144,0.6603983474628385), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.52, y0=0.20000000000000007, m=1.3199999999999998: y0+m*(x-x0), 'deriv': lambda x, m=1.3199999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (0.06106815712344854,0.9789318428765514), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.52, y0=4.2, m=-2.68: y0+m*(x-x0), 'deriv': lambda x, m=-2.68: np.zeros_like(x,dtype=float)+m, 'domain': (0.25431120500833687,0.7856887949916631), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.52, y0=8.2, m=-6.68: y0+m*(x-x0), 'deriv': lambda x, m=-6.68: np.zeros_like(x,dtype=float)+m, 'domain': (0.4074813459933246,0.6325186540066755), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.52, y0=-7.8, m=13.32: y0+m*(x-x0), 'deriv': lambda x, m=13.32: np.zeros_like(x,dtype=float)+m, 'domain': (4.463103060810661,4.576896939189338), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.52, y0=-3.8, m=9.32: y0+m*(x-x0), 'deriv': lambda x, m=9.32: np.zeros_like(x,dtype=float)+m, 'domain': (4.4389203134127655,4.601079686587234), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.52, y0=0.20000000000000007, m=5.319999999999999: y0+m*(x-x0), 'deriv': lambda x, m=5.319999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (4.379601652537161,4.660398347462838), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.52, y0=4.2, m=1.3199999999999994: y0+m*(x-x0), 'deriv': lambda x, m=1.3199999999999994: np.zeros_like(x,dtype=float)+m, 'domain': (4.061068157123448,4.978931842876551), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.52, y0=8.2, m=-2.6799999999999997: y0+m*(x-x0), 'deriv': lambda x, m=-2.6799999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (4.254311205008336,4.785688794991663), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.52, y0=-7.8, m=17.32: y0+m*(x-x0), 'deriv': lambda x, m=17.32: np.zeros_like(x,dtype=float)+m, 'domain': (8.476193047500931,8.563806952499068), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.52, y0=-3.8, m=13.32: y0+m*(x-x0), 'deriv': lambda x, m=13.32: np.zeros_like(x,dtype=float)+m, 'domain': (8.463103060810662,8.576896939189337), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.52, y0=0.20000000000000007, m=9.32: y0+m*(x-x0), 'deriv': lambda x, m=9.32: np.zeros_like(x,dtype=float)+m, 'domain': (8.438920313412765,8.601079686587234), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.52, y0=4.2, m=5.319999999999999: y0+m*(x-x0), 'deriv': lambda x, m=5.319999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (8.379601652537161,8.660398347462838), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.52, y0=8.2, m=1.3200000000000003: y0+m*(x-x0), 'deriv': lambda x, m=1.3200000000000003: np.zeros_like(x,dtype=float)+m, 'domain': (8.061068157123449,8.97893184287655), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d2_02_v1_fa_field_apply.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.35, y0=-7.73, m=8.35: y0+m*(x-x0), 'deriv': lambda x, m=8.35: np.zeros_like(x,dtype=float)+m, 'domain': (-8.42491378621444,-8.27508621378556), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.35, y0=-3.73, m=8.35: y0+m*(x-x0), 'deriv': lambda x, m=8.35: np.zeros_like(x,dtype=float)+m, 'domain': (-8.42491378621444,-8.27508621378556), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.35, y0=0.27, m=8.35: y0+m*(x-x0), 'deriv': lambda x, m=8.35: np.zeros_like(x,dtype=float)+m, 'domain': (-8.42491378621444,-8.27508621378556), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.35, y0=4.27, m=8.35: y0+m*(x-x0), 'deriv': lambda x, m=8.35: np.zeros_like(x,dtype=float)+m, 'domain': (-8.42491378621444,-8.27508621378556), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.35, y0=8.27, m=8.35: y0+m*(x-x0), 'deriv': lambda x, m=8.35: np.zeros_like(x,dtype=float)+m, 'domain': (-8.42491378621444,-8.27508621378556), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.35, y0=-7.73, m=4.35: y0+m*(x-x0), 'deriv': lambda x, m=4.35: np.zeros_like(x,dtype=float)+m, 'domain': (-4.491146018432197,-4.2088539815678025), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.35, y0=-3.73, m=4.35: y0+m*(x-x0), 'deriv': lambda x, m=4.35: np.zeros_like(x,dtype=float)+m, 'domain': (-4.491146018432197,-4.2088539815678025), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.35, y0=0.27, m=4.35: y0+m*(x-x0), 'deriv': lambda x, m=4.35: np.zeros_like(x,dtype=float)+m, 'domain': (-4.491146018432197,-4.2088539815678025), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.35, y0=4.27, m=4.35: y0+m*(x-x0), 'deriv': lambda x, m=4.35: np.zeros_like(x,dtype=float)+m, 'domain': (-4.491146018432197,-4.2088539815678025), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.35, y0=8.27, m=4.35: y0+m*(x-x0), 'deriv': lambda x, m=4.35: np.zeros_like(x,dtype=float)+m, 'domain': (-4.491146018432197,-4.2088539815678025), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.35, y0=-7.73, m=0.35: y0+m*(x-x0), 'deriv': lambda x, m=0.35: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9446307645105909,0.24463076451059096), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.35, y0=-3.73, m=0.35: y0+m*(x-x0), 'deriv': lambda x, m=0.35: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9446307645105909,0.24463076451059096), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.35, y0=0.27, m=0.35: y0+m*(x-x0), 'deriv': lambda x, m=0.35: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9446307645105909,0.24463076451059096), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.35, y0=4.27, m=0.35: y0+m*(x-x0), 'deriv': lambda x, m=0.35: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9446307645105909,0.24463076451059096), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.35, y0=8.27, m=0.35: y0+m*(x-x0), 'deriv': lambda x, m=0.35: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9446307645105909,0.24463076451059096), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.65, y0=-7.73, m=-3.65: y0+m*(x-x0), 'deriv': lambda x, m=-3.65: np.zeros_like(x,dtype=float)+m, 'domain': (3.483531857340482,3.816468142659518), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.65, y0=-3.73, m=-3.65: y0+m*(x-x0), 'deriv': lambda x, m=-3.65: np.zeros_like(x,dtype=float)+m, 'domain': (3.483531857340482,3.816468142659518), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.65, y0=0.27, m=-3.65: y0+m*(x-x0), 'deriv': lambda x, m=-3.65: np.zeros_like(x,dtype=float)+m, 'domain': (3.483531857340482,3.816468142659518), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.65, y0=4.27, m=-3.65: y0+m*(x-x0), 'deriv': lambda x, m=-3.65: np.zeros_like(x,dtype=float)+m, 'domain': (3.483531857340482,3.816468142659518), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.65, y0=8.27, m=-3.65: y0+m*(x-x0), 'deriv': lambda x, m=-3.65: np.zeros_like(x,dtype=float)+m, 'domain': (3.483531857340482,3.816468142659518), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.65, y0=-7.73, m=-7.65: y0+m*(x-x0), 'deriv': lambda x, m=-7.65: np.zeros_like(x,dtype=float)+m, 'domain': (7.568341768954579,7.731658231045421), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.65, y0=-3.73, m=-7.65: y0+m*(x-x0), 'deriv': lambda x, m=-7.65: np.zeros_like(x,dtype=float)+m, 'domain': (7.568341768954579,7.731658231045421), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.65, y0=0.27, m=-7.65: y0+m*(x-x0), 'deriv': lambda x, m=-7.65: np.zeros_like(x,dtype=float)+m, 'domain': (7.568341768954579,7.731658231045421), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.65, y0=4.27, m=-7.65: y0+m*(x-x0), 'deriv': lambda x, m=-7.65: np.zeros_like(x,dtype=float)+m, 'domain': (7.568341768954579,7.731658231045421), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.65, y0=8.27, m=-7.65: y0+m*(x-x0), 'deriv': lambda x, m=-7.65: np.zeros_like(x,dtype=float)+m, 'domain': (7.568341768954579,7.731658231045421), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d2_02_v2_fa_field_apply.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.02, y0=-7.8, m=-8.02: y0+m*(x-x0), 'deriv': lambda x, m=-8.02: np.zeros_like(x,dtype=float)+m, 'domain': (-8.086814287584463,-7.953185712415537), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.02, y0=-3.8, m=-8.02: y0+m*(x-x0), 'deriv': lambda x, m=-8.02: np.zeros_like(x,dtype=float)+m, 'domain': (-8.086814287584463,-7.953185712415537), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.02, y0=0.20000000000000007, m=-8.02: y0+m*(x-x0), 'deriv': lambda x, m=-8.02: np.zeros_like(x,dtype=float)+m, 'domain': (-8.086814287584463,-7.953185712415537), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.02, y0=4.2, m=-8.02: y0+m*(x-x0), 'deriv': lambda x, m=-8.02: np.zeros_like(x,dtype=float)+m, 'domain': (-8.086814287584463,-7.953185712415537), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.02, y0=8.2, m=-8.02: y0+m*(x-x0), 'deriv': lambda x, m=-8.02: np.zeros_like(x,dtype=float)+m, 'domain': (-8.086814287584463,-7.953185712415537), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.02, y0=-7.8, m=-4.02: y0+m*(x-x0), 'deriv': lambda x, m=-4.02: np.zeros_like(x,dtype=float)+m, 'domain': (-4.150355709099567,-3.8896442909004327), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.02, y0=-3.8, m=-4.02: y0+m*(x-x0), 'deriv': lambda x, m=-4.02: np.zeros_like(x,dtype=float)+m, 'domain': (-4.150355709099567,-3.8896442909004327), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.02, y0=0.20000000000000007, m=-4.02: y0+m*(x-x0), 'deriv': lambda x, m=-4.02: np.zeros_like(x,dtype=float)+m, 'domain': (-4.150355709099567,-3.8896442909004327), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.02, y0=4.2, m=-4.02: y0+m*(x-x0), 'deriv': lambda x, m=-4.02: np.zeros_like(x,dtype=float)+m, 'domain': (-4.150355709099567,-3.8896442909004327), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.02, y0=8.2, m=-4.02: y0+m*(x-x0), 'deriv': lambda x, m=-4.02: np.zeros_like(x,dtype=float)+m, 'domain': (-4.150355709099567,-3.8896442909004327), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.020000000000000018, y0=-7.8, m=-0.020000000000000018: y0+m*(x-x0), 'deriv': lambda x, m=-0.020000000000000018: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5598920323892039,0.5198920323892039), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.020000000000000018, y0=-3.8, m=-0.020000000000000018: y0+m*(x-x0), 'deriv': lambda x, m=-0.020000000000000018: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5598920323892039,0.5198920323892039), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.020000000000000018, y0=0.20000000000000007, m=-0.020000000000000018: y0+m*(x-x0), 'deriv': lambda x, m=-0.020000000000000018: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5598920323892039,0.5198920323892039), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.020000000000000018, y0=4.2, m=-0.020000000000000018: y0+m*(x-x0), 'deriv': lambda x, m=-0.020000000000000018: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5598920323892039,0.5198920323892039), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.020000000000000018, y0=8.2, m=-0.020000000000000018: y0+m*(x-x0), 'deriv': lambda x, m=-0.020000000000000018: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5598920323892039,0.5198920323892039), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.98, y0=-7.8, m=3.98: y0+m*(x-x0), 'deriv': lambda x, m=3.98: np.zeros_like(x,dtype=float)+m, 'domain': (3.848411614514432,4.111588385485568), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.98, y0=-3.8, m=3.98: y0+m*(x-x0), 'deriv': lambda x, m=3.98: np.zeros_like(x,dtype=float)+m, 'domain': (3.848411614514432,4.111588385485568), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.98, y0=0.20000000000000007, m=3.98: y0+m*(x-x0), 'deriv': lambda x, m=3.98: np.zeros_like(x,dtype=float)+m, 'domain': (3.848411614514432,4.111588385485568), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.98, y0=4.2, m=3.98: y0+m*(x-x0), 'deriv': lambda x, m=3.98: np.zeros_like(x,dtype=float)+m, 'domain': (3.848411614514432,4.111588385485568), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.98, y0=8.2, m=3.98: y0+m*(x-x0), 'deriv': lambda x, m=3.98: np.zeros_like(x,dtype=float)+m, 'domain': (3.848411614514432,4.111588385485568), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.98, y0=-7.8, m=7.98: y0+m*(x-x0), 'deriv': lambda x, m=7.98: np.zeros_like(x,dtype=float)+m, 'domain': (7.912855968893054,8.047144031106948), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.98, y0=-3.8, m=7.98: y0+m*(x-x0), 'deriv': lambda x, m=7.98: np.zeros_like(x,dtype=float)+m, 'domain': (7.912855968893054,8.047144031106948), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.98, y0=0.20000000000000007, m=7.98: y0+m*(x-x0), 'deriv': lambda x, m=7.98: np.zeros_like(x,dtype=float)+m, 'domain': (7.912855968893054,8.047144031106948), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.98, y0=4.2, m=7.98: y0+m*(x-x0), 'deriv': lambda x, m=7.98: np.zeros_like(x,dtype=float)+m, 'domain': (7.912855968893054,8.047144031106948), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.98, y0=8.2, m=7.98: y0+m*(x-x0), 'deriv': lambda x, m=7.98: np.zeros_like(x,dtype=float)+m, 'domain': (7.912855968893054,8.047144031106948), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d2_02_v3_fa_field_apply.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.73, y0=-7.99, m=-0.7400000000000002: y0+m*(x-x0), 'deriv': lambda x, m=-0.7400000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (-8.180151463553736,-7.279848536446264), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.73, y0=-3.99, m=-4.74: y0+m*(x-x0), 'deriv': lambda x, m=-4.74: np.zeros_like(x,dtype=float)+m, 'domain': (-7.845598896519434,-7.614401103480567), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.73, y0=0.010000000000000009, m=-8.74: y0+m*(x-x0), 'deriv': lambda x, m=-8.74: np.zeros_like(x,dtype=float)+m, 'domain': (-7.793657904433476,-7.666342095566525), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.73, y0=4.01, m=-12.74: y0+m*(x-x0), 'deriv': lambda x, m=-12.74: np.zeros_like(x,dtype=float)+m, 'domain': (-7.773821256778253,-7.686178743221748), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.73, y0=8.01, m=-16.740000000000002: y0+m*(x-x0), 'deriv': lambda x, m=-16.740000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (-7.763393278285025,-7.696606721714976), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.73, y0=-7.99, m=3.26: y0+m*(x-x0), 'deriv': lambda x, m=3.26: np.zeros_like(x,dtype=float)+m, 'domain': (-3.8942264047473403,-3.5657735952526597), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.73, y0=-3.99, m=-0.7399999999999998: y0+m*(x-x0), 'deriv': lambda x, m=-0.7399999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (-4.180151463553736,-3.2798485364462633), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.73, y0=0.010000000000000009, m=-4.74: y0+m*(x-x0), 'deriv': lambda x, m=-4.74: np.zeros_like(x,dtype=float)+m, 'domain': (-3.845598896519433,-3.6144011034805668), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.73, y0=4.01, m=-8.74: y0+m*(x-x0), 'deriv': lambda x, m=-8.74: np.zeros_like(x,dtype=float)+m, 'domain': (-3.7936579044334757,-3.6663420955665242), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.73, y0=8.01, m=-12.74: y0+m*(x-x0), 'deriv': lambda x, m=-12.74: np.zeros_like(x,dtype=float)+m, 'domain': (-3.7738212567782528,-3.686178743221747), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.27, y0=-7.99, m=7.26: y0+m*(x-x0), 'deriv': lambda x, m=7.26: np.zeros_like(x,dtype=float)+m, 'domain': (0.19358648912295143,0.34641351087704864), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.27, y0=-3.99, m=3.26: y0+m*(x-x0), 'deriv': lambda x, m=3.26: np.zeros_like(x,dtype=float)+m, 'domain': (0.10577359525265972,0.4342264047473403), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.27, y0=0.010000000000000009, m=-0.74: y0+m*(x-x0), 'deriv': lambda x, m=-0.74: np.zeros_like(x,dtype=float)+m, 'domain': (-0.1801514635537365,0.7201514635537365), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.27, y0=4.01, m=-4.74: y0+m*(x-x0), 'deriv': lambda x, m=-4.74: np.zeros_like(x,dtype=float)+m, 'domain': (0.15440110348056693,0.38559889651943313), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.27, y0=8.01, m=-8.74: y0+m*(x-x0), 'deriv': lambda x, m=-8.74: np.zeros_like(x,dtype=float)+m, 'domain': (0.2063420955665244,0.33365790443347565), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.27, y0=-7.99, m=11.26: y0+m*(x-x0), 'deriv': lambda x, m=11.26: np.zeros_like(x,dtype=float)+m, 'domain': (4.22046140670043,4.319538593299569), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.27, y0=-3.99, m=7.26: y0+m*(x-x0), 'deriv': lambda x, m=7.26: np.zeros_like(x,dtype=float)+m, 'domain': (4.193586489122951,4.346413510877048), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.27, y0=0.010000000000000009, m=3.26: y0+m*(x-x0), 'deriv': lambda x, m=3.26: np.zeros_like(x,dtype=float)+m, 'domain': (4.105773595252659,4.43422640474734), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.27, y0=4.01, m=-0.7400000000000002: y0+m*(x-x0), 'deriv': lambda x, m=-0.7400000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (3.819848536446263,4.720151463553736), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.27, y0=8.01, m=-4.74: y0+m*(x-x0), 'deriv': lambda x, m=-4.74: np.zeros_like(x,dtype=float)+m, 'domain': (4.154401103480566,4.385598896519433), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.27, y0=-7.99, m=15.259999999999998: y0+m*(x-x0), 'deriv': lambda x, m=15.259999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (8.233381293657276,8.306618706342723), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.27, y0=-3.99, m=11.26: y0+m*(x-x0), 'deriv': lambda x, m=11.26: np.zeros_like(x,dtype=float)+m, 'domain': (8.22046140670043,8.31953859329957), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.27, y0=0.010000000000000009, m=7.26: y0+m*(x-x0), 'deriv': lambda x, m=7.26: np.zeros_like(x,dtype=float)+m, 'domain': (8.193586489122952,8.346413510877047), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.27, y0=4.01, m=3.26: y0+m*(x-x0), 'deriv': lambda x, m=3.26: np.zeros_like(x,dtype=float)+m, 'domain': (8.105773595252659,8.43422640474734), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.27, y0=8.01, m=-0.7400000000000002: y0+m*(x-x0), 'deriv': lambda x, m=-0.7400000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (7.819848536446263,8.720151463553735), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d2_02_v4_fa_field_apply.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.9399999999999995, y0=-7.5, m=-7.9399999999999995: y0+m*(x-x0), 'deriv': lambda x, m=-7.9399999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (-8.019972763915147,-7.860027236084852), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.9399999999999995, y0=-3.5, m=-7.9399999999999995: y0+m*(x-x0), 'deriv': lambda x, m=-7.9399999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (-8.019972763915147,-7.860027236084852), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.9399999999999995, y0=0.5000000000000001, m=-7.9399999999999995: y0+m*(x-x0), 'deriv': lambda x, m=-7.9399999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (-8.019972763915147,-7.860027236084852), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.9399999999999995, y0=4.5, m=-7.9399999999999995: y0+m*(x-x0), 'deriv': lambda x, m=-7.9399999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (-8.019972763915147,-7.860027236084852), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.9399999999999995, y0=8.5, m=-7.9399999999999995: y0+m*(x-x0), 'deriv': lambda x, m=-7.9399999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (-8.019972763915147,-7.860027236084852), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.94, y0=-7.5, m=-3.94: y0+m*(x-x0), 'deriv': lambda x, m=-3.94: np.zeros_like(x,dtype=float)+m, 'domain': (-4.097444551976734,-3.7825554480232664), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.94, y0=-3.5, m=-3.94: y0+m*(x-x0), 'deriv': lambda x, m=-3.94: np.zeros_like(x,dtype=float)+m, 'domain': (-4.097444551976734,-3.7825554480232664), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.94, y0=0.5000000000000001, m=-3.94: y0+m*(x-x0), 'deriv': lambda x, m=-3.94: np.zeros_like(x,dtype=float)+m, 'domain': (-4.097444551976734,-3.7825554480232664), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.94, y0=4.5, m=-3.94: y0+m*(x-x0), 'deriv': lambda x, m=-3.94: np.zeros_like(x,dtype=float)+m, 'domain': (-4.097444551976734,-3.7825554480232664), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.94, y0=8.5, m=-3.94: y0+m*(x-x0), 'deriv': lambda x, m=-3.94: np.zeros_like(x,dtype=float)+m, 'domain': (-4.097444551976734,-3.7825554480232664), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.06000000000000005, y0=-7.5, m=0.06000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=0.06000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5788511010980983,0.6988511010980984), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.06000000000000005, y0=-3.5, m=0.06000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=0.06000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5788511010980983,0.6988511010980984), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.06000000000000005, y0=0.5000000000000001, m=0.06000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=0.06000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5788511010980983,0.6988511010980984), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.06000000000000005, y0=4.5, m=0.06000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=0.06000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5788511010980983,0.6988511010980984), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.06000000000000005, y0=8.5, m=0.06000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=0.06000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5788511010980983,0.6988511010980984), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.0600000000000005, y0=-7.5, m=4.0600000000000005: y0+m*(x-x0), 'deriv': lambda x, m=4.0600000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (3.906939001183472,4.213060998816529), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.0600000000000005, y0=-3.5, m=4.0600000000000005: y0+m*(x-x0), 'deriv': lambda x, m=4.0600000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (3.906939001183472,4.213060998816529), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.0600000000000005, y0=0.5000000000000001, m=4.0600000000000005: y0+m*(x-x0), 'deriv': lambda x, m=4.0600000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (3.906939001183472,4.213060998816529), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.0600000000000005, y0=4.5, m=4.0600000000000005: y0+m*(x-x0), 'deriv': lambda x, m=4.0600000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (3.906939001183472,4.213060998816529), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.0600000000000005, y0=8.5, m=4.0600000000000005: y0+m*(x-x0), 'deriv': lambda x, m=4.0600000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (3.906939001183472,4.213060998816529), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.06, y0=-7.5, m=8.06: y0+m*(x-x0), 'deriv': lambda x, m=8.06: np.zeros_like(x,dtype=float)+m, 'domain': (7.981199713000214,8.138800286999787), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.06, y0=-3.5, m=8.06: y0+m*(x-x0), 'deriv': lambda x, m=8.06: np.zeros_like(x,dtype=float)+m, 'domain': (7.981199713000214,8.138800286999787), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.06, y0=0.5000000000000001, m=8.06: y0+m*(x-x0), 'deriv': lambda x, m=8.06: np.zeros_like(x,dtype=float)+m, 'domain': (7.981199713000214,8.138800286999787), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.06, y0=4.5, m=8.06: y0+m*(x-x0), 'deriv': lambda x, m=8.06: np.zeros_like(x,dtype=float)+m, 'domain': (7.981199713000214,8.138800286999787), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.06, y0=8.5, m=8.06: y0+m*(x-x0), 'deriv': lambda x, m=8.06: np.zeros_like(x,dtype=float)+m, 'domain': (7.981199713000214,8.138800286999787), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d2_02_v5_fa_field_apply.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.36, y0=-7.57, m=7.57: y0+m*(x-x0), 'deriv': lambda x, m=7.57: np.zeros_like(x,dtype=float)+m, 'domain': (-7.468699004066624,-7.251300995933376), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.36, y0=-3.57, m=3.57: y0+m*(x-x0), 'deriv': lambda x, m=3.57: np.zeros_like(x,dtype=float)+m, 'domain': (-7.583875876987441,-7.13612412301256), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.36, y0=0.43000000000000005, m=-0.43000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=-0.43000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-8.12249542569143,-6.597504574308571), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.36, y0=4.43, m=-4.43: y0+m*(x-x0), 'deriv': lambda x, m=-4.43: np.zeros_like(x,dtype=float)+m, 'domain': (-7.542760428067097,-7.177239571932904), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.36, y0=8.43, m=-8.43: y0+m*(x-x0), 'deriv': lambda x, m=-8.43: np.zeros_like(x,dtype=float)+m, 'domain': (-7.4577723825200755,-7.262227617479925), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.3600000000000003, y0=-7.57, m=7.57: y0+m*(x-x0), 'deriv': lambda x, m=7.57: np.zeros_like(x,dtype=float)+m, 'domain': (-3.468699004066624,-3.2513009959333767), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.3600000000000003, y0=-3.57, m=3.57: y0+m*(x-x0), 'deriv': lambda x, m=3.57: np.zeros_like(x,dtype=float)+m, 'domain': (-3.583875876987441,-3.13612412301256), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.3600000000000003, y0=0.43000000000000005, m=-0.43000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=-0.43000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-4.1224954256914295,-2.5975045743085707), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.3600000000000003, y0=4.43, m=-4.43: y0+m*(x-x0), 'deriv': lambda x, m=-4.43: np.zeros_like(x,dtype=float)+m, 'domain': (-3.5427604280670963,-3.1772395719329043), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.3600000000000003, y0=8.43, m=-8.43: y0+m*(x-x0), 'deriv': lambda x, m=-8.43: np.zeros_like(x,dtype=float)+m, 'domain': (-3.457772382520076,-3.2622276174799247), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6399999999999999, y0=-7.57, m=7.57: y0+m*(x-x0), 'deriv': lambda x, m=7.57: np.zeros_like(x,dtype=float)+m, 'domain': (0.5313009959333762,0.7486990040666236), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6399999999999999, y0=-3.57, m=3.57: y0+m*(x-x0), 'deriv': lambda x, m=3.57: np.zeros_like(x,dtype=float)+m, 'domain': (0.4161241230125594,0.8638758769874404), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6399999999999999, y0=0.43000000000000005, m=-0.43000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=-0.43000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-0.12249542569142968,1.4024954256914295), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6399999999999999, y0=4.43, m=-4.43: y0+m*(x-x0), 'deriv': lambda x, m=-4.43: np.zeros_like(x,dtype=float)+m, 'domain': (0.45723957193290377,0.822760428067096), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6399999999999999, y0=8.43, m=-8.43: y0+m*(x-x0), 'deriv': lambda x, m=-8.43: np.zeros_like(x,dtype=float)+m, 'domain': (0.5422276174799244,0.7377723825200754), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.64, y0=-7.57, m=7.57: y0+m*(x-x0), 'deriv': lambda x, m=7.57: np.zeros_like(x,dtype=float)+m, 'domain': (4.531300995933376,4.748699004066624), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.64, y0=-3.57, m=3.57: y0+m*(x-x0), 'deriv': lambda x, m=3.57: np.zeros_like(x,dtype=float)+m, 'domain': (4.416124123012559,4.86387587698744), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.64, y0=0.43000000000000005, m=-0.43000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=-0.43000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (3.87750457430857,5.402495425691429), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.64, y0=4.43, m=-4.43: y0+m*(x-x0), 'deriv': lambda x, m=-4.43: np.zeros_like(x,dtype=float)+m, 'domain': (4.457239571932903,4.822760428067096), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.64, y0=8.43, m=-8.43: y0+m*(x-x0), 'deriv': lambda x, m=-8.43: np.zeros_like(x,dtype=float)+m, 'domain': (4.5422276174799245,4.737772382520075), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.64, y0=-7.57, m=7.57: y0+m*(x-x0), 'deriv': lambda x, m=7.57: np.zeros_like(x,dtype=float)+m, 'domain': (8.531300995933377,8.748699004066625), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.64, y0=-3.57, m=3.57: y0+m*(x-x0), 'deriv': lambda x, m=3.57: np.zeros_like(x,dtype=float)+m, 'domain': (8.41612412301256,8.863875876987441), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.64, y0=0.43000000000000005, m=-0.43000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=-0.43000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (7.8775045743085705,9.40249542569143), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.64, y0=4.43, m=-4.43: y0+m*(x-x0), 'deriv': lambda x, m=-4.43: np.zeros_like(x,dtype=float)+m, 'domain': (8.457239571932904,8.822760428067097), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.64, y0=8.43, m=-8.43: y0+m*(x-x0), 'deriv': lambda x, m=-8.43: np.zeros_like(x,dtype=float)+m, 'domain': (8.542227617479925,8.737772382520076), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d2_02_v6_fa_field_apply.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.02, y0=-8.4, m=8.02: y0+m*(x-x0), 'deriv': lambda x, m=8.02: np.zeros_like(x,dtype=float)+m, 'domain': (-8.122696034620564,-7.917303965379436), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.02, y0=-4.4, m=8.02: y0+m*(x-x0), 'deriv': lambda x, m=8.02: np.zeros_like(x,dtype=float)+m, 'domain': (-8.122696034620564,-7.917303965379436), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.02, y0=-0.39999999999999997, m=8.02: y0+m*(x-x0), 'deriv': lambda x, m=8.02: np.zeros_like(x,dtype=float)+m, 'domain': (-8.122696034620564,-7.917303965379436), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.02, y0=3.6, m=8.02: y0+m*(x-x0), 'deriv': lambda x, m=8.02: np.zeros_like(x,dtype=float)+m, 'domain': (-8.122696034620564,-7.917303965379436), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.02, y0=7.6, m=8.02: y0+m*(x-x0), 'deriv': lambda x, m=8.02: np.zeros_like(x,dtype=float)+m, 'domain': (-8.122696034620564,-7.917303965379436), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.02, y0=-8.4, m=4.02: y0+m*(x-x0), 'deriv': lambda x, m=4.02: np.zeros_like(x,dtype=float)+m, 'domain': (-4.22036155287526,-3.8196384471247393), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.02, y0=-4.4, m=4.02: y0+m*(x-x0), 'deriv': lambda x, m=4.02: np.zeros_like(x,dtype=float)+m, 'domain': (-4.22036155287526,-3.8196384471247393), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.02, y0=-0.39999999999999997, m=4.02: y0+m*(x-x0), 'deriv': lambda x, m=4.02: np.zeros_like(x,dtype=float)+m, 'domain': (-4.22036155287526,-3.8196384471247393), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.02, y0=3.6, m=4.02: y0+m*(x-x0), 'deriv': lambda x, m=4.02: np.zeros_like(x,dtype=float)+m, 'domain': (-4.22036155287526,-3.8196384471247393), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.02, y0=7.6, m=4.02: y0+m*(x-x0), 'deriv': lambda x, m=4.02: np.zeros_like(x,dtype=float)+m, 'domain': (-4.22036155287526,-3.8196384471247393), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.020000000000000018, y0=-8.4, m=0.020000000000000018: y0+m*(x-x0), 'deriv': lambda x, m=0.020000000000000018: np.zeros_like(x,dtype=float)+m, 'domain': (-0.8498340497834059,0.8098340497834059), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.020000000000000018, y0=-4.4, m=0.020000000000000018: y0+m*(x-x0), 'deriv': lambda x, m=0.020000000000000018: np.zeros_like(x,dtype=float)+m, 'domain': (-0.8498340497834059,0.8098340497834059), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.020000000000000018, y0=-0.39999999999999997, m=0.020000000000000018: y0+m*(x-x0), 'deriv': lambda x, m=0.020000000000000018: np.zeros_like(x,dtype=float)+m, 'domain': (-0.8498340497834059,0.8098340497834059), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.020000000000000018, y0=3.6, m=0.020000000000000018: y0+m*(x-x0), 'deriv': lambda x, m=0.020000000000000018: np.zeros_like(x,dtype=float)+m, 'domain': (-0.8498340497834059,0.8098340497834059), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.020000000000000018, y0=7.6, m=0.020000000000000018: y0+m*(x-x0), 'deriv': lambda x, m=0.020000000000000018: np.zeros_like(x,dtype=float)+m, 'domain': (-0.8498340497834059,0.8098340497834059), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.98, y0=-8.4, m=-3.98: y0+m*(x-x0), 'deriv': lambda x, m=-3.98: np.zeros_like(x,dtype=float)+m, 'domain': (3.777743777864775,4.182256222135225), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.98, y0=-4.4, m=-3.98: y0+m*(x-x0), 'deriv': lambda x, m=-3.98: np.zeros_like(x,dtype=float)+m, 'domain': (3.777743777864775,4.182256222135225), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.98, y0=-0.39999999999999997, m=-3.98: y0+m*(x-x0), 'deriv': lambda x, m=-3.98: np.zeros_like(x,dtype=float)+m, 'domain': (3.777743777864775,4.182256222135225), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.98, y0=3.6, m=-3.98: y0+m*(x-x0), 'deriv': lambda x, m=-3.98: np.zeros_like(x,dtype=float)+m, 'domain': (3.777743777864775,4.182256222135225), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.98, y0=7.6, m=-3.98: y0+m*(x-x0), 'deriv': lambda x, m=-3.98: np.zeros_like(x,dtype=float)+m, 'domain': (3.777743777864775,4.182256222135225), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.98, y0=-8.4, m=-7.98: y0+m*(x-x0), 'deriv': lambda x, m=-7.98: np.zeros_like(x,dtype=float)+m, 'domain': (7.876797137372657,8.083202862627344), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.98, y0=-4.4, m=-7.98: y0+m*(x-x0), 'deriv': lambda x, m=-7.98: np.zeros_like(x,dtype=float)+m, 'domain': (7.876797137372657,8.083202862627344), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.98, y0=-0.39999999999999997, m=-7.98: y0+m*(x-x0), 'deriv': lambda x, m=-7.98: np.zeros_like(x,dtype=float)+m, 'domain': (7.876797137372657,8.083202862627344), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.98, y0=3.6, m=-7.98: y0+m*(x-x0), 'deriv': lambda x, m=-7.98: np.zeros_like(x,dtype=float)+m, 'domain': (7.876797137372657,8.083202862627344), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.98, y0=7.6, m=-7.98: y0+m*(x-x0), 'deriv': lambda x, m=-7.98: np.zeros_like(x,dtype=float)+m, 'domain': (7.876797137372657,8.083202862627344), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d2_02_v7_fa_field_apply.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.1, y0=-8.52, m=-8.52: y0+m*(x-x0), 'deriv': lambda x, m=-8.52: np.zeros_like(x,dtype=float)+m, 'domain': (-8.161782472969035,-8.038217527030964), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.1, y0=-4.52, m=-4.52: y0+m*(x-x0), 'deriv': lambda x, m=-4.52: np.zeros_like(x,dtype=float)+m, 'domain': (-8.21448820094617,-7.985511799053829), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.1, y0=-0.52, m=-0.52: y0+m*(x-x0), 'deriv': lambda x, m=-0.52: np.zeros_like(x,dtype=float)+m, 'domain': (-8.570224904654335,-7.629775095345664), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.1, y0=3.48, m=3.48: y0+m*(x-x0), 'deriv': lambda x, m=3.48: np.zeros_like(x,dtype=float)+m, 'domain': (-8.246375333800236,-7.953624666199763), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.1, y0=7.48, m=7.48: y0+m*(x-x0), 'deriv': lambda x, m=7.48: np.zeros_like(x,dtype=float)+m, 'domain': (-8.170230778123809,-8.02976922187619), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.1, y0=-8.52, m=-8.52: y0+m*(x-x0), 'deriv': lambda x, m=-8.52: np.zeros_like(x,dtype=float)+m, 'domain': (-4.161782472969034,-4.038217527030965), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.1, y0=-4.52, m=-4.52: y0+m*(x-x0), 'deriv': lambda x, m=-4.52: np.zeros_like(x,dtype=float)+m, 'domain': (-4.21448820094617,-3.985511799053829), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.1, y0=-0.52, m=-0.52: y0+m*(x-x0), 'deriv': lambda x, m=-0.52: np.zeros_like(x,dtype=float)+m, 'domain': (-4.570224904654335,-3.6297750953456642), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.1, y0=3.48, m=3.48: y0+m*(x-x0), 'deriv': lambda x, m=3.48: np.zeros_like(x,dtype=float)+m, 'domain': (-4.246375333800236,-3.9536246661997634), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.1, y0=7.48, m=7.48: y0+m*(x-x0), 'deriv': lambda x, m=7.48: np.zeros_like(x,dtype=float)+m, 'domain': (-4.170230778123809,-4.02976922187619), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.09999999999999998, y0=-8.52, m=-8.52: y0+m*(x-x0), 'deriv': lambda x, m=-8.52: np.zeros_like(x,dtype=float)+m, 'domain': (-0.16178247296903467,-0.038217527030965294), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.09999999999999998, y0=-4.52, m=-4.52: y0+m*(x-x0), 'deriv': lambda x, m=-4.52: np.zeros_like(x,dtype=float)+m, 'domain': (-0.21448820094617044,0.014488200946170468), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.09999999999999998, y0=-0.52, m=-0.52: y0+m*(x-x0), 'deriv': lambda x, m=-0.52: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5702249046543354,0.37022490465433544), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.09999999999999998, y0=3.48, m=3.48: y0+m*(x-x0), 'deriv': lambda x, m=3.48: np.zeros_like(x,dtype=float)+m, 'domain': (-0.2463753338002362,0.04637533380023626), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.09999999999999998, y0=7.48, m=7.48: y0+m*(x-x0), 'deriv': lambda x, m=7.48: np.zeros_like(x,dtype=float)+m, 'domain': (-0.17023077812380913,-0.029769221876190827), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.9, y0=-8.52, m=-8.52: y0+m*(x-x0), 'deriv': lambda x, m=-8.52: np.zeros_like(x,dtype=float)+m, 'domain': (3.8382175270309653,3.9617824729690345), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.9, y0=-4.52, m=-4.52: y0+m*(x-x0), 'deriv': lambda x, m=-4.52: np.zeros_like(x,dtype=float)+m, 'domain': (3.7855117990538294,4.01448820094617), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.9, y0=-0.52, m=-0.52: y0+m*(x-x0), 'deriv': lambda x, m=-0.52: np.zeros_like(x,dtype=float)+m, 'domain': (3.4297750953456645,4.370224904654336), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.9, y0=3.48, m=3.48: y0+m*(x-x0), 'deriv': lambda x, m=3.48: np.zeros_like(x,dtype=float)+m, 'domain': (3.7536246661997636,4.046375333800236), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.9, y0=7.48, m=7.48: y0+m*(x-x0), 'deriv': lambda x, m=7.48: np.zeros_like(x,dtype=float)+m, 'domain': (3.829769221876191,3.970230778123809), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.9, y0=-8.52, m=-8.52: y0+m*(x-x0), 'deriv': lambda x, m=-8.52: np.zeros_like(x,dtype=float)+m, 'domain': (7.838217527030966,7.961782472969035), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.9, y0=-4.52, m=-4.52: y0+m*(x-x0), 'deriv': lambda x, m=-4.52: np.zeros_like(x,dtype=float)+m, 'domain': (7.78551179905383,8.01448820094617), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.9, y0=-0.52, m=-0.52: y0+m*(x-x0), 'deriv': lambda x, m=-0.52: np.zeros_like(x,dtype=float)+m, 'domain': (7.429775095345665,8.370224904654336), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.9, y0=3.48, m=3.48: y0+m*(x-x0), 'deriv': lambda x, m=3.48: np.zeros_like(x,dtype=float)+m, 'domain': (7.753624666199764,8.046375333800237), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.9, y0=7.48, m=7.48: y0+m*(x-x0), 'deriv': lambda x, m=7.48: np.zeros_like(x,dtype=float)+m, 'domain': (7.829769221876191,7.97023077812381), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d2_02_v8_fa_field_apply.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.5, y0=-8.59, m=-16.09: y0+m*(x-x0), 'deriv': lambda x, m=-16.09: np.zeros_like(x,dtype=float)+m, 'domain': (-7.539079351876547,-7.460920648123453), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.5, y0=-4.59, m=-12.09: y0+m*(x-x0), 'deriv': lambda x, m=-12.09: np.zeros_like(x,dtype=float)+m, 'domain': (-7.551931839710112,-7.448068160289888), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.5, y0=-0.59, m=-8.09: y0+m*(x-x0), 'deriv': lambda x, m=-8.09: np.zeros_like(x,dtype=float)+m, 'domain': (-7.577285721526048,-7.422714278473952), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.5, y0=3.41, m=-4.09: y0+m*(x-x0), 'deriv': lambda x, m=-4.09: np.zeros_like(x,dtype=float)+m, 'domain': (-7.649626822419037,-7.350373177580963), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.5, y0=7.41, m=-0.08999999999999986: y0+m*(x-x0), 'deriv': lambda x, m=-0.08999999999999986: np.zeros_like(x,dtype=float)+m, 'domain': (-8.127463896471234,-6.872536103528766), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.5, y0=-8.59, m=-12.09: y0+m*(x-x0), 'deriv': lambda x, m=-12.09: np.zeros_like(x,dtype=float)+m, 'domain': (-3.5519318397101123,-3.4480681602898877), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.5, y0=-4.59, m=-8.09: y0+m*(x-x0), 'deriv': lambda x, m=-8.09: np.zeros_like(x,dtype=float)+m, 'domain': (-3.5772857215260476,-3.4227142784739524), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.5, y0=-0.59, m=-4.09: y0+m*(x-x0), 'deriv': lambda x, m=-4.09: np.zeros_like(x,dtype=float)+m, 'domain': (-3.649626822419037,-3.350373177580963), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.5, y0=3.41, m=-0.08999999999999986: y0+m*(x-x0), 'deriv': lambda x, m=-0.08999999999999986: np.zeros_like(x,dtype=float)+m, 'domain': (-4.127463896471234,-2.872536103528766), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.5, y0=7.41, m=3.91: y0+m*(x-x0), 'deriv': lambda x, m=3.91: np.zeros_like(x,dtype=float)+m, 'domain': (-3.656100874466058,-3.343899125533942), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5, y0=-8.59, m=-8.09: y0+m*(x-x0), 'deriv': lambda x, m=-8.09: np.zeros_like(x,dtype=float)+m, 'domain': (0.4227142784739524,0.5772857215260476), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5, y0=-4.59, m=-4.09: y0+m*(x-x0), 'deriv': lambda x, m=-4.09: np.zeros_like(x,dtype=float)+m, 'domain': (0.350373177580963,0.649626822419037), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5, y0=-0.59, m=-0.08999999999999997: y0+m*(x-x0), 'deriv': lambda x, m=-0.08999999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (-0.12746389647123402,1.127463896471234), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5, y0=3.41, m=3.91: y0+m*(x-x0), 'deriv': lambda x, m=3.91: np.zeros_like(x,dtype=float)+m, 'domain': (0.3438991255339421,0.6561008744660579), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5, y0=7.41, m=7.91: y0+m*(x-x0), 'deriv': lambda x, m=7.91: np.zeros_like(x,dtype=float)+m, 'domain': (0.4209829277704972,0.5790170722295028), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.5, y0=-8.59, m=-4.09: y0+m*(x-x0), 'deriv': lambda x, m=-4.09: np.zeros_like(x,dtype=float)+m, 'domain': (4.350373177580963,4.649626822419037), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.5, y0=-4.59, m=-0.08999999999999986: y0+m*(x-x0), 'deriv': lambda x, m=-0.08999999999999986: np.zeros_like(x,dtype=float)+m, 'domain': (3.872536103528766,5.127463896471234), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.5, y0=-0.59, m=3.91: y0+m*(x-x0), 'deriv': lambda x, m=3.91: np.zeros_like(x,dtype=float)+m, 'domain': (4.343899125533942,4.656100874466058), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.5, y0=3.41, m=7.91: y0+m*(x-x0), 'deriv': lambda x, m=7.91: np.zeros_like(x,dtype=float)+m, 'domain': (4.420982927770497,4.579017072229503), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.5, y0=7.41, m=11.91: y0+m*(x-x0), 'deriv': lambda x, m=11.91: np.zeros_like(x,dtype=float)+m, 'domain': (4.447288749990888,4.552711250009112), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.5, y0=-8.59, m=-0.08999999999999986: y0+m*(x-x0), 'deriv': lambda x, m=-0.08999999999999986: np.zeros_like(x,dtype=float)+m, 'domain': (7.872536103528766,9.127463896471234), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.5, y0=-4.59, m=3.91: y0+m*(x-x0), 'deriv': lambda x, m=3.91: np.zeros_like(x,dtype=float)+m, 'domain': (8.343899125533943,8.656100874466057), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.5, y0=-0.59, m=7.91: y0+m*(x-x0), 'deriv': lambda x, m=7.91: np.zeros_like(x,dtype=float)+m, 'domain': (8.420982927770497,8.579017072229503), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.5, y0=3.41, m=11.91: y0+m*(x-x0), 'deriv': lambda x, m=11.91: np.zeros_like(x,dtype=float)+m, 'domain': (8.44728874999089,8.55271125000911), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.5, y0=7.41, m=15.91: y0+m*(x-x0), 'deriv': lambda x, m=15.91: np.zeros_like(x,dtype=float)+m, 'domain': (8.460480248532265,8.539519751467735), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d3_02_v1_fa_field_reason.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.98, y0=-8.52, m=-16.5: y0+m*(x-x0), 'deriv': lambda x, m=-16.5: np.zeros_like(x,dtype=float)+m, 'domain': (-8.033235653153664,-7.926764346846337), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.98, y0=-4.52, m=-12.5: y0+m*(x-x0), 'deriv': lambda x, m=-12.5: np.zeros_like(x,dtype=float)+m, 'domain': (-8.050175795608943,-7.909824204391057), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.98, y0=-0.52, m=-8.5: y0+m*(x-x0), 'deriv': lambda x, m=-8.5: np.zeros_like(x,dtype=float)+m, 'domain': (-8.08282029785931,-7.877179702140691), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.98, y0=3.48, m=-4.5: y0+m*(x-x0), 'deriv': lambda x, m=-4.5: np.zeros_like(x,dtype=float)+m, 'domain': (-8.170898802880417,-7.789101197119583), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.98, y0=7.48, m=-0.5: y0+m*(x-x0), 'deriv': lambda x, m=-0.5: np.zeros_like(x,dtype=float)+m, 'domain': (-8.767095928079927,-7.192904071920075), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.98, y0=-8.52, m=-12.5: y0+m*(x-x0), 'deriv': lambda x, m=-12.5: np.zeros_like(x,dtype=float)+m, 'domain': (-4.0501757956089435,-3.909824204391057), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.98, y0=-4.52, m=-8.5: y0+m*(x-x0), 'deriv': lambda x, m=-8.5: np.zeros_like(x,dtype=float)+m, 'domain': (-4.0828202978593096,-3.8771797021406904), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.98, y0=-0.52, m=-4.5: y0+m*(x-x0), 'deriv': lambda x, m=-4.5: np.zeros_like(x,dtype=float)+m, 'domain': (-4.1708988028804175,-3.7891011971195825), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.98, y0=3.48, m=-0.5: y0+m*(x-x0), 'deriv': lambda x, m=-0.5: np.zeros_like(x,dtype=float)+m, 'domain': (-4.767095928079926,-3.192904071920074), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.98, y0=7.48, m=3.5000000000000004: y0+m*(x-x0), 'deriv': lambda x, m=3.5000000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-4.221754592549693,-3.7382454074503073), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.020000000000000018, y0=-8.52, m=-8.5: y0+m*(x-x0), 'deriv': lambda x, m=-8.5: np.zeros_like(x,dtype=float)+m, 'domain': (-0.08282029785930951,0.12282029785930955), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.020000000000000018, y0=-4.52, m=-4.5: y0+m*(x-x0), 'deriv': lambda x, m=-4.5: np.zeros_like(x,dtype=float)+m, 'domain': (-0.17089880288041742,0.21089880288041746), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.020000000000000018, y0=-0.52, m=-0.5: y0+m*(x-x0), 'deriv': lambda x, m=-0.5: np.zeros_like(x,dtype=float)+m, 'domain': (-0.7670959280799259,0.8070959280799259), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.020000000000000018, y0=3.48, m=3.5: y0+m*(x-x0), 'deriv': lambda x, m=3.5: np.zeros_like(x,dtype=float)+m, 'domain': (-0.22175459254969268,0.2617545925496927), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.020000000000000018, y0=7.48, m=7.5: y0+m*(x-x0), 'deriv': lambda x, m=7.5: np.zeros_like(x,dtype=float)+m, 'domain': (-0.09630407368009578,0.1363040736800958), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.02, y0=-8.52, m=-4.5: y0+m*(x-x0), 'deriv': lambda x, m=-4.5: np.zeros_like(x,dtype=float)+m, 'domain': (3.829101197119582,4.210898802880417), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.02, y0=-4.52, m=-0.5: y0+m*(x-x0), 'deriv': lambda x, m=-0.5: np.zeros_like(x,dtype=float)+m, 'domain': (3.232904071920074,4.807095928079925), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.02, y0=-0.52, m=3.4999999999999996: y0+m*(x-x0), 'deriv': lambda x, m=3.4999999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (3.778245407450307,4.261754592549693), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.02, y0=3.48, m=7.5: y0+m*(x-x0), 'deriv': lambda x, m=7.5: np.zeros_like(x,dtype=float)+m, 'domain': (3.903695926319904,4.136304073680096), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.02, y0=7.48, m=11.5: y0+m*(x-x0), 'deriv': lambda x, m=11.5: np.zeros_like(x,dtype=float)+m, 'domain': (3.9437659375989385,4.09623406240106), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.02, y0=-8.52, m=-0.5: y0+m*(x-x0), 'deriv': lambda x, m=-0.5: np.zeros_like(x,dtype=float)+m, 'domain': (7.232904071920074,8.807095928079926), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.02, y0=-4.52, m=3.5: y0+m*(x-x0), 'deriv': lambda x, m=3.5: np.zeros_like(x,dtype=float)+m, 'domain': (7.7782454074503065,8.261754592549693), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.02, y0=-0.52, m=7.5: y0+m*(x-x0), 'deriv': lambda x, m=7.5: np.zeros_like(x,dtype=float)+m, 'domain': (7.903695926319903,8.136304073680096), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.02, y0=3.48, m=11.5: y0+m*(x-x0), 'deriv': lambda x, m=11.5: np.zeros_like(x,dtype=float)+m, 'domain': (7.943765937598939,8.096234062401061), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.02, y0=7.48, m=15.5: y0+m*(x-x0), 'deriv': lambda x, m=15.5: np.zeros_like(x,dtype=float)+m, 'domain': (7.96334359536179,8.076656404638209), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d3_02_v2_fa_field_reason.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.71, y0=-7.93, m=7.71: y0+m*(x-x0), 'deriv': lambda x, m=7.71: np.zeros_like(x,dtype=float)+m, 'domain': (-7.780743367938154,-7.639256632061846), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.71, y0=-3.9299999999999997, m=7.71: y0+m*(x-x0), 'deriv': lambda x, m=7.71: np.zeros_like(x,dtype=float)+m, 'domain': (-7.780743367938154,-7.639256632061846), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.71, y0=0.07000000000000006, m=7.71: y0+m*(x-x0), 'deriv': lambda x, m=7.71: np.zeros_like(x,dtype=float)+m, 'domain': (-7.780743367938154,-7.639256632061846), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.71, y0=4.07, m=7.71: y0+m*(x-x0), 'deriv': lambda x, m=7.71: np.zeros_like(x,dtype=float)+m, 'domain': (-7.780743367938154,-7.639256632061846), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.71, y0=8.07, m=7.71: y0+m*(x-x0), 'deriv': lambda x, m=7.71: np.zeros_like(x,dtype=float)+m, 'domain': (-7.780743367938154,-7.639256632061846), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.71, y0=-7.93, m=3.71: y0+m*(x-x0), 'deriv': lambda x, m=3.71: np.zeros_like(x,dtype=float)+m, 'domain': (-3.85313940263903,-3.5668605973609697), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.71, y0=-3.9299999999999997, m=3.71: y0+m*(x-x0), 'deriv': lambda x, m=3.71: np.zeros_like(x,dtype=float)+m, 'domain': (-3.85313940263903,-3.5668605973609697), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.71, y0=0.07000000000000006, m=3.71: y0+m*(x-x0), 'deriv': lambda x, m=3.71: np.zeros_like(x,dtype=float)+m, 'domain': (-3.85313940263903,-3.5668605973609697), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.71, y0=4.07, m=3.71: y0+m*(x-x0), 'deriv': lambda x, m=3.71: np.zeros_like(x,dtype=float)+m, 'domain': (-3.85313940263903,-3.5668605973609697), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.71, y0=8.07, m=3.71: y0+m*(x-x0), 'deriv': lambda x, m=3.71: np.zeros_like(x,dtype=float)+m, 'domain': (-3.85313940263903,-3.5668605973609697), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.29000000000000004, y0=-7.93, m=-0.29000000000000004: y0+m*(x-x0), 'deriv': lambda x, m=-0.29000000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-0.23823602632830254,0.8182360263283026), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.29000000000000004, y0=-3.9299999999999997, m=-0.29000000000000004: y0+m*(x-x0), 'deriv': lambda x, m=-0.29000000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-0.23823602632830254,0.8182360263283026), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.29000000000000004, y0=0.07000000000000006, m=-0.29000000000000004: y0+m*(x-x0), 'deriv': lambda x, m=-0.29000000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-0.23823602632830254,0.8182360263283026), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.29000000000000004, y0=4.07, m=-0.29000000000000004: y0+m*(x-x0), 'deriv': lambda x, m=-0.29000000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-0.23823602632830254,0.8182360263283026), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.29000000000000004, y0=8.07, m=-0.29000000000000004: y0+m*(x-x0), 'deriv': lambda x, m=-0.29000000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-0.23823602632830254,0.8182360263283026), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.29, y0=-7.93, m=-4.29: y0+m*(x-x0), 'deriv': lambda x, m=-4.29: np.zeros_like(x,dtype=float)+m, 'domain': (4.165142125521162,4.414857874478838), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.29, y0=-3.9299999999999997, m=-4.29: y0+m*(x-x0), 'deriv': lambda x, m=-4.29: np.zeros_like(x,dtype=float)+m, 'domain': (4.165142125521162,4.414857874478838), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.29, y0=0.07000000000000006, m=-4.29: y0+m*(x-x0), 'deriv': lambda x, m=-4.29: np.zeros_like(x,dtype=float)+m, 'domain': (4.165142125521162,4.414857874478838), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.29, y0=4.07, m=-4.29: y0+m*(x-x0), 'deriv': lambda x, m=-4.29: np.zeros_like(x,dtype=float)+m, 'domain': (4.165142125521162,4.414857874478838), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.29, y0=8.07, m=-4.29: y0+m*(x-x0), 'deriv': lambda x, m=-4.29: np.zeros_like(x,dtype=float)+m, 'domain': (4.165142125521162,4.414857874478838), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.29, y0=-7.93, m=-8.29: y0+m*(x-x0), 'deriv': lambda x, m=-8.29: np.zeros_like(x,dtype=float)+m, 'domain': (8.224132492290204,8.355867507709794), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.29, y0=-3.9299999999999997, m=-8.29: y0+m*(x-x0), 'deriv': lambda x, m=-8.29: np.zeros_like(x,dtype=float)+m, 'domain': (8.224132492290204,8.355867507709794), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.29, y0=0.07000000000000006, m=-8.29: y0+m*(x-x0), 'deriv': lambda x, m=-8.29: np.zeros_like(x,dtype=float)+m, 'domain': (8.224132492290204,8.355867507709794), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.29, y0=4.07, m=-8.29: y0+m*(x-x0), 'deriv': lambda x, m=-8.29: np.zeros_like(x,dtype=float)+m, 'domain': (8.224132492290204,8.355867507709794), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.29, y0=8.07, m=-8.29: y0+m*(x-x0), 'deriv': lambda x, m=-8.29: np.zeros_like(x,dtype=float)+m, 'domain': (8.224132492290204,8.355867507709794), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d3_02_v3_fa_field_reason.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.66, y0=-7.77, m=-1.8900000000000006: y0+m*(x-x0), 'deriv': lambda x, m=-1.8900000000000006: np.zeros_like(x,dtype=float)+m, 'domain': (-8.973340796022482,-8.346659203977518), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.66, y0=-3.77, m=-5.890000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-5.890000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-8.772147281912567,-8.547852718087434), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.66, y0=0.22999999999999998, m=-9.89: y0+m*(x-x0), 'deriv': lambda x, m=-9.89: np.zeros_like(x,dtype=float)+m, 'domain': (-8.72740152737015,-8.59259847262985), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.66, y0=4.23, m=-13.89: y0+m*(x-x0), 'deriv': lambda x, m=-13.89: np.zeros_like(x,dtype=float)+m, 'domain': (-8.708111616897183,-8.611888383102817), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.66, y0=8.23, m=-17.89: y0+m*(x-x0), 'deriv': lambda x, m=-17.89: np.zeros_like(x,dtype=float)+m, 'domain': (-8.69739271894142,-8.62260728105858), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.66, y0=-7.77, m=2.1099999999999994: y0+m*(x-x0), 'deriv': lambda x, m=2.1099999999999994: np.zeros_like(x,dtype=float)+m, 'domain': (-4.946941190784902,-4.373058809215098), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.66, y0=-3.77, m=-1.8900000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-1.8900000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-4.973340796022482,-4.346659203977518), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.66, y0=0.22999999999999998, m=-5.890000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-5.890000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-4.772147281912566,-4.547852718087435), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.66, y0=4.23, m=-9.89: y0+m*(x-x0), 'deriv': lambda x, m=-9.89: np.zeros_like(x,dtype=float)+m, 'domain': (-4.72740152737015,-4.5925984726298505), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.66, y0=8.23, m=-13.89: y0+m*(x-x0), 'deriv': lambda x, m=-13.89: np.zeros_like(x,dtype=float)+m, 'domain': (-4.708111616897184,-4.611888383102817), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.66, y0=-7.77, m=6.109999999999999: y0+m*(x-x0), 'deriv': lambda x, m=6.109999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-0.7682165032018762,-0.5517834967981239), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.66, y0=-3.77, m=2.11: y0+m*(x-x0), 'deriv': lambda x, m=2.11: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9469411907849026,-0.37305880921509754), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.66, y0=0.22999999999999998, m=-1.8900000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-1.8900000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9733407960224816,-0.34665920397751854), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.66, y0=4.23, m=-5.890000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-5.890000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-0.7721472819125655,-0.5478527180874345), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.66, y0=8.23, m=-9.89: y0+m*(x-x0), 'deriv': lambda x, m=-9.89: np.zeros_like(x,dtype=float)+m, 'domain': (-0.7274015273701497,-0.5925984726298503), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.34, y0=-7.77, m=10.11: y0+m*(x-x0), 'deriv': lambda x, m=10.11: np.zeros_like(x,dtype=float)+m, 'domain': (3.274050805501558,3.405949194498442), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.34, y0=-3.77, m=6.109999999999999: y0+m*(x-x0), 'deriv': lambda x, m=6.109999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (3.2317834967981236,3.448216503201876), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.34, y0=0.22999999999999998, m=2.11: y0+m*(x-x0), 'deriv': lambda x, m=2.11: np.zeros_like(x,dtype=float)+m, 'domain': (3.053058809215097,3.6269411907849025), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.34, y0=4.23, m=-1.8900000000000006: y0+m*(x-x0), 'deriv': lambda x, m=-1.8900000000000006: np.zeros_like(x,dtype=float)+m, 'domain': (3.0266592039775184,3.6533407960224813), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.34, y0=8.23, m=-5.890000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-5.890000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (3.2278527180874343,3.4521472819125654), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.34, y0=-7.77, m=14.11: y0+m*(x-x0), 'deriv': lambda x, m=14.11: np.zeros_like(x,dtype=float)+m, 'domain': (7.292634750254009,7.387365249745991), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.34, y0=-3.77, m=10.11: y0+m*(x-x0), 'deriv': lambda x, m=10.11: np.zeros_like(x,dtype=float)+m, 'domain': (7.274050805501558,7.405949194498442), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.34, y0=0.22999999999999998, m=6.109999999999999: y0+m*(x-x0), 'deriv': lambda x, m=6.109999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (7.231783496798124,7.448216503201876), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.34, y0=4.23, m=2.1099999999999994: y0+m*(x-x0), 'deriv': lambda x, m=2.1099999999999994: np.zeros_like(x,dtype=float)+m, 'domain': (7.053058809215098,7.626941190784902), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.34, y0=8.23, m=-1.8900000000000006: y0+m*(x-x0), 'deriv': lambda x, m=-1.8900000000000006: np.zeros_like(x,dtype=float)+m, 'domain': (7.026659203977518,7.653340796022482), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d3_02_v4_fa_field_reason.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.54, y0=-8.59, m=-17.13: y0+m*(x-x0), 'deriv': lambda x, m=-17.13: np.zeros_like(x,dtype=float)+m, 'domain': (-7.59186732954313,-7.48813267045687), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.54, y0=-4.59, m=-13.129999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-13.129999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-7.6075879605964865,-7.472412039403514), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.54, y0=-0.59, m=-9.13: y0+m*(x-x0), 'deriv': lambda x, m=-9.13: np.zeros_like(x,dtype=float)+m, 'domain': (-7.636901321724876,-7.443098678275124), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.54, y0=3.41, m=-5.13: y0+m*(x-x0), 'deriv': lambda x, m=-5.13: np.zeros_like(x,dtype=float)+m, 'domain': (-7.710284179169762,-7.369715820830238), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.54, y0=7.41, m=-1.13: y0+m*(x-x0), 'deriv': lambda x, m=-1.13: np.zeros_like(x,dtype=float)+m, 'domain': (-8.12981800805088,-6.95018199194912), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.54, y0=-8.59, m=-13.129999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-13.129999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-3.607587960596487,-3.472412039403513), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.54, y0=-4.59, m=-9.129999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-9.129999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-3.6369013217248756,-3.4430986782751245), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.54, y0=-0.59, m=-5.13: y0+m*(x-x0), 'deriv': lambda x, m=-5.13: np.zeros_like(x,dtype=float)+m, 'domain': (-3.710284179169762,-3.3697158208302382), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.54, y0=3.41, m=-1.13: y0+m*(x-x0), 'deriv': lambda x, m=-1.13: np.zeros_like(x,dtype=float)+m, 'domain': (-4.12981800805088,-2.9501819919491203), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.54, y0=7.41, m=2.87: y0+m*(x-x0), 'deriv': lambda x, m=2.87: np.zeros_like(x,dtype=float)+m, 'domain': (-3.8328376315737427,-3.2471623684262574), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.45999999999999996, y0=-8.59, m=-9.129999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-9.129999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (0.3630986782751242,0.5569013217248757), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.45999999999999996, y0=-4.59, m=-5.13: y0+m*(x-x0), 'deriv': lambda x, m=-5.13: np.zeros_like(x,dtype=float)+m, 'domain': (0.2897158208302381,0.6302841791697618), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.45999999999999996, y0=-0.59, m=-1.13: y0+m*(x-x0), 'deriv': lambda x, m=-1.13: np.zeros_like(x,dtype=float)+m, 'domain': (-0.12981800805087984,1.0498180080508797), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.45999999999999996, y0=3.41, m=2.87: y0+m*(x-x0), 'deriv': lambda x, m=2.87: np.zeros_like(x,dtype=float)+m, 'domain': (0.16716236842625737,0.7528376315737426), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.45999999999999996, y0=7.41, m=6.87: y0+m*(x-x0), 'deriv': lambda x, m=6.87: np.zeros_like(x,dtype=float)+m, 'domain': (0.3318022349652373,0.5881977650347626), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.46, y0=-8.59, m=-5.13: y0+m*(x-x0), 'deriv': lambda x, m=-5.13: np.zeros_like(x,dtype=float)+m, 'domain': (4.289715820830238,4.630284179169762), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.46, y0=-4.59, m=-1.13: y0+m*(x-x0), 'deriv': lambda x, m=-1.13: np.zeros_like(x,dtype=float)+m, 'domain': (3.8701819919491203,5.04981800805088), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.46, y0=-0.59, m=2.87: y0+m*(x-x0), 'deriv': lambda x, m=2.87: np.zeros_like(x,dtype=float)+m, 'domain': (4.167162368426258,4.752837631573742), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.46, y0=3.41, m=6.87: y0+m*(x-x0), 'deriv': lambda x, m=6.87: np.zeros_like(x,dtype=float)+m, 'domain': (4.331802234965237,4.588197765034763), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.46, y0=7.41, m=10.870000000000001: y0+m*(x-x0), 'deriv': lambda x, m=10.870000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (4.378467565802454,4.541532434197546), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.46, y0=-8.59, m=-1.129999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-1.129999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (7.870181991949121,9.049818008050881), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.46, y0=-4.59, m=2.870000000000001: y0+m*(x-x0), 'deriv': lambda x, m=2.870000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (8.167162368426258,8.752837631573744), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.46, y0=-0.59, m=6.870000000000001: y0+m*(x-x0), 'deriv': lambda x, m=6.870000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (8.331802234965238,8.588197765034764), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.46, y0=3.41, m=10.870000000000001: y0+m*(x-x0), 'deriv': lambda x, m=10.870000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (8.378467565802456,8.541532434197546), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.46, y0=7.41, m=14.870000000000001: y0+m*(x-x0), 'deriv': lambda x, m=14.870000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (8.400282831847663,8.519717168152338), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d3_02_v5_fa_field_reason.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.09, y0=-7.6, m=-7.6: y0+m*(x-x0), 'deriv': lambda x, m=-7.6: np.zeros_like(x,dtype=float)+m, 'domain': (-8.159140891662835,-8.020859108337165), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.09, y0=-3.6, m=-3.6: y0+m*(x-x0), 'deriv': lambda x, m=-3.6: np.zeros_like(x,dtype=float)+m, 'domain': (-8.23185124780663,-7.94814875219337), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.09, y0=0.4, m=0.4: y0+m*(x-x0), 'deriv': lambda x, m=0.4: np.zeros_like(x,dtype=float)+m, 'domain': (-8.582092646169187,-7.597907353830813), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.09, y0=4.4, m=4.4: y0+m*(x-x0), 'deriv': lambda x, m=4.4: np.zeros_like(x,dtype=float)+m, 'domain': (-8.207459184902152,-7.9725408150978465), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.09, y0=8.4, m=8.4: y0+m*(x-x0), 'deriv': lambda x, m=8.4: np.zeros_like(x,dtype=float)+m, 'domain': (-8.15265283158611,-8.02734716841389), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.09, y0=-7.6, m=-7.6: y0+m*(x-x0), 'deriv': lambda x, m=-7.6: np.zeros_like(x,dtype=float)+m, 'domain': (-4.159140891662835,-4.020859108337165), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.09, y0=-3.6, m=-3.6: y0+m*(x-x0), 'deriv': lambda x, m=-3.6: np.zeros_like(x,dtype=float)+m, 'domain': (-4.23185124780663,-3.9481487521933696), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.09, y0=0.4, m=0.4: y0+m*(x-x0), 'deriv': lambda x, m=0.4: np.zeros_like(x,dtype=float)+m, 'domain': (-4.582092646169187,-3.5979073538308124), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.09, y0=4.4, m=4.4: y0+m*(x-x0), 'deriv': lambda x, m=4.4: np.zeros_like(x,dtype=float)+m, 'domain': (-4.207459184902153,-3.972540815097847), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.09, y0=8.4, m=8.4: y0+m*(x-x0), 'deriv': lambda x, m=8.4: np.zeros_like(x,dtype=float)+m, 'domain': (-4.152652831586109,-4.027347168413891), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.08999999999999997, y0=-7.6, m=-7.6: y0+m*(x-x0), 'deriv': lambda x, m=-7.6: np.zeros_like(x,dtype=float)+m, 'domain': (-0.15914089166283527,-0.020859108337164672), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.08999999999999997, y0=-3.6, m=-3.6: y0+m*(x-x0), 'deriv': lambda x, m=-3.6: np.zeros_like(x,dtype=float)+m, 'domain': (-0.2318512478066301,0.05185124780663017), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.08999999999999997, y0=0.4, m=0.4: y0+m*(x-x0), 'deriv': lambda x, m=0.4: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5820926461691873,0.40209264616918744), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.08999999999999997, y0=4.4, m=4.4: y0+m*(x-x0), 'deriv': lambda x, m=4.4: np.zeros_like(x,dtype=float)+m, 'domain': (-0.2074591849021531,0.02745918490215314), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.08999999999999997, y0=8.4, m=8.4: y0+m*(x-x0), 'deriv': lambda x, m=8.4: np.zeros_like(x,dtype=float)+m, 'domain': (-0.1526528315861095,-0.027347168413890427), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.91, y0=-7.6, m=-7.6: y0+m*(x-x0), 'deriv': lambda x, m=-7.6: np.zeros_like(x,dtype=float)+m, 'domain': (3.840859108337165,3.9791408916628352), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.91, y0=-3.6, m=-3.6: y0+m*(x-x0), 'deriv': lambda x, m=-3.6: np.zeros_like(x,dtype=float)+m, 'domain': (3.76814875219337,4.05185124780663), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.91, y0=0.4, m=0.4: y0+m*(x-x0), 'deriv': lambda x, m=0.4: np.zeros_like(x,dtype=float)+m, 'domain': (3.4179073538308127,4.402092646169187), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.91, y0=4.4, m=4.4: y0+m*(x-x0), 'deriv': lambda x, m=4.4: np.zeros_like(x,dtype=float)+m, 'domain': (3.7925408150978472,4.0274591849021535), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.91, y0=8.4, m=8.4: y0+m*(x-x0), 'deriv': lambda x, m=8.4: np.zeros_like(x,dtype=float)+m, 'domain': (3.8473471684138905,3.97265283158611), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.91, y0=-7.6, m=-7.6: y0+m*(x-x0), 'deriv': lambda x, m=-7.6: np.zeros_like(x,dtype=float)+m, 'domain': (7.840859108337165,7.979140891662835), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.91, y0=-3.6, m=-3.6: y0+m*(x-x0), 'deriv': lambda x, m=-3.6: np.zeros_like(x,dtype=float)+m, 'domain': (7.76814875219337,8.05185124780663), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.91, y0=0.4, m=0.4: y0+m*(x-x0), 'deriv': lambda x, m=0.4: np.zeros_like(x,dtype=float)+m, 'domain': (7.417907353830813,8.402092646169187), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.91, y0=4.4, m=4.4: y0+m*(x-x0), 'deriv': lambda x, m=4.4: np.zeros_like(x,dtype=float)+m, 'domain': (7.792540815097847,8.027459184902153), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.91, y0=8.4, m=8.4: y0+m*(x-x0), 'deriv': lambda x, m=8.4: np.zeros_like(x,dtype=float)+m, 'domain': (7.847347168413891,7.972652831586109), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d3_02_v6_fa_field_reason.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.32, y0=-8.1, m=8.32: y0+m*(x-x0), 'deriv': lambda x, m=8.32: np.zeros_like(x,dtype=float)+m, 'domain': (-8.399953405852882,-8.240046594147119), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.32, y0=-4.1, m=8.32: y0+m*(x-x0), 'deriv': lambda x, m=8.32: np.zeros_like(x,dtype=float)+m, 'domain': (-8.399953405852882,-8.240046594147119), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.32, y0=-0.09999999999999998, m=8.32: y0+m*(x-x0), 'deriv': lambda x, m=8.32: np.zeros_like(x,dtype=float)+m, 'domain': (-8.399953405852882,-8.240046594147119), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.32, y0=3.9, m=8.32: y0+m*(x-x0), 'deriv': lambda x, m=8.32: np.zeros_like(x,dtype=float)+m, 'domain': (-8.399953405852882,-8.240046594147119), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.32, y0=7.9, m=8.32: y0+m*(x-x0), 'deriv': lambda x, m=8.32: np.zeros_like(x,dtype=float)+m, 'domain': (-8.399953405852882,-8.240046594147119), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.32, y0=-8.1, m=4.32: y0+m*(x-x0), 'deriv': lambda x, m=4.32: np.zeros_like(x,dtype=float)+m, 'domain': (-4.471097242727007,-4.168902757272994), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.32, y0=-4.1, m=4.32: y0+m*(x-x0), 'deriv': lambda x, m=4.32: np.zeros_like(x,dtype=float)+m, 'domain': (-4.471097242727007,-4.168902757272994), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.32, y0=-0.09999999999999998, m=4.32: y0+m*(x-x0), 'deriv': lambda x, m=4.32: np.zeros_like(x,dtype=float)+m, 'domain': (-4.471097242727007,-4.168902757272994), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.32, y0=3.9, m=4.32: y0+m*(x-x0), 'deriv': lambda x, m=4.32: np.zeros_like(x,dtype=float)+m, 'domain': (-4.471097242727007,-4.168902757272994), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.32, y0=7.9, m=4.32: y0+m*(x-x0), 'deriv': lambda x, m=4.32: np.zeros_like(x,dtype=float)+m, 'domain': (-4.471097242727007,-4.168902757272994), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.32, y0=-8.1, m=0.32: y0+m*(x-x0), 'deriv': lambda x, m=0.32: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9581241786235473,0.3181241786235472), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.32, y0=-4.1, m=0.32: y0+m*(x-x0), 'deriv': lambda x, m=0.32: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9581241786235473,0.3181241786235472), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.32, y0=-0.09999999999999998, m=0.32: y0+m*(x-x0), 'deriv': lambda x, m=0.32: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9581241786235473,0.3181241786235472), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.32, y0=3.9, m=0.32: y0+m*(x-x0), 'deriv': lambda x, m=0.32: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9581241786235473,0.3181241786235472), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.32, y0=7.9, m=0.32: y0+m*(x-x0), 'deriv': lambda x, m=0.32: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9581241786235473,0.3181241786235472), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.68, y0=-8.1, m=-3.68: y0+m*(x-x0), 'deriv': lambda x, m=-3.68: np.zeros_like(x,dtype=float)+m, 'domain': (3.504306069100141,3.8556939308998595), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.68, y0=-4.1, m=-3.68: y0+m*(x-x0), 'deriv': lambda x, m=-3.68: np.zeros_like(x,dtype=float)+m, 'domain': (3.504306069100141,3.8556939308998595), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.68, y0=-0.09999999999999998, m=-3.68: y0+m*(x-x0), 'deriv': lambda x, m=-3.68: np.zeros_like(x,dtype=float)+m, 'domain': (3.504306069100141,3.8556939308998595), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.68, y0=3.9, m=-3.68: y0+m*(x-x0), 'deriv': lambda x, m=-3.68: np.zeros_like(x,dtype=float)+m, 'domain': (3.504306069100141,3.8556939308998595), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.68, y0=7.9, m=-3.68: y0+m*(x-x0), 'deriv': lambda x, m=-3.68: np.zeros_like(x,dtype=float)+m, 'domain': (3.504306069100141,3.8556939308998595), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.68, y0=-8.1, m=-7.68: y0+m*(x-x0), 'deriv': lambda x, m=-7.68: np.zeros_like(x,dtype=float)+m, 'domain': (7.593490682970944,7.766509317029056), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.68, y0=-4.1, m=-7.68: y0+m*(x-x0), 'deriv': lambda x, m=-7.68: np.zeros_like(x,dtype=float)+m, 'domain': (7.593490682970944,7.766509317029056), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.68, y0=-0.09999999999999998, m=-7.68: y0+m*(x-x0), 'deriv': lambda x, m=-7.68: np.zeros_like(x,dtype=float)+m, 'domain': (7.593490682970944,7.766509317029056), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.68, y0=3.9, m=-7.68: y0+m*(x-x0), 'deriv': lambda x, m=-7.68: np.zeros_like(x,dtype=float)+m, 'domain': (7.593490682970944,7.766509317029056), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.68, y0=7.9, m=-7.68: y0+m*(x-x0), 'deriv': lambda x, m=-7.68: np.zeros_like(x,dtype=float)+m, 'domain': (7.593490682970944,7.766509317029056), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d3_02_v7_fa_field_reason.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.72, y0=-7.5, m=8.72: y0+m*(x-x0), 'deriv': lambda x, m=8.72: np.zeros_like(x,dtype=float)+m, 'domain': (-8.783802015726321,-8.65619798427368), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.72, y0=-3.5, m=8.72: y0+m*(x-x0), 'deriv': lambda x, m=8.72: np.zeros_like(x,dtype=float)+m, 'domain': (-8.783802015726321,-8.65619798427368), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.72, y0=0.5000000000000001, m=8.72: y0+m*(x-x0), 'deriv': lambda x, m=8.72: np.zeros_like(x,dtype=float)+m, 'domain': (-8.783802015726321,-8.65619798427368), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.72, y0=4.5, m=8.72: y0+m*(x-x0), 'deriv': lambda x, m=8.72: np.zeros_like(x,dtype=float)+m, 'domain': (-8.783802015726321,-8.65619798427368), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.72, y0=8.5, m=8.72: y0+m*(x-x0), 'deriv': lambda x, m=8.72: np.zeros_like(x,dtype=float)+m, 'domain': (-8.783802015726321,-8.65619798427368), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.72, y0=-7.5, m=4.72: y0+m*(x-x0), 'deriv': lambda x, m=4.72: np.zeros_like(x,dtype=float)+m, 'domain': (-4.836067722770628,-4.603932277229371), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.72, y0=-3.5, m=4.72: y0+m*(x-x0), 'deriv': lambda x, m=4.72: np.zeros_like(x,dtype=float)+m, 'domain': (-4.836067722770628,-4.603932277229371), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.72, y0=0.5000000000000001, m=4.72: y0+m*(x-x0), 'deriv': lambda x, m=4.72: np.zeros_like(x,dtype=float)+m, 'domain': (-4.836067722770628,-4.603932277229371), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.72, y0=4.5, m=4.72: y0+m*(x-x0), 'deriv': lambda x, m=4.72: np.zeros_like(x,dtype=float)+m, 'domain': (-4.836067722770628,-4.603932277229371), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.72, y0=8.5, m=4.72: y0+m*(x-x0), 'deriv': lambda x, m=4.72: np.zeros_like(x,dtype=float)+m, 'domain': (-4.836067722770628,-4.603932277229371), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.72, y0=-7.5, m=0.72: y0+m*(x-x0), 'deriv': lambda x, m=0.72: np.zeros_like(x,dtype=float)+m, 'domain': (-1.1744592312128368,-0.2655407687871631), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.72, y0=-3.5, m=0.72: y0+m*(x-x0), 'deriv': lambda x, m=0.72: np.zeros_like(x,dtype=float)+m, 'domain': (-1.1744592312128368,-0.2655407687871631), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.72, y0=0.5000000000000001, m=0.72: y0+m*(x-x0), 'deriv': lambda x, m=0.72: np.zeros_like(x,dtype=float)+m, 'domain': (-1.1744592312128368,-0.2655407687871631), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.72, y0=4.5, m=0.72: y0+m*(x-x0), 'deriv': lambda x, m=0.72: np.zeros_like(x,dtype=float)+m, 'domain': (-1.1744592312128368,-0.2655407687871631), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.72, y0=8.5, m=0.72: y0+m*(x-x0), 'deriv': lambda x, m=0.72: np.zeros_like(x,dtype=float)+m, 'domain': (-1.1744592312128368,-0.2655407687871631), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.2800000000000002, y0=-7.5, m=-3.2800000000000002: y0+m*(x-x0), 'deriv': lambda x, m=-3.2800000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (3.1166895738837566,3.443310426116244), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.2800000000000002, y0=-3.5, m=-3.2800000000000002: y0+m*(x-x0), 'deriv': lambda x, m=-3.2800000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (3.1166895738837566,3.443310426116244), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.2800000000000002, y0=0.5000000000000001, m=-3.2800000000000002: y0+m*(x-x0), 'deriv': lambda x, m=-3.2800000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (3.1166895738837566,3.443310426116244), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.2800000000000002, y0=4.5, m=-3.2800000000000002: y0+m*(x-x0), 'deriv': lambda x, m=-3.2800000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (3.1166895738837566,3.443310426116244), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.2800000000000002, y0=8.5, m=-3.2800000000000002: y0+m*(x-x0), 'deriv': lambda x, m=-3.2800000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (3.1166895738837566,3.443310426116244), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.28, y0=-7.5, m=-7.28: y0+m*(x-x0), 'deriv': lambda x, m=-7.28: np.zeros_like(x,dtype=float)+m, 'domain': (7.20379252343873,7.356207476561271), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.28, y0=-3.5, m=-7.28: y0+m*(x-x0), 'deriv': lambda x, m=-7.28: np.zeros_like(x,dtype=float)+m, 'domain': (7.20379252343873,7.356207476561271), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.28, y0=0.5000000000000001, m=-7.28: y0+m*(x-x0), 'deriv': lambda x, m=-7.28: np.zeros_like(x,dtype=float)+m, 'domain': (7.20379252343873,7.356207476561271), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.28, y0=4.5, m=-7.28: y0+m*(x-x0), 'deriv': lambda x, m=-7.28: np.zeros_like(x,dtype=float)+m, 'domain': (7.20379252343873,7.356207476561271), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.28, y0=8.5, m=-7.28: y0+m*(x-x0), 'deriv': lambda x, m=-7.28: np.zeros_like(x,dtype=float)+m, 'domain': (7.20379252343873,7.356207476561271), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d3_02_v8_fa_field_reason.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.43, y0=-7.8, m=-8.43: y0+m*(x-x0), 'deriv': lambda x, m=-8.43: np.zeros_like(x,dtype=float)+m, 'domain': (-8.537196226618395,-8.322803773381604), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.43, y0=-3.8, m=-8.43: y0+m*(x-x0), 'deriv': lambda x, m=-8.43: np.zeros_like(x,dtype=float)+m, 'domain': (-8.537196226618395,-8.322803773381604), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.43, y0=0.20000000000000007, m=-8.43: y0+m*(x-x0), 'deriv': lambda x, m=-8.43: np.zeros_like(x,dtype=float)+m, 'domain': (-8.537196226618395,-8.322803773381604), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.43, y0=4.2, m=-8.43: y0+m*(x-x0), 'deriv': lambda x, m=-8.43: np.zeros_like(x,dtype=float)+m, 'domain': (-8.537196226618395,-8.322803773381604), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.43, y0=8.2, m=-8.43: y0+m*(x-x0), 'deriv': lambda x, m=-8.43: np.zeros_like(x,dtype=float)+m, 'domain': (-8.537196226618395,-8.322803773381604), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.43, y0=-7.8, m=-4.43: y0+m*(x-x0), 'deriv': lambda x, m=-4.43: np.zeros_like(x,dtype=float)+m, 'domain': (-4.630375891013322,-4.229624108986678), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.43, y0=-3.8, m=-4.43: y0+m*(x-x0), 'deriv': lambda x, m=-4.43: np.zeros_like(x,dtype=float)+m, 'domain': (-4.630375891013322,-4.229624108986678), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.43, y0=0.20000000000000007, m=-4.43: y0+m*(x-x0), 'deriv': lambda x, m=-4.43: np.zeros_like(x,dtype=float)+m, 'domain': (-4.630375891013322,-4.229624108986678), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.43, y0=4.2, m=-4.43: y0+m*(x-x0), 'deriv': lambda x, m=-4.43: np.zeros_like(x,dtype=float)+m, 'domain': (-4.630375891013322,-4.229624108986678), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.43, y0=8.2, m=-4.43: y0+m*(x-x0), 'deriv': lambda x, m=-4.43: np.zeros_like(x,dtype=float)+m, 'domain': (-4.630375891013322,-4.229624108986678), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.43, y0=-7.8, m=-0.43: y0+m*(x-x0), 'deriv': lambda x, m=-0.43: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2659889606978325,0.40598896069783247), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.43, y0=-3.8, m=-0.43: y0+m*(x-x0), 'deriv': lambda x, m=-0.43: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2659889606978325,0.40598896069783247), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.43, y0=0.20000000000000007, m=-0.43: y0+m*(x-x0), 'deriv': lambda x, m=-0.43: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2659889606978325,0.40598896069783247), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.43, y0=4.2, m=-0.43: y0+m*(x-x0), 'deriv': lambda x, m=-0.43: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2659889606978325,0.40598896069783247), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.43, y0=8.2, m=-0.43: y0+m*(x-x0), 'deriv': lambda x, m=-0.43: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2659889606978325,0.40598896069783247), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.57, y0=-7.8, m=3.57: y0+m*(x-x0), 'deriv': lambda x, m=3.57: np.zeros_like(x,dtype=float)+m, 'domain': (3.3245457252306374,3.8154542747693623), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.57, y0=-3.8, m=3.57: y0+m*(x-x0), 'deriv': lambda x, m=3.57: np.zeros_like(x,dtype=float)+m, 'domain': (3.3245457252306374,3.8154542747693623), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.57, y0=0.20000000000000007, m=3.57: y0+m*(x-x0), 'deriv': lambda x, m=3.57: np.zeros_like(x,dtype=float)+m, 'domain': (3.3245457252306374,3.8154542747693623), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.57, y0=4.2, m=3.57: y0+m*(x-x0), 'deriv': lambda x, m=3.57: np.zeros_like(x,dtype=float)+m, 'domain': (3.3245457252306374,3.8154542747693623), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.57, y0=8.2, m=3.57: y0+m*(x-x0), 'deriv': lambda x, m=3.57: np.zeros_like(x,dtype=float)+m, 'domain': (3.3245457252306374,3.8154542747693623), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.57, y0=-7.8, m=7.57: y0+m*(x-x0), 'deriv': lambda x, m=7.57: np.zeros_like(x,dtype=float)+m, 'domain': (7.45082398349322,7.689176016506781), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.57, y0=-3.8, m=7.57: y0+m*(x-x0), 'deriv': lambda x, m=7.57: np.zeros_like(x,dtype=float)+m, 'domain': (7.45082398349322,7.689176016506781), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.57, y0=0.20000000000000007, m=7.57: y0+m*(x-x0), 'deriv': lambda x, m=7.57: np.zeros_like(x,dtype=float)+m, 'domain': (7.45082398349322,7.689176016506781), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.57, y0=4.2, m=7.57: y0+m*(x-x0), 'deriv': lambda x, m=7.57: np.zeros_like(x,dtype=float)+m, 'domain': (7.45082398349322,7.689176016506781), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.57, y0=8.2, m=7.57: y0+m*(x-x0), 'deriv': lambda x, m=7.57: np.zeros_like(x,dtype=float)+m, 'domain': (7.45082398349322,7.689176016506781), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d4_04_v1_fa_synthesis_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.43, y0=-8.59, m=0.16000000000000014: y0+m*(x-x0), 'deriv': lambda x, m=0.16000000000000014: np.zeros_like(x,dtype=float)+m, 'domain': (-9.190329286575864,-7.669670713424137), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.43, y0=-4.59, m=-3.84: y0+m*(x-x0), 'deriv': lambda x, m=-3.84: np.zeros_like(x,dtype=float)+m, 'domain': (-8.624048869943708,-8.235951130056291), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.43, y0=-0.59, m=-7.84: y0+m*(x-x0), 'deriv': lambda x, m=-7.84: np.zeros_like(x,dtype=float)+m, 'domain': (-8.527424966974355,-8.332575033025645), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.43, y0=3.41, m=-11.84: y0+m*(x-x0), 'deriv': lambda x, m=-11.84: np.zeros_like(x,dtype=float)+m, 'domain': (-8.49480306142271,-8.36519693857729), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.43, y0=7.41, m=-15.84: y0+m*(x-x0), 'deriv': lambda x, m=-15.84: np.zeros_like(x,dtype=float)+m, 'domain': (-8.478514528402178,-8.381485471597822), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.43, y0=-8.59, m=4.16: y0+m*(x-x0), 'deriv': lambda x, m=4.16: np.zeros_like(x,dtype=float)+m, 'domain': (-4.609969429951749,-4.25003057004825), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.43, y0=-4.59, m=0.16000000000000014: y0+m*(x-x0), 'deriv': lambda x, m=0.16000000000000014: np.zeros_like(x,dtype=float)+m, 'domain': (-5.190329286575863,-3.6696707134241366), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.43, y0=-0.59, m=-3.84: y0+m*(x-x0), 'deriv': lambda x, m=-3.84: np.zeros_like(x,dtype=float)+m, 'domain': (-4.624048869943708,-4.235951130056291), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.43, y0=3.41, m=-7.84: y0+m*(x-x0), 'deriv': lambda x, m=-7.84: np.zeros_like(x,dtype=float)+m, 'domain': (-4.527424966974355,-4.332575033025645), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.43, y0=7.41, m=-11.84: y0+m*(x-x0), 'deriv': lambda x, m=-11.84: np.zeros_like(x,dtype=float)+m, 'domain': (-4.49480306142271,-4.36519693857729), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.43, y0=-8.59, m=8.16: y0+m*(x-x0), 'deriv': lambda x, m=8.16: np.zeros_like(x,dtype=float)+m, 'domain': (-0.523662045537009,-0.336337954462991), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.43, y0=-4.59, m=4.16: y0+m*(x-x0), 'deriv': lambda x, m=4.16: np.zeros_like(x,dtype=float)+m, 'domain': (-0.6099694299517493,-0.2500305700482507), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.43, y0=-0.59, m=0.15999999999999998: y0+m*(x-x0), 'deriv': lambda x, m=0.15999999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (-1.190329286575863,0.3303292865758631), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.43, y0=3.41, m=-3.8400000000000003: y0+m*(x-x0), 'deriv': lambda x, m=-3.8400000000000003: np.zeros_like(x,dtype=float)+m, 'domain': (-0.6240488699437083,-0.23595113005629165), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.43, y0=7.41, m=-7.84: y0+m*(x-x0), 'deriv': lambda x, m=-7.84: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5274249669743549,-0.33257503302564506), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.57, y0=-8.59, m=12.16: y0+m*(x-x0), 'deriv': lambda x, m=12.16: np.zeros_like(x,dtype=float)+m, 'domain': (3.5068906729686526,3.633109327031347), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.57, y0=-4.59, m=8.16: y0+m*(x-x0), 'deriv': lambda x, m=8.16: np.zeros_like(x,dtype=float)+m, 'domain': (3.476337954462991,3.663662045537009), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.57, y0=-0.59, m=4.16: y0+m*(x-x0), 'deriv': lambda x, m=4.16: np.zeros_like(x,dtype=float)+m, 'domain': (3.3900305700482507,3.749969429951749), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.57, y0=3.41, m=0.1599999999999997: y0+m*(x-x0), 'deriv': lambda x, m=0.1599999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (2.8096707134241363,4.330329286575863), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.57, y0=7.41, m=-3.8400000000000003: y0+m*(x-x0), 'deriv': lambda x, m=-3.8400000000000003: np.zeros_like(x,dtype=float)+m, 'domain': (3.3759511300562917,3.764048869943708), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.57, y0=-8.59, m=16.16: y0+m*(x-x0), 'deriv': lambda x, m=16.16: np.zeros_like(x,dtype=float)+m, 'domain': (7.522442453760266,7.617557546239735), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.57, y0=-4.59, m=12.16: y0+m*(x-x0), 'deriv': lambda x, m=12.16: np.zeros_like(x,dtype=float)+m, 'domain': (7.506890672968653,7.633109327031348), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.57, y0=-0.59, m=8.16: y0+m*(x-x0), 'deriv': lambda x, m=8.16: np.zeros_like(x,dtype=float)+m, 'domain': (7.476337954462991,7.663662045537009), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.57, y0=3.41, m=4.16: y0+m*(x-x0), 'deriv': lambda x, m=4.16: np.zeros_like(x,dtype=float)+m, 'domain': (7.390030570048251,7.74996942995175), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.57, y0=7.41, m=0.16000000000000014: y0+m*(x-x0), 'deriv': lambda x, m=0.16000000000000014: np.zeros_like(x,dtype=float)+m, 'domain': (6.809670713424137,8.330329286575864), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d4_04_v2_fa_synthesis_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.91, y0=-7.54, m=7.91: y0+m*(x-x0), 'deriv': lambda x, m=7.91: np.zeros_like(x,dtype=float)+m, 'domain': (-7.982745876020812,-7.837254123979188), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.91, y0=-3.54, m=7.91: y0+m*(x-x0), 'deriv': lambda x, m=7.91: np.zeros_like(x,dtype=float)+m, 'domain': (-7.982745876020812,-7.837254123979188), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.91, y0=0.4600000000000001, m=7.91: y0+m*(x-x0), 'deriv': lambda x, m=7.91: np.zeros_like(x,dtype=float)+m, 'domain': (-7.982745876020812,-7.837254123979188), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.91, y0=4.46, m=7.91: y0+m*(x-x0), 'deriv': lambda x, m=7.91: np.zeros_like(x,dtype=float)+m, 'domain': (-7.982745876020812,-7.837254123979188), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.91, y0=8.46, m=7.91: y0+m*(x-x0), 'deriv': lambda x, m=7.91: np.zeros_like(x,dtype=float)+m, 'domain': (-7.982745876020812,-7.837254123979188), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.91, y0=-7.54, m=3.91: y0+m*(x-x0), 'deriv': lambda x, m=3.91: np.zeros_like(x,dtype=float)+m, 'domain': (-4.053711916175101,-3.7662880838248993), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.91, y0=-3.54, m=3.91: y0+m*(x-x0), 'deriv': lambda x, m=3.91: np.zeros_like(x,dtype=float)+m, 'domain': (-4.053711916175101,-3.7662880838248993), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.91, y0=0.4600000000000001, m=3.91: y0+m*(x-x0), 'deriv': lambda x, m=3.91: np.zeros_like(x,dtype=float)+m, 'domain': (-4.053711916175101,-3.7662880838248993), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.91, y0=4.46, m=3.91: y0+m*(x-x0), 'deriv': lambda x, m=3.91: np.zeros_like(x,dtype=float)+m, 'domain': (-4.053711916175101,-3.7662880838248993), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.91, y0=8.46, m=3.91: y0+m*(x-x0), 'deriv': lambda x, m=3.91: np.zeros_like(x,dtype=float)+m, 'domain': (-4.053711916175101,-3.7662880838248993), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.08999999999999997, y0=-7.54, m=-0.08999999999999997: y0+m*(x-x0), 'deriv': lambda x, m=-0.08999999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (-0.48766517452907276,0.6676651745290727), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.08999999999999997, y0=-3.54, m=-0.08999999999999997: y0+m*(x-x0), 'deriv': lambda x, m=-0.08999999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (-0.48766517452907276,0.6676651745290727), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.08999999999999997, y0=0.4600000000000001, m=-0.08999999999999997: y0+m*(x-x0), 'deriv': lambda x, m=-0.08999999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (-0.48766517452907276,0.6676651745290727), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.08999999999999997, y0=4.46, m=-0.08999999999999997: y0+m*(x-x0), 'deriv': lambda x, m=-0.08999999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (-0.48766517452907276,0.6676651745290727), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.08999999999999997, y0=8.46, m=-0.08999999999999997: y0+m*(x-x0), 'deriv': lambda x, m=-0.08999999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (-0.48766517452907276,0.6676651745290727), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.09, y0=-7.54, m=-4.09: y0+m*(x-x0), 'deriv': lambda x, m=-4.09: np.zeros_like(x,dtype=float)+m, 'domain': (3.9522483222173945,4.227751677782606), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.09, y0=-3.54, m=-4.09: y0+m*(x-x0), 'deriv': lambda x, m=-4.09: np.zeros_like(x,dtype=float)+m, 'domain': (3.9522483222173945,4.227751677782606), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.09, y0=0.4600000000000001, m=-4.09: y0+m*(x-x0), 'deriv': lambda x, m=-4.09: np.zeros_like(x,dtype=float)+m, 'domain': (3.9522483222173945,4.227751677782606), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.09, y0=4.46, m=-4.09: y0+m*(x-x0), 'deriv': lambda x, m=-4.09: np.zeros_like(x,dtype=float)+m, 'domain': (3.9522483222173945,4.227751677782606), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.09, y0=8.46, m=-4.09: y0+m*(x-x0), 'deriv': lambda x, m=-4.09: np.zeros_like(x,dtype=float)+m, 'domain': (3.9522483222173945,4.227751677782606), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.09, y0=-7.54, m=-8.09: y0+m*(x-x0), 'deriv': lambda x, m=-8.09: np.zeros_like(x,dtype=float)+m, 'domain': (8.018848065896654,8.161151934103346), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.09, y0=-3.54, m=-8.09: y0+m*(x-x0), 'deriv': lambda x, m=-8.09: np.zeros_like(x,dtype=float)+m, 'domain': (8.018848065896654,8.161151934103346), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.09, y0=0.4600000000000001, m=-8.09: y0+m*(x-x0), 'deriv': lambda x, m=-8.09: np.zeros_like(x,dtype=float)+m, 'domain': (8.018848065896654,8.161151934103346), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.09, y0=4.46, m=-8.09: y0+m*(x-x0), 'deriv': lambda x, m=-8.09: np.zeros_like(x,dtype=float)+m, 'domain': (8.018848065896654,8.161151934103346), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.09, y0=8.46, m=-8.09: y0+m*(x-x0), 'deriv': lambda x, m=-8.09: np.zeros_like(x,dtype=float)+m, 'domain': (8.018848065896654,8.161151934103346), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d4_04_v3_fa_synthesis_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.34, y0=-7.87, m=7.87: y0+m*(x-x0), 'deriv': lambda x, m=7.87: np.zeros_like(x,dtype=float)+m, 'domain': (-7.418151805488151,-7.261848194511849), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.34, y0=-3.87, m=3.87: y0+m*(x-x0), 'deriv': lambda x, m=3.87: np.zeros_like(x,dtype=float)+m, 'domain': (-7.495112011927518,-7.184887988072481), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.34, y0=0.13, m=-0.13: y0+m*(x-x0), 'deriv': lambda x, m=-0.13: np.zeros_like(x,dtype=float)+m, 'domain': (-7.954826482752807,-6.7251735172471925), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.34, y0=4.13, m=-4.13: y0+m*(x-x0), 'deriv': lambda x, m=-4.13: np.zeros_like(x,dtype=float)+m, 'domain': (-7.485904972489566,-7.194095027510434), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.34, y0=8.13, m=-8.13: y0+m*(x-x0), 'deriv': lambda x, m=-8.13: np.zeros_like(x,dtype=float)+m, 'domain': (-7.415690340992834,-7.264309659007166), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.34, y0=-7.87, m=7.87: y0+m*(x-x0), 'deriv': lambda x, m=7.87: np.zeros_like(x,dtype=float)+m, 'domain': (-3.4181518054881503,-3.2618481945118494), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.34, y0=-3.87, m=3.87: y0+m*(x-x0), 'deriv': lambda x, m=3.87: np.zeros_like(x,dtype=float)+m, 'domain': (-3.4951120119275183,-3.1848879880724814), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.34, y0=0.13, m=-0.13: y0+m*(x-x0), 'deriv': lambda x, m=-0.13: np.zeros_like(x,dtype=float)+m, 'domain': (-3.9548264827528072,-2.7251735172471925), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.34, y0=4.13, m=-4.13: y0+m*(x-x0), 'deriv': lambda x, m=-4.13: np.zeros_like(x,dtype=float)+m, 'domain': (-3.4859049724895663,-3.1940950275104334), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.34, y0=8.13, m=-8.13: y0+m*(x-x0), 'deriv': lambda x, m=-8.13: np.zeros_like(x,dtype=float)+m, 'domain': (-3.4156903409928336,-3.264309659007166), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6599999999999999, y0=-7.87, m=7.87: y0+m*(x-x0), 'deriv': lambda x, m=7.87: np.zeros_like(x,dtype=float)+m, 'domain': (0.5818481945118493,0.7381518054881505), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6599999999999999, y0=-3.87, m=3.87: y0+m*(x-x0), 'deriv': lambda x, m=3.87: np.zeros_like(x,dtype=float)+m, 'domain': (0.5048879880724817,0.8151120119275181), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6599999999999999, y0=0.13, m=-0.13: y0+m*(x-x0), 'deriv': lambda x, m=-0.13: np.zeros_like(x,dtype=float)+m, 'domain': (0.04517351724719254,1.2748264827528073), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6599999999999999, y0=4.13, m=-4.13: y0+m*(x-x0), 'deriv': lambda x, m=-4.13: np.zeros_like(x,dtype=float)+m, 'domain': (0.5140950275104335,0.8059049724895664), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6599999999999999, y0=8.13, m=-8.13: y0+m*(x-x0), 'deriv': lambda x, m=-8.13: np.zeros_like(x,dtype=float)+m, 'domain': (0.5843096590071659,0.7356903409928339), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.66, y0=-7.87, m=7.87: y0+m*(x-x0), 'deriv': lambda x, m=7.87: np.zeros_like(x,dtype=float)+m, 'domain': (4.581848194511849,4.738151805488151), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.66, y0=-3.87, m=3.87: y0+m*(x-x0), 'deriv': lambda x, m=3.87: np.zeros_like(x,dtype=float)+m, 'domain': (4.504887988072482,4.815112011927519), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.66, y0=0.13, m=-0.13: y0+m*(x-x0), 'deriv': lambda x, m=-0.13: np.zeros_like(x,dtype=float)+m, 'domain': (4.045173517247193,5.2748264827528075), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.66, y0=4.13, m=-4.13: y0+m*(x-x0), 'deriv': lambda x, m=-4.13: np.zeros_like(x,dtype=float)+m, 'domain': (4.514095027510434,4.805904972489566), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.66, y0=8.13, m=-8.13: y0+m*(x-x0), 'deriv': lambda x, m=-8.13: np.zeros_like(x,dtype=float)+m, 'domain': (4.584309659007166,4.735690340992834), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.66, y0=-7.87, m=7.87: y0+m*(x-x0), 'deriv': lambda x, m=7.87: np.zeros_like(x,dtype=float)+m, 'domain': (8.58184819451185,8.738151805488151), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.66, y0=-3.87, m=3.87: y0+m*(x-x0), 'deriv': lambda x, m=3.87: np.zeros_like(x,dtype=float)+m, 'domain': (8.504887988072483,8.815112011927518), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.66, y0=0.13, m=-0.13: y0+m*(x-x0), 'deriv': lambda x, m=-0.13: np.zeros_like(x,dtype=float)+m, 'domain': (8.045173517247193,9.274826482752808), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.66, y0=4.13, m=-4.13: y0+m*(x-x0), 'deriv': lambda x, m=-4.13: np.zeros_like(x,dtype=float)+m, 'domain': (8.514095027510434,8.805904972489566), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.66, y0=8.13, m=-8.13: y0+m*(x-x0), 'deriv': lambda x, m=-8.13: np.zeros_like(x,dtype=float)+m, 'domain': (8.584309659007166,8.735690340992834), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d4_04_v4_fa_synthesis_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.5, y0=-8.16, m=1.6600000000000001: y0+m*(x-x0), 'deriv': lambda x, m=1.6600000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-7.9437708650638506,-7.0562291349361494), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.5, y0=-4.16, m=-2.34: y0+m*(x-x0), 'deriv': lambda x, m=-2.34: np.zeros_like(x,dtype=float)+m, 'domain': (-7.837954663745081,-7.162045336254919), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.5, y0=-0.15999999999999998, m=-6.34: y0+m*(x-x0), 'deriv': lambda x, m=-6.34: np.zeros_like(x,dtype=float)+m, 'domain': (-7.633990199870232,-7.366009800129768), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.5, y0=3.84, m=-10.34: y0+m*(x-x0), 'deriv': lambda x, m=-10.34: np.zeros_like(x,dtype=float)+m, 'domain': (-7.58278589272717,-7.41721410727283), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.5, y0=7.84, m=-14.34: y0+m*(x-x0), 'deriv': lambda x, m=-14.34: np.zeros_like(x,dtype=float)+m, 'domain': (-7.559826814382626,-7.440173185617374), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.5, y0=-8.16, m=5.66: y0+m*(x-x0), 'deriv': lambda x, m=5.66: np.zeros_like(x,dtype=float)+m, 'domain': (-3.649626098466024,-3.350373901533976), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.5, y0=-4.16, m=1.6600000000000001: y0+m*(x-x0), 'deriv': lambda x, m=1.6600000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-3.943770865063851,-3.056229134936149), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.5, y0=-0.15999999999999998, m=-2.34: y0+m*(x-x0), 'deriv': lambda x, m=-2.34: np.zeros_like(x,dtype=float)+m, 'domain': (-3.8379546637450805,-3.1620453362549195), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.5, y0=3.84, m=-6.34: y0+m*(x-x0), 'deriv': lambda x, m=-6.34: np.zeros_like(x,dtype=float)+m, 'domain': (-3.633990199870232,-3.366009800129768), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.5, y0=7.84, m=-10.34: y0+m*(x-x0), 'deriv': lambda x, m=-10.34: np.zeros_like(x,dtype=float)+m, 'domain': (-3.58278589272717,-3.41721410727283), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5, y0=-8.16, m=9.66: y0+m*(x-x0), 'deriv': lambda x, m=9.66: np.zeros_like(x,dtype=float)+m, 'domain': (0.411446305413281,0.588553694586719), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5, y0=-4.16, m=5.66: y0+m*(x-x0), 'deriv': lambda x, m=5.66: np.zeros_like(x,dtype=float)+m, 'domain': (0.3503739015339763,0.6496260984660237), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5, y0=-0.15999999999999998, m=1.66: y0+m*(x-x0), 'deriv': lambda x, m=1.66: np.zeros_like(x,dtype=float)+m, 'domain': (0.056229134936149006,0.943770865063851), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5, y0=3.84, m=-2.34: y0+m*(x-x0), 'deriv': lambda x, m=-2.34: np.zeros_like(x,dtype=float)+m, 'domain': (0.16204533625491935,0.8379546637450807), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5, y0=7.84, m=-6.34: y0+m*(x-x0), 'deriv': lambda x, m=-6.34: np.zeros_like(x,dtype=float)+m, 'domain': (0.3660098001297677,0.6339901998702323), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.5, y0=-8.16, m=13.66: y0+m*(x-x0), 'deriv': lambda x, m=13.66: np.zeros_like(x,dtype=float)+m, 'domain': (4.437210485173197,4.562789514826803), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.5, y0=-4.16, m=9.66: y0+m*(x-x0), 'deriv': lambda x, m=9.66: np.zeros_like(x,dtype=float)+m, 'domain': (4.411446305413281,4.588553694586719), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.5, y0=-0.15999999999999998, m=5.66: y0+m*(x-x0), 'deriv': lambda x, m=5.66: np.zeros_like(x,dtype=float)+m, 'domain': (4.350373901533977,4.649626098466023), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.5, y0=3.84, m=1.6600000000000001: y0+m*(x-x0), 'deriv': lambda x, m=1.6600000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (4.0562291349361494,4.9437708650638506), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.5, y0=7.84, m=-2.34: y0+m*(x-x0), 'deriv': lambda x, m=-2.34: np.zeros_like(x,dtype=float)+m, 'domain': (4.162045336254919,4.837954663745081), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.5, y0=-8.16, m=17.66: y0+m*(x-x0), 'deriv': lambda x, m=17.66: np.zeros_like(x,dtype=float)+m, 'domain': (8.451380263197088,8.548619736802912), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.5, y0=-4.16, m=13.66: y0+m*(x-x0), 'deriv': lambda x, m=13.66: np.zeros_like(x,dtype=float)+m, 'domain': (8.437210485173198,8.562789514826802), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.5, y0=-0.15999999999999998, m=9.66: y0+m*(x-x0), 'deriv': lambda x, m=9.66: np.zeros_like(x,dtype=float)+m, 'domain': (8.41144630541328,8.58855369458672), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.5, y0=3.84, m=5.66: y0+m*(x-x0), 'deriv': lambda x, m=5.66: np.zeros_like(x,dtype=float)+m, 'domain': (8.350373901533976,8.649626098466024), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.5, y0=7.84, m=1.6600000000000001: y0+m*(x-x0), 'deriv': lambda x, m=1.6600000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (8.05622913493615,8.94377086506385), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d4_04_v5_fa_synthesis_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.79, y0=-8.58, m=-0.20999999999999996: y0+m*(x-x0), 'deriv': lambda x, m=-0.20999999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (-8.641428542043714,-6.938571457956286), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.79, y0=-4.58, m=-4.21: y0+m*(x-x0), 'deriv': lambda x, m=-4.21: np.zeros_like(x,dtype=float)+m, 'domain': (-7.991056804898492,-7.588943195101508), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.79, y0=-0.58, m=-8.21: y0+m*(x-x0), 'deriv': lambda x, m=-8.21: np.zeros_like(x,dtype=float)+m, 'domain': (-7.895190903661842,-7.684809096338158), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.79, y0=3.42, m=-12.21: y0+m*(x-x0), 'deriv': lambda x, m=-12.21: np.zeros_like(x,dtype=float)+m, 'domain': (-7.861015297375331,-7.718984702624669), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.79, y0=7.42, m=-16.21: y0+m*(x-x0), 'deriv': lambda x, m=-16.21: np.zeros_like(x,dtype=float)+m, 'domain': (-7.843568737381618,-7.736431262618382), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.79, y0=-8.58, m=3.79: y0+m*(x-x0), 'deriv': lambda x, m=3.79: np.zeros_like(x,dtype=float)+m, 'domain': (-4.0119553869642175,-3.5680446130357826), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.79, y0=-4.58, m=-0.20999999999999996: y0+m*(x-x0), 'deriv': lambda x, m=-0.20999999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (-4.641428542043714,-2.938571457956286), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.79, y0=-0.58, m=-4.21: y0+m*(x-x0), 'deriv': lambda x, m=-4.21: np.zeros_like(x,dtype=float)+m, 'domain': (-3.991056804898492,-3.5889431951015083), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.79, y0=3.42, m=-8.21: y0+m*(x-x0), 'deriv': lambda x, m=-8.21: np.zeros_like(x,dtype=float)+m, 'domain': (-3.8951909036618417,-3.6848090963381583), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.79, y0=7.42, m=-12.21: y0+m*(x-x0), 'deriv': lambda x, m=-12.21: np.zeros_like(x,dtype=float)+m, 'domain': (-3.861015297375331,-3.718984702624669), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.20999999999999996, y0=-8.58, m=7.789999999999999: y0+m*(x-x0), 'deriv': lambda x, m=7.789999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (0.09922732680631444,0.3207726731936855), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.20999999999999996, y0=-4.58, m=3.79: y0+m*(x-x0), 'deriv': lambda x, m=3.79: np.zeros_like(x,dtype=float)+m, 'domain': (-0.011955386964217551,0.4319553869642175), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.20999999999999996, y0=-0.58, m=-0.21000000000000008: y0+m*(x-x0), 'deriv': lambda x, m=-0.21000000000000008: np.zeros_like(x,dtype=float)+m, 'domain': (-0.6414285420437144,1.0614285420437142), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.20999999999999996, y0=3.42, m=-4.21: y0+m*(x-x0), 'deriv': lambda x, m=-4.21: np.zeros_like(x,dtype=float)+m, 'domain': (0.008943195101508178,0.4110568048984917), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.20999999999999996, y0=7.42, m=-8.21: y0+m*(x-x0), 'deriv': lambda x, m=-8.21: np.zeros_like(x,dtype=float)+m, 'domain': (0.10480909633815846,0.31519090366184144), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.21, y0=-8.58, m=11.79: y0+m*(x-x0), 'deriv': lambda x, m=11.79: np.zeros_like(x,dtype=float)+m, 'domain': (4.136472656481124,4.283527343518876), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.21, y0=-4.58, m=7.789999999999999: y0+m*(x-x0), 'deriv': lambda x, m=7.789999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (4.099227326806314,4.3207726731936855), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.21, y0=-0.58, m=3.79: y0+m*(x-x0), 'deriv': lambda x, m=3.79: np.zeros_like(x,dtype=float)+m, 'domain': (3.9880446130357825,4.431955386964217), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.21, y0=3.42, m=-0.20999999999999996: y0+m*(x-x0), 'deriv': lambda x, m=-0.20999999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (3.3585714579562858,5.061428542043714), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.21, y0=7.42, m=-4.21: y0+m*(x-x0), 'deriv': lambda x, m=-4.21: np.zeros_like(x,dtype=float)+m, 'domain': (4.008943195101508,4.411056804898492), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.21, y0=-8.58, m=15.79: y0+m*(x-x0), 'deriv': lambda x, m=15.79: np.zeros_like(x,dtype=float)+m, 'domain': (8.155012000389933,8.264987999610069), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.21, y0=-4.58, m=11.790000000000001: y0+m*(x-x0), 'deriv': lambda x, m=11.790000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (8.136472656481125,8.283527343518877), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.21, y0=-0.58, m=7.790000000000001: y0+m*(x-x0), 'deriv': lambda x, m=7.790000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (8.099227326806316,8.320772673193686), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.21, y0=3.42, m=3.790000000000001: y0+m*(x-x0), 'deriv': lambda x, m=3.790000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (7.988044613035783,8.43195538696422), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.21, y0=7.42, m=-0.20999999999999908: y0+m*(x-x0), 'deriv': lambda x, m=-0.20999999999999908: np.zeros_like(x,dtype=float)+m, 'domain': (7.358571457956287,9.061428542043716), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d4_04_v6_fa_synthesis_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.48, y0=-7.71, m=-1.7700000000000005: y0+m*(x-x0), 'deriv': lambda x, m=-1.7700000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-8.903029839613149,-8.056970160386852), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.48, y0=-3.71, m=-5.7700000000000005: y0+m*(x-x0), 'deriv': lambda x, m=-5.7700000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-8.626857572335245,-8.333142427664756), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.48, y0=0.29000000000000004, m=-9.77: y0+m*(x-x0), 'deriv': lambda x, m=-9.77: np.zeros_like(x,dtype=float)+m, 'domain': (-8.567567067554624,-8.392432932445377), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.48, y0=4.29, m=-13.77: y0+m*(x-x0), 'deriv': lambda x, m=-13.77: np.zeros_like(x,dtype=float)+m, 'domain': (-8.542290570268365,-8.417709429731635), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.48, y0=8.29, m=-17.77: y0+m*(x-x0), 'deriv': lambda x, m=-17.77: np.zeros_like(x,dtype=float)+m, 'domain': (-8.528319723559415,-8.431680276440586), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.48, y0=-7.71, m=2.2299999999999995: y0+m*(x-x0), 'deriv': lambda x, m=2.2299999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (-4.831889112072014,-4.128110887927987), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.48, y0=-3.71, m=-1.7700000000000005: y0+m*(x-x0), 'deriv': lambda x, m=-1.7700000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-4.903029839613149,-4.056970160386852), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.48, y0=0.29000000000000004, m=-5.7700000000000005: y0+m*(x-x0), 'deriv': lambda x, m=-5.7700000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-4.626857572335245,-4.333142427664756), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.48, y0=4.29, m=-9.77: y0+m*(x-x0), 'deriv': lambda x, m=-9.77: np.zeros_like(x,dtype=float)+m, 'domain': (-4.567567067554623,-4.392432932445378), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.48, y0=8.29, m=-13.77: y0+m*(x-x0), 'deriv': lambda x, m=-13.77: np.zeros_like(x,dtype=float)+m, 'domain': (-4.5422905702683645,-4.417709429731636), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.48, y0=-7.71, m=6.23: y0+m*(x-x0), 'deriv': lambda x, m=6.23: np.zeros_like(x,dtype=float)+m, 'domain': (-0.6162970778078797,-0.3437029221921203), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.48, y0=-3.71, m=2.23: y0+m*(x-x0), 'deriv': lambda x, m=2.23: np.zeros_like(x,dtype=float)+m, 'domain': (-0.8318891120720133,-0.12811088792798664), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.48, y0=0.29000000000000004, m=-1.77: y0+m*(x-x0), 'deriv': lambda x, m=-1.77: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9030298396131485,-0.05697016038685143), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.48, y0=4.29, m=-5.77: y0+m*(x-x0), 'deriv': lambda x, m=-5.77: np.zeros_like(x,dtype=float)+m, 'domain': (-0.6268575723352452,-0.33314242766475477), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.48, y0=8.29, m=-9.77: y0+m*(x-x0), 'deriv': lambda x, m=-9.77: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5675670675546225,-0.3924329324453774), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.52, y0=-7.71, m=10.23: y0+m*(x-x0), 'deriv': lambda x, m=10.23: np.zeros_like(x,dtype=float)+m, 'domain': (3.4363323174147915,3.6036676825852085), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.52, y0=-3.71, m=6.23: y0+m*(x-x0), 'deriv': lambda x, m=6.23: np.zeros_like(x,dtype=float)+m, 'domain': (3.3837029221921204,3.6562970778078796), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.52, y0=0.29000000000000004, m=2.23: y0+m*(x-x0), 'deriv': lambda x, m=2.23: np.zeros_like(x,dtype=float)+m, 'domain': (3.1681108879279867,3.8718891120720134), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.52, y0=4.29, m=-1.77: y0+m*(x-x0), 'deriv': lambda x, m=-1.77: np.zeros_like(x,dtype=float)+m, 'domain': (3.0969701603868516,3.9430298396131485), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.52, y0=8.29, m=-5.77: y0+m*(x-x0), 'deriv': lambda x, m=-5.77: np.zeros_like(x,dtype=float)+m, 'domain': (3.373142427664755,3.6668575723352452), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.52, y0=-7.71, m=14.23: y0+m*(x-x0), 'deriv': lambda x, m=14.23: np.zeros_like(x,dtype=float)+m, 'domain': (7.459712979498769,7.58028702050123), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.52, y0=-3.71, m=10.23: y0+m*(x-x0), 'deriv': lambda x, m=10.23: np.zeros_like(x,dtype=float)+m, 'domain': (7.436332317414791,7.6036676825852085), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.52, y0=0.29000000000000004, m=6.2299999999999995: y0+m*(x-x0), 'deriv': lambda x, m=6.2299999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (7.3837029221921195,7.65629707780788), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.52, y0=4.29, m=2.2299999999999995: y0+m*(x-x0), 'deriv': lambda x, m=2.2299999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (7.168110887927986,7.871889112072013), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.52, y0=8.29, m=-1.7699999999999996: y0+m*(x-x0), 'deriv': lambda x, m=-1.7699999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (7.096970160386851,7.943029839613148), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d4_04_v7_fa_synthesis_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.03, y0=-7.7, m=-16.73: y0+m*(x-x0), 'deriv': lambda x, m=-16.73: np.zeros_like(x,dtype=float)+m, 'domain': (-8.083103069562485,-7.976896930437513), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.03, y0=-3.7, m=-12.73: y0+m*(x-x0), 'deriv': lambda x, m=-12.73: np.zeros_like(x,dtype=float)+m, 'domain': (-8.099698870934906,-7.960301129065091), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.03, y0=0.30000000000000004, m=-8.73: y0+m*(x-x0), 'deriv': lambda x, m=-8.73: np.zeros_like(x,dtype=float)+m, 'domain': (-8.131284986498907,-7.928715013501091), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.03, y0=4.3, m=-4.7299999999999995: y0+m*(x-x0), 'deriv': lambda x, m=-4.7299999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (-8.214091487174189,-7.845908512825809), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.03, y0=8.3, m=-0.7299999999999986: y0+m*(x-x0), 'deriv': lambda x, m=-0.7299999999999986: np.zeros_like(x,dtype=float)+m, 'domain': (-8.748841421843494,-7.311158578156506), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.03, y0=-7.7, m=-12.73: y0+m*(x-x0), 'deriv': lambda x, m=-12.73: np.zeros_like(x,dtype=float)+m, 'domain': (-4.099698870934908,-3.9603011290650922), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.03, y0=-3.7, m=-8.73: y0+m*(x-x0), 'deriv': lambda x, m=-8.73: np.zeros_like(x,dtype=float)+m, 'domain': (-4.131284986498908,-3.9287150135010926), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.03, y0=0.30000000000000004, m=-4.73: y0+m*(x-x0), 'deriv': lambda x, m=-4.73: np.zeros_like(x,dtype=float)+m, 'domain': (-4.214091487174191,-3.8459085128258104), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.03, y0=4.3, m=-0.7300000000000004: y0+m*(x-x0), 'deriv': lambda x, m=-0.7300000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-4.748841421843493,-3.3111585781565074), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.03, y0=8.3, m=3.2700000000000005: y0+m*(x-x0), 'deriv': lambda x, m=3.2700000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-4.290272862726039,-3.769727137273961), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.030000000000000027, y0=-7.7, m=-8.73: y0+m*(x-x0), 'deriv': lambda x, m=-8.73: np.zeros_like(x,dtype=float)+m, 'domain': (-0.13128498649890769,0.07128498649890765), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.030000000000000027, y0=-3.7, m=-4.73: y0+m*(x-x0), 'deriv': lambda x, m=-4.73: np.zeros_like(x,dtype=float)+m, 'domain': (-0.2140914871741899,0.15409148717418986), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.030000000000000027, y0=0.30000000000000004, m=-0.73: y0+m*(x-x0), 'deriv': lambda x, m=-0.73: np.zeros_like(x,dtype=float)+m, 'domain': (-0.7488414218434933,0.6888414218434933), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.030000000000000027, y0=4.3, m=3.2699999999999996: y0+m*(x-x0), 'deriv': lambda x, m=3.2699999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (-0.2902728627260391,0.23027286272603903), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.030000000000000027, y0=8.3, m=7.270000000000001: y0+m*(x-x0), 'deriv': lambda x, m=7.270000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-0.15127895858619095,0.0912789585861909), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.9699999999999998, y0=-7.7, m=-4.73: y0+m*(x-x0), 'deriv': lambda x, m=-4.73: np.zeros_like(x,dtype=float)+m, 'domain': (3.78590851282581,4.15409148717419), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.9699999999999998, y0=-3.7, m=-0.7300000000000004: y0+m*(x-x0), 'deriv': lambda x, m=-0.7300000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (3.251158578156507,4.688841421843493), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.9699999999999998, y0=0.30000000000000004, m=3.2699999999999996: y0+m*(x-x0), 'deriv': lambda x, m=3.2699999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (3.7097271372739606,4.230272862726038), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.9699999999999998, y0=4.3, m=7.27: y0+m*(x-x0), 'deriv': lambda x, m=7.27: np.zeros_like(x,dtype=float)+m, 'domain': (3.8487210414138087,4.091278958586191), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.9699999999999998, y0=8.3, m=11.27: y0+m*(x-x0), 'deriv': lambda x, m=11.27: np.zeros_like(x,dtype=float)+m, 'domain': (3.89133833430363,4.048661665696369), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.97, y0=-7.7, m=-0.7300000000000004: y0+m*(x-x0), 'deriv': lambda x, m=-0.7300000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (7.251158578156507,8.688841421843494), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.97, y0=-3.7, m=3.2699999999999996: y0+m*(x-x0), 'deriv': lambda x, m=3.2699999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (7.709727137273961,8.23027286272604), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.97, y0=0.30000000000000004, m=7.27: y0+m*(x-x0), 'deriv': lambda x, m=7.27: np.zeros_like(x,dtype=float)+m, 'domain': (7.848721041413809,8.09127895858619), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.97, y0=4.3, m=11.27: y0+m*(x-x0), 'deriv': lambda x, m=11.27: np.zeros_like(x,dtype=float)+m, 'domain': (7.89133833430363,8.04866166569637), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.97, y0=8.3, m=15.27: y0+m*(x-x0), 'deriv': lambda x, m=15.27: np.zeros_like(x,dtype=float)+m, 'domain': (7.9118403626962825,8.028159637303718), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_fa_d4_04_v8_fa_synthesis_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.96, y0=-8.15, m=-17.11: y0+m*(x-x0), 'deriv': lambda x, m=-17.11: np.zeros_like(x,dtype=float)+m, 'domain': (-8.00375934103328,-7.91624065896672), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.96, y0=-4.15, m=-13.11: y0+m*(x-x0), 'deriv': lambda x, m=-13.11: np.zeros_like(x,dtype=float)+m, 'domain': (-8.017042533947937,-7.9029574660520625), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.96, y0=-0.14999999999999997, m=-9.11: y0+m*(x-x0), 'deriv': lambda x, m=-9.11: np.zeros_like(x,dtype=float)+m, 'domain': (-8.041835557148632,-7.878164442851368), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.96, y0=3.85, m=-5.109999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-5.109999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-8.104038863010182,-7.815961136989818), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.96, y0=7.85, m=-1.1100000000000003: y0+m*(x-x0), 'deriv': lambda x, m=-1.1100000000000003: np.zeros_like(x,dtype=float)+m, 'domain': (-8.462000835151596,-7.457999164848405), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.96, y0=-8.15, m=-13.11: y0+m*(x-x0), 'deriv': lambda x, m=-13.11: np.zeros_like(x,dtype=float)+m, 'domain': (-4.017042533947937,-3.902957466052063), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.96, y0=-4.15, m=-9.11: y0+m*(x-x0), 'deriv': lambda x, m=-9.11: np.zeros_like(x,dtype=float)+m, 'domain': (-4.041835557148632,-3.878164442851368), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.96, y0=-0.14999999999999997, m=-5.11: y0+m*(x-x0), 'deriv': lambda x, m=-5.11: np.zeros_like(x,dtype=float)+m, 'domain': (-4.104038863010182,-3.8159611369898183), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.96, y0=3.85, m=-1.1099999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-1.1099999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-4.462000835151596,-3.4579991648484047), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.96, y0=7.85, m=2.8899999999999997: y0+m*(x-x0), 'deriv': lambda x, m=2.8899999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (-4.205248687812668,-3.7147513121873312), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.040000000000000036, y0=-8.15, m=-9.11: y0+m*(x-x0), 'deriv': lambda x, m=-9.11: np.zeros_like(x,dtype=float)+m, 'domain': (-0.04183555714863198,0.12183555714863205), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.040000000000000036, y0=-4.15, m=-5.11: y0+m*(x-x0), 'deriv': lambda x, m=-5.11: np.zeros_like(x,dtype=float)+m, 'domain': (-0.10403886301018184,0.1840388630101819), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.040000000000000036, y0=-0.14999999999999997, m=-1.1099999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-1.1099999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-0.4620008351515952,0.5420008351515953), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.040000000000000036, y0=3.85, m=2.89: y0+m*(x-x0), 'deriv': lambda x, m=2.89: np.zeros_like(x,dtype=float)+m, 'domain': (-0.20524868781266864,0.2852486878126687), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.040000000000000036, y0=7.85, m=6.89: y0+m*(x-x0), 'deriv': lambda x, m=6.89: np.zeros_like(x,dtype=float)+m, 'domain': (-0.06772471298702963,0.14772471298702972), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.04, y0=-8.15, m=-5.11: y0+m*(x-x0), 'deriv': lambda x, m=-5.11: np.zeros_like(x,dtype=float)+m, 'domain': (3.8959611369898184,4.184038863010182), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.04, y0=-4.15, m=-1.1100000000000003: y0+m*(x-x0), 'deriv': lambda x, m=-1.1100000000000003: np.zeros_like(x,dtype=float)+m, 'domain': (3.537999164848405,4.542000835151595), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.04, y0=-0.14999999999999997, m=2.89: y0+m*(x-x0), 'deriv': lambda x, m=2.89: np.zeros_like(x,dtype=float)+m, 'domain': (3.7947513121873313,4.285248687812668), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.04, y0=3.85, m=6.890000000000001: y0+m*(x-x0), 'deriv': lambda x, m=6.890000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (3.9322752870129705,4.14772471298703), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.04, y0=7.85, m=10.89: y0+m*(x-x0), 'deriv': lambda x, m=10.89: np.zeros_like(x,dtype=float)+m, 'domain': (3.971418020301097,4.108581979698903), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.04, y0=-8.15, m=-1.1100000000000012: y0+m*(x-x0), 'deriv': lambda x, m=-1.1100000000000012: np.zeros_like(x,dtype=float)+m, 'domain': (7.537999164848404,8.542000835151594), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.04, y0=-4.15, m=2.889999999999999: y0+m*(x-x0), 'deriv': lambda x, m=2.889999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (7.79475131218733,8.285248687812668), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.04, y0=-0.14999999999999997, m=6.889999999999999: y0+m*(x-x0), 'deriv': lambda x, m=6.889999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (7.93227528701297,8.14772471298703), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.04, y0=3.85, m=10.889999999999999: y0+m*(x-x0), 'deriv': lambda x, m=10.889999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (7.971418020301096,8.108581979698902), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.04, y0=7.85, m=14.889999999999999: y0+m*(x-x0), 'deriv': lambda x, m=14.889999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (7.989743833614743,8.090256166385254), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s2_cyu_c02_solution_tendency.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.37, y0=-8.19, m=8.19: y0+m*(x-x0), 'deriv': lambda x, m=8.19: np.zeros_like(x,dtype=float)+m, 'domain': (-8.449992008543786,-8.290007991456212), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.37, y0=-4.19, m=4.19: y0+m*(x-x0), 'deriv': lambda x, m=4.19: np.zeros_like(x,dtype=float)+m, 'domain': (-8.52321475229691,-8.216785247703088), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.37, y0=-0.19, m=0.19: y0+m*(x-x0), 'deriv': lambda x, m=0.19: np.zeros_like(x,dtype=float)+m, 'domain': (-9.018400138122939,-7.721599861877061), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.37, y0=3.81, m=-3.81: y0+m*(x-x0), 'deriv': lambda x, m=-3.81: np.zeros_like(x,dtype=float)+m, 'domain': (-8.537553170675602,-8.202446829324396), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.37, y0=7.81, m=-7.81: y0+m*(x-x0), 'deriv': lambda x, m=-7.81: np.zeros_like(x,dtype=float)+m, 'domain': (-8.453822720221334,-8.286177279778665), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.37, y0=-8.19, m=8.19: y0+m*(x-x0), 'deriv': lambda x, m=8.19: np.zeros_like(x,dtype=float)+m, 'domain': (-4.449992008543787,-4.290007991456213), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.37, y0=-4.19, m=4.19: y0+m*(x-x0), 'deriv': lambda x, m=4.19: np.zeros_like(x,dtype=float)+m, 'domain': (-4.523214752296911,-4.216785247703089), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.37, y0=-0.19, m=0.19: y0+m*(x-x0), 'deriv': lambda x, m=0.19: np.zeros_like(x,dtype=float)+m, 'domain': (-5.018400138122939,-3.7215998618770616), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.37, y0=3.81, m=-3.81: y0+m*(x-x0), 'deriv': lambda x, m=-3.81: np.zeros_like(x,dtype=float)+m, 'domain': (-4.537553170675604,-4.202446829324396), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.37, y0=7.81, m=-7.81: y0+m*(x-x0), 'deriv': lambda x, m=-7.81: np.zeros_like(x,dtype=float)+m, 'domain': (-4.453822720221335,-4.2861772797786655), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.37, y0=-8.19, m=8.19: y0+m*(x-x0), 'deriv': lambda x, m=8.19: np.zeros_like(x,dtype=float)+m, 'domain': (-0.4499920085437869,-0.2900079914562131), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.37, y0=-4.19, m=4.19: y0+m*(x-x0), 'deriv': lambda x, m=4.19: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5232147522969108,-0.21678524770308916), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.37, y0=-0.19, m=0.19: y0+m*(x-x0), 'deriv': lambda x, m=0.19: np.zeros_like(x,dtype=float)+m, 'domain': (-1.0184001381229386,0.2784001381229386), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.37, y0=3.81, m=-3.81: y0+m*(x-x0), 'deriv': lambda x, m=-3.81: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5375531706756038,-0.20244682932439614), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.37, y0=7.81, m=-7.81: y0+m*(x-x0), 'deriv': lambda x, m=-7.81: np.zeros_like(x,dtype=float)+m, 'domain': (-0.45382272022133485,-0.28617727977866514), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.63, y0=-8.19, m=8.19: y0+m*(x-x0), 'deriv': lambda x, m=8.19: np.zeros_like(x,dtype=float)+m, 'domain': (3.550007991456213,3.7099920085437867), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.63, y0=-4.19, m=4.19: y0+m*(x-x0), 'deriv': lambda x, m=4.19: np.zeros_like(x,dtype=float)+m, 'domain': (3.476785247703089,3.7832147522969106), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.63, y0=-0.19, m=0.19: y0+m*(x-x0), 'deriv': lambda x, m=0.19: np.zeros_like(x,dtype=float)+m, 'domain': (2.9815998618770614,4.278400138122938), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.63, y0=3.81, m=-3.81: y0+m*(x-x0), 'deriv': lambda x, m=-3.81: np.zeros_like(x,dtype=float)+m, 'domain': (3.462446829324396,3.7975531706756036), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.63, y0=7.81, m=-7.81: y0+m*(x-x0), 'deriv': lambda x, m=-7.81: np.zeros_like(x,dtype=float)+m, 'domain': (3.5461772797786653,3.7138227202213345), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.63, y0=-8.19, m=8.19: y0+m*(x-x0), 'deriv': lambda x, m=8.19: np.zeros_like(x,dtype=float)+m, 'domain': (7.550007991456213,7.709992008543787), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.63, y0=-4.19, m=4.19: y0+m*(x-x0), 'deriv': lambda x, m=4.19: np.zeros_like(x,dtype=float)+m, 'domain': (7.476785247703089,7.783214752296911), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.63, y0=-0.19, m=0.19: y0+m*(x-x0), 'deriv': lambda x, m=0.19: np.zeros_like(x,dtype=float)+m, 'domain': (6.981599861877061,8.278400138122938), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.63, y0=3.81, m=-3.81: y0+m*(x-x0), 'deriv': lambda x, m=-3.81: np.zeros_like(x,dtype=float)+m, 'domain': (7.462446829324396,7.797553170675604), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.63, y0=7.81, m=-7.81: y0+m*(x-x0), 'deriv': lambda x, m=-7.81: np.zeros_like(x,dtype=float)+m, 'domain': (7.546177279778665,7.7138227202213345), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s2_et_b_c02_slope_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.35, y0=-7.57, m=0.22000000000000064: y0+m*(x-x0), 'deriv': lambda x, m=0.22000000000000064: np.zeros_like(x,dtype=float)+m, 'domain': (-7.984818903358308,-6.715181096641691), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.35, y0=-3.57, m=-3.78: y0+m*(x-x0), 'deriv': lambda x, m=-3.78: np.zeros_like(x,dtype=float)+m, 'domain': (-7.516238778154617,-7.183761221845383), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.35, y0=0.43000000000000005, m=-7.779999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-7.779999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-7.432865841655359,-7.26713415834464), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.35, y0=4.43, m=-11.78: y0+m*(x-x0), 'deriv': lambda x, m=-11.78: np.zeros_like(x,dtype=float)+m, 'domain': (-7.4049805221269684,-7.295019477873031), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.35, y0=8.43, m=-15.78: y0+m*(x-x0), 'deriv': lambda x, m=-15.78: np.zeros_like(x,dtype=float)+m, 'domain': (-7.391108918958151,-7.308891081041848), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.35, y0=-7.57, m=4.220000000000001: y0+m*(x-x0), 'deriv': lambda x, m=4.220000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-3.4998778416964065,-3.2001221583035937), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.35, y0=-3.57, m=0.21999999999999975: y0+m*(x-x0), 'deriv': lambda x, m=0.21999999999999975: np.zeros_like(x,dtype=float)+m, 'domain': (-3.984818903358309,-2.7151810966416914), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.35, y0=0.43000000000000005, m=-3.7800000000000002: y0+m*(x-x0), 'deriv': lambda x, m=-3.7800000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (-3.5162387781546176,-3.1837612218453826), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.35, y0=4.43, m=-7.779999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-7.779999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-3.4328658416553597,-3.2671341583446405), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.35, y0=8.43, m=-11.78: y0+m*(x-x0), 'deriv': lambda x, m=-11.78: np.zeros_like(x,dtype=float)+m, 'domain': (-3.404980522126969,-3.2950194778730313), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6499999999999999, y0=-7.57, m=8.22: y0+m*(x-x0), 'deriv': lambda x, m=8.22: np.zeros_like(x,dtype=float)+m, 'domain': (0.5715033091370765,0.7284966908629233), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6499999999999999, y0=-3.57, m=4.22: y0+m*(x-x0), 'deriv': lambda x, m=4.22: np.zeros_like(x,dtype=float)+m, 'domain': (0.5001221583035933,0.7998778416964065), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6499999999999999, y0=0.43000000000000005, m=0.21999999999999986: y0+m*(x-x0), 'deriv': lambda x, m=0.21999999999999986: np.zeros_like(x,dtype=float)+m, 'domain': (0.015181096641691427,1.2848189033583084), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6499999999999999, y0=4.43, m=-3.78: y0+m*(x-x0), 'deriv': lambda x, m=-3.78: np.zeros_like(x,dtype=float)+m, 'domain': (0.4837612218453826,0.8162387781546172), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.6499999999999999, y0=8.43, m=-7.779999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-7.779999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (0.5671341583446404,0.7328658416553594), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.65, y0=-7.57, m=12.22: y0+m*(x-x0), 'deriv': lambda x, m=12.22: np.zeros_like(x,dtype=float)+m, 'domain': (4.596985723472419,4.703014276527582), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.65, y0=-3.57, m=8.22: y0+m*(x-x0), 'deriv': lambda x, m=8.22: np.zeros_like(x,dtype=float)+m, 'domain': (4.571503309137077,4.728496690862924), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.65, y0=0.43000000000000005, m=4.220000000000001: y0+m*(x-x0), 'deriv': lambda x, m=4.220000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (4.5001221583035935,4.799877841696407), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.65, y0=4.43, m=0.22000000000000064: y0+m*(x-x0), 'deriv': lambda x, m=0.22000000000000064: np.zeros_like(x,dtype=float)+m, 'domain': (4.015181096641692,5.284818903358309), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.65, y0=8.43, m=-3.7799999999999994: y0+m*(x-x0), 'deriv': lambda x, m=-3.7799999999999994: np.zeros_like(x,dtype=float)+m, 'domain': (4.483761221845383,4.816238778154617), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.65, y0=-7.57, m=16.22: y0+m*(x-x0), 'deriv': lambda x, m=16.22: np.zeros_like(x,dtype=float)+m, 'domain': (8.610001961512513,8.689998038487488), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.65, y0=-3.57, m=12.22: y0+m*(x-x0), 'deriv': lambda x, m=12.22: np.zeros_like(x,dtype=float)+m, 'domain': (8.59698572347242,8.70301427652758), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.65, y0=0.43000000000000005, m=8.22: y0+m*(x-x0), 'deriv': lambda x, m=8.22: np.zeros_like(x,dtype=float)+m, 'domain': (8.571503309137077,8.728496690862924), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.65, y0=4.43, m=4.220000000000001: y0+m*(x-x0), 'deriv': lambda x, m=4.220000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (8.500122158303594,8.799877841696407), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.65, y0=8.43, m=0.22000000000000064: y0+m*(x-x0), 'deriv': lambda x, m=0.22000000000000064: np.zeros_like(x,dtype=float)+m, 'domain': (8.015181096641692,9.284818903358309), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s2_et_c_c01_slope_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.59, y0=-8.11, m=0.5199999999999996: y0+m*(x-x0), 'deriv': lambda x, m=0.5199999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (-8.15781875279014,-7.022181247209859), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.59, y0=-4.11, m=-3.4799999999999995: y0+m*(x-x0), 'deriv': lambda x, m=-3.4799999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (-7.766755120060663,-7.413244879939337), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.59, y0=-0.10999999999999999, m=-7.4799999999999995: y0+m*(x-x0), 'deriv': lambda x, m=-7.4799999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (-7.674806977357052,-7.505193022642947), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.59, y0=3.89, m=-11.48: y0+m*(x-x0), 'deriv': lambda x, m=-11.48: np.zeros_like(x,dtype=float)+m, 'domain': (-7.6455388182234945,-7.534461181776505), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.59, y0=7.89, m=-15.48: y0+m*(x-x0), 'deriv': lambda x, m=-15.48: np.zeros_like(x,dtype=float)+m, 'domain': (-7.631257672817458,-7.548742327182541), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.59, y0=-8.11, m=4.52: y0+m*(x-x0), 'deriv': lambda x, m=4.52: np.zeros_like(x,dtype=float)+m, 'domain': (-3.728249903029338,-3.451750096970662), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.59, y0=-4.11, m=0.5200000000000005: y0+m*(x-x0), 'deriv': lambda x, m=0.5200000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-4.15781875279014,-3.022181247209859), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.59, y0=-0.10999999999999999, m=-3.48: y0+m*(x-x0), 'deriv': lambda x, m=-3.48: np.zeros_like(x,dtype=float)+m, 'domain': (-3.7667551200606626,-3.413244879939337), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.59, y0=3.89, m=-7.48: y0+m*(x-x0), 'deriv': lambda x, m=-7.48: np.zeros_like(x,dtype=float)+m, 'domain': (-3.6748069773570524,-3.5051930226429473), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.59, y0=7.89, m=-11.48: y0+m*(x-x0), 'deriv': lambda x, m=-11.48: np.zeros_like(x,dtype=float)+m, 'domain': (-3.6455388182234945,-3.5344611817765053), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.4099999999999999, y0=-8.11, m=8.52: y0+m*(x-x0), 'deriv': lambda x, m=8.52: np.zeros_like(x,dtype=float)+m, 'domain': (0.33539474962229765,0.4846052503777022), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.4099999999999999, y0=-4.11, m=4.5200000000000005: y0+m*(x-x0), 'deriv': lambda x, m=4.5200000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (0.27175009697066205,0.5482499030293377), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.4099999999999999, y0=-0.10999999999999999, m=0.5199999999999999: y0+m*(x-x0), 'deriv': lambda x, m=0.5199999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-0.1578187527901409,0.9778187527901407), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.4099999999999999, y0=3.89, m=-3.4800000000000004: y0+m*(x-x0), 'deriv': lambda x, m=-3.4800000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (0.23324487993933732,0.5867551200606625), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.4099999999999999, y0=7.89, m=-7.4799999999999995: y0+m*(x-x0), 'deriv': lambda x, m=-7.4799999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (0.32519302264294736,0.4948069773570525), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.41, y0=-8.11, m=12.52: y0+m*(x-x0), 'deriv': lambda x, m=12.52: np.zeros_like(x,dtype=float)+m, 'domain': (4.359044069169244,4.460955930830757), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.41, y0=-4.11, m=8.52: y0+m*(x-x0), 'deriv': lambda x, m=8.52: np.zeros_like(x,dtype=float)+m, 'domain': (4.3353947496222975,4.484605250377703), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.41, y0=-0.10999999999999999, m=4.5200000000000005: y0+m*(x-x0), 'deriv': lambda x, m=4.5200000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (4.2717500969706625,4.548249903029338), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.41, y0=3.89, m=0.52: y0+m*(x-x0), 'deriv': lambda x, m=0.52: np.zeros_like(x,dtype=float)+m, 'domain': (3.8421812472098593,4.9778187527901405), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.41, y0=7.89, m=-3.4799999999999995: y0+m*(x-x0), 'deriv': lambda x, m=-3.4799999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (4.233244879939337,4.586755120060663), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.41, y0=-8.11, m=16.52: y0+m*(x-x0), 'deriv': lambda x, m=16.52: np.zeros_like(x,dtype=float)+m, 'domain': (8.371329862801327,8.448670137198674), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.41, y0=-4.11, m=12.52: y0+m*(x-x0), 'deriv': lambda x, m=12.52: np.zeros_like(x,dtype=float)+m, 'domain': (8.359044069169244,8.460955930830757), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.41, y0=-0.10999999999999999, m=8.52: y0+m*(x-x0), 'deriv': lambda x, m=8.52: np.zeros_like(x,dtype=float)+m, 'domain': (8.335394749622298,8.484605250377703), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.41, y0=3.89, m=4.52: y0+m*(x-x0), 'deriv': lambda x, m=4.52: np.zeros_like(x,dtype=float)+m, 'domain': (8.271750096970662,8.548249903029339), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.41, y0=7.89, m=0.5200000000000005: y0+m*(x-x0), 'deriv': lambda x, m=0.5200000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (7.84218124720986,8.97781875279014), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s2_note_ex01_slope_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.58, y0=-7.46, m=-7.46: y0+m*(x-x0), 'deriv': lambda x, m=-7.46: np.zeros_like(x,dtype=float)+m, 'domain': (-7.673001927589315,-7.486998072410685), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.58, y0=-3.46, m=-3.46: y0+m*(x-x0), 'deriv': lambda x, m=-3.46: np.zeros_like(x,dtype=float)+m, 'domain': (-7.774357482926776,-7.385642517073224), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.58, y0=0.5399999999999999, m=0.5399999999999999: y0+m*(x-x0), 'deriv': lambda x, m=0.5399999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-8.195933778360034,-6.964066221639966), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.58, y0=4.54, m=4.54: y0+m*(x-x0), 'deriv': lambda x, m=4.54: np.zeros_like(x,dtype=float)+m, 'domain': (-7.730575592111217,-7.429424407888783), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.58, y0=8.54, m=8.54: y0+m*(x-x0), 'deriv': lambda x, m=8.54: np.zeros_like(x,dtype=float)+m, 'domain': (-7.6614109808940905,-7.49858901910591), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.58, y0=-7.46, m=-7.46: y0+m*(x-x0), 'deriv': lambda x, m=-7.46: np.zeros_like(x,dtype=float)+m, 'domain': (-3.6730019275893153,-3.486998072410685), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.58, y0=-3.46, m=-3.46: y0+m*(x-x0), 'deriv': lambda x, m=-3.46: np.zeros_like(x,dtype=float)+m, 'domain': (-3.774357482926776,-3.385642517073224), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.58, y0=0.5399999999999999, m=0.5399999999999999: y0+m*(x-x0), 'deriv': lambda x, m=0.5399999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-4.195933778360034,-2.9640662216399654), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.58, y0=4.54, m=4.54: y0+m*(x-x0), 'deriv': lambda x, m=4.54: np.zeros_like(x,dtype=float)+m, 'domain': (-3.730575592111217,-3.429424407888783), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.58, y0=8.54, m=8.54: y0+m*(x-x0), 'deriv': lambda x, m=8.54: np.zeros_like(x,dtype=float)+m, 'domain': (-3.661410980894091,-3.4985890191059092), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.41999999999999993, y0=-7.46, m=-7.46: y0+m*(x-x0), 'deriv': lambda x, m=-7.46: np.zeros_like(x,dtype=float)+m, 'domain': (0.3269980724106848,0.5130019275893151), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.41999999999999993, y0=-3.46, m=-3.46: y0+m*(x-x0), 'deriv': lambda x, m=-3.46: np.zeros_like(x,dtype=float)+m, 'domain': (0.2256425170732237,0.6143574829267762), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.41999999999999993, y0=0.5399999999999999, m=0.5399999999999999: y0+m*(x-x0), 'deriv': lambda x, m=0.5399999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-0.19593377836003478,1.0359337783600346), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.41999999999999993, y0=4.54, m=4.54: y0+m*(x-x0), 'deriv': lambda x, m=4.54: np.zeros_like(x,dtype=float)+m, 'domain': (0.2694244078887832,0.5705755921112167), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.41999999999999993, y0=8.54, m=8.54: y0+m*(x-x0), 'deriv': lambda x, m=8.54: np.zeros_like(x,dtype=float)+m, 'domain': (0.3385890191059091,0.5014109808940908), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.42, y0=-7.46, m=-7.46: y0+m*(x-x0), 'deriv': lambda x, m=-7.46: np.zeros_like(x,dtype=float)+m, 'domain': (4.326998072410685,4.513001927589315), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.42, y0=-3.46, m=-3.46: y0+m*(x-x0), 'deriv': lambda x, m=-3.46: np.zeros_like(x,dtype=float)+m, 'domain': (4.225642517073224,4.614357482926776), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.42, y0=0.5399999999999999, m=0.5399999999999999: y0+m*(x-x0), 'deriv': lambda x, m=0.5399999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (3.804066221639965,5.035933778360034), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.42, y0=4.54, m=4.54: y0+m*(x-x0), 'deriv': lambda x, m=4.54: np.zeros_like(x,dtype=float)+m, 'domain': (4.269424407888783,4.570575592111217), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.42, y0=8.54, m=8.54: y0+m*(x-x0), 'deriv': lambda x, m=8.54: np.zeros_like(x,dtype=float)+m, 'domain': (4.3385890191059095,4.50141098089409), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.42, y0=-7.46, m=-7.46: y0+m*(x-x0), 'deriv': lambda x, m=-7.46: np.zeros_like(x,dtype=float)+m, 'domain': (8.326998072410685,8.513001927589315), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.42, y0=-3.46, m=-3.46: y0+m*(x-x0), 'deriv': lambda x, m=-3.46: np.zeros_like(x,dtype=float)+m, 'domain': (8.225642517073224,8.614357482926776), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.42, y0=0.5399999999999999, m=0.5399999999999999: y0+m*(x-x0), 'deriv': lambda x, m=0.5399999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (7.804066221639966,9.035933778360034), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.42, y0=4.54, m=4.54: y0+m*(x-x0), 'deriv': lambda x, m=4.54: np.zeros_like(x,dtype=float)+m, 'domain': (8.269424407888783,8.570575592111217), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.42, y0=8.54, m=8.54: y0+m*(x-x0), 'deriv': lambda x, m=8.54: np.zeros_like(x,dtype=float)+m, 'domain': (8.338589019105909,8.501410980894091), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s2_note_ex03_solution_tendency.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.9399999999999995, y0=-8.47, m=1.5300000000000011: y0+m*(x-x0), 'deriv': lambda x, m=1.5300000000000011: np.zeros_like(x,dtype=float)+m, 'domain': (-8.437862773398074,-7.4421372266019254), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.9399999999999995, y0=-4.47, m=-2.4699999999999998: y0+m*(x-x0), 'deriv': lambda x, m=-2.4699999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (-8.281495271012952,-7.598504728987048), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.9399999999999995, y0=-0.47, m=-6.47: y0+m*(x-x0), 'deriv': lambda x, m=-6.47: np.zeros_like(x,dtype=float)+m, 'domain': (-8.078998703437353,-7.801001296562646), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.9399999999999995, y0=3.5300000000000002, m=-10.469999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-10.469999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-8.026521252661102,-7.853478747338897), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.9399999999999995, y0=7.53, m=-14.469999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-14.469999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-8.002739093295022,-7.8772609067049775), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.94, y0=-8.47, m=5.530000000000001: y0+m*(x-x0), 'deriv': lambda x, m=5.530000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-4.1019306805429885,-3.778069319457011), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.94, y0=-4.47, m=1.5299999999999998: y0+m*(x-x0), 'deriv': lambda x, m=1.5299999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (-4.4378627733980744,-3.442137226601926), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.94, y0=-0.47, m=-2.4699999999999998: y0+m*(x-x0), 'deriv': lambda x, m=-2.4699999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (-4.281495271012952,-3.5985047289870478), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.94, y0=3.5300000000000002, m=-6.470000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-6.470000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-4.078998703437354,-3.801001296562646), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.94, y0=7.53, m=-10.47: y0+m*(x-x0), 'deriv': lambda x, m=-10.47: np.zeros_like(x,dtype=float)+m, 'domain': (-4.026521252661102,-3.8534787473388983), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.06000000000000005, y0=-8.47, m=9.530000000000001: y0+m*(x-x0), 'deriv': lambda x, m=9.530000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-0.034966540968449256,0.15496654096844936), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.06000000000000005, y0=-4.47, m=5.529999999999999: y0+m*(x-x0), 'deriv': lambda x, m=5.529999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-0.10193068054298896,0.22193068054298906), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.06000000000000005, y0=-0.47, m=1.53: y0+m*(x-x0), 'deriv': lambda x, m=1.53: np.zeros_like(x,dtype=float)+m, 'domain': (-0.43786277339807417,0.5578627733980743), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.06000000000000005, y0=3.5300000000000002, m=-2.47: y0+m*(x-x0), 'deriv': lambda x, m=-2.47: np.zeros_like(x,dtype=float)+m, 'domain': (-0.281495271012952,0.4014952710129521), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.06000000000000005, y0=7.53, m=-6.470000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-6.470000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-0.0789987034373539,0.198998703437354), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.0600000000000005, y0=-8.47, m=13.530000000000001: y0+m*(x-x0), 'deriv': lambda x, m=13.530000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (3.992925009213094,4.127074990786907), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.0600000000000005, y0=-4.47, m=9.530000000000001: y0+m*(x-x0), 'deriv': lambda x, m=9.530000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (3.965033459031551,4.1549665409684495), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.0600000000000005, y0=-0.47, m=5.53: y0+m*(x-x0), 'deriv': lambda x, m=5.53: np.zeros_like(x,dtype=float)+m, 'domain': (3.8980693194570115,4.2219306805429895), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.0600000000000005, y0=3.5300000000000002, m=1.5300000000000002: y0+m*(x-x0), 'deriv': lambda x, m=1.5300000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (3.5621372266019264,4.5578627733980746), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.0600000000000005, y0=7.53, m=-2.4699999999999998: y0+m*(x-x0), 'deriv': lambda x, m=-2.4699999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (3.7185047289870483,4.401495271012952), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.06, y0=-8.47, m=17.53: y0+m*(x-x0), 'deriv': lambda x, m=17.53: np.zeros_like(x,dtype=float)+m, 'domain': (8.008173247553195,8.111826752446806), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.06, y0=-4.47, m=13.530000000000001: y0+m*(x-x0), 'deriv': lambda x, m=13.530000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (7.992925009213094,8.127074990786907), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.06, y0=-0.47, m=9.530000000000001: y0+m*(x-x0), 'deriv': lambda x, m=9.530000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (7.9650334590315515,8.15496654096845), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.06, y0=3.5300000000000002, m=5.53: y0+m*(x-x0), 'deriv': lambda x, m=5.53: np.zeros_like(x,dtype=float)+m, 'domain': (7.8980693194570115,8.22193068054299), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.06, y0=7.53, m=1.5300000000000002: y0+m*(x-x0), 'deriv': lambda x, m=1.5300000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (7.562137226601926,8.557862773398075), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s2_note_yti02_solution_tendency.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.36, y0=-7.51, m=8.36: y0+m*(x-x0), 'deriv': lambda x, m=8.36: np.zeros_like(x,dtype=float)+m, 'domain': (-8.437200852076993,-8.282799147923006), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.36, y0=-3.51, m=8.36: y0+m*(x-x0), 'deriv': lambda x, m=8.36: np.zeros_like(x,dtype=float)+m, 'domain': (-8.437200852076993,-8.282799147923006), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.36, y0=0.4900000000000001, m=8.36: y0+m*(x-x0), 'deriv': lambda x, m=8.36: np.zeros_like(x,dtype=float)+m, 'domain': (-8.437200852076993,-8.282799147923006), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.36, y0=4.49, m=8.36: y0+m*(x-x0), 'deriv': lambda x, m=8.36: np.zeros_like(x,dtype=float)+m, 'domain': (-8.437200852076993,-8.282799147923006), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.36, y0=8.49, m=8.36: y0+m*(x-x0), 'deriv': lambda x, m=8.36: np.zeros_like(x,dtype=float)+m, 'domain': (-8.437200852076993,-8.282799147923006), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.36, y0=-7.51, m=4.36: y0+m*(x-x0), 'deriv': lambda x, m=4.36: np.zeros_like(x,dtype=float)+m, 'domain': (-4.505309548429774,-4.214690451570227), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.36, y0=-3.51, m=4.36: y0+m*(x-x0), 'deriv': lambda x, m=4.36: np.zeros_like(x,dtype=float)+m, 'domain': (-4.505309548429774,-4.214690451570227), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.36, y0=0.4900000000000001, m=4.36: y0+m*(x-x0), 'deriv': lambda x, m=4.36: np.zeros_like(x,dtype=float)+m, 'domain': (-4.505309548429774,-4.214690451570227), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.36, y0=4.49, m=4.36: y0+m*(x-x0), 'deriv': lambda x, m=4.36: np.zeros_like(x,dtype=float)+m, 'domain': (-4.505309548429774,-4.214690451570227), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.36, y0=8.49, m=4.36: y0+m*(x-x0), 'deriv': lambda x, m=4.36: np.zeros_like(x,dtype=float)+m, 'domain': (-4.505309548429774,-4.214690451570227), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.36, y0=-7.51, m=0.36: y0+m*(x-x0), 'deriv': lambda x, m=0.36: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9715768177146724,0.25157681771467244), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.36, y0=-3.51, m=0.36: y0+m*(x-x0), 'deriv': lambda x, m=0.36: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9715768177146724,0.25157681771467244), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.36, y0=0.4900000000000001, m=0.36: y0+m*(x-x0), 'deriv': lambda x, m=0.36: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9715768177146724,0.25157681771467244), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.36, y0=4.49, m=0.36: y0+m*(x-x0), 'deriv': lambda x, m=0.36: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9715768177146724,0.25157681771467244), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.36, y0=8.49, m=0.36: y0+m*(x-x0), 'deriv': lambda x, m=0.36: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9715768177146724,0.25157681771467244), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.64, y0=-7.51, m=-3.64: y0+m*(x-x0), 'deriv': lambda x, m=-3.64: np.zeros_like(x,dtype=float)+m, 'domain': (3.467808376840049,3.812191623159951), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.64, y0=-3.51, m=-3.64: y0+m*(x-x0), 'deriv': lambda x, m=-3.64: np.zeros_like(x,dtype=float)+m, 'domain': (3.467808376840049,3.812191623159951), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.64, y0=0.4900000000000001, m=-3.64: y0+m*(x-x0), 'deriv': lambda x, m=-3.64: np.zeros_like(x,dtype=float)+m, 'domain': (3.467808376840049,3.812191623159951), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.64, y0=4.49, m=-3.64: y0+m*(x-x0), 'deriv': lambda x, m=-3.64: np.zeros_like(x,dtype=float)+m, 'domain': (3.467808376840049,3.812191623159951), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.64, y0=8.49, m=-3.64: y0+m*(x-x0), 'deriv': lambda x, m=-3.64: np.zeros_like(x,dtype=float)+m, 'domain': (3.467808376840049,3.812191623159951), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.64, y0=-7.51, m=-7.64: y0+m*(x-x0), 'deriv': lambda x, m=-7.64: np.zeros_like(x,dtype=float)+m, 'domain': (7.55564102481138,7.724358975188619), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.64, y0=-3.51, m=-7.64: y0+m*(x-x0), 'deriv': lambda x, m=-7.64: np.zeros_like(x,dtype=float)+m, 'domain': (7.55564102481138,7.724358975188619), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.64, y0=0.4900000000000001, m=-7.64: y0+m*(x-x0), 'deriv': lambda x, m=-7.64: np.zeros_like(x,dtype=float)+m, 'domain': (7.55564102481138,7.724358975188619), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.64, y0=4.49, m=-7.64: y0+m*(x-x0), 'deriv': lambda x, m=-7.64: np.zeros_like(x,dtype=float)+m, 'domain': (7.55564102481138,7.724358975188619), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.64, y0=8.49, m=-7.64: y0+m*(x-x0), 'deriv': lambda x, m=-7.64: np.zeros_like(x,dtype=float)+m, 'domain': (7.55564102481138,7.724358975188619), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s2_ps1_c02_solution_tendency.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.07, y0=-8.53, m=-16.6: y0+m*(x-x0), 'deriv': lambda x, m=-16.6: np.zeros_like(x,dtype=float)+m, 'domain': (-8.115700285057999,-8.024299714942002), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.07, y0=-4.53, m=-12.600000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-12.600000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-8.130128388758573,-8.009871611241428), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.07, y0=-0.53, m=-8.6: y0+m*(x-x0), 'deriv': lambda x, m=-8.6: np.zeros_like(x,dtype=float)+m, 'domain': (-8.157780652212669,-7.982219347787331), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.07, y0=3.4699999999999998, m=-4.6000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=-4.6000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-8.231446529691594,-7.908553470308407), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.07, y0=7.47, m=-0.6000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=-0.6000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-8.721694623541534,-7.418305376458467), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.07, y0=-8.53, m=-12.6: y0+m*(x-x0), 'deriv': lambda x, m=-12.6: np.zeros_like(x,dtype=float)+m, 'domain': (-4.130128388758573,-4.009871611241428), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.07, y0=-4.53, m=-8.600000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-8.600000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-4.1577806522126695,-3.9822193477873316), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.07, y0=-0.53, m=-4.6000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=-4.6000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-4.231446529691594,-3.9085534703084073), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.07, y0=3.4699999999999998, m=-0.6000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=-0.6000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-4.721694623541533,-3.4183053764584668), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.07, y0=7.47, m=3.3999999999999995: y0+m*(x-x0), 'deriv': lambda x, m=3.3999999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (-4.284446406233578,-3.8555535937664223), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.06999999999999995, y0=-8.53, m=-8.6: y0+m*(x-x0), 'deriv': lambda x, m=-8.6: np.zeros_like(x,dtype=float)+m, 'domain': (-0.15778065221266888,0.017780652212668974), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.06999999999999995, y0=-4.53, m=-4.6000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=-4.6000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-0.23144652969159313,0.09144652969159323), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.06999999999999995, y0=-0.53, m=-0.6: y0+m*(x-x0), 'deriv': lambda x, m=-0.6: np.zeros_like(x,dtype=float)+m, 'domain': (-0.7216946235415336,0.5816946235415337), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.06999999999999995, y0=3.4699999999999998, m=3.4: y0+m*(x-x0), 'deriv': lambda x, m=3.4: np.zeros_like(x,dtype=float)+m, 'domain': (-0.2844464062335781,0.14444640623357818), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.06999999999999995, y0=7.47, m=7.3999999999999995: y0+m*(x-x0), 'deriv': lambda x, m=7.3999999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (-0.1717776009610602,0.03177760096106029), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.93, y0=-8.53, m=-4.6: y0+m*(x-x0), 'deriv': lambda x, m=-4.6: np.zeros_like(x,dtype=float)+m, 'domain': (3.768553470308407,4.091446529691593), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.93, y0=-4.53, m=-0.6000000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-0.6000000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (3.2783053764584666,4.581694623541534), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.93, y0=-0.53, m=3.4000000000000004: y0+m*(x-x0), 'deriv': lambda x, m=3.4000000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (3.715553593766422,4.144446406233579), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.93, y0=3.4699999999999998, m=7.4: y0+m*(x-x0), 'deriv': lambda x, m=7.4: np.zeros_like(x,dtype=float)+m, 'domain': (3.82822239903894,4.03177760096106), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.93, y0=7.47, m=11.4: y0+m*(x-x0), 'deriv': lambda x, m=11.4: np.zeros_like(x,dtype=float)+m, 'domain': (3.863588351736719,3.9964116482632814), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.93, y0=-8.53, m=-0.5999999999999996: y0+m*(x-x0), 'deriv': lambda x, m=-0.5999999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (7.278305376458466,8.581694623541534), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.93, y0=-4.53, m=3.3999999999999995: y0+m*(x-x0), 'deriv': lambda x, m=3.3999999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (7.715553593766422,8.144446406233579), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.93, y0=-0.53, m=7.3999999999999995: y0+m*(x-x0), 'deriv': lambda x, m=7.3999999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (7.82822239903894,8.03177760096106), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.93, y0=3.4699999999999998, m=11.399999999999999: y0+m*(x-x0), 'deriv': lambda x, m=11.399999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (7.863588351736718,7.996411648263281), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.93, y0=7.47, m=15.399999999999999: y0+m*(x-x0), 'deriv': lambda x, m=15.399999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (7.880753067819503,7.979246932180496), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s2_ps2_c01_solution_tendency.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.98, y0=-7.77, m=0.7899999999999991: y0+m*(x-x0), 'deriv': lambda x, m=0.7899999999999991: np.zeros_like(x,dtype=float)+m, 'domain': (-8.654826405762195,-7.3051735942378055), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.98, y0=-3.77, m=-3.210000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-3.210000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-8.235788170602559,-7.724211829397443), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.98, y0=0.22999999999999998, m=-7.210000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-7.210000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-8.09814780849651,-7.861852191503491), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.98, y0=4.23, m=-11.21: y0+m*(x-x0), 'deriv': lambda x, m=-11.21: np.zeros_like(x,dtype=float)+m, 'domain': (-8.056413779427084,-7.903586220572917), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.98, y0=8.23, m=-15.21: y0+m*(x-x0), 'deriv': lambda x, m=-15.21: np.zeros_like(x,dtype=float)+m, 'domain': (-8.036419940896282,-7.923580059103719), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.98, y0=-7.77, m=4.789999999999999: y0+m*(x-x0), 'deriv': lambda x, m=4.789999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-4.15575156155977,-3.80424843844023), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.98, y0=-3.77, m=0.79: y0+m*(x-x0), 'deriv': lambda x, m=0.79: np.zeros_like(x,dtype=float)+m, 'domain': (-4.654826405762194,-3.3051735942378055), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.98, y0=0.22999999999999998, m=-3.21: y0+m*(x-x0), 'deriv': lambda x, m=-3.21: np.zeros_like(x,dtype=float)+m, 'domain': (-4.235788170602558,-3.7242118293974418), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.98, y0=4.23, m=-7.210000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-7.210000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-4.098147808496509,-3.8618521915034907), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.98, y0=8.23, m=-11.21: y0+m*(x-x0), 'deriv': lambda x, m=-11.21: np.zeros_like(x,dtype=float)+m, 'domain': (-4.056413779427084,-3.903586220572916), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.020000000000000018, y0=-7.77, m=8.79: y0+m*(x-x0), 'deriv': lambda x, m=8.79: np.zeros_like(x,dtype=float)+m, 'domain': (-0.07721139011042014,0.11721139011042017), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.020000000000000018, y0=-3.77, m=4.79: y0+m*(x-x0), 'deriv': lambda x, m=4.79: np.zeros_like(x,dtype=float)+m, 'domain': (-0.15575156155976966,0.1957515615597697), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.020000000000000018, y0=0.22999999999999998, m=0.79: y0+m*(x-x0), 'deriv': lambda x, m=0.79: np.zeros_like(x,dtype=float)+m, 'domain': (-0.6548264057621942,0.6948264057621942), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.020000000000000018, y0=4.23, m=-3.210000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-3.210000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-0.2357881706025579,0.27578817060255795), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.020000000000000018, y0=8.23, m=-7.210000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-7.210000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-0.09814780849650927,0.13814780849650932), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.02, y0=-7.77, m=12.79: y0+m*(x-x0), 'deriv': lambda x, m=12.79: np.zeros_like(x,dtype=float)+m, 'domain': (3.9529645525189716,4.0870354474810275), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.02, y0=-3.77, m=8.79: y0+m*(x-x0), 'deriv': lambda x, m=8.79: np.zeros_like(x,dtype=float)+m, 'domain': (3.9227886098895794,4.11721139011042), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.02, y0=0.22999999999999998, m=4.789999999999999: y0+m*(x-x0), 'deriv': lambda x, m=4.789999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (3.84424843844023,4.195751561559769), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.02, y0=4.23, m=0.7899999999999991: y0+m*(x-x0), 'deriv': lambda x, m=0.7899999999999991: np.zeros_like(x,dtype=float)+m, 'domain': (3.345173594237805,4.6948264057621945), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.02, y0=8.23, m=-3.210000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-3.210000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (3.764211829397442,4.275788170602557), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.02, y0=-7.77, m=16.79: y0+m*(x-x0), 'deriv': lambda x, m=16.79: np.zeros_like(x,dtype=float)+m, 'domain': (7.968869642360884,8.071130357639115), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.02, y0=-3.77, m=12.79: y0+m*(x-x0), 'deriv': lambda x, m=12.79: np.zeros_like(x,dtype=float)+m, 'domain': (7.952964552518972,8.087035447481028), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.02, y0=0.22999999999999998, m=8.79: y0+m*(x-x0), 'deriv': lambda x, m=8.79: np.zeros_like(x,dtype=float)+m, 'domain': (7.9227886098895794,8.11721139011042), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.02, y0=4.23, m=4.789999999999999: y0+m*(x-x0), 'deriv': lambda x, m=4.789999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (7.84424843844023,8.195751561559769), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.02, y0=8.23, m=0.7899999999999991: y0+m*(x-x0), 'deriv': lambda x, m=0.7899999999999991: np.zeros_like(x,dtype=float)+m, 'domain': (7.345173594237805,8.694826405762194), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s2_ps2_c05_slope_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.85, y0=-7.71, m=-1.1399999999999997: y0+m*(x-x0), 'deriv': lambda x, m=-1.1399999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (-8.43030528011116,-7.2696947198888395), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.85, y0=-3.71, m=-5.14: y0+m*(x-x0), 'deriv': lambda x, m=-5.14: np.zeros_like(x,dtype=float)+m, 'domain': (-8.018055261767076,-7.681944738232923), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.85, y0=0.29000000000000004, m=-9.14: y0+m*(x-x0), 'deriv': lambda x, m=-9.14: np.zeros_like(x,dtype=float)+m, 'domain': (-7.945708955756452,-7.754291044243547), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.85, y0=4.29, m=-13.14: y0+m*(x-x0), 'deriv': lambda x, m=-13.14: np.zeros_like(x,dtype=float)+m, 'domain': (-7.916777979422532,-7.783222020577467), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.85, y0=8.29, m=-17.14: y0+m*(x-x0), 'deriv': lambda x, m=-17.14: np.zeros_like(x,dtype=float)+m, 'domain': (-7.901254731117238,-7.798745268882762), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.85, y0=-7.71, m=2.86: y0+m*(x-x0), 'deriv': lambda x, m=2.86: np.zeros_like(x,dtype=float)+m, 'domain': (-4.140449607220243,-3.559550392779757), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.85, y0=-3.71, m=-1.1400000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-1.1400000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-4.43030528011116,-3.2696947198888404), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.85, y0=0.29000000000000004, m=-5.140000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-5.140000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-4.018055261767076,-3.6819447382329242), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.85, y0=4.29, m=-9.14: y0+m*(x-x0), 'deriv': lambda x, m=-9.14: np.zeros_like(x,dtype=float)+m, 'domain': (-3.9457089557564524,-3.7542910442435478), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.85, y0=8.29, m=-13.139999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-13.139999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-3.9167779794225326,-3.7832220205774676), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.15000000000000002, y0=-7.71, m=6.86: y0+m*(x-x0), 'deriv': lambda x, m=6.86: np.zeros_like(x,dtype=float)+m, 'domain': (0.02306172384340649,0.27693827615659355), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.15000000000000002, y0=-3.71, m=2.86: y0+m*(x-x0), 'deriv': lambda x, m=2.86: np.zeros_like(x,dtype=float)+m, 'domain': (-0.14044960722024336,0.4404496072202434), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.15000000000000002, y0=0.29000000000000004, m=-1.1400000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-1.1400000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-0.4303052801111594,0.7303052801111595), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.15000000000000002, y0=4.29, m=-5.14: y0+m*(x-x0), 'deriv': lambda x, m=-5.14: np.zeros_like(x,dtype=float)+m, 'domain': (-0.018055261767075892,0.31805526176707594), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.15000000000000002, y0=8.29, m=-9.139999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-9.139999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (0.05429104424354746,0.24570895575645257), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.15, y0=-7.71, m=10.86: y0+m*(x-x0), 'deriv': lambda x, m=10.86: np.zeros_like(x,dtype=float)+m, 'domain': (4.0693100520453935,4.230689947954607), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.15, y0=-3.71, m=6.86: y0+m*(x-x0), 'deriv': lambda x, m=6.86: np.zeros_like(x,dtype=float)+m, 'domain': (4.023061723843407,4.276938276156594), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.15, y0=0.29000000000000004, m=2.8600000000000003: y0+m*(x-x0), 'deriv': lambda x, m=2.8600000000000003: np.zeros_like(x,dtype=float)+m, 'domain': (3.859550392779757,4.440449607220244), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.15, y0=4.29, m=-1.1399999999999997: y0+m*(x-x0), 'deriv': lambda x, m=-1.1399999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (3.5696947198888407,4.7303052801111605), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.15, y0=8.29, m=-5.139999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-5.139999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (3.9819447382329245,4.318055261767077), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.15, y0=-7.71, m=14.86: y0+m*(x-x0), 'deriv': lambda x, m=14.86: np.zeros_like(x,dtype=float)+m, 'domain': (8.090914255351441,8.20908574464856), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.15, y0=-3.71, m=10.86: y0+m*(x-x0), 'deriv': lambda x, m=10.86: np.zeros_like(x,dtype=float)+m, 'domain': (8.069310052045394,8.230689947954607), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.15, y0=0.29000000000000004, m=6.86: y0+m*(x-x0), 'deriv': lambda x, m=6.86: np.zeros_like(x,dtype=float)+m, 'domain': (8.023061723843407,8.276938276156594), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.15, y0=4.29, m=2.8600000000000003: y0+m*(x-x0), 'deriv': lambda x, m=2.8600000000000003: np.zeros_like(x,dtype=float)+m, 'domain': (7.859550392779757,8.440449607220243), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.15, y0=8.29, m=-1.1399999999999988: y0+m*(x-x0), 'deriv': lambda x, m=-1.1399999999999988: np.zeros_like(x,dtype=float)+m, 'domain': (7.56969471988884,8.73030528011116), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s2_xp_c03_slope_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.08, y0=-7.6, m=0.5199999999999996: y0+m*(x-x0), 'deriv': lambda x, m=0.5199999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (-8.851878617074098,-7.308121382925902), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.08, y0=-3.6, m=-3.4800000000000004: y0+m*(x-x0), 'deriv': lambda x, m=-3.4800000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-8.320276491332462,-7.839723508667537), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.08, y0=0.4, m=-7.48: y0+m*(x-x0), 'deriv': lambda x, m=-7.48: np.zeros_like(x,dtype=float)+m, 'domain': (-8.195284484844743,-7.964715515155257), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.08, y0=4.4, m=-11.48: y0+m*(x-x0), 'deriv': lambda x, m=-11.48: np.zeros_like(x,dtype=float)+m, 'domain': (-8.155498081022563,-8.004501918977438), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.08, y0=8.4, m=-15.48: y0+m*(x-x0), 'deriv': lambda x, m=-15.48: np.zeros_like(x,dtype=float)+m, 'domain': (-8.136084648986234,-8.023915351013766), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.08, y0=-7.6, m=4.52: y0+m*(x-x0), 'deriv': lambda x, m=4.52: np.zeros_like(x,dtype=float)+m, 'domain': (-4.2679334619305065,-3.8920665380694937), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.08, y0=-3.6, m=0.52: y0+m*(x-x0), 'deriv': lambda x, m=0.52: np.zeros_like(x,dtype=float)+m, 'domain': (-4.851878617074098,-3.3081213829259024), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.08, y0=0.4, m=-3.4800000000000004: y0+m*(x-x0), 'deriv': lambda x, m=-3.4800000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-4.320276491332463,-3.8397235086675368), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.08, y0=4.4, m=-7.48: y0+m*(x-x0), 'deriv': lambda x, m=-7.48: np.zeros_like(x,dtype=float)+m, 'domain': (-4.195284484844743,-3.9647155151552567), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.08, y0=8.4, m=-11.48: y0+m*(x-x0), 'deriv': lambda x, m=-11.48: np.zeros_like(x,dtype=float)+m, 'domain': (-4.1554980810225635,-4.004501918977437), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.07999999999999996, y0=-7.6, m=8.52: y0+m*(x-x0), 'deriv': lambda x, m=8.52: np.zeros_like(x,dtype=float)+m, 'domain': (-0.18141651223218896,0.021416512232189044), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.07999999999999996, y0=-3.6, m=4.52: y0+m*(x-x0), 'deriv': lambda x, m=4.52: np.zeros_like(x,dtype=float)+m, 'domain': (-0.26793346193050616,0.10793346193050624), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.07999999999999996, y0=0.4, m=0.52: y0+m*(x-x0), 'deriv': lambda x, m=0.52: np.zeros_like(x,dtype=float)+m, 'domain': (-0.8518786170740977,0.6918786170740978), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.07999999999999996, y0=4.4, m=-3.4800000000000004: y0+m*(x-x0), 'deriv': lambda x, m=-3.4800000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-0.32027649133246316,0.16027649133246327), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.07999999999999996, y0=8.4, m=-7.48: y0+m*(x-x0), 'deriv': lambda x, m=-7.48: np.zeros_like(x,dtype=float)+m, 'domain': (-0.19528448484474326,0.03528448484474335), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.92, y0=-7.6, m=12.52: y0+m*(x-x0), 'deriv': lambda x, m=12.52: np.zeros_like(x,dtype=float)+m, 'domain': (3.8507317815269406,3.9892682184730592), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.92, y0=-3.6, m=8.52: y0+m*(x-x0), 'deriv': lambda x, m=8.52: np.zeros_like(x,dtype=float)+m, 'domain': (3.818583487767811,4.0214165122321885), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.92, y0=0.4, m=4.52: y0+m*(x-x0), 'deriv': lambda x, m=4.52: np.zeros_like(x,dtype=float)+m, 'domain': (3.7320665380694935,4.107933461930506), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.92, y0=4.4, m=0.5199999999999996: y0+m*(x-x0), 'deriv': lambda x, m=0.5199999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (3.1481213829259023,4.691878617074098), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.92, y0=8.4, m=-3.4800000000000004: y0+m*(x-x0), 'deriv': lambda x, m=-3.4800000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (3.6797235086675366,4.160276491332463), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.92, y0=-7.6, m=16.52: y0+m*(x-x0), 'deriv': lambda x, m=16.52: np.zeros_like(x,dtype=float)+m, 'domain': (7.8674327822455545,7.972567217754445), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.92, y0=-3.6, m=12.52: y0+m*(x-x0), 'deriv': lambda x, m=12.52: np.zeros_like(x,dtype=float)+m, 'domain': (7.850731781526941,7.989268218473059), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.92, y0=0.4, m=8.52: y0+m*(x-x0), 'deriv': lambda x, m=8.52: np.zeros_like(x,dtype=float)+m, 'domain': (7.818583487767811,8.02141651223219), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.92, y0=4.4, m=4.52: y0+m*(x-x0), 'deriv': lambda x, m=4.52: np.zeros_like(x,dtype=float)+m, 'domain': (7.7320665380694935,8.107933461930505), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.92, y0=8.4, m=0.5199999999999996: y0+m*(x-x0), 'deriv': lambda x, m=0.5199999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (7.148121382925902,8.691878617074098), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s2_xp_c05_solution_tendency.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.21, y0=-8.54, m=-17.75: y0+m*(x-x0), 'deriv': lambda x, m=-17.75: np.zeros_like(x,dtype=float)+m, 'domain': (-8.256686531173347,-8.163313468826654), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.21, y0=-4.54, m=-13.75: y0+m*(x-x0), 'deriv': lambda x, m=-13.75: np.zeros_like(x,dtype=float)+m, 'domain': (-8.270204627496422,-8.14979537250358), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.21, y0=-0.54, m=-9.75: y0+m*(x-x0), 'deriv': lambda x, m=-9.75: np.zeros_like(x,dtype=float)+m, 'domain': (-8.294683958363814,-8.125316041636188), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.21, y0=3.46, m=-5.750000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-5.750000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-8.352213175043536,-8.067786824956466), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.21, y0=7.46, m=-1.7500000000000009: y0+m*(x-x0), 'deriv': lambda x, m=-1.7500000000000009: np.zeros_like(x,dtype=float)+m, 'domain': (-8.621795318836172,-7.798204681163829), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.21, y0=-8.54, m=-13.75: y0+m*(x-x0), 'deriv': lambda x, m=-13.75: np.zeros_like(x,dtype=float)+m, 'domain': (-4.270204627496422,-4.149795372503578), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.21, y0=-4.54, m=-9.75: y0+m*(x-x0), 'deriv': lambda x, m=-9.75: np.zeros_like(x,dtype=float)+m, 'domain': (-4.294683958363812,-4.125316041636188), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.21, y0=-0.54, m=-5.75: y0+m*(x-x0), 'deriv': lambda x, m=-5.75: np.zeros_like(x,dtype=float)+m, 'domain': (-4.352213175043536,-4.067786824956464), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.21, y0=3.46, m=-1.75: y0+m*(x-x0), 'deriv': lambda x, m=-1.75: np.zeros_like(x,dtype=float)+m, 'domain': (-4.621795318836172,-3.7982046811638277), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.21, y0=7.46, m=2.25: y0+m*(x-x0), 'deriv': lambda x, m=2.25: np.zeros_like(x,dtype=float)+m, 'domain': (-4.547094926824362,-3.8729050731756383), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.20999999999999996, y0=-8.54, m=-9.75: y0+m*(x-x0), 'deriv': lambda x, m=-9.75: np.zeros_like(x,dtype=float)+m, 'domain': (-0.2946839583638126,-0.12531604163618731), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.20999999999999996, y0=-4.54, m=-5.75: y0+m*(x-x0), 'deriv': lambda x, m=-5.75: np.zeros_like(x,dtype=float)+m, 'domain': (-0.35221317504353555,-0.06778682495646435), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.20999999999999996, y0=-0.54, m=-1.75: y0+m*(x-x0), 'deriv': lambda x, m=-1.75: np.zeros_like(x,dtype=float)+m, 'domain': (-0.6217953188361721,0.2017953188361722), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.20999999999999996, y0=3.46, m=2.25: y0+m*(x-x0), 'deriv': lambda x, m=2.25: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5470949268243616,0.12709492682436163), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.20999999999999996, y0=7.46, m=6.25: y0+m*(x-x0), 'deriv': lambda x, m=6.25: np.zeros_like(x,dtype=float)+m, 'domain': (-0.3411321159185384,-0.07886788408146148), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.79, y0=-8.54, m=-5.749999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-5.749999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (3.6477868249564644,3.9322131750435356), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.79, y0=-4.54, m=-1.75: y0+m*(x-x0), 'deriv': lambda x, m=-1.75: np.zeros_like(x,dtype=float)+m, 'domain': (3.3782046811638278,4.201795318836172), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.79, y0=-0.54, m=2.25: y0+m*(x-x0), 'deriv': lambda x, m=2.25: np.zeros_like(x,dtype=float)+m, 'domain': (3.4529050731756383,4.127094926824362), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.79, y0=3.46, m=6.25: y0+m*(x-x0), 'deriv': lambda x, m=6.25: np.zeros_like(x,dtype=float)+m, 'domain': (3.6588678840814617,3.9211321159185384), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.79, y0=7.46, m=10.25: y0+m*(x-x0), 'deriv': lambda x, m=10.25: np.zeros_like(x,dtype=float)+m, 'domain': (3.7094070296430965,3.8705929703569035), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.79, y0=-8.54, m=-1.7499999999999991: y0+m*(x-x0), 'deriv': lambda x, m=-1.7499999999999991: np.zeros_like(x,dtype=float)+m, 'domain': (7.378204681163828,8.201795318836172), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.79, y0=-4.54, m=2.25: y0+m*(x-x0), 'deriv': lambda x, m=2.25: np.zeros_like(x,dtype=float)+m, 'domain': (7.452905073175638,8.12709492682436), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.79, y0=-0.54, m=6.25: y0+m*(x-x0), 'deriv': lambda x, m=6.25: np.zeros_like(x,dtype=float)+m, 'domain': (7.658867884081461,7.921132115918539), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.79, y0=3.46, m=10.25: y0+m*(x-x0), 'deriv': lambda x, m=10.25: np.zeros_like(x,dtype=float)+m, 'domain': (7.7094070296430965,7.8705929703569035), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.79, y0=7.46, m=14.25: y0+m*(x-x0), 'deriv': lambda x, m=14.25: np.zeros_like(x,dtype=float)+m, 'domain': (7.731897276370994,7.848102723629006), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s3_cyu_l02_slope_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.42, y0=-8.04, m=-8.04: y0+m*(x-x0), 'deriv': lambda x, m=-8.04: np.zeros_like(x,dtype=float)+m, 'domain': (-7.502696137148617,-7.337303862851383), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.42, y0=-4.04, m=-4.04: y0+m*(x-x0), 'deriv': lambda x, m=-4.04: np.zeros_like(x,dtype=float)+m, 'domain': (-7.580983290450205,-7.259016709549795), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.42, y0=-0.039999999999999925, m=-0.039999999999999925: y0+m*(x-x0), 'deriv': lambda x, m=-0.039999999999999925: np.zeros_like(x,dtype=float)+m, 'domain': (-8.089464642343598,-6.750535357656401), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.42, y0=3.96, m=3.96: y0+m*(x-x0), 'deriv': lambda x, m=3.96: np.zeros_like(x,dtype=float)+m, 'domain': (-7.584042338253541,-7.255957661746459), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.42, y0=7.96, m=7.96: y0+m*(x-x0), 'deriv': lambda x, m=7.96: np.zeros_like(x,dtype=float)+m, 'domain': (-7.503514404118819,-7.3364855958811805), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.42, y0=-8.04, m=-8.04: y0+m*(x-x0), 'deriv': lambda x, m=-8.04: np.zeros_like(x,dtype=float)+m, 'domain': (-3.502696137148617,-3.3373038628513827), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.42, y0=-4.04, m=-4.04: y0+m*(x-x0), 'deriv': lambda x, m=-4.04: np.zeros_like(x,dtype=float)+m, 'domain': (-3.580983290450205,-3.259016709549795), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.42, y0=-0.039999999999999925, m=-0.039999999999999925: y0+m*(x-x0), 'deriv': lambda x, m=-0.039999999999999925: np.zeros_like(x,dtype=float)+m, 'domain': (-4.089464642343599,-2.750535357656401), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.42, y0=3.96, m=3.96: y0+m*(x-x0), 'deriv': lambda x, m=3.96: np.zeros_like(x,dtype=float)+m, 'domain': (-3.5840423382535413,-3.2559576617464585), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.42, y0=7.96, m=7.96: y0+m*(x-x0), 'deriv': lambda x, m=7.96: np.zeros_like(x,dtype=float)+m, 'domain': (-3.503514404118819,-3.336485595881181), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5800000000000001, y0=-8.04, m=-8.04: y0+m*(x-x0), 'deriv': lambda x, m=-8.04: np.zeros_like(x,dtype=float)+m, 'domain': (0.497303862851383,0.6626961371486171), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5800000000000001, y0=-4.04, m=-4.04: y0+m*(x-x0), 'deriv': lambda x, m=-4.04: np.zeros_like(x,dtype=float)+m, 'domain': (0.41901670954979486,0.7409832904502053), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5800000000000001, y0=-0.039999999999999925, m=-0.039999999999999925: y0+m*(x-x0), 'deriv': lambda x, m=-0.039999999999999925: np.zeros_like(x,dtype=float)+m, 'domain': (-0.08946464234359885,1.2494646423435989), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5800000000000001, y0=3.96, m=3.96: y0+m*(x-x0), 'deriv': lambda x, m=3.96: np.zeros_like(x,dtype=float)+m, 'domain': (0.4159576617464589,0.7440423382535413), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5800000000000001, y0=7.96, m=7.96: y0+m*(x-x0), 'deriv': lambda x, m=7.96: np.zeros_like(x,dtype=float)+m, 'domain': (0.4964855958811809,0.6635144041188192), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.58, y0=-8.04, m=-8.04: y0+m*(x-x0), 'deriv': lambda x, m=-8.04: np.zeros_like(x,dtype=float)+m, 'domain': (4.497303862851383,4.662696137148617), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.58, y0=-4.04, m=-4.04: y0+m*(x-x0), 'deriv': lambda x, m=-4.04: np.zeros_like(x,dtype=float)+m, 'domain': (4.419016709549795,4.740983290450205), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.58, y0=-0.039999999999999925, m=-0.039999999999999925: y0+m*(x-x0), 'deriv': lambda x, m=-0.039999999999999925: np.zeros_like(x,dtype=float)+m, 'domain': (3.9105353576564013,5.249464642343599), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.58, y0=3.96, m=3.96: y0+m*(x-x0), 'deriv': lambda x, m=3.96: np.zeros_like(x,dtype=float)+m, 'domain': (4.415957661746459,4.744042338253541), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.58, y0=7.96, m=7.96: y0+m*(x-x0), 'deriv': lambda x, m=7.96: np.zeros_like(x,dtype=float)+m, 'domain': (4.496485595881181,4.6635144041188195), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.58, y0=-8.04, m=-8.04: y0+m*(x-x0), 'deriv': lambda x, m=-8.04: np.zeros_like(x,dtype=float)+m, 'domain': (8.497303862851384,8.662696137148616), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.58, y0=-4.04, m=-4.04: y0+m*(x-x0), 'deriv': lambda x, m=-4.04: np.zeros_like(x,dtype=float)+m, 'domain': (8.419016709549794,8.740983290450206), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.58, y0=-0.039999999999999925, m=-0.039999999999999925: y0+m*(x-x0), 'deriv': lambda x, m=-0.039999999999999925: np.zeros_like(x,dtype=float)+m, 'domain': (7.910535357656401,9.249464642343598), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.58, y0=3.96, m=3.96: y0+m*(x-x0), 'deriv': lambda x, m=3.96: np.zeros_like(x,dtype=float)+m, 'domain': (8.415957661746459,8.744042338253541), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.58, y0=7.96, m=7.96: y0+m*(x-x0), 'deriv': lambda x, m=7.96: np.zeros_like(x,dtype=float)+m, 'domain': (8.49648559588118,8.66351440411882), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s3_et_a_l01_slope_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.92, y0=-7.47, m=-14.39: y0+m*(x-x0), 'deriv': lambda x, m=-14.39: np.zeros_like(x,dtype=float)+m, 'domain': (-7.958129030642127,-7.881870969357873), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.92, y0=-3.47, m=-10.39: y0+m*(x-x0), 'deriv': lambda x, m=-10.39: np.zeros_like(x,dtype=float)+m, 'domain': (-7.972692024702444,-7.867307975297556), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.92, y0=0.5299999999999999, m=-6.39: y0+m*(x-x0), 'deriv': lambda x, m=-6.39: np.zeros_like(x,dtype=float)+m, 'domain': (-8.005036985427452,-7.834963014572547), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.92, y0=4.53, m=-2.3899999999999997: y0+m*(x-x0), 'deriv': lambda x, m=-2.3899999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (-8.132291927583097,-7.707708072416903), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.92, y0=8.53, m=1.6099999999999994: y0+m*(x-x0), 'deriv': lambda x, m=1.6099999999999994: np.zeros_like(x,dtype=float)+m, 'domain': (-8.210194034399136,-7.629805965600865), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.92, y0=-7.47, m=-10.39: y0+m*(x-x0), 'deriv': lambda x, m=-10.39: np.zeros_like(x,dtype=float)+m, 'domain': (-3.9726920247024435,-3.8673079752975563), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.92, y0=-3.47, m=-6.390000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-6.390000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-4.005036985427453,-3.8349630145725473), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.92, y0=0.5299999999999999, m=-2.39: y0+m*(x-x0), 'deriv': lambda x, m=-2.39: np.zeros_like(x,dtype=float)+m, 'domain': (-4.132291927583097,-3.707708072416903), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.92, y0=4.53, m=1.6100000000000003: y0+m*(x-x0), 'deriv': lambda x, m=1.6100000000000003: np.zeros_like(x,dtype=float)+m, 'domain': (-4.210194034399135,-3.629805965600865), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.92, y0=8.53, m=5.609999999999999: y0+m*(x-x0), 'deriv': lambda x, m=5.609999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-4.016517819436649,-3.823482180563351), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.07999999999999996, y0=-7.47, m=-6.39: y0+m*(x-x0), 'deriv': lambda x, m=-6.39: np.zeros_like(x,dtype=float)+m, 'domain': (-0.005036985427452775,0.1650369854274527), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.07999999999999996, y0=-3.47, m=-2.39: y0+m*(x-x0), 'deriv': lambda x, m=-2.39: np.zeros_like(x,dtype=float)+m, 'domain': (-0.13229192758309727,0.2922919275830972), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.07999999999999996, y0=0.5299999999999999, m=1.6099999999999999: y0+m*(x-x0), 'deriv': lambda x, m=1.6099999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-0.21019403439913498,0.3701940343991349), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.07999999999999996, y0=4.53, m=5.61: y0+m*(x-x0), 'deriv': lambda x, m=5.61: np.zeros_like(x,dtype=float)+m, 'domain': (-0.016517819436648776,0.17651781943664868), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.07999999999999996, y0=8.53, m=9.61: y0+m*(x-x0), 'deriv': lambda x, m=9.61: np.zeros_like(x,dtype=float)+m, 'domain': (0.023075314030567944,0.13692468596943197), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.08, y0=-7.47, m=-2.3899999999999997: y0+m*(x-x0), 'deriv': lambda x, m=-2.3899999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (3.8677080724169026,4.292291927583097), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.08, y0=-3.47, m=1.6099999999999999: y0+m*(x-x0), 'deriv': lambda x, m=1.6099999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (3.7898059656008654,4.370194034399135), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.08, y0=0.5299999999999999, m=5.61: y0+m*(x-x0), 'deriv': lambda x, m=5.61: np.zeros_like(x,dtype=float)+m, 'domain': (3.9834821805633513,4.176517819436649), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.08, y0=4.53, m=9.61: y0+m*(x-x0), 'deriv': lambda x, m=9.61: np.zeros_like(x,dtype=float)+m, 'domain': (4.023075314030568,4.136924685969432), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.08, y0=8.53, m=13.61: y0+m*(x-x0), 'deriv': lambda x, m=13.61: np.zeros_like(x,dtype=float)+m, 'domain': (4.039697181454663,4.120302818545337), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.08, y0=-7.47, m=1.6100000000000003: y0+m*(x-x0), 'deriv': lambda x, m=1.6100000000000003: np.zeros_like(x,dtype=float)+m, 'domain': (7.789805965600865,8.370194034399136), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.08, y0=-3.47, m=5.609999999999999: y0+m*(x-x0), 'deriv': lambda x, m=5.609999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (7.983482180563351,8.176517819436649), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.08, y0=0.5299999999999999, m=9.61: y0+m*(x-x0), 'deriv': lambda x, m=9.61: np.zeros_like(x,dtype=float)+m, 'domain': (8.023075314030567,8.136924685969433), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.08, y0=4.53, m=13.61: y0+m*(x-x0), 'deriv': lambda x, m=13.61: np.zeros_like(x,dtype=float)+m, 'domain': (8.039697181454663,8.120302818545337), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.08, y0=8.53, m=17.61: y0+m*(x-x0), 'deriv': lambda x, m=17.61: np.zeros_like(x,dtype=float)+m, 'domain': (8.048817980533837,8.111182019466163), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s3_et_c_l01_solution_tendency.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.92, y0=-8.03, m=-15.95: y0+m*(x-x0), 'deriv': lambda x, m=-15.95: np.zeros_like(x,dtype=float)+m, 'domain': (-7.975064297143626,-7.864935702856374), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.92, y0=-4.03, m=-11.95: y0+m*(x-x0), 'deriv': lambda x, m=-11.95: np.zeros_like(x,dtype=float)+m, 'domain': (-7.9933836744743125,-7.846616325525687), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.92, y0=-0.030000000000000027, m=-7.95: y0+m*(x-x0), 'deriv': lambda x, m=-7.95: np.zeros_like(x,dtype=float)+m, 'domain': (-8.029826388442405,-7.810173611557594), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.92, y0=3.9699999999999998, m=-3.95: y0+m*(x-x0), 'deriv': lambda x, m=-3.95: np.zeros_like(x,dtype=float)+m, 'domain': (-8.135971242933941,-7.704028757066059), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.92, y0=7.97, m=0.04999999999999982: y0+m*(x-x0), 'deriv': lambda x, m=0.04999999999999982: np.zeros_like(x,dtype=float)+m, 'domain': (-8.798902058212503,-7.041097941787497), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.92, y0=-8.03, m=-11.95: y0+m*(x-x0), 'deriv': lambda x, m=-11.95: np.zeros_like(x,dtype=float)+m, 'domain': (-3.9933836744743125,-3.8466163255256873), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.92, y0=-4.03, m=-7.95: y0+m*(x-x0), 'deriv': lambda x, m=-7.95: np.zeros_like(x,dtype=float)+m, 'domain': (-4.029826388442406,-3.810173611557594), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.92, y0=-0.030000000000000027, m=-3.95: y0+m*(x-x0), 'deriv': lambda x, m=-3.95: np.zeros_like(x,dtype=float)+m, 'domain': (-4.135971242933941,-3.7040287570660584), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.92, y0=3.9699999999999998, m=0.04999999999999982: y0+m*(x-x0), 'deriv': lambda x, m=0.04999999999999982: np.zeros_like(x,dtype=float)+m, 'domain': (-4.798902058212503,-3.0410979417874966), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.92, y0=7.97, m=4.05: y0+m*(x-x0), 'deriv': lambda x, m=4.05: np.zeros_like(x,dtype=float)+m, 'domain': (-4.130948698837319,-3.709051301162681), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.07999999999999996, y0=-8.03, m=-7.949999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-7.949999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-0.029826388442406024,0.18982638844240596), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.07999999999999996, y0=-4.03, m=-3.95: y0+m*(x-x0), 'deriv': lambda x, m=-3.95: np.zeros_like(x,dtype=float)+m, 'domain': (-0.13597124293394153,0.29597124293394145), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.07999999999999996, y0=-0.030000000000000027, m=0.04999999999999993: y0+m*(x-x0), 'deriv': lambda x, m=0.04999999999999993: np.zeros_like(x,dtype=float)+m, 'domain': (-0.7989020582125034,0.9589020582125033), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.07999999999999996, y0=3.9699999999999998, m=4.05: y0+m*(x-x0), 'deriv': lambda x, m=4.05: np.zeros_like(x,dtype=float)+m, 'domain': (-0.130948698837319,0.2909486988373189), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.07999999999999996, y0=7.97, m=8.049999999999999: y0+m*(x-x0), 'deriv': lambda x, m=8.049999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-0.028482947172052075,0.188482947172052), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.08, y0=-8.03, m=-3.9499999999999993: y0+m*(x-x0), 'deriv': lambda x, m=-3.9499999999999993: np.zeros_like(x,dtype=float)+m, 'domain': (3.8640287570660585,4.295971242933941), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.08, y0=-4.03, m=0.04999999999999982: y0+m*(x-x0), 'deriv': lambda x, m=0.04999999999999982: np.zeros_like(x,dtype=float)+m, 'domain': (3.2010979417874967,4.958902058212503), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.08, y0=-0.030000000000000027, m=4.05: y0+m*(x-x0), 'deriv': lambda x, m=4.05: np.zeros_like(x,dtype=float)+m, 'domain': (3.8690513011626813,4.290948698837319), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.08, y0=3.9699999999999998, m=8.05: y0+m*(x-x0), 'deriv': lambda x, m=8.05: np.zeros_like(x,dtype=float)+m, 'domain': (3.971517052827948,4.188482947172052), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.08, y0=7.97, m=12.05: y0+m*(x-x0), 'deriv': lambda x, m=12.05: np.zeros_like(x,dtype=float)+m, 'domain': (4.007221135957976,4.152778864042024), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.08, y0=-8.03, m=0.05000000000000071: y0+m*(x-x0), 'deriv': lambda x, m=0.05000000000000071: np.zeros_like(x,dtype=float)+m, 'domain': (7.201097941787497,8.958902058212503), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.08, y0=-4.03, m=4.05: y0+m*(x-x0), 'deriv': lambda x, m=4.05: np.zeros_like(x,dtype=float)+m, 'domain': (7.869051301162681,8.290948698837319), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.08, y0=-0.030000000000000027, m=8.05: y0+m*(x-x0), 'deriv': lambda x, m=8.05: np.zeros_like(x,dtype=float)+m, 'domain': (7.971517052827948,8.188482947172051), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.08, y0=3.9699999999999998, m=12.05: y0+m*(x-x0), 'deriv': lambda x, m=12.05: np.zeros_like(x,dtype=float)+m, 'domain': (8.007221135957977,8.152778864042023), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.08, y0=7.97, m=16.05: y0+m*(x-x0), 'deriv': lambda x, m=16.05: np.zeros_like(x,dtype=float)+m, 'domain': (8.025277451778427,8.134722548221573), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s3_ps1_l03_slope_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.15, y0=-7.54, m=-14.690000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-14.690000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-8.185995659664382,-8.114004340335619), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.15, y0=-3.54, m=-10.690000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-10.690000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-8.19936353266001,-8.10063646733999), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.15, y0=0.4600000000000001, m=-6.69: y0+m*(x-x0), 'deriv': lambda x, m=-6.69: np.zeros_like(x,dtype=float)+m, 'domain': (-8.228352231413195,-8.071647768586805), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.15, y0=4.46, m=-2.6900000000000004: y0+m*(x-x0), 'deriv': lambda x, m=-2.6900000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-8.334677958088236,-7.965322041911765), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.15, y0=8.46, m=1.3100000000000005: y0+m*(x-x0), 'deriv': lambda x, m=1.3100000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-8.47159033777169,-7.828409662228311), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.15, y0=-7.54, m=-10.690000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-10.690000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-4.19936353266001,-4.10063646733999), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.15, y0=-3.54, m=-6.69: y0+m*(x-x0), 'deriv': lambda x, m=-6.69: np.zeros_like(x,dtype=float)+m, 'domain': (-4.228352231413195,-4.071647768586805), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.15, y0=0.4600000000000001, m=-2.6900000000000004: y0+m*(x-x0), 'deriv': lambda x, m=-2.6900000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-4.3346779580882355,-3.9653220419117647), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.15, y0=4.46, m=1.3099999999999996: y0+m*(x-x0), 'deriv': lambda x, m=1.3099999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (-4.47159033777169,-3.8284096622283106), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.15, y0=8.46, m=5.3100000000000005: y0+m*(x-x0), 'deriv': lambda x, m=5.3100000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-4.24808745093051,-4.051912549069491), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.15000000000000002, y0=-7.54, m=-6.69: y0+m*(x-x0), 'deriv': lambda x, m=-6.69: np.zeros_like(x,dtype=float)+m, 'domain': (-0.228352231413195,-0.07164776858680505), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.15000000000000002, y0=-3.54, m=-2.69: y0+m*(x-x0), 'deriv': lambda x, m=-2.69: np.zeros_like(x,dtype=float)+m, 'domain': (-0.3346779580882355,0.03467795808823551), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.15000000000000002, y0=0.4600000000000001, m=1.31: y0+m*(x-x0), 'deriv': lambda x, m=1.31: np.zeros_like(x,dtype=float)+m, 'domain': (-0.4715903377716897,0.17159033777168964), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.15000000000000002, y0=4.46, m=5.31: y0+m*(x-x0), 'deriv': lambda x, m=5.31: np.zeros_like(x,dtype=float)+m, 'domain': (-0.24808745093050955,-0.05191254906949051), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.15000000000000002, y0=8.46, m=9.31: y0+m*(x-x0), 'deriv': lambda x, m=9.31: np.zeros_like(x,dtype=float)+m, 'domain': (-0.2066024536742379,-0.09339754632576212), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.85, y0=-7.54, m=-2.69: y0+m*(x-x0), 'deriv': lambda x, m=-2.69: np.zeros_like(x,dtype=float)+m, 'domain': (3.6653220419117645,4.034677958088236), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.85, y0=-3.54, m=1.31: y0+m*(x-x0), 'deriv': lambda x, m=1.31: np.zeros_like(x,dtype=float)+m, 'domain': (3.5284096622283103,4.17159033777169), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.85, y0=0.4600000000000001, m=5.3100000000000005: y0+m*(x-x0), 'deriv': lambda x, m=5.3100000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (3.7519125490694907,3.9480874509305095), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.85, y0=4.46, m=9.31: y0+m*(x-x0), 'deriv': lambda x, m=9.31: np.zeros_like(x,dtype=float)+m, 'domain': (3.793397546325762,3.906602453674238), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.85, y0=8.46, m=13.31: y0+m*(x-x0), 'deriv': lambda x, m=13.31: np.zeros_like(x,dtype=float)+m, 'domain': (3.810292227856489,3.8897077721435114), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.85, y0=-7.54, m=1.3099999999999996: y0+m*(x-x0), 'deriv': lambda x, m=1.3099999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (7.52840966222831,8.17159033777169), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.85, y0=-3.54, m=5.31: y0+m*(x-x0), 'deriv': lambda x, m=5.31: np.zeros_like(x,dtype=float)+m, 'domain': (7.75191254906949,7.948087450930509), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.85, y0=0.4600000000000001, m=9.31: y0+m*(x-x0), 'deriv': lambda x, m=9.31: np.zeros_like(x,dtype=float)+m, 'domain': (7.793397546325762,7.906602453674237), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.85, y0=4.46, m=13.309999999999999: y0+m*(x-x0), 'deriv': lambda x, m=13.309999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (7.810292227856488,7.889707772143511), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.85, y0=8.46, m=17.310000000000002: y0+m*(x-x0), 'deriv': lambda x, m=17.310000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (7.819432824873261,7.880567175126738), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s3_ps1_l05_solution_tendency.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.72, y0=-7.85, m=0.1299999999999999: y0+m*(x-x0), 'deriv': lambda x, m=0.1299999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-8.443908600660563,-6.996091399339436), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.72, y0=-3.85, m=-3.8699999999999997: y0+m*(x-x0), 'deriv': lambda x, m=-3.8699999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (-7.9026318850114325,-7.537368114988567), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.72, y0=0.15000000000000002, m=-7.87: y0+m*(x-x0), 'deriv': lambda x, m=-7.87: np.zeros_like(x,dtype=float)+m, 'domain': (-7.812017448397338,-7.627982551602662), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.72, y0=4.15, m=-11.870000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-11.870000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-7.781282490465624,-7.658717509534376), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.72, y0=8.15, m=-15.870000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-15.870000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-7.765907691604586,-7.674092308395413), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.7199999999999998, y0=-7.85, m=4.13: y0+m*(x-x0), 'deriv': lambda x, m=4.13: np.zeros_like(x,dtype=float)+m, 'domain': (-3.8917913385764247,-3.548208661423575), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.7199999999999998, y0=-3.85, m=0.13000000000000034: y0+m*(x-x0), 'deriv': lambda x, m=0.13000000000000034: np.zeros_like(x,dtype=float)+m, 'domain': (-4.4439086006605635,-2.996091399339436), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.7199999999999998, y0=0.15000000000000002, m=-3.8699999999999997: y0+m*(x-x0), 'deriv': lambda x, m=-3.8699999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (-3.9026318850114325,-3.537368114988567), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.7199999999999998, y0=4.15, m=-7.87: y0+m*(x-x0), 'deriv': lambda x, m=-7.87: np.zeros_like(x,dtype=float)+m, 'domain': (-3.8120174483973384,-3.627982551602661), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.7199999999999998, y0=8.15, m=-11.870000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-11.870000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-3.781282490465624,-3.6587175095343754), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.28, y0=-7.85, m=8.129999999999999: y0+m*(x-x0), 'deriv': lambda x, m=8.129999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (0.19088072754069546,0.3691192724593046), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.28, y0=-3.85, m=4.13: y0+m*(x-x0), 'deriv': lambda x, m=4.13: np.zeros_like(x,dtype=float)+m, 'domain': (0.10820866142357505,0.451791338576425), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.28, y0=0.15000000000000002, m=0.13: y0+m*(x-x0), 'deriv': lambda x, m=0.13: np.zeros_like(x,dtype=float)+m, 'domain': (-0.4439086006605636,1.0039086006605635), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.28, y0=4.15, m=-3.87: y0+m*(x-x0), 'deriv': lambda x, m=-3.87: np.zeros_like(x,dtype=float)+m, 'domain': (0.09736811498856726,0.4626318850114328), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.28, y0=8.15, m=-7.87: y0+m*(x-x0), 'deriv': lambda x, m=-7.87: np.zeros_like(x,dtype=float)+m, 'domain': (0.18798255160266142,0.37201744839733863), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.28, y0=-7.85, m=12.129999999999999: y0+m*(x-x0), 'deriv': lambda x, m=12.129999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (4.2200221029990495,4.339977897000951), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.28, y0=-3.85, m=8.13: y0+m*(x-x0), 'deriv': lambda x, m=8.13: np.zeros_like(x,dtype=float)+m, 'domain': (4.190880727540696,4.369119272459304), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.28, y0=0.15000000000000002, m=4.13: y0+m*(x-x0), 'deriv': lambda x, m=4.13: np.zeros_like(x,dtype=float)+m, 'domain': (4.108208661423575,4.451791338576426), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.28, y0=4.15, m=0.1299999999999999: y0+m*(x-x0), 'deriv': lambda x, m=0.1299999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (3.5560913993394365,5.003908600660564), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.28, y0=8.15, m=-3.87: y0+m*(x-x0), 'deriv': lambda x, m=-3.87: np.zeros_like(x,dtype=float)+m, 'domain': (4.0973681149885675,4.462631885011433), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.28, y0=-7.85, m=16.13: y0+m*(x-x0), 'deriv': lambda x, m=16.13: np.zeros_like(x,dtype=float)+m, 'domain': (8.234829439585003,8.325170560414996), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.28, y0=-3.85, m=12.129999999999999: y0+m*(x-x0), 'deriv': lambda x, m=12.129999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (8.220022102999048,8.339977897000951), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.28, y0=0.15000000000000002, m=8.129999999999999: y0+m*(x-x0), 'deriv': lambda x, m=8.129999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (8.190880727540694,8.369119272459304), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.28, y0=4.15, m=4.129999999999999: y0+m*(x-x0), 'deriv': lambda x, m=4.129999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (8.108208661423575,8.451791338576424), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.28, y0=8.15, m=0.129999999999999: y0+m*(x-x0), 'deriv': lambda x, m=0.129999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (7.556091399339436,9.003908600660562), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s3_ps2_l02_slope_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.75, y0=-7.96, m=-8.75: y0+m*(x-x0), 'deriv': lambda x, m=-8.75: np.zeros_like(x,dtype=float)+m, 'domain': (-8.815857022873224,-8.684142977126776), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.75, y0=-3.96, m=-8.75: y0+m*(x-x0), 'deriv': lambda x, m=-8.75: np.zeros_like(x,dtype=float)+m, 'domain': (-8.815857022873224,-8.684142977126776), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.75, y0=0.040000000000000036, m=-8.75: y0+m*(x-x0), 'deriv': lambda x, m=-8.75: np.zeros_like(x,dtype=float)+m, 'domain': (-8.815857022873224,-8.684142977126776), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.75, y0=4.04, m=-8.75: y0+m*(x-x0), 'deriv': lambda x, m=-8.75: np.zeros_like(x,dtype=float)+m, 'domain': (-8.815857022873224,-8.684142977126776), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.75, y0=8.04, m=-8.75: y0+m*(x-x0), 'deriv': lambda x, m=-8.75: np.zeros_like(x,dtype=float)+m, 'domain': (-8.815857022873224,-8.684142977126776), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.75, y0=-7.96, m=-4.75: y0+m*(x-x0), 'deriv': lambda x, m=-4.75: np.zeros_like(x,dtype=float)+m, 'domain': (-4.869486079008908,-4.630513920991092), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.75, y0=-3.96, m=-4.75: y0+m*(x-x0), 'deriv': lambda x, m=-4.75: np.zeros_like(x,dtype=float)+m, 'domain': (-4.869486079008908,-4.630513920991092), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.75, y0=0.040000000000000036, m=-4.75: y0+m*(x-x0), 'deriv': lambda x, m=-4.75: np.zeros_like(x,dtype=float)+m, 'domain': (-4.869486079008908,-4.630513920991092), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.75, y0=4.04, m=-4.75: y0+m*(x-x0), 'deriv': lambda x, m=-4.75: np.zeros_like(x,dtype=float)+m, 'domain': (-4.869486079008908,-4.630513920991092), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.75, y0=8.04, m=-4.75: y0+m*(x-x0), 'deriv': lambda x, m=-4.75: np.zeros_like(x,dtype=float)+m, 'domain': (-4.869486079008908,-4.630513920991092), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.75, y0=-7.96, m=-0.75: y0+m*(x-x0), 'deriv': lambda x, m=-0.75: np.zeros_like(x,dtype=float)+m, 'domain': (-1.214,-0.2859999999999999), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.75, y0=-3.96, m=-0.75: y0+m*(x-x0), 'deriv': lambda x, m=-0.75: np.zeros_like(x,dtype=float)+m, 'domain': (-1.214,-0.2859999999999999), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.75, y0=0.040000000000000036, m=-0.75: y0+m*(x-x0), 'deriv': lambda x, m=-0.75: np.zeros_like(x,dtype=float)+m, 'domain': (-1.214,-0.2859999999999999), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.75, y0=4.04, m=-0.75: y0+m*(x-x0), 'deriv': lambda x, m=-0.75: np.zeros_like(x,dtype=float)+m, 'domain': (-1.214,-0.2859999999999999), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.75, y0=8.04, m=-0.75: y0+m*(x-x0), 'deriv': lambda x, m=-0.75: np.zeros_like(x,dtype=float)+m, 'domain': (-1.214,-0.2859999999999999), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.25, y0=-7.96, m=3.25: y0+m*(x-x0), 'deriv': lambda x, m=3.25: np.zeros_like(x,dtype=float)+m, 'domain': (3.0794302076742364,3.4205697923257636), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.25, y0=-3.96, m=3.25: y0+m*(x-x0), 'deriv': lambda x, m=3.25: np.zeros_like(x,dtype=float)+m, 'domain': (3.0794302076742364,3.4205697923257636), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.25, y0=0.040000000000000036, m=3.25: y0+m*(x-x0), 'deriv': lambda x, m=3.25: np.zeros_like(x,dtype=float)+m, 'domain': (3.0794302076742364,3.4205697923257636), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.25, y0=4.04, m=3.25: y0+m*(x-x0), 'deriv': lambda x, m=3.25: np.zeros_like(x,dtype=float)+m, 'domain': (3.0794302076742364,3.4205697923257636), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.25, y0=8.04, m=3.25: y0+m*(x-x0), 'deriv': lambda x, m=3.25: np.zeros_like(x,dtype=float)+m, 'domain': (3.0794302076742364,3.4205697923257636), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.25, y0=-7.96, m=7.25: y0+m*(x-x0), 'deriv': lambda x, m=7.25: np.zeros_like(x,dtype=float)+m, 'domain': (7.170750309660522,7.329249690339478), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.25, y0=-3.96, m=7.25: y0+m*(x-x0), 'deriv': lambda x, m=7.25: np.zeros_like(x,dtype=float)+m, 'domain': (7.170750309660522,7.329249690339478), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.25, y0=0.040000000000000036, m=7.25: y0+m*(x-x0), 'deriv': lambda x, m=7.25: np.zeros_like(x,dtype=float)+m, 'domain': (7.170750309660522,7.329249690339478), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.25, y0=4.04, m=7.25: y0+m*(x-x0), 'deriv': lambda x, m=7.25: np.zeros_like(x,dtype=float)+m, 'domain': (7.170750309660522,7.329249690339478), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.25, y0=8.04, m=7.25: y0+m*(x-x0), 'deriv': lambda x, m=7.25: np.zeros_like(x,dtype=float)+m, 'domain': (7.170750309660522,7.329249690339478), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s3_ps2_l04_solution_tendency.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.83, y0=-7.72, m=-1.1100000000000003: y0+m*(x-x0), 'deriv': lambda x, m=-1.1100000000000003: np.zeros_like(x,dtype=float)+m, 'domain': (-8.332000835151595,-7.327999164848405), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.83, y0=-3.7199999999999998, m=-5.11: y0+m*(x-x0), 'deriv': lambda x, m=-5.11: np.zeros_like(x,dtype=float)+m, 'domain': (-7.974038863010182,-7.685961136989818), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.83, y0=0.28, m=-9.11: y0+m*(x-x0), 'deriv': lambda x, m=-9.11: np.zeros_like(x,dtype=float)+m, 'domain': (-7.911835557148632,-7.748164442851368), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.83, y0=4.28, m=-13.11: y0+m*(x-x0), 'deriv': lambda x, m=-13.11: np.zeros_like(x,dtype=float)+m, 'domain': (-7.8870425339479375,-7.772957466052063), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.83, y0=8.28, m=-17.11: y0+m*(x-x0), 'deriv': lambda x, m=-17.11: np.zeros_like(x,dtype=float)+m, 'domain': (-7.8737593410332805,-7.78624065896672), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.83, y0=-7.72, m=2.8899999999999997: y0+m*(x-x0), 'deriv': lambda x, m=2.8899999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (-4.075248687812668,-3.5847513121873313), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.83, y0=-3.7199999999999998, m=-1.1100000000000003: y0+m*(x-x0), 'deriv': lambda x, m=-1.1100000000000003: np.zeros_like(x,dtype=float)+m, 'domain': (-4.332000835151595,-3.327999164848405), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.83, y0=0.28, m=-5.11: y0+m*(x-x0), 'deriv': lambda x, m=-5.11: np.zeros_like(x,dtype=float)+m, 'domain': (-3.9740388630101817,-3.6859611369898184), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.83, y0=4.28, m=-9.11: y0+m*(x-x0), 'deriv': lambda x, m=-9.11: np.zeros_like(x,dtype=float)+m, 'domain': (-3.911835557148632,-3.748164442851368), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.83, y0=8.28, m=-13.11: y0+m*(x-x0), 'deriv': lambda x, m=-13.11: np.zeros_like(x,dtype=float)+m, 'domain': (-3.887042533947937,-3.772957466052063), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.17000000000000004, y0=-7.72, m=6.89: y0+m*(x-x0), 'deriv': lambda x, m=6.89: np.zeros_like(x,dtype=float)+m, 'domain': (0.06227528701297037,0.2777247129870297), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.17000000000000004, y0=-3.7199999999999998, m=2.8899999999999997: y0+m*(x-x0), 'deriv': lambda x, m=2.8899999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (-0.07524868781266866,0.41524868781266877), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.17000000000000004, y0=0.28, m=-1.1099999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-1.1099999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-0.3320008351515952,0.6720008351515953), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.17000000000000004, y0=4.28, m=-5.11: y0+m*(x-x0), 'deriv': lambda x, m=-5.11: np.zeros_like(x,dtype=float)+m, 'domain': (0.025961136989818168,0.3140388630101819), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.17000000000000004, y0=8.28, m=-9.11: y0+m*(x-x0), 'deriv': lambda x, m=-9.11: np.zeros_like(x,dtype=float)+m, 'domain': (0.08816444285136803,0.25183555714863204), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.17, y0=-7.72, m=10.89: y0+m*(x-x0), 'deriv': lambda x, m=10.89: np.zeros_like(x,dtype=float)+m, 'domain': (4.101418020301097,4.238581979698903), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.17, y0=-3.7199999999999998, m=6.89: y0+m*(x-x0), 'deriv': lambda x, m=6.89: np.zeros_like(x,dtype=float)+m, 'domain': (4.06227528701297,4.2777247129870295), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.17, y0=0.28, m=2.8899999999999997: y0+m*(x-x0), 'deriv': lambda x, m=2.8899999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (3.924751312187331,4.415248687812668), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.17, y0=4.28, m=-1.1100000000000003: y0+m*(x-x0), 'deriv': lambda x, m=-1.1100000000000003: np.zeros_like(x,dtype=float)+m, 'domain': (3.6679991648484047,4.672000835151595), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.17, y0=8.28, m=-5.109999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-5.109999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (4.025961136989818,4.314038863010182), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.17, y0=-7.72, m=14.89: y0+m*(x-x0), 'deriv': lambda x, m=14.89: np.zeros_like(x,dtype=float)+m, 'domain': (8.119743833614745,8.220256166385255), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.17, y0=-3.7199999999999998, m=10.89: y0+m*(x-x0), 'deriv': lambda x, m=10.89: np.zeros_like(x,dtype=float)+m, 'domain': (8.101418020301097,8.238581979698903), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.17, y0=0.28, m=6.89: y0+m*(x-x0), 'deriv': lambda x, m=6.89: np.zeros_like(x,dtype=float)+m, 'domain': (8.06227528701297,8.27772471298703), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.17, y0=4.28, m=2.8899999999999997: y0+m*(x-x0), 'deriv': lambda x, m=2.8899999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (7.924751312187332,8.415248687812669), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.17, y0=8.28, m=-1.1099999999999994: y0+m*(x-x0), 'deriv': lambda x, m=-1.1099999999999994: np.zeros_like(x,dtype=float)+m, 'domain': (7.667999164848404,8.672000835151595), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s3_wu2_l01_solution_tendency.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.2, y0=-7.4, m=-1.799999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-1.799999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-8.457390753524674,-7.942609246475324), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.2, y0=-3.4, m=-5.799999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-5.799999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-8.290050665749051,-8.109949334250947), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.2, y0=0.6, m=-9.799999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-9.799999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-8.253802254647098,-8.1461977453529), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.2, y0=4.6, m=-13.799999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-13.799999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-8.238305358106588,-8.16169464189341), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.2, y0=8.6, m=-17.799999999999997: y0+m*(x-x0), 'deriv': lambda x, m=-17.799999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (-8.229728403951611,-8.170271596048387), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.2, y0=-7.4, m=2.2: y0+m*(x-x0), 'deriv': lambda x, m=2.2: np.zeros_like(x,dtype=float)+m, 'domain': (-4.419315560479627,-3.9806844395203727), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.2, y0=-3.4, m=-1.8000000000000003: y0+m*(x-x0), 'deriv': lambda x, m=-1.8000000000000003: np.zeros_like(x,dtype=float)+m, 'domain': (-4.457390753524675,-3.942609246475325), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.2, y0=0.6, m=-5.8: y0+m*(x-x0), 'deriv': lambda x, m=-5.8: np.zeros_like(x,dtype=float)+m, 'domain': (-4.290050665749052,-4.109949334250948), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.2, y0=4.6, m=-9.8: y0+m*(x-x0), 'deriv': lambda x, m=-9.8: np.zeros_like(x,dtype=float)+m, 'domain': (-4.253802254647099,-4.146197745352901), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.2, y0=8.6, m=-13.8: y0+m*(x-x0), 'deriv': lambda x, m=-13.8: np.zeros_like(x,dtype=float)+m, 'domain': (-4.238305358106589,-4.161694641893411), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.19999999999999996, y0=-7.4, m=6.2: y0+m*(x-x0), 'deriv': lambda x, m=6.2: np.zeros_like(x,dtype=float)+m, 'domain': (-0.2843931925770488,-0.1156068074229511), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.19999999999999996, y0=-3.4, m=2.2: y0+m*(x-x0), 'deriv': lambda x, m=2.2: np.zeros_like(x,dtype=float)+m, 'domain': (-0.4193155604796275,0.01931556047962754), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.19999999999999996, y0=0.6, m=-1.7999999999999998: y0+m*(x-x0), 'deriv': lambda x, m=-1.7999999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (-0.457390753524675,0.0573907535246751), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.19999999999999996, y0=4.6, m=-5.8: y0+m*(x-x0), 'deriv': lambda x, m=-5.8: np.zeros_like(x,dtype=float)+m, 'domain': (-0.29005066574905247,-0.10994933425094747), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.19999999999999996, y0=8.6, m=-9.799999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-9.799999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-0.25380225464709905,-0.14619774535290087), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.8, y0=-7.4, m=10.2: y0+m*(x-x0), 'deriv': lambda x, m=10.2: np.zeros_like(x,dtype=float)+m, 'domain': (3.7482871452633733,3.8517128547366264), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.8, y0=-3.4, m=6.199999999999999: y0+m*(x-x0), 'deriv': lambda x, m=6.199999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (3.715606807422951,3.8843931925770487), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.8, y0=0.6, m=2.1999999999999997: y0+m*(x-x0), 'deriv': lambda x, m=2.1999999999999997: np.zeros_like(x,dtype=float)+m, 'domain': (3.5806844395203723,4.019315560479628), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.8, y0=4.6, m=-1.7999999999999998: y0+m*(x-x0), 'deriv': lambda x, m=-1.7999999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (3.5426092464753247,4.0573907535246745), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.8, y0=8.6, m=-5.8: y0+m*(x-x0), 'deriv': lambda x, m=-5.8: np.zeros_like(x,dtype=float)+m, 'domain': (3.7099493342509473,3.8900506657490523), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.8, y0=-7.4, m=14.2: y0+m*(x-x0), 'deriv': lambda x, m=14.2: np.zeros_like(x,dtype=float)+m, 'domain': (7.762768264452762,7.837231735547237), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.8, y0=-3.4, m=10.2: y0+m*(x-x0), 'deriv': lambda x, m=10.2: np.zeros_like(x,dtype=float)+m, 'domain': (7.748287145263373,7.851712854736626), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.8, y0=0.6, m=6.2: y0+m*(x-x0), 'deriv': lambda x, m=6.2: np.zeros_like(x,dtype=float)+m, 'domain': (7.715606807422951,7.884393192577049), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.8, y0=4.6, m=2.2: y0+m*(x-x0), 'deriv': lambda x, m=2.2: np.zeros_like(x,dtype=float)+m, 'domain': (7.580684439520373,8.019315560479628), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.8, y0=8.6, m=-1.7999999999999998: y0+m*(x-x0), 'deriv': lambda x, m=-1.7999999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (7.542609246475325,8.057390753524675), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_s3_xp_l02_solution_tendency.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.2))
make_context_graph(ax, [
    {'expr': lambda x, A=50.0, k=0.18: A*np.exp(k*x), 'deriv': lambda x, A=50.0, k=0.18: A*k*np.exp(k*x), 'color': 'steelblue', 'label': None}
], xmin=0, xmax=8, ymin=0, ymax=250, xlabel='Time', ylabel='Quantity', title='')
save_graph(fig, 'u7_s5_et_a_c01_exponential_graph.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.2))
make_context_graph(ax, [
    {'expr': lambda x, A=50.0, k=-0.2: A*np.exp(k*x), 'deriv': lambda x, A=50.0, k=-0.2: A*k*np.exp(k*x), 'color': 'steelblue', 'label': None}
], xmin=0, xmax=8, ymin=0, ymax=60, xlabel='Time', ylabel='Quantity', title='')
save_graph(fig, 'u7_s5_note_yti04_exponential_graph.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.2))
make_context_graph(ax, [
    {'expr': lambda x, A=40.0, k=-0.14: A*np.exp(k*x), 'deriv': lambda x, A=40.0, k=-0.14: A*k*np.exp(k*x), 'color': 'steelblue', 'label': None}
], xmin=0, xmax=10, ymin=0, ymax=60, xlabel='Time', ylabel='Quantity', title='')
save_graph(fig, 'u7_s5_ps1_c04_exponential_graph.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.2))
make_context_graph(ax, [
    {'expr': lambda x, A=50.0, k=0.24: A*np.exp(k*x), 'deriv': lambda x, A=50.0, k=0.24: A*k*np.exp(k*x), 'color': 'steelblue', 'label': None}
], xmin=0, xmax=6, ymin=0, ymax=250, xlabel='Time', ylabel='Quantity', title='')
save_graph(fig, 'u7_s5_ps2_c03_exponential_graph.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.2))
make_context_graph(ax, [
    {'expr': lambda x, A=40.0, k=0.18: A*np.exp(k*x), 'deriv': lambda x, A=40.0, k=0.18: A*k*np.exp(k*x), 'color': 'steelblue', 'label': None}
], xmin=0, xmax=8, ymin=0, ymax=200, xlabel='Time', ylabel='Quantity', title='')
save_graph(fig, 'u7_s5_wu2_c01_exponential_graph.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.2))
make_context_graph(ax, [
    {'expr': lambda x, A=50.0, k=-0.14: A*np.exp(k*x), 'deriv': lambda x, A=50.0, k=-0.14: A*k*np.exp(k*x), 'color': 'steelblue', 'label': None}
], xmin=0, xmax=6, ymin=0, ymax=60, xlabel='Time', ylabel='Quantity', title='')
save_graph(fig, 'u7_s5_xp_c01_exponential_graph.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.92, y0=-8.58, m=1.6600000000000001: y0+m*(x-x0), 'deriv': lambda x, m=1.6600000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-8.198646822249394,-7.641353177750605), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.92, y0=-4.58, m=-2.34: y0+m*(x-x0), 'deriv': lambda x, m=-2.34: np.zeros_like(x,dtype=float)+m, 'domain': (-8.132204091188772,-7.707795908811228), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.92, y0=-0.58, m=-6.34: y0+m*(x-x0), 'deriv': lambda x, m=-6.34: np.zeros_like(x,dtype=float)+m, 'domain': (-8.004133381313867,-7.835866618686133), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.92, y0=3.42, m=-10.34: y0+m*(x-x0), 'deriv': lambda x, m=-10.34: np.zeros_like(x,dtype=float)+m, 'domain': (-7.971981839619386,-7.868018160380614), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.92, y0=7.42, m=-14.34: y0+m*(x-x0), 'deriv': lambda x, m=-14.34: np.zeros_like(x,dtype=float)+m, 'domain': (-7.95756567414723,-7.88243432585277), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.92, y0=-8.58, m=5.66: y0+m*(x-x0), 'deriv': lambda x, m=5.66: np.zeros_like(x,dtype=float)+m, 'domain': (-4.013951271129828,-3.826048728870171), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.92, y0=-4.58, m=1.6600000000000001: y0+m*(x-x0), 'deriv': lambda x, m=1.6600000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-4.198646822249395,-3.641353177750605), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.92, y0=-0.58, m=-2.34: y0+m*(x-x0), 'deriv': lambda x, m=-2.34: np.zeros_like(x,dtype=float)+m, 'domain': (-4.132204091188772,-3.7077959088112284), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.92, y0=3.42, m=-6.34: y0+m*(x-x0), 'deriv': lambda x, m=-6.34: np.zeros_like(x,dtype=float)+m, 'domain': (-4.004133381313867,-3.835866618686133), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.92, y0=7.42, m=-10.34: y0+m*(x-x0), 'deriv': lambda x, m=-10.34: np.zeros_like(x,dtype=float)+m, 'domain': (-3.9719818396193856,-3.8680181603806143), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.07999999999999996, y0=-8.58, m=9.66: y0+m*(x-x0), 'deriv': lambda x, m=9.66: np.zeros_like(x,dtype=float)+m, 'domain': (0.02439651735252523,0.13560348264747468), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.07999999999999996, y0=-4.58, m=5.66: y0+m*(x-x0), 'deriv': lambda x, m=5.66: np.zeros_like(x,dtype=float)+m, 'domain': (-0.013951271129828857,0.17395127112982878), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.07999999999999996, y0=-0.58, m=1.66: y0+m*(x-x0), 'deriv': lambda x, m=1.66: np.zeros_like(x,dtype=float)+m, 'domain': (-0.19864682224939484,0.35864682224939476), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.07999999999999996, y0=3.42, m=-2.34: y0+m*(x-x0), 'deriv': lambda x, m=-2.34: np.zeros_like(x,dtype=float)+m, 'domain': (-0.13220409118877158,0.2922040911887715), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.07999999999999996, y0=7.42, m=-6.34: y0+m*(x-x0), 'deriv': lambda x, m=-6.34: np.zeros_like(x,dtype=float)+m, 'domain': (-0.004133381313866827,0.16413338131386673), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.08, y0=-8.58, m=13.66: y0+m*(x-x0), 'deriv': lambda x, m=13.66: np.zeros_like(x,dtype=float)+m, 'domain': (4.040574025573868,4.119425974426132), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.08, y0=-4.58, m=9.66: y0+m*(x-x0), 'deriv': lambda x, m=9.66: np.zeros_like(x,dtype=float)+m, 'domain': (4.024396517352526,4.135603482647475), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.08, y0=-0.58, m=5.66: y0+m*(x-x0), 'deriv': lambda x, m=5.66: np.zeros_like(x,dtype=float)+m, 'domain': (3.986048728870171,4.1739512711298286), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.08, y0=3.42, m=1.6600000000000001: y0+m*(x-x0), 'deriv': lambda x, m=1.6600000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (3.801353177750605,4.358646822249395), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.08, y0=7.42, m=-2.34: y0+m*(x-x0), 'deriv': lambda x, m=-2.34: np.zeros_like(x,dtype=float)+m, 'domain': (3.8677959088112286,4.292204091188772), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.08, y0=-8.58, m=17.66: y0+m*(x-x0), 'deriv': lambda x, m=17.66: np.zeros_like(x,dtype=float)+m, 'domain': (8.049471328053984,8.110528671946016), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.08, y0=-4.58, m=13.66: y0+m*(x-x0), 'deriv': lambda x, m=13.66: np.zeros_like(x,dtype=float)+m, 'domain': (8.040574025573868,8.119425974426132), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.08, y0=-0.58, m=9.66: y0+m*(x-x0), 'deriv': lambda x, m=9.66: np.zeros_like(x,dtype=float)+m, 'domain': (8.024396517352525,8.135603482647475), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.08, y0=3.42, m=5.66: y0+m*(x-x0), 'deriv': lambda x, m=5.66: np.zeros_like(x,dtype=float)+m, 'domain': (7.986048728870172,8.173951271129829), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.08, y0=7.42, m=1.6600000000000001: y0+m*(x-x0), 'deriv': lambda x, m=1.6600000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (7.801353177750605,8.358646822249394), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_sum_v1_q07_sum_field_mc.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.51, y0=-7.98, m=-0.5299999999999994: y0+m*(x-x0), 'deriv': lambda x, m=-0.5299999999999994: np.zeros_like(x,dtype=float)+m, 'domain': (-9.287544151973137,-7.732455848026863), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.51, y0=-3.98, m=-4.529999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-4.529999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-8.699693503368287,-8.320306496631712), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.51, y0=0.020000000000000018, m=-8.53: y0+m*(x-x0), 'deriv': lambda x, m=-8.53: np.zeros_like(x,dtype=float)+m, 'domain': (-8.612463590108279,-8.407536409891721), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.51, y0=4.02, m=-12.53: y0+m*(x-x0), 'deriv': lambda x, m=-12.53: np.zeros_like(x,dtype=float)+m, 'domain': (-8.580008841615477,-8.439991158384522), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.51, y0=8.02, m=-16.53: y0+m*(x-x0), 'deriv': lambda x, m=-16.53: np.zeros_like(x,dtype=float)+m, 'domain': (-8.563139389347329,-8.45686061065267), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.51, y0=-7.98, m=3.4700000000000006: y0+m*(x-x0), 'deriv': lambda x, m=3.4700000000000006: np.zeros_like(x,dtype=float)+m, 'domain': (-4.7536850520418525,-4.266314947958147), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.51, y0=-3.98, m=-0.5299999999999998: y0+m*(x-x0), 'deriv': lambda x, m=-0.5299999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (-5.287544151973137,-3.732455848026863), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.51, y0=0.020000000000000018, m=-4.529999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-4.529999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-4.699693503368288,-4.320306496631711), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.51, y0=4.02, m=-8.53: y0+m*(x-x0), 'deriv': lambda x, m=-8.53: np.zeros_like(x,dtype=float)+m, 'domain': (-4.612463590108279,-4.407536409891721), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.51, y0=8.02, m=-12.53: y0+m*(x-x0), 'deriv': lambda x, m=-12.53: np.zeros_like(x,dtype=float)+m, 'domain': (-4.580008841615477,-4.439991158384522), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.51, y0=-7.98, m=7.470000000000001: y0+m*(x-x0), 'deriv': lambda x, m=7.470000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-0.6267629512806121,-0.39323704871938797), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.51, y0=-3.98, m=3.4699999999999998: y0+m*(x-x0), 'deriv': lambda x, m=3.4699999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (-0.7536850520418525,-0.2663149479581475), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.51, y0=0.020000000000000018, m=-0.53: y0+m*(x-x0), 'deriv': lambda x, m=-0.53: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2875441519731368,0.26754415197313675), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.51, y0=4.02, m=-4.529999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-4.529999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-0.6996935033682885,-0.32030649663171157), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.51, y0=8.02, m=-8.53: y0+m*(x-x0), 'deriv': lambda x, m=-8.53: np.zeros_like(x,dtype=float)+m, 'domain': (-0.6124635901082787,-0.4075364098917214), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.49, y0=-7.98, m=11.47: y0+m*(x-x0), 'deriv': lambda x, m=11.47: np.zeros_like(x,dtype=float)+m, 'domain': (3.413568048183649,3.5664319518163516), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.49, y0=-3.98, m=7.470000000000001: y0+m*(x-x0), 'deriv': lambda x, m=7.470000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (3.373237048719388,3.6067629512806123), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.49, y0=0.020000000000000018, m=3.47: y0+m*(x-x0), 'deriv': lambda x, m=3.47: np.zeros_like(x,dtype=float)+m, 'domain': (3.246314947958148,3.7336850520418525), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.49, y0=4.02, m=-0.5299999999999994: y0+m*(x-x0), 'deriv': lambda x, m=-0.5299999999999994: np.zeros_like(x,dtype=float)+m, 'domain': (2.7124558480268632,4.267544151973137), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.49, y0=8.02, m=-4.529999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-4.529999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (3.3003064966317117,3.6796935033682887), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.49, y0=-7.98, m=15.47: y0+m*(x-x0), 'deriv': lambda x, m=15.47: np.zeros_like(x,dtype=float)+m, 'domain': (7.433234181893062,7.546765818106938), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.49, y0=-3.98, m=11.47: y0+m*(x-x0), 'deriv': lambda x, m=11.47: np.zeros_like(x,dtype=float)+m, 'domain': (7.413568048183649,7.566431951816352), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.49, y0=0.020000000000000018, m=7.470000000000001: y0+m*(x-x0), 'deriv': lambda x, m=7.470000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (7.373237048719388,7.606762951280612), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.49, y0=4.02, m=3.4700000000000006: y0+m*(x-x0), 'deriv': lambda x, m=3.4700000000000006: np.zeros_like(x,dtype=float)+m, 'domain': (7.2463149479581475,7.733685052041853), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.49, y0=8.02, m=-0.5299999999999994: y0+m*(x-x0), 'deriv': lambda x, m=-0.5299999999999994: np.zeros_like(x,dtype=float)+m, 'domain': (6.712455848026863,8.267544151973137), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_sum_v1_q17_sum_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.2))
make_context_graph(ax, [
    {'expr': lambda x, A=30.0, k=-0.2: A*np.exp(k*x), 'deriv': lambda x, A=30.0, k=-0.2: A*k*np.exp(k*x), 'color': 'steelblue', 'label': None}
], xmin=0, xmax=6, ymin=0, ymax=60, xlabel='Time', ylabel='Quantity', title='')
save_graph(fig, 'u7_sum_v1_q20_sum_exp_graph.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.41, y0=-8.56, m=8.41: y0+m*(x-x0), 'deriv': lambda x, m=8.41: np.zeros_like(x,dtype=float)+m, 'domain': (-8.473760115679392,-8.346239884320608), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.41, y0=-4.56, m=8.41: y0+m*(x-x0), 'deriv': lambda x, m=8.41: np.zeros_like(x,dtype=float)+m, 'domain': (-8.473760115679392,-8.346239884320608), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.41, y0=-0.5599999999999999, m=8.41: y0+m*(x-x0), 'deriv': lambda x, m=8.41: np.zeros_like(x,dtype=float)+m, 'domain': (-8.473760115679392,-8.346239884320608), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.41, y0=3.44, m=8.41: y0+m*(x-x0), 'deriv': lambda x, m=8.41: np.zeros_like(x,dtype=float)+m, 'domain': (-8.473760115679392,-8.346239884320608), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.41, y0=7.44, m=8.41: y0+m*(x-x0), 'deriv': lambda x, m=8.41: np.zeros_like(x,dtype=float)+m, 'domain': (-8.473760115679392,-8.346239884320608), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.41, y0=-8.56, m=4.41: y0+m*(x-x0), 'deriv': lambda x, m=4.41: np.zeros_like(x,dtype=float)+m, 'domain': (-4.529417308795651,-4.290582691204349), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.41, y0=-4.56, m=4.41: y0+m*(x-x0), 'deriv': lambda x, m=4.41: np.zeros_like(x,dtype=float)+m, 'domain': (-4.529417308795651,-4.290582691204349), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.41, y0=-0.5599999999999999, m=4.41: y0+m*(x-x0), 'deriv': lambda x, m=4.41: np.zeros_like(x,dtype=float)+m, 'domain': (-4.529417308795651,-4.290582691204349), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.41, y0=3.44, m=4.41: y0+m*(x-x0), 'deriv': lambda x, m=4.41: np.zeros_like(x,dtype=float)+m, 'domain': (-4.529417308795651,-4.290582691204349), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.41, y0=7.44, m=4.41: y0+m*(x-x0), 'deriv': lambda x, m=4.41: np.zeros_like(x,dtype=float)+m, 'domain': (-4.529417308795651,-4.290582691204349), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.41, y0=-8.56, m=0.41: y0+m*(x-x0), 'deriv': lambda x, m=0.41: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9096360288125358,0.0896360288125358), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.41, y0=-4.56, m=0.41: y0+m*(x-x0), 'deriv': lambda x, m=0.41: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9096360288125358,0.0896360288125358), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.41, y0=-0.5599999999999999, m=0.41: y0+m*(x-x0), 'deriv': lambda x, m=0.41: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9096360288125358,0.0896360288125358), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.41, y0=3.44, m=0.41: y0+m*(x-x0), 'deriv': lambda x, m=0.41: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9096360288125358,0.0896360288125358), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.41, y0=7.44, m=0.41: y0+m*(x-x0), 'deriv': lambda x, m=0.41: np.zeros_like(x,dtype=float)+m, 'domain': (-0.9096360288125358,0.0896360288125358), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.59, y0=-8.56, m=-3.59: y0+m*(x-x0), 'deriv': lambda x, m=-3.59: np.zeros_like(x,dtype=float)+m, 'domain': (3.4450986798020824,3.7349013201979173), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.59, y0=-4.56, m=-3.59: y0+m*(x-x0), 'deriv': lambda x, m=-3.59: np.zeros_like(x,dtype=float)+m, 'domain': (3.4450986798020824,3.7349013201979173), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.59, y0=-0.5599999999999999, m=-3.59: y0+m*(x-x0), 'deriv': lambda x, m=-3.59: np.zeros_like(x,dtype=float)+m, 'domain': (3.4450986798020824,3.7349013201979173), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.59, y0=3.44, m=-3.59: y0+m*(x-x0), 'deriv': lambda x, m=-3.59: np.zeros_like(x,dtype=float)+m, 'domain': (3.4450986798020824,3.7349013201979173), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.59, y0=7.44, m=-3.59: y0+m*(x-x0), 'deriv': lambda x, m=-3.59: np.zeros_like(x,dtype=float)+m, 'domain': (3.4450986798020824,3.7349013201979173), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.59, y0=-8.56, m=-7.59: y0+m*(x-x0), 'deriv': lambda x, m=-7.59: np.zeros_like(x,dtype=float)+m, 'domain': (7.5194633323291455,7.660536667670854), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.59, y0=-4.56, m=-7.59: y0+m*(x-x0), 'deriv': lambda x, m=-7.59: np.zeros_like(x,dtype=float)+m, 'domain': (7.5194633323291455,7.660536667670854), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.59, y0=-0.5599999999999999, m=-7.59: y0+m*(x-x0), 'deriv': lambda x, m=-7.59: np.zeros_like(x,dtype=float)+m, 'domain': (7.5194633323291455,7.660536667670854), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.59, y0=3.44, m=-7.59: y0+m*(x-x0), 'deriv': lambda x, m=-7.59: np.zeros_like(x,dtype=float)+m, 'domain': (7.5194633323291455,7.660536667670854), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.59, y0=7.44, m=-7.59: y0+m*(x-x0), 'deriv': lambda x, m=-7.59: np.zeros_like(x,dtype=float)+m, 'domain': (7.5194633323291455,7.660536667670854), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_sum_v2_q12_sum_field_mc.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.83, y0=-8.48, m=-7.83: y0+m*(x-x0), 'deriv': lambda x, m=-7.83: np.zeros_like(x,dtype=float)+m, 'domain': (-7.903477259748742,-7.756522740251258), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.83, y0=-4.48, m=-7.83: y0+m*(x-x0), 'deriv': lambda x, m=-7.83: np.zeros_like(x,dtype=float)+m, 'domain': (-7.903477259748742,-7.756522740251258), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.83, y0=-0.48, m=-7.83: y0+m*(x-x0), 'deriv': lambda x, m=-7.83: np.zeros_like(x,dtype=float)+m, 'domain': (-7.903477259748742,-7.756522740251258), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.83, y0=3.52, m=-7.83: y0+m*(x-x0), 'deriv': lambda x, m=-7.83: np.zeros_like(x,dtype=float)+m, 'domain': (-7.903477259748742,-7.756522740251258), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.83, y0=7.52, m=-7.83: y0+m*(x-x0), 'deriv': lambda x, m=-7.83: np.zeros_like(x,dtype=float)+m, 'domain': (-7.903477259748742,-7.756522740251258), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.83, y0=-8.48, m=-3.83: y0+m*(x-x0), 'deriv': lambda x, m=-3.83: np.zeros_like(x,dtype=float)+m, 'domain': (-3.976523990906478,-3.683476009093522), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.83, y0=-4.48, m=-3.83: y0+m*(x-x0), 'deriv': lambda x, m=-3.83: np.zeros_like(x,dtype=float)+m, 'domain': (-3.976523990906478,-3.683476009093522), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.83, y0=-0.48, m=-3.83: y0+m*(x-x0), 'deriv': lambda x, m=-3.83: np.zeros_like(x,dtype=float)+m, 'domain': (-3.976523990906478,-3.683476009093522), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.83, y0=3.52, m=-3.83: y0+m*(x-x0), 'deriv': lambda x, m=-3.83: np.zeros_like(x,dtype=float)+m, 'domain': (-3.976523990906478,-3.683476009093522), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.83, y0=7.52, m=-3.83: y0+m*(x-x0), 'deriv': lambda x, m=-3.83: np.zeros_like(x,dtype=float)+m, 'domain': (-3.976523990906478,-3.683476009093522), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.17000000000000004, y0=-8.48, m=0.17000000000000004: y0+m*(x-x0), 'deriv': lambda x, m=0.17000000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-0.4017963910684942,0.7417963910684943), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.17000000000000004, y0=-4.48, m=0.17000000000000004: y0+m*(x-x0), 'deriv': lambda x, m=0.17000000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-0.4017963910684942,0.7417963910684943), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.17000000000000004, y0=-0.48, m=0.17000000000000004: y0+m*(x-x0), 'deriv': lambda x, m=0.17000000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-0.4017963910684942,0.7417963910684943), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.17000000000000004, y0=3.52, m=0.17000000000000004: y0+m*(x-x0), 'deriv': lambda x, m=0.17000000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-0.4017963910684942,0.7417963910684943), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.17000000000000004, y0=7.52, m=0.17000000000000004: y0+m*(x-x0), 'deriv': lambda x, m=0.17000000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-0.4017963910684942,0.7417963910684943), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.17, y0=-8.48, m=4.17: y0+m*(x-x0), 'deriv': lambda x, m=4.17: np.zeros_like(x,dtype=float)+m, 'domain': (4.034745999927683,4.305254000072317), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.17, y0=-4.48, m=4.17: y0+m*(x-x0), 'deriv': lambda x, m=4.17: np.zeros_like(x,dtype=float)+m, 'domain': (4.034745999927683,4.305254000072317), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.17, y0=-0.48, m=4.17: y0+m*(x-x0), 'deriv': lambda x, m=4.17: np.zeros_like(x,dtype=float)+m, 'domain': (4.034745999927683,4.305254000072317), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.17, y0=3.52, m=4.17: y0+m*(x-x0), 'deriv': lambda x, m=4.17: np.zeros_like(x,dtype=float)+m, 'domain': (4.034745999927683,4.305254000072317), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.17, y0=7.52, m=4.17: y0+m*(x-x0), 'deriv': lambda x, m=4.17: np.zeros_like(x,dtype=float)+m, 'domain': (4.034745999927683,4.305254000072317), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.17, y0=-8.48, m=8.17: y0+m*(x-x0), 'deriv': lambda x, m=8.17: np.zeros_like(x,dtype=float)+m, 'domain': (8.09953444620182,8.24046555379818), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.17, y0=-4.48, m=8.17: y0+m*(x-x0), 'deriv': lambda x, m=8.17: np.zeros_like(x,dtype=float)+m, 'domain': (8.09953444620182,8.24046555379818), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.17, y0=-0.48, m=8.17: y0+m*(x-x0), 'deriv': lambda x, m=8.17: np.zeros_like(x,dtype=float)+m, 'domain': (8.09953444620182,8.24046555379818), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.17, y0=3.52, m=8.17: y0+m*(x-x0), 'deriv': lambda x, m=8.17: np.zeros_like(x,dtype=float)+m, 'domain': (8.09953444620182,8.24046555379818), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.17, y0=7.52, m=8.17: y0+m*(x-x0), 'deriv': lambda x, m=8.17: np.zeros_like(x,dtype=float)+m, 'domain': (8.09953444620182,8.24046555379818), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_sum_v2_q17_sum_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.2))
make_context_graph(ax, [
    {'expr': lambda x, A=20.0, k=-0.14: A*np.exp(k*x), 'deriv': lambda x, A=20.0, k=-0.14: A*k*np.exp(k*x), 'color': 'steelblue', 'label': None}
], xmin=0, xmax=10, ymin=0, ymax=60, xlabel='Time', ylabel='Quantity', title='')
save_graph(fig, 'u7_sum_v2_q20_sum_exp_graph.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.76, y0=-8.58, m=-8.58: y0+m*(x-x0), 'deriv': lambda x, m=-8.58: np.zeros_like(x,dtype=float)+m, 'domain': (-7.834090551589492,-7.6859094484105075), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.76, y0=-4.58, m=-4.58: y0+m*(x-x0), 'deriv': lambda x, m=-4.58: np.zeros_like(x,dtype=float)+m, 'domain': (-7.896521703393093,-7.623478296606907), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.76, y0=-0.58, m=-0.58: y0+m*(x-x0), 'deriv': lambda x, m=-0.58: np.zeros_like(x,dtype=float)+m, 'domain': (-8.313619961127554,-7.206380038872446), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.76, y0=3.42, m=3.42: y0+m*(x-x0), 'deriv': lambda x, m=3.42: np.zeros_like(x,dtype=float)+m, 'domain': (-7.939613789522051,-7.580386210477949), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.76, y0=7.42, m=7.42: y0+m*(x-x0), 'deriv': lambda x, m=7.42: np.zeros_like(x,dtype=float)+m, 'domain': (-7.845480562675719,-7.674519437324281), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.76, y0=-8.58, m=-8.58: y0+m*(x-x0), 'deriv': lambda x, m=-8.58: np.zeros_like(x,dtype=float)+m, 'domain': (-3.834090551589492,-3.6859094484105075), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.76, y0=-4.58, m=-4.58: y0+m*(x-x0), 'deriv': lambda x, m=-4.58: np.zeros_like(x,dtype=float)+m, 'domain': (-3.896521703393093,-3.6234782966069066), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.76, y0=-0.58, m=-0.58: y0+m*(x-x0), 'deriv': lambda x, m=-0.58: np.zeros_like(x,dtype=float)+m, 'domain': (-4.313619961127554,-3.2063800388724455), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.76, y0=3.42, m=3.42: y0+m*(x-x0), 'deriv': lambda x, m=3.42: np.zeros_like(x,dtype=float)+m, 'domain': (-3.9396137895220513,-3.5803862104779483), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.76, y0=7.42, m=7.42: y0+m*(x-x0), 'deriv': lambda x, m=7.42: np.zeros_like(x,dtype=float)+m, 'domain': (-3.8454805626757187,-3.674519437324281), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.24, y0=-8.58, m=-8.58: y0+m*(x-x0), 'deriv': lambda x, m=-8.58: np.zeros_like(x,dtype=float)+m, 'domain': (0.1659094484105076,0.3140905515894924), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.24, y0=-4.58, m=-4.58: y0+m*(x-x0), 'deriv': lambda x, m=-4.58: np.zeros_like(x,dtype=float)+m, 'domain': (0.10347829660690683,0.37652170339309315), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.24, y0=-0.58, m=-0.58: y0+m*(x-x0), 'deriv': lambda x, m=-0.58: np.zeros_like(x,dtype=float)+m, 'domain': (-0.3136199611275542,0.7936199611275542), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.24, y0=3.42, m=3.42: y0+m*(x-x0), 'deriv': lambda x, m=3.42: np.zeros_like(x,dtype=float)+m, 'domain': (0.06038621047794865,0.4196137895220513), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.24, y0=7.42, m=7.42: y0+m*(x-x0), 'deriv': lambda x, m=7.42: np.zeros_like(x,dtype=float)+m, 'domain': (0.15451943732428094,0.325480562675719), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.24, y0=-8.58, m=-8.58: y0+m*(x-x0), 'deriv': lambda x, m=-8.58: np.zeros_like(x,dtype=float)+m, 'domain': (4.165909448410508,4.3140905515894925), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.24, y0=-4.58, m=-4.58: y0+m*(x-x0), 'deriv': lambda x, m=-4.58: np.zeros_like(x,dtype=float)+m, 'domain': (4.103478296606907,4.376521703393093), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.24, y0=-0.58, m=-0.58: y0+m*(x-x0), 'deriv': lambda x, m=-0.58: np.zeros_like(x,dtype=float)+m, 'domain': (3.686380038872446,4.793619961127554), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.24, y0=3.42, m=3.42: y0+m*(x-x0), 'deriv': lambda x, m=3.42: np.zeros_like(x,dtype=float)+m, 'domain': (4.060386210477949,4.419613789522051), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.24, y0=7.42, m=7.42: y0+m*(x-x0), 'deriv': lambda x, m=7.42: np.zeros_like(x,dtype=float)+m, 'domain': (4.154519437324281,4.325480562675719), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.24, y0=-8.58, m=-8.58: y0+m*(x-x0), 'deriv': lambda x, m=-8.58: np.zeros_like(x,dtype=float)+m, 'domain': (8.165909448410508,8.314090551589493), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.24, y0=-4.58, m=-4.58: y0+m*(x-x0), 'deriv': lambda x, m=-4.58: np.zeros_like(x,dtype=float)+m, 'domain': (8.103478296606907,8.376521703393093), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.24, y0=-0.58, m=-0.58: y0+m*(x-x0), 'deriv': lambda x, m=-0.58: np.zeros_like(x,dtype=float)+m, 'domain': (7.686380038872446,8.793619961127554), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.24, y0=3.42, m=3.42: y0+m*(x-x0), 'deriv': lambda x, m=3.42: np.zeros_like(x,dtype=float)+m, 'domain': (8.06038621047795,8.419613789522051), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.24, y0=7.42, m=7.42: y0+m*(x-x0), 'deriv': lambda x, m=7.42: np.zeros_like(x,dtype=float)+m, 'domain': (8.15451943732428,8.32548056267572), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_sum_v3_q02_sum_field_mc.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.58, y0=-7.68, m=8.58: y0+m*(x-x0), 'deriv': lambda x, m=8.58: np.zeros_like(x,dtype=float)+m, 'domain': (-8.664509535406765,-8.495490464593235), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.58, y0=-3.6799999999999997, m=8.58: y0+m*(x-x0), 'deriv': lambda x, m=8.58: np.zeros_like(x,dtype=float)+m, 'domain': (-8.664509535406765,-8.495490464593235), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.58, y0=0.32000000000000006, m=8.58: y0+m*(x-x0), 'deriv': lambda x, m=8.58: np.zeros_like(x,dtype=float)+m, 'domain': (-8.664509535406765,-8.495490464593235), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.58, y0=4.32, m=8.58: y0+m*(x-x0), 'deriv': lambda x, m=8.58: np.zeros_like(x,dtype=float)+m, 'domain': (-8.664509535406765,-8.495490464593235), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.58, y0=8.32, m=8.58: y0+m*(x-x0), 'deriv': lambda x, m=8.58: np.zeros_like(x,dtype=float)+m, 'domain': (-8.664509535406765,-8.495490464593235), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.58, y0=-7.68, m=4.58: y0+m*(x-x0), 'deriv': lambda x, m=4.58: np.zeros_like(x,dtype=float)+m, 'domain': (-4.735720067932747,-4.424279932067253), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.58, y0=-3.6799999999999997, m=4.58: y0+m*(x-x0), 'deriv': lambda x, m=4.58: np.zeros_like(x,dtype=float)+m, 'domain': (-4.735720067932747,-4.424279932067253), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.58, y0=0.32000000000000006, m=4.58: y0+m*(x-x0), 'deriv': lambda x, m=4.58: np.zeros_like(x,dtype=float)+m, 'domain': (-4.735720067932747,-4.424279932067253), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.58, y0=4.32, m=4.58: y0+m*(x-x0), 'deriv': lambda x, m=4.58: np.zeros_like(x,dtype=float)+m, 'domain': (-4.735720067932747,-4.424279932067253), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.58, y0=8.32, m=4.58: y0+m*(x-x0), 'deriv': lambda x, m=4.58: np.zeros_like(x,dtype=float)+m, 'domain': (-4.735720067932747,-4.424279932067253), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.58, y0=-7.68, m=0.58: y0+m*(x-x0), 'deriv': lambda x, m=0.58: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2114727681611164,0.051472768161116456), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.58, y0=-3.6799999999999997, m=0.58: y0+m*(x-x0), 'deriv': lambda x, m=0.58: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2114727681611164,0.051472768161116456), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.58, y0=0.32000000000000006, m=0.58: y0+m*(x-x0), 'deriv': lambda x, m=0.58: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2114727681611164,0.051472768161116456), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.58, y0=4.32, m=0.58: y0+m*(x-x0), 'deriv': lambda x, m=0.58: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2114727681611164,0.051472768161116456), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.58, y0=8.32, m=0.58: y0+m*(x-x0), 'deriv': lambda x, m=0.58: np.zeros_like(x,dtype=float)+m, 'domain': (-1.2114727681611164,0.051472768161116456), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.42, y0=-7.68, m=-3.42: y0+m*(x-x0), 'deriv': lambda x, m=-3.42: np.zeros_like(x,dtype=float)+m, 'domain': (3.21512802132641,3.62487197867359), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.42, y0=-3.6799999999999997, m=-3.42: y0+m*(x-x0), 'deriv': lambda x, m=-3.42: np.zeros_like(x,dtype=float)+m, 'domain': (3.21512802132641,3.62487197867359), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.42, y0=0.32000000000000006, m=-3.42: y0+m*(x-x0), 'deriv': lambda x, m=-3.42: np.zeros_like(x,dtype=float)+m, 'domain': (3.21512802132641,3.62487197867359), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.42, y0=4.32, m=-3.42: y0+m*(x-x0), 'deriv': lambda x, m=-3.42: np.zeros_like(x,dtype=float)+m, 'domain': (3.21512802132641,3.62487197867359), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.42, y0=8.32, m=-3.42: y0+m*(x-x0), 'deriv': lambda x, m=-3.42: np.zeros_like(x,dtype=float)+m, 'domain': (3.21512802132641,3.62487197867359), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.42, y0=-7.68, m=-7.42: y0+m*(x-x0), 'deriv': lambda x, m=-7.42: np.zeros_like(x,dtype=float)+m, 'domain': (7.322498733198008,7.517501266801992), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.42, y0=-3.6799999999999997, m=-7.42: y0+m*(x-x0), 'deriv': lambda x, m=-7.42: np.zeros_like(x,dtype=float)+m, 'domain': (7.322498733198008,7.517501266801992), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.42, y0=0.32000000000000006, m=-7.42: y0+m*(x-x0), 'deriv': lambda x, m=-7.42: np.zeros_like(x,dtype=float)+m, 'domain': (7.322498733198008,7.517501266801992), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.42, y0=4.32, m=-7.42: y0+m*(x-x0), 'deriv': lambda x, m=-7.42: np.zeros_like(x,dtype=float)+m, 'domain': (7.322498733198008,7.517501266801992), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.42, y0=8.32, m=-7.42: y0+m*(x-x0), 'deriv': lambda x, m=-7.42: np.zeros_like(x,dtype=float)+m, 'domain': (7.322498733198008,7.517501266801992), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_sum_v3_q17_sum_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.2))
make_context_graph(ax, [
    {'expr': lambda x, A=40.0, k=-0.2: A*np.exp(k*x), 'deriv': lambda x, A=40.0, k=-0.2: A*k*np.exp(k*x), 'color': 'steelblue', 'label': None}
], xmin=0, xmax=10, ymin=0, ymax=60, xlabel='Time', ylabel='Quantity', title='')
save_graph(fig, 'u7_sum_v3_q20_sum_exp_graph.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.63, y0=-7.82, m=1.1900000000000004: y0+m*(x-x0), 'deriv': lambda x, m=1.1900000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-7.977405193939479,-7.282594806060521), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.63, y0=-3.82, m=-2.81: y0+m*(x-x0), 'deriv': lambda x, m=-2.81: np.zeros_like(x,dtype=float)+m, 'domain': (-7.81104808343667,-7.44895191656333), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.63, y0=0.18000000000000005, m=-6.81: y0+m*(x-x0), 'deriv': lambda x, m=-6.81: np.zeros_like(x,dtype=float)+m, 'domain': (-7.708453822123294,-7.5515461778767055), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.63, y0=4.18, m=-10.809999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-10.809999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-7.679741368207317,-7.580258631792683), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.63, y0=8.18, m=-14.809999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-14.809999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-7.66637901457191,-7.59362098542809), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.63, y0=-7.82, m=5.19: y0+m*(x-x0), 'deriv': lambda x, m=5.19: np.zeros_like(x,dtype=float)+m, 'domain': (-3.732167054207994,-3.5278329457920057), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.63, y0=-3.82, m=1.19: y0+m*(x-x0), 'deriv': lambda x, m=1.19: np.zeros_like(x,dtype=float)+m, 'domain': (-3.977405193939479,-3.282594806060521), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.63, y0=0.18000000000000005, m=-2.81: y0+m*(x-x0), 'deriv': lambda x, m=-2.81: np.zeros_like(x,dtype=float)+m, 'domain': (-3.8110480834366696,-3.4489519165633302), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.63, y0=4.18, m=-6.81: y0+m*(x-x0), 'deriv': lambda x, m=-6.81: np.zeros_like(x,dtype=float)+m, 'domain': (-3.7084538221232943,-3.5515461778767055), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.63, y0=8.18, m=-10.809999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-10.809999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-3.6797413682073166,-3.580258631792683), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.3700000000000001, y0=-7.82, m=9.190000000000001: y0+m*(x-x0), 'deriv': lambda x, m=9.190000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (0.31158528994245904,0.42841471005754117), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.3700000000000001, y0=-3.82, m=5.1899999999999995: y0+m*(x-x0), 'deriv': lambda x, m=5.1899999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (0.26783294579200595,0.47216705420799426), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.3700000000000001, y0=0.18000000000000005, m=1.19: y0+m*(x-x0), 'deriv': lambda x, m=1.19: np.zeros_like(x,dtype=float)+m, 'domain': (0.02259480606052139,0.7174051939394788), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.3700000000000001, y0=4.18, m=-2.8099999999999996: y0+m*(x-x0), 'deriv': lambda x, m=-2.8099999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (0.18895191656333035,0.5510480834366699), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.3700000000000001, y0=8.18, m=-6.81: y0+m*(x-x0), 'deriv': lambda x, m=-6.81: np.zeros_like(x,dtype=float)+m, 'domain': (0.2915461778767056,0.4484538221232946), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.37, y0=-7.82, m=13.190000000000001: y0+m*(x-x0), 'deriv': lambda x, m=13.190000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (4.329177049195465,4.410822950804535), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.37, y0=-3.82, m=9.19: y0+m*(x-x0), 'deriv': lambda x, m=9.19: np.zeros_like(x,dtype=float)+m, 'domain': (4.311585289942459,4.428414710057541), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.37, y0=0.18000000000000005, m=5.19: y0+m*(x-x0), 'deriv': lambda x, m=5.19: np.zeros_like(x,dtype=float)+m, 'domain': (4.267832945792006,4.472167054207994), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.37, y0=4.18, m=1.1900000000000004: y0+m*(x-x0), 'deriv': lambda x, m=1.1900000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (4.022594806060521,4.717405193939479), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.37, y0=8.18, m=-2.8099999999999996: y0+m*(x-x0), 'deriv': lambda x, m=-2.8099999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (4.18895191656333,4.55104808343667), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.370000000000001, y0=-7.82, m=17.19: y0+m*(x-x0), 'deriv': lambda x, m=17.19: np.zeros_like(x,dtype=float)+m, 'domain': (8.338639406925708,8.401360593074294), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.370000000000001, y0=-3.82, m=13.190000000000001: y0+m*(x-x0), 'deriv': lambda x, m=13.190000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (8.329177049195465,8.410822950804537), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.370000000000001, y0=0.18000000000000005, m=9.190000000000001: y0+m*(x-x0), 'deriv': lambda x, m=9.190000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (8.31158528994246,8.428414710057542), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.370000000000001, y0=4.18, m=5.190000000000001: y0+m*(x-x0), 'deriv': lambda x, m=5.190000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (8.267832945792007,8.472167054207995), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.370000000000001, y0=8.18, m=1.1900000000000013: y0+m*(x-x0), 'deriv': lambda x, m=1.1900000000000013: np.zeros_like(x,dtype=float)+m, 'domain': (8.022594806060523,8.717405193939479), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_sum_v4_q07_sum_field_mc.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.18, y0=-7.4, m=-1.7799999999999994: y0+m*(x-x0), 'deriv': lambda x, m=-1.7799999999999994: np.zeros_like(x,dtype=float)+m, 'domain': (-8.542449100357905,-7.817550899642095), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.18, y0=-3.4, m=-5.779999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-5.779999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-8.30615355127372,-8.05384644872628), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.18, y0=0.6, m=-9.78: y0+m*(x-x0), 'deriv': lambda x, m=-9.78: np.zeros_like(x,dtype=float)+m, 'domain': (-8.255272161159352,-8.104727838840647), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.18, y0=4.6, m=-13.78: y0+m*(x-x0), 'deriv': lambda x, m=-13.78: np.zeros_like(x,dtype=float)+m, 'domain': (-8.233560170499516,-8.126439829500484), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.18, y0=8.6, m=-17.78: y0+m*(x-x0), 'deriv': lambda x, m=-17.78: np.zeros_like(x,dtype=float)+m, 'domain': (-8.221554125924703,-8.138445874075297), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.18, y0=-7.4, m=2.2200000000000006: y0+m*(x-x0), 'deriv': lambda x, m=2.2200000000000006: np.zeros_like(x,dtype=float)+m, 'domain': (-4.483922582196902,-3.8760774178030974), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.18, y0=-3.4, m=-1.7799999999999998: y0+m*(x-x0), 'deriv': lambda x, m=-1.7799999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (-4.542449100357905,-3.8175508996420953), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.18, y0=0.6, m=-5.779999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-5.779999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-4.30615355127372,-4.053846448726279), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.18, y0=4.6, m=-9.78: y0+m*(x-x0), 'deriv': lambda x, m=-9.78: np.zeros_like(x,dtype=float)+m, 'domain': (-4.255272161159352,-4.104727838840647), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.18, y0=8.6, m=-13.78: y0+m*(x-x0), 'deriv': lambda x, m=-13.78: np.zeros_like(x,dtype=float)+m, 'domain': (-4.233560170499516,-4.126439829500484), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.18000000000000005, y0=-7.4, m=6.220000000000001: y0+m*(x-x0), 'deriv': lambda x, m=6.220000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-0.29746268508680096,-0.06253731491319911), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.18000000000000005, y0=-3.4, m=2.2199999999999998: y0+m*(x-x0), 'deriv': lambda x, m=2.2199999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (-0.48392258219690243,0.12392258219690233), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.18000000000000005, y0=0.6, m=-1.78: y0+m*(x-x0), 'deriv': lambda x, m=-1.78: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5424491003579045,0.1824491003579044), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.18000000000000005, y0=4.6, m=-5.779999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-5.779999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-0.30615355127372035,-0.05384644872627972), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.18000000000000005, y0=8.6, m=-9.78: y0+m*(x-x0), 'deriv': lambda x, m=-9.78: np.zeros_like(x,dtype=float)+m, 'domain': (-0.2552721611593524,-0.10472783884064768), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.82, y0=-7.4, m=10.22: y0+m*(x-x0), 'deriv': lambda x, m=10.22: np.zeros_like(x,dtype=float)+m, 'domain': (3.747937102120364,3.8920628978796357), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.82, y0=-3.4, m=6.22: y0+m*(x-x0), 'deriv': lambda x, m=6.22: np.zeros_like(x,dtype=float)+m, 'domain': (3.702537314913199,3.9374626850868006), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.82, y0=0.6, m=2.2199999999999998: y0+m*(x-x0), 'deriv': lambda x, m=2.2199999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (3.5160774178030976,4.1239225821969026), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.82, y0=4.6, m=-1.7799999999999998: y0+m*(x-x0), 'deriv': lambda x, m=-1.7799999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (3.4575508996420954,4.182449100357904), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.82, y0=8.6, m=-5.779999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-5.779999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (3.6938464487262794,3.9461535512737203), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.82, y0=-7.4, m=14.22: y0+m*(x-x0), 'deriv': lambda x, m=14.22: np.zeros_like(x,dtype=float)+m, 'domain': (7.768088821129058,7.871911178870943), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.82, y0=-3.4, m=10.22: y0+m*(x-x0), 'deriv': lambda x, m=10.22: np.zeros_like(x,dtype=float)+m, 'domain': (7.747937102120364,7.8920628978796366), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.82, y0=0.6, m=6.220000000000001: y0+m*(x-x0), 'deriv': lambda x, m=6.220000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (7.702537314913199,7.9374626850868015), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.82, y0=4.6, m=2.2200000000000006: y0+m*(x-x0), 'deriv': lambda x, m=2.2200000000000006: np.zeros_like(x,dtype=float)+m, 'domain': (7.516077417803098,8.123922582196903), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.82, y0=8.6, m=-1.7799999999999994: y0+m*(x-x0), 'deriv': lambda x, m=-1.7799999999999994: np.zeros_like(x,dtype=float)+m, 'domain': (7.457550899642095,8.182449100357905), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_sum_v4_q17_sum_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.2))
make_context_graph(ax, [
    {'expr': lambda x, A=30.0, k=-0.14: A*np.exp(k*x), 'deriv': lambda x, A=30.0, k=-0.14: A*k*np.exp(k*x), 'color': 'steelblue', 'label': None}
], xmin=0, xmax=8, ymin=0, ymax=60, xlabel='Time', ylabel='Quantity', title='')
save_graph(fig, 'u7_sum_v4_q20_sum_exp_graph.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.83, y0=-7.84, m=-14.67: y0+m*(x-x0), 'deriv': lambda x, m=-14.67: np.zeros_like(x,dtype=float)+m, 'domain': (-7.874885612093701,-7.785114387906299), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.83, y0=-3.84, m=-10.67: y0+m*(x-x0), 'deriv': lambda x, m=-10.67: np.zeros_like(x,dtype=float)+m, 'domain': (-7.891585789828836,-7.768414210171164), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.83, y0=0.16000000000000003, m=-6.67: y0+m*(x-x0), 'deriv': lambda x, m=-6.67: np.zeros_like(x,dtype=float)+m, 'domain': (-7.927856846911966,-7.7321431530880345), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.83, y0=4.16, m=-2.67: y0+m*(x-x0), 'deriv': lambda x, m=-2.67: np.zeros_like(x,dtype=float)+m, 'domain': (-8.061487766698296,-7.598512233301704), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.83, y0=8.16, m=1.33: y0+m*(x-x0), 'deriv': lambda x, m=1.33: np.zeros_like(x,dtype=float)+m, 'domain': (-8.226634328891958,-7.433365671108041), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.83, y0=-7.84, m=-10.67: y0+m*(x-x0), 'deriv': lambda x, m=-10.67: np.zeros_like(x,dtype=float)+m, 'domain': (-3.8915857898288366,-3.7684142101711635), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.83, y0=-3.84, m=-6.67: y0+m*(x-x0), 'deriv': lambda x, m=-6.67: np.zeros_like(x,dtype=float)+m, 'domain': (-3.9278568469119657,-3.7321431530880345), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.83, y0=0.16000000000000003, m=-2.67: y0+m*(x-x0), 'deriv': lambda x, m=-2.67: np.zeros_like(x,dtype=float)+m, 'domain': (-4.0614877666982965,-3.5985122333017037), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.83, y0=4.16, m=1.33: y0+m*(x-x0), 'deriv': lambda x, m=1.33: np.zeros_like(x,dtype=float)+m, 'domain': (-4.226634328891959,-3.4333656711080414), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.83, y0=8.16, m=5.33: y0+m*(x-x0), 'deriv': lambda x, m=5.33: np.zeros_like(x,dtype=float)+m, 'domain': (-3.951703913961205,-3.708296086038795), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.17000000000000004, y0=-7.84, m=-6.67: y0+m*(x-x0), 'deriv': lambda x, m=-6.67: np.zeros_like(x,dtype=float)+m, 'domain': (0.07214315308803425,0.2678568469119658), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.17000000000000004, y0=-3.84, m=-2.67: y0+m*(x-x0), 'deriv': lambda x, m=-2.67: np.zeros_like(x,dtype=float)+m, 'domain': (-0.061487766698296226,0.40148776669829633), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.17000000000000004, y0=0.16000000000000003, m=1.33: y0+m*(x-x0), 'deriv': lambda x, m=1.33: np.zeros_like(x,dtype=float)+m, 'domain': (-0.22663432889195861,0.5666343288919586), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.17000000000000004, y0=4.16, m=5.33: y0+m*(x-x0), 'deriv': lambda x, m=5.33: np.zeros_like(x,dtype=float)+m, 'domain': (0.048296086038795,0.2917039139612051), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.17000000000000004, y0=8.16, m=9.33: y0+m*(x-x0), 'deriv': lambda x, m=9.33: np.zeros_like(x,dtype=float)+m, 'domain': (0.09966330330031539,0.2403366966996847), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.17, y0=-7.84, m=-2.67: y0+m*(x-x0), 'deriv': lambda x, m=-2.67: np.zeros_like(x,dtype=float)+m, 'domain': (3.9385122333017035,4.401487766698296), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.17, y0=-3.84, m=1.33: y0+m*(x-x0), 'deriv': lambda x, m=1.33: np.zeros_like(x,dtype=float)+m, 'domain': (3.773365671108041,4.566634328891959), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.17, y0=0.16000000000000003, m=5.33: y0+m*(x-x0), 'deriv': lambda x, m=5.33: np.zeros_like(x,dtype=float)+m, 'domain': (4.048296086038795,4.2917039139612045), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.17, y0=4.16, m=9.33: y0+m*(x-x0), 'deriv': lambda x, m=9.33: np.zeros_like(x,dtype=float)+m, 'domain': (4.099663303300315,4.240336696699685), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.17, y0=8.16, m=13.33: y0+m*(x-x0), 'deriv': lambda x, m=13.33: np.zeros_like(x,dtype=float)+m, 'domain': (4.120626359802369,4.219373640197631), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.17, y0=-7.84, m=1.33: y0+m*(x-x0), 'deriv': lambda x, m=1.33: np.zeros_like(x,dtype=float)+m, 'domain': (7.773365671108041,8.566634328891958), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.17, y0=-3.84, m=5.33: y0+m*(x-x0), 'deriv': lambda x, m=5.33: np.zeros_like(x,dtype=float)+m, 'domain': (8.048296086038794,8.291703913961205), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.17, y0=0.16000000000000003, m=9.33: y0+m*(x-x0), 'deriv': lambda x, m=9.33: np.zeros_like(x,dtype=float)+m, 'domain': (8.099663303300316,8.240336696699684), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.17, y0=4.16, m=13.33: y0+m*(x-x0), 'deriv': lambda x, m=13.33: np.zeros_like(x,dtype=float)+m, 'domain': (8.120626359802369,8.219373640197631), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.17, y0=8.16, m=17.33: y0+m*(x-x0), 'deriv': lambda x, m=17.33: np.zeros_like(x,dtype=float)+m, 'domain': (8.13197899936328,8.20802100063672), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_sum_v5_q12_sum_field_mc.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.38, y0=-7.79, m=-16.17: y0+m*(x-x0), 'deriv': lambda x, m=-16.17: np.zeros_like(x,dtype=float)+m, 'domain': (-8.425676497457633,-8.334323502542368), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.38, y0=-3.79, m=-12.170000000000002: y0+m*(x-x0), 'deriv': lambda x, m=-12.170000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (-8.440601020354256,-8.319398979645745), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.38, y0=0.21000000000000008, m=-8.17: y0+m*(x-x0), 'deriv': lambda x, m=-8.17: np.zeros_like(x,dtype=float)+m, 'domain': (-8.469904327259748,-8.290095672740254), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.38, y0=4.21, m=-4.170000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-4.170000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-8.552565448368128,-8.207434551631874), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.38, y0=8.21, m=-0.16999999999999993: y0+m*(x-x0), 'deriv': lambda x, m=-0.16999999999999993: np.zeros_like(x,dtype=float)+m, 'domain': (-9.109533326535665,-7.650466673464336), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.38, y0=-7.79, m=-12.17: y0+m*(x-x0), 'deriv': lambda x, m=-12.17: np.zeros_like(x,dtype=float)+m, 'domain': (-4.440601020354255,-4.319398979645745), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.38, y0=-3.79, m=-8.17: y0+m*(x-x0), 'deriv': lambda x, m=-8.17: np.zeros_like(x,dtype=float)+m, 'domain': (-4.469904327259746,-4.290095672740254), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.38, y0=0.21000000000000008, m=-4.17: y0+m*(x-x0), 'deriv': lambda x, m=-4.17: np.zeros_like(x,dtype=float)+m, 'domain': (-4.552565448368128,-4.207434551631872), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.38, y0=4.21, m=-0.16999999999999993: y0+m*(x-x0), 'deriv': lambda x, m=-0.16999999999999993: np.zeros_like(x,dtype=float)+m, 'domain': (-5.109533326535665,-3.6504666734643347), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.38, y0=8.21, m=3.830000000000001: y0+m*(x-x0), 'deriv': lambda x, m=3.830000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-4.566944402191023,-4.193055597808977), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.38, y0=-7.79, m=-8.17: y0+m*(x-x0), 'deriv': lambda x, m=-8.17: np.zeros_like(x,dtype=float)+m, 'domain': (-0.4699043272597462,-0.29009567274025383), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.38, y0=-3.79, m=-4.17: y0+m*(x-x0), 'deriv': lambda x, m=-4.17: np.zeros_like(x,dtype=float)+m, 'domain': (-0.552565448368128,-0.20743455163187194), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.38, y0=0.21000000000000008, m=-0.16999999999999993: y0+m*(x-x0), 'deriv': lambda x, m=-0.16999999999999993: np.zeros_like(x,dtype=float)+m, 'domain': (-1.109533326535665,0.34953332653566493), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.38, y0=4.21, m=3.83: y0+m*(x-x0), 'deriv': lambda x, m=3.83: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5669444021910235,-0.19305559780897658), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.38, y0=8.21, m=7.830000000000001: y0+m*(x-x0), 'deriv': lambda x, m=7.830000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-0.47374684864494654,-0.28625315135505347), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.62, y0=-7.79, m=-4.17: y0+m*(x-x0), 'deriv': lambda x, m=-4.17: np.zeros_like(x,dtype=float)+m, 'domain': (3.447434551631872,3.792565448368128), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.62, y0=-3.79, m=-0.16999999999999993: y0+m*(x-x0), 'deriv': lambda x, m=-0.16999999999999993: np.zeros_like(x,dtype=float)+m, 'domain': (2.890466673464335,4.349533326535665), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.62, y0=0.21000000000000008, m=3.83: y0+m*(x-x0), 'deriv': lambda x, m=3.83: np.zeros_like(x,dtype=float)+m, 'domain': (3.4330555978089765,3.8069444021910237), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.62, y0=4.21, m=7.83: y0+m*(x-x0), 'deriv': lambda x, m=7.83: np.zeros_like(x,dtype=float)+m, 'domain': (3.5262531513550535,3.7137468486449468), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.62, y0=8.21, m=11.830000000000002: y0+m*(x-x0), 'deriv': lambda x, m=11.830000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (3.557669462170462,3.682330537829538), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.62, y0=-7.79, m=-0.16999999999999993: y0+m*(x-x0), 'deriv': lambda x, m=-0.16999999999999993: np.zeros_like(x,dtype=float)+m, 'domain': (6.890466673464335,8.349533326535665), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.62, y0=-3.79, m=3.83: y0+m*(x-x0), 'deriv': lambda x, m=3.83: np.zeros_like(x,dtype=float)+m, 'domain': (7.433055597808977,7.806944402191023), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.62, y0=0.21000000000000008, m=7.83: y0+m*(x-x0), 'deriv': lambda x, m=7.83: np.zeros_like(x,dtype=float)+m, 'domain': (7.5262531513550535,7.713746848644947), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.62, y0=4.21, m=11.83: y0+m*(x-x0), 'deriv': lambda x, m=11.83: np.zeros_like(x,dtype=float)+m, 'domain': (7.557669462170462,7.682330537829539), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.62, y0=8.21, m=15.830000000000002: y0+m*(x-x0), 'deriv': lambda x, m=15.830000000000002: np.zeros_like(x,dtype=float)+m, 'domain': (7.573346311901063,7.666653688098937), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_sum_v5_q17_sum_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.2))
make_context_graph(ax, [
    {'expr': lambda x, A=20.0, k=-0.14: A*np.exp(k*x), 'deriv': lambda x, A=20.0, k=-0.14: A*k*np.exp(k*x), 'color': 'steelblue', 'label': None}
], xmin=0, xmax=8, ymin=0, ymax=60, xlabel='Time', ylabel='Quantity', title='')
save_graph(fig, 'u7_sum_v5_q20_sum_exp_graph.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.41, y0=-7.6899999999999995, m=-7.6899999999999995: y0+m*(x-x0), 'deriv': lambda x, m=-7.6899999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (-7.500267291249871,-7.319732708750129), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.41, y0=-3.69, m=-3.69: y0+m*(x-x0), 'deriv': lambda x, m=-3.69: np.zeros_like(x,dtype=float)+m, 'domain': (-7.5930974463089616,-7.226902553691039), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.41, y0=0.31000000000000005, m=0.31000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=0.31000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-8.078610128322117,-6.741389871677883), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.41, y0=4.3100000000000005, m=4.3100000000000005: y0+m*(x-x0), 'deriv': lambda x, m=4.3100000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-7.568210372327238,-7.251789627672762), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.41, y0=8.31, m=8.31: y0+m*(x-x0), 'deriv': lambda x, m=8.31: np.zeros_like(x,dtype=float)+m, 'domain': (-7.493632496629085,-7.326367503370915), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.41, y0=-7.6899999999999995, m=-7.6899999999999995: y0+m*(x-x0), 'deriv': lambda x, m=-7.6899999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (-3.500267291249872,-3.3197327087501285), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.41, y0=-3.69, m=-3.69: y0+m*(x-x0), 'deriv': lambda x, m=-3.69: np.zeros_like(x,dtype=float)+m, 'domain': (-3.593097446308961,-3.226902553691039), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.41, y0=0.31000000000000005, m=0.31000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=0.31000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-4.078610128322118,-2.7413898716778826), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.41, y0=4.3100000000000005, m=4.3100000000000005: y0+m*(x-x0), 'deriv': lambda x, m=4.3100000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-3.568210372327238,-3.2517896276727623), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.41, y0=8.31, m=8.31: y0+m*(x-x0), 'deriv': lambda x, m=8.31: np.zeros_like(x,dtype=float)+m, 'domain': (-3.4936324966290853,-3.326367503370915), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5900000000000001, y0=-7.6899999999999995, m=-7.6899999999999995: y0+m*(x-x0), 'deriv': lambda x, m=-7.6899999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (0.4997327087501285,0.6802672912498717), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5900000000000001, y0=-3.69, m=-3.69: y0+m*(x-x0), 'deriv': lambda x, m=-3.69: np.zeros_like(x,dtype=float)+m, 'domain': (0.4069025536910391,0.773097446308961), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5900000000000001, y0=0.31000000000000005, m=0.31000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=0.31000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (-0.07861012832211733,1.2586101283221174), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5900000000000001, y0=4.3100000000000005, m=4.3100000000000005: y0+m*(x-x0), 'deriv': lambda x, m=4.3100000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (0.4317896276727623,0.7482103723272379), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.5900000000000001, y0=8.31, m=8.31: y0+m*(x-x0), 'deriv': lambda x, m=8.31: np.zeros_like(x,dtype=float)+m, 'domain': (0.506367503370915,0.6736324966290852), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.59, y0=-7.6899999999999995, m=-7.6899999999999995: y0+m*(x-x0), 'deriv': lambda x, m=-7.6899999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (4.499732708750129,4.680267291249871), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.59, y0=-3.69, m=-3.69: y0+m*(x-x0), 'deriv': lambda x, m=-3.69: np.zeros_like(x,dtype=float)+m, 'domain': (4.4069025536910384,4.773097446308961), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.59, y0=0.31000000000000005, m=0.31000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=0.31000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (3.9213898716778823,5.258610128322117), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.59, y0=4.3100000000000005, m=4.3100000000000005: y0+m*(x-x0), 'deriv': lambda x, m=4.3100000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (4.431789627672762,4.748210372327238), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.59, y0=8.31, m=8.31: y0+m*(x-x0), 'deriv': lambda x, m=8.31: np.zeros_like(x,dtype=float)+m, 'domain': (4.506367503370915,4.673632496629085), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.59, y0=-7.6899999999999995, m=-7.6899999999999995: y0+m*(x-x0), 'deriv': lambda x, m=-7.6899999999999995: np.zeros_like(x,dtype=float)+m, 'domain': (8.499732708750129,8.680267291249871), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.59, y0=-3.69, m=-3.69: y0+m*(x-x0), 'deriv': lambda x, m=-3.69: np.zeros_like(x,dtype=float)+m, 'domain': (8.406902553691038,8.773097446308961), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.59, y0=0.31000000000000005, m=0.31000000000000005: y0+m*(x-x0), 'deriv': lambda x, m=0.31000000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (7.921389871677882,9.258610128322116), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.59, y0=4.3100000000000005, m=4.3100000000000005: y0+m*(x-x0), 'deriv': lambda x, m=4.3100000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (8.431789627672762,8.748210372327238), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.59, y0=8.31, m=8.31: y0+m*(x-x0), 'deriv': lambda x, m=8.31: np.zeros_like(x,dtype=float)+m, 'domain': (8.506367503370914,8.673632496629086), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_sum_v6_q02_sum_field_mc.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.53, y0=-8.13, m=-17.66: y0+m*(x-x0), 'deriv': lambda x, m=-17.66: np.zeros_like(x,dtype=float)+m, 'domain': (-8.568443512820908,-8.491556487179091), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.53, y0=-4.13, m=-13.66: y0+m*(x-x0), 'deriv': lambda x, m=-13.66: np.zeros_like(x,dtype=float)+m, 'domain': (-8.579647523351424,-8.480352476648575), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.53, y0=-0.13, m=-9.66: y0+m*(x-x0), 'deriv': lambda x, m=-9.66: np.zeros_like(x,dtype=float)+m, 'domain': (-8.600019200370893,-8.459980799629106), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.53, y0=3.87, m=-5.659999999999999: y0+m*(x-x0), 'deriv': lambda x, m=-5.659999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (-8.648309008089413,-8.411690991910586), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.53, y0=7.87, m=-1.6599999999999993: y0+m*(x-x0), 'deriv': lambda x, m=-1.6599999999999993: np.zeros_like(x,dtype=float)+m, 'domain': (-8.88088859098072,-8.17911140901928), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.53, y0=-8.13, m=-13.66: y0+m*(x-x0), 'deriv': lambda x, m=-13.66: np.zeros_like(x,dtype=float)+m, 'domain': (-4.579647523351426,-4.480352476648575), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.53, y0=-4.13, m=-9.66: y0+m*(x-x0), 'deriv': lambda x, m=-9.66: np.zeros_like(x,dtype=float)+m, 'domain': (-4.600019200370895,-4.459980799629106), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.53, y0=-0.13, m=-5.66: y0+m*(x-x0), 'deriv': lambda x, m=-5.66: np.zeros_like(x,dtype=float)+m, 'domain': (-4.648309008089415,-4.411690991910586), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.53, y0=3.87, m=-1.6600000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-1.6600000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-4.880888590980719,-4.179111409019281), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.53, y0=7.87, m=2.34: y0+m*(x-x0), 'deriv': lambda x, m=2.34: np.zeros_like(x,dtype=float)+m, 'domain': (-4.797219966682157,-4.262780033317844), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.53, y0=-8.13, m=-9.66: y0+m*(x-x0), 'deriv': lambda x, m=-9.66: np.zeros_like(x,dtype=float)+m, 'domain': (-0.6000192003708942,-0.45998079962910593), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.53, y0=-4.13, m=-5.66: y0+m*(x-x0), 'deriv': lambda x, m=-5.66: np.zeros_like(x,dtype=float)+m, 'domain': (-0.648309008089414,-0.41169099191058595), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.53, y0=-0.13, m=-1.6600000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-1.6600000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-0.8808885909807194,-0.1791114090192807), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.53, y0=3.87, m=2.34: y0+m*(x-x0), 'deriv': lambda x, m=2.34: np.zeros_like(x,dtype=float)+m, 'domain': (-0.7972199666821568,-0.2627800333178433), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.53, y0=7.87, m=6.34: y0+m*(x-x0), 'deriv': lambda x, m=6.34: np.zeros_like(x,dtype=float)+m, 'domain': (-0.6359457394322767,-0.42405426056772333), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.4699999999999998, y0=-8.13, m=-5.660000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-5.660000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (3.3516909919105857,3.5883090080894138), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.4699999999999998, y0=-4.13, m=-1.6600000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-1.6600000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (3.1191114090192804,3.820888590980719), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.4699999999999998, y0=-0.13, m=2.34: y0+m*(x-x0), 'deriv': lambda x, m=2.34: np.zeros_like(x,dtype=float)+m, 'domain': (3.202780033317843,3.7372199666821566), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.4699999999999998, y0=3.87, m=6.34: y0+m*(x-x0), 'deriv': lambda x, m=6.34: np.zeros_like(x,dtype=float)+m, 'domain': (3.3640542605677233,3.5759457394322762), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.4699999999999998, y0=7.87, m=10.34: y0+m*(x-x0), 'deriv': lambda x, m=10.34: np.zeros_like(x,dtype=float)+m, 'domain': (3.4045413871459584,3.535458612854041), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.47, y0=-8.13, m=-1.660000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-1.660000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (7.119111409019281,7.820888590980719), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.47, y0=-4.13, m=2.34: y0+m*(x-x0), 'deriv': lambda x, m=2.34: np.zeros_like(x,dtype=float)+m, 'domain': (7.202780033317843,7.737219966682156), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.47, y0=-0.13, m=6.34: y0+m*(x-x0), 'deriv': lambda x, m=6.34: np.zeros_like(x,dtype=float)+m, 'domain': (7.364054260567723,7.575945739432276), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.47, y0=3.87, m=10.34: y0+m*(x-x0), 'deriv': lambda x, m=10.34: np.zeros_like(x,dtype=float)+m, 'domain': (7.404541387145959,7.535458612854041), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.47, y0=7.87, m=14.34: y0+m*(x-x0), 'deriv': lambda x, m=14.34: np.zeros_like(x,dtype=float)+m, 'domain': (7.4226950769997835,7.517304923000216), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_sum_v6_q17_sum_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.2))
make_context_graph(ax, [
    {'expr': lambda x, A=40.0, k=-0.2: A*np.exp(k*x), 'deriv': lambda x, A=40.0, k=-0.2: A*k*np.exp(k*x), 'color': 'steelblue', 'label': None}
], xmin=0, xmax=8, ymin=0, ymax=60, xlabel='Time', ylabel='Quantity', title='')
save_graph(fig, 'u7_sum_v6_q20_sum_exp_graph.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.42, y0=-7.84, m=-17.259999999999998: y0+m*(x-x0), 'deriv': lambda x, m=-17.259999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (-8.467429153505536,-8.372570846494463), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.42, y0=-3.84, m=-13.26: y0+m*(x-x0), 'deriv': lambda x, m=-13.26: np.zeros_like(x,dtype=float)+m, 'domain': (-8.481665012822114,-8.358334987177885), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.42, y0=0.16000000000000003, m=-9.26: y0+m*(x-x0), 'deriv': lambda x, m=-9.26: np.zeros_like(x,dtype=float)+m, 'domain': (-8.508041030710645,-8.331958969289355), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.42, y0=4.16, m=-5.26: y0+m*(x-x0), 'deriv': lambda x, m=-5.26: np.zeros_like(x,dtype=float)+m, 'domain': (-8.57315041723968,-8.26684958276032), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.42, y0=8.16, m=-1.2599999999999998: y0+m*(x-x0), 'deriv': lambda x, m=-1.2599999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (-8.929759444842489,-7.910240555157512), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.42, y0=-7.84, m=-13.26: y0+m*(x-x0), 'deriv': lambda x, m=-13.26: np.zeros_like(x,dtype=float)+m, 'domain': (-4.4816650128221145,-4.358334987177885), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.42, y0=-3.84, m=-9.26: y0+m*(x-x0), 'deriv': lambda x, m=-9.26: np.zeros_like(x,dtype=float)+m, 'domain': (-4.508041030710644,-4.331958969289356), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.42, y0=0.16000000000000003, m=-5.26: y0+m*(x-x0), 'deriv': lambda x, m=-5.26: np.zeros_like(x,dtype=float)+m, 'domain': (-4.573150417239679,-4.266849582760321), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.42, y0=4.16, m=-1.2599999999999998: y0+m*(x-x0), 'deriv': lambda x, m=-1.2599999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (-4.929759444842488,-3.910240555157512), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.42, y0=8.16, m=2.74: y0+m*(x-x0), 'deriv': lambda x, m=2.74: np.zeros_like(x,dtype=float)+m, 'domain': (-4.701132025162483,-4.138867974837517), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.42, y0=-7.84, m=-9.26: y0+m*(x-x0), 'deriv': lambda x, m=-9.26: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5080410307106445,-0.3319589692893554), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.42, y0=-3.84, m=-5.26: y0+m*(x-x0), 'deriv': lambda x, m=-5.26: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5731504172396794,-0.26684958276032056), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.42, y0=0.16000000000000003, m=-1.26: y0+m*(x-x0), 'deriv': lambda x, m=-1.26: np.zeros_like(x,dtype=float)+m, 'domain': (-0.929759444842488,0.08975944484248805), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.42, y0=4.16, m=2.74: y0+m*(x-x0), 'deriv': lambda x, m=2.74: np.zeros_like(x,dtype=float)+m, 'domain': (-0.7011320251624834,-0.13886797483751656), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.42, y0=8.16, m=6.74: y0+m*(x-x0), 'deriv': lambda x, m=6.74: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5403443582493942,-0.2996556417506058), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.58, y0=-7.84, m=-5.26: y0+m*(x-x0), 'deriv': lambda x, m=-5.26: np.zeros_like(x,dtype=float)+m, 'domain': (3.4268495827603207,3.7331504172396794), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.58, y0=-3.84, m=-1.2599999999999998: y0+m*(x-x0), 'deriv': lambda x, m=-1.2599999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (3.070240555157512,4.089759444842488), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.58, y0=0.16000000000000003, m=2.74: y0+m*(x-x0), 'deriv': lambda x, m=2.74: np.zeros_like(x,dtype=float)+m, 'domain': (3.2988679748375165,3.8611320251624837), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.58, y0=4.16, m=6.74: y0+m*(x-x0), 'deriv': lambda x, m=6.74: np.zeros_like(x,dtype=float)+m, 'domain': (3.4596556417506057,3.7003443582493944), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.58, y0=8.16, m=10.74: y0+m*(x-x0), 'deriv': lambda x, m=10.74: np.zeros_like(x,dtype=float)+m, 'domain': (3.5039787269859364,3.6560212730140638), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.58, y0=-7.84, m=-1.2599999999999998: y0+m*(x-x0), 'deriv': lambda x, m=-1.2599999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (7.070240555157512,8.089759444842489), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.58, y0=-3.84, m=2.74: y0+m*(x-x0), 'deriv': lambda x, m=2.74: np.zeros_like(x,dtype=float)+m, 'domain': (7.298867974837517,7.861132025162483), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.58, y0=0.16000000000000003, m=6.74: y0+m*(x-x0), 'deriv': lambda x, m=6.74: np.zeros_like(x,dtype=float)+m, 'domain': (7.459655641750606,7.700344358249394), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.58, y0=4.16, m=10.74: y0+m*(x-x0), 'deriv': lambda x, m=10.74: np.zeros_like(x,dtype=float)+m, 'domain': (7.503978726985936,7.656021273014064), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.58, y0=8.16, m=14.74: y0+m*(x-x0), 'deriv': lambda x, m=14.74: np.zeros_like(x,dtype=float)+m, 'domain': (7.524496647530108,7.635503352469892), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_sum_v7_q07_sum_field_mc.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-8.14, y0=-7.6, m=-1.540000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-1.540000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-8.59202251756816,-7.68797748243184), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.14, y0=-3.6, m=-5.540000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-5.540000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-8.287436838427595,-7.992563161572406), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.14, y0=0.4, m=-9.540000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-9.540000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-8.22652802697784,-8.05347197302216), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.14, y0=4.4, m=-13.540000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-13.540000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-8.201133350136375,-8.078866649863626), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-8.14, y0=8.4, m=-17.54: y0+m*(x-x0), 'deriv': lambda x, m=-17.54: np.zeros_like(x,dtype=float)+m, 'domain': (-8.187243691625019,-8.092756308374982), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.14, y0=-7.6, m=2.46: y0+m*(x-x0), 'deriv': lambda x, m=2.46: np.zeros_like(x,dtype=float)+m, 'domain': (-4.452560618659716,-3.827439381340283), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.14, y0=-3.6, m=-1.5399999999999996: y0+m*(x-x0), 'deriv': lambda x, m=-1.5399999999999996: np.zeros_like(x,dtype=float)+m, 'domain': (-4.592022517568161,-3.687977482431839), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.14, y0=0.4, m=-5.54: y0+m*(x-x0), 'deriv': lambda x, m=-5.54: np.zeros_like(x,dtype=float)+m, 'domain': (-4.2874368384275945,-3.992563161572405), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.14, y0=4.4, m=-9.54: y0+m*(x-x0), 'deriv': lambda x, m=-9.54: np.zeros_like(x,dtype=float)+m, 'domain': (-4.226528026977839,-4.053471973022161), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-4.14, y0=8.4, m=-13.54: y0+m*(x-x0), 'deriv': lambda x, m=-13.54: np.zeros_like(x,dtype=float)+m, 'domain': (-4.201133350136375,-4.078866649863625), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.14, y0=-7.6, m=6.46: y0+m*(x-x0), 'deriv': lambda x, m=6.46: np.zeros_like(x,dtype=float)+m, 'domain': (-0.26697070037127907,-0.013029299628720958), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.14, y0=-3.6, m=2.46: y0+m*(x-x0), 'deriv': lambda x, m=2.46: np.zeros_like(x,dtype=float)+m, 'domain': (-0.4525606186597167,0.17256061865971667), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.14, y0=0.4, m=-1.54: y0+m*(x-x0), 'deriv': lambda x, m=-1.54: np.zeros_like(x,dtype=float)+m, 'domain': (-0.5920225175681608,0.3120225175681608), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.14, y0=4.4, m=-5.54: y0+m*(x-x0), 'deriv': lambda x, m=-5.54: np.zeros_like(x,dtype=float)+m, 'domain': (-0.287436838427595,0.007436838427594961), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-0.14, y0=8.4, m=-9.540000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-9.540000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (-0.22652802697783941,-0.0534719730221606), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.86, y0=-7.6, m=10.459999999999999: y0+m*(x-x0), 'deriv': lambda x, m=10.459999999999999: np.zeros_like(x,dtype=float)+m, 'domain': (3.7810102497485443,3.9389897502514555), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.86, y0=-3.6, m=6.46: y0+m*(x-x0), 'deriv': lambda x, m=6.46: np.zeros_like(x,dtype=float)+m, 'domain': (3.7330292996287207,3.986970700371279), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.86, y0=0.4, m=2.46: y0+m*(x-x0), 'deriv': lambda x, m=2.46: np.zeros_like(x,dtype=float)+m, 'domain': (3.547439381340283,4.172560618659716), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.86, y0=4.4, m=-1.5400000000000005: y0+m*(x-x0), 'deriv': lambda x, m=-1.5400000000000005: np.zeros_like(x,dtype=float)+m, 'domain': (3.407977482431839,4.312022517568161), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=3.86, y0=8.4, m=-5.540000000000001: y0+m*(x-x0), 'deriv': lambda x, m=-5.540000000000001: np.zeros_like(x,dtype=float)+m, 'domain': (3.712563161572405,4.007436838427595), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.86, y0=-7.6, m=14.46: y0+m*(x-x0), 'deriv': lambda x, m=14.46: np.zeros_like(x,dtype=float)+m, 'domain': (7.802737045990599,7.917262954009401), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.86, y0=-3.6, m=10.46: y0+m*(x-x0), 'deriv': lambda x, m=10.46: np.zeros_like(x,dtype=float)+m, 'domain': (7.781010249748545,7.938989750251456), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.86, y0=0.4, m=6.46: y0+m*(x-x0), 'deriv': lambda x, m=6.46: np.zeros_like(x,dtype=float)+m, 'domain': (7.733029299628721,7.9869707003712795), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.86, y0=4.4, m=2.46: y0+m*(x-x0), 'deriv': lambda x, m=2.46: np.zeros_like(x,dtype=float)+m, 'domain': (7.547439381340284,8.172560618659716), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=7.86, y0=8.4, m=-1.54: y0+m*(x-x0), 'deriv': lambda x, m=-1.54: np.zeros_like(x,dtype=float)+m, 'domain': (7.407977482431839,8.312022517568161), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_sum_v7_q17_sum_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.2))
make_context_graph(ax, [
    {'expr': lambda x, A=50.0, k=0.24: A*np.exp(k*x), 'deriv': lambda x, A=50.0, k=0.24: A*k*np.exp(k*x), 'color': 'steelblue', 'label': None}
], xmin=0, xmax=10, ymin=0, ymax=350, xlabel='Time', ylabel='Quantity', title='')
save_graph(fig, 'u7_sum_v7_q20_sum_exp_graph.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.57, y0=-7.85, m=-16.42: y0+m*(x-x0), 'deriv': lambda x, m=-16.42: np.zeros_like(x,dtype=float)+m, 'domain': (-7.616199421358431,-7.52380057864157), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.57, y0=-3.85, m=-12.42: y0+m*(x-x0), 'deriv': lambda x, m=-12.42: np.zeros_like(x,dtype=float)+m, 'domain': (-7.6309942417015435,-7.509005758298457), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.57, y0=0.15000000000000002, m=-8.42: y0+m*(x-x0), 'deriv': lambda x, m=-8.42: np.zeros_like(x,dtype=float)+m, 'domain': (-7.659631366952613,-7.480368633047387), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.57, y0=4.15, m=-4.42: y0+m*(x-x0), 'deriv': lambda x, m=-4.42: np.zeros_like(x,dtype=float)+m, 'domain': (-7.737707094491737,-7.402292905508263), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.57, y0=8.15, m=-0.41999999999999993: y0+m*(x-x0), 'deriv': lambda x, m=-0.41999999999999993: np.zeros_like(x,dtype=float)+m, 'domain': (-8.270706400261595,-6.8692935997384055), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.5700000000000003, y0=-7.85, m=-12.42: y0+m*(x-x0), 'deriv': lambda x, m=-12.42: np.zeros_like(x,dtype=float)+m, 'domain': (-3.630994241701544,-3.5090057582984566), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.5700000000000003, y0=-3.85, m=-8.42: y0+m*(x-x0), 'deriv': lambda x, m=-8.42: np.zeros_like(x,dtype=float)+m, 'domain': (-3.6596313669526137,-3.480368633047387), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.5700000000000003, y0=0.15000000000000002, m=-4.42: y0+m*(x-x0), 'deriv': lambda x, m=-4.42: np.zeros_like(x,dtype=float)+m, 'domain': (-3.7377070944917374,-3.402292905508263), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.5700000000000003, y0=4.15, m=-0.41999999999999993: y0+m*(x-x0), 'deriv': lambda x, m=-0.41999999999999993: np.zeros_like(x,dtype=float)+m, 'domain': (-4.270706400261595,-2.8692935997384055), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.5700000000000003, y0=8.15, m=3.58: y0+m*(x-x0), 'deriv': lambda x, m=3.58: np.zeros_like(x,dtype=float)+m, 'domain': (-3.7744636659534305,-3.36553633404657), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.42999999999999994, y0=-7.85, m=-8.42: y0+m*(x-x0), 'deriv': lambda x, m=-8.42: np.zeros_like(x,dtype=float)+m, 'domain': (0.3403686330473866,0.5196313669526132), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.42999999999999994, y0=-3.85, m=-4.42: y0+m*(x-x0), 'deriv': lambda x, m=-4.42: np.zeros_like(x,dtype=float)+m, 'domain': (0.2622929055082627,0.5977070944917372), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.42999999999999994, y0=0.15000000000000002, m=-0.42000000000000004: y0+m*(x-x0), 'deriv': lambda x, m=-0.42000000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (-0.27070640026159465,1.1307064002615945), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.42999999999999994, y0=4.15, m=3.58: y0+m*(x-x0), 'deriv': lambda x, m=3.58: np.zeros_like(x,dtype=float)+m, 'domain': (0.2255363340465696,0.6344636659534303), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.42999999999999994, y0=8.15, m=7.58: y0+m*(x-x0), 'deriv': lambda x, m=7.58: np.zeros_like(x,dtype=float)+m, 'domain': (0.33059744230486565,0.5294025576951342), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.43, y0=-7.85, m=-4.42: y0+m*(x-x0), 'deriv': lambda x, m=-4.42: np.zeros_like(x,dtype=float)+m, 'domain': (4.262292905508263,4.597707094491737), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.43, y0=-3.85, m=-0.4200000000000004: y0+m*(x-x0), 'deriv': lambda x, m=-0.4200000000000004: np.zeros_like(x,dtype=float)+m, 'domain': (3.729293599738405,5.1307064002615945), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.43, y0=0.15000000000000002, m=3.58: y0+m*(x-x0), 'deriv': lambda x, m=3.58: np.zeros_like(x,dtype=float)+m, 'domain': (4.225536334046569,4.63446366595343), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.43, y0=4.15, m=7.58: y0+m*(x-x0), 'deriv': lambda x, m=7.58: np.zeros_like(x,dtype=float)+m, 'domain': (4.330597442304866,4.529402557695134), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.43, y0=8.15, m=11.58: y0+m*(x-x0), 'deriv': lambda x, m=11.58: np.zeros_like(x,dtype=float)+m, 'domain': (4.364612956027692,4.495387043972308), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.43, y0=-7.85, m=-0.41999999999999993: y0+m*(x-x0), 'deriv': lambda x, m=-0.41999999999999993: np.zeros_like(x,dtype=float)+m, 'domain': (7.729293599738405,9.130706400261595), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.43, y0=-3.85, m=3.58: y0+m*(x-x0), 'deriv': lambda x, m=3.58: np.zeros_like(x,dtype=float)+m, 'domain': (8.22553633404657,8.63446366595343), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.43, y0=0.15000000000000002, m=7.58: y0+m*(x-x0), 'deriv': lambda x, m=7.58: np.zeros_like(x,dtype=float)+m, 'domain': (8.330597442304866,8.529402557695134), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.43, y0=4.15, m=11.58: y0+m*(x-x0), 'deriv': lambda x, m=11.58: np.zeros_like(x,dtype=float)+m, 'domain': (8.364612956027692,8.495387043972308), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.43, y0=8.15, m=15.579999999999998: y0+m*(x-x0), 'deriv': lambda x, m=15.579999999999998: np.zeros_like(x,dtype=float)+m, 'domain': (8.381319683168291,8.478680316831708), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_sum_v8_q12_sum_field_mc.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
make_piecewise_graph(ax, [
    {'expr': lambda x, x0=-7.84, y0=-7.95, m=7.95: y0+m*(x-x0), 'deriv': lambda x, m=7.95: np.zeros_like(x,dtype=float)+m, 'domain': (-7.938594144169887,-7.741405855830113), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.84, y0=-3.95, m=3.95: y0+m*(x-x0), 'deriv': lambda x, m=3.95: np.zeros_like(x,dtype=float)+m, 'domain': (-8.033883274906607,-7.6461167250933935), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.84, y0=0.050000000000000044, m=-0.050000000000000044: y0+m*(x-x0), 'deriv': lambda x, m=-0.050000000000000044: np.zeros_like(x,dtype=float)+m, 'domain': (-8.629014347713497,-7.050985652286503), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.84, y0=4.05, m=-4.05: y0+m*(x-x0), 'deriv': lambda x, m=-4.05: np.zeros_like(x,dtype=float)+m, 'domain': (-8.029374400092593,-7.650625599907407), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-7.84, y0=8.05, m=-8.05: y0+m*(x-x0), 'deriv': lambda x, m=-8.05: np.zeros_like(x,dtype=float)+m, 'domain': (-7.937388100302183,-7.742611899697817), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.84, y0=-7.95, m=7.95: y0+m*(x-x0), 'deriv': lambda x, m=7.95: np.zeros_like(x,dtype=float)+m, 'domain': (-3.938594144169887,-3.741405855830113), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.84, y0=-3.95, m=3.95: y0+m*(x-x0), 'deriv': lambda x, m=3.95: np.zeros_like(x,dtype=float)+m, 'domain': (-4.033883274906606,-3.6461167250933935), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.84, y0=0.050000000000000044, m=-0.050000000000000044: y0+m*(x-x0), 'deriv': lambda x, m=-0.050000000000000044: np.zeros_like(x,dtype=float)+m, 'domain': (-4.629014347713497,-3.0509856522865024), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.84, y0=4.05, m=-4.05: y0+m*(x-x0), 'deriv': lambda x, m=-4.05: np.zeros_like(x,dtype=float)+m, 'domain': (-4.029374400092593,-3.6506255999074066), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=-3.84, y0=8.05, m=-8.05: y0+m*(x-x0), 'deriv': lambda x, m=-8.05: np.zeros_like(x,dtype=float)+m, 'domain': (-3.937388100302183,-3.742611899697817), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.16000000000000003, y0=-7.95, m=7.95: y0+m*(x-x0), 'deriv': lambda x, m=7.95: np.zeros_like(x,dtype=float)+m, 'domain': (0.061405855830112835,0.25859414416988724), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.16000000000000003, y0=-3.95, m=3.95: y0+m*(x-x0), 'deriv': lambda x, m=3.95: np.zeros_like(x,dtype=float)+m, 'domain': (-0.03388327490660653,0.3538832749066066), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.16000000000000003, y0=0.050000000000000044, m=-0.050000000000000044: y0+m*(x-x0), 'deriv': lambda x, m=-0.050000000000000044: np.zeros_like(x,dtype=float)+m, 'domain': (-0.6290143477134973,0.9490143477134974), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.16000000000000003, y0=4.05, m=-4.05: y0+m*(x-x0), 'deriv': lambda x, m=-4.05: np.zeros_like(x,dtype=float)+m, 'domain': (-0.029374400092593123,0.3493744000925932), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=0.16000000000000003, y0=8.05, m=-8.05: y0+m*(x-x0), 'deriv': lambda x, m=-8.05: np.zeros_like(x,dtype=float)+m, 'domain': (0.06261189969781698,0.25738810030218306), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.16, y0=-7.95, m=7.95: y0+m*(x-x0), 'deriv': lambda x, m=7.95: np.zeros_like(x,dtype=float)+m, 'domain': (4.061405855830113,4.258594144169887), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.16, y0=-3.95, m=3.95: y0+m*(x-x0), 'deriv': lambda x, m=3.95: np.zeros_like(x,dtype=float)+m, 'domain': (3.9661167250933937,4.3538832749066065), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.16, y0=0.050000000000000044, m=-0.050000000000000044: y0+m*(x-x0), 'deriv': lambda x, m=-0.050000000000000044: np.zeros_like(x,dtype=float)+m, 'domain': (3.3709856522865027,4.949014347713497), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.16, y0=4.05, m=-4.05: y0+m*(x-x0), 'deriv': lambda x, m=-4.05: np.zeros_like(x,dtype=float)+m, 'domain': (3.970625599907407,4.349374400092593), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=4.16, y0=8.05, m=-8.05: y0+m*(x-x0), 'deriv': lambda x, m=-8.05: np.zeros_like(x,dtype=float)+m, 'domain': (4.062611899697817,4.257388100302183), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.16, y0=-7.95, m=7.95: y0+m*(x-x0), 'deriv': lambda x, m=7.95: np.zeros_like(x,dtype=float)+m, 'domain': (8.061405855830113,8.258594144169887), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.16, y0=-3.95, m=3.95: y0+m*(x-x0), 'deriv': lambda x, m=3.95: np.zeros_like(x,dtype=float)+m, 'domain': (7.966116725093394,8.353883274906607), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.16, y0=0.050000000000000044, m=-0.050000000000000044: y0+m*(x-x0), 'deriv': lambda x, m=-0.050000000000000044: np.zeros_like(x,dtype=float)+m, 'domain': (7.370985652286503,8.949014347713497), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.16, y0=4.05, m=-4.05: y0+m*(x-x0), 'deriv': lambda x, m=-4.05: np.zeros_like(x,dtype=float)+m, 'domain': (7.970625599907407,8.349374400092593), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False},
    {'expr': lambda x, x0=8.16, y0=8.05, m=-8.05: y0+m*(x-x0), 'deriv': lambda x, m=-8.05: np.zeros_like(x,dtype=float)+m, 'domain': (8.062611899697817,8.257388100302183), 'include_left': True, 'include_right': True, 'arrow_left': False, 'arrow_right': False}
], title='', dot_scale=0)
save_graph(fig, 'u7_sum_v8_q17_sum_field.png')
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.2))
make_context_graph(ax, [
    {'expr': lambda x, A=40.0, k=-0.2: A*np.exp(k*x), 'deriv': lambda x, A=40.0, k=-0.2: A*k*np.exp(k*x), 'color': 'steelblue', 'label': None}
], xmin=0, xmax=6, ymin=0, ymax=60, xlabel='Time', ylabel='Quantity', title='')
save_graph(fig, 'u7_sum_v8_q20_sum_exp_graph.png')
plt.close(fig)
