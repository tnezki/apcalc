# =============================================================================
# ~graph_tool_v14.py — AUTHORITATIVE GRAPH ENTRYPOINT
# =============================================================================
#
# v14 preserves v13 line-weight styling and makes standard Cartesian exit
# arrows visibly render on BOTH ends of each axis. It also ensures horizontal
# curves that continue through the left/right edges receive visible curve
# arrows on both ends. Other curve exit arrows remain inherited from v13.
#
# Context/modeling graphs remain domain-aware: their axes are not forced to
# extend into directions excluded by the modeled domain.
#
# Do not manually override graph styling in generation blocks.
# =============================================================================

from pathlib import Path
import importlib.util
import os


def _find_v13():
    here = Path(__file__).resolve()
    candidates = [here.with_name('~graph_tool_v13.py')]

    env_root = os.environ.get('CURRICULUM_MEMORIES_ROOT')
    if env_root:
        candidates.append(Path(env_root) / 'Tools' / '~graph_tool_v13.py')

    candidates.append(Path.home() / 'Documents' / 'GitHub' / 'memories' / 'Tools' / '~graph_tool_v13.py')

    for parent in here.parents:
        candidates.append(parent / 'Tools' / '~graph_tool_v13.py')
        candidates.append(parent / 'memories' / 'Tools' / '~graph_tool_v13.py')

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        'Graph tool v14 requires Tools/~graph_tool_v13.py. '
        'Stage v13 beside v14 or set CURRICULUM_MEMORIES_ROOT.'
    )


_BASE_PATH = _find_v13()
_spec = importlib.util.spec_from_file_location('_curriculum_graph_tool_v13', str(_BASE_PATH))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_v13_module = _base

for _name in dir(_v13_module):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_v13_module, _name)

# A copied authoritative wrapper must write figures beside the copied file,
# not beside its imported base module.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


_STANDARD_MIN = -10
_STANDARD_MAX = 10
_STANDARD_VIEW_PAD = 0.8
_STANDARD_ARROW_LENGTH = 0.45
_STANDARD_ARROW_WIDTH = 1.2


def _reveal_standard_exit_arrows(ax):
    """Leave the v13 graph unchanged except for enough viewport to show exits."""
    ax.set_xlim(_STANDARD_MIN - _STANDARD_VIEW_PAD, _STANDARD_MAX + _STANDARD_VIEW_PAD)
    ax.set_ylim(_STANDARD_MIN - _STANDARD_VIEW_PAD, _STANDARD_MAX + _STANDARD_VIEW_PAD)


def _add_horizontal_curve_exit_arrows(ax, functions):
    """v12/v13 skip exit arrows when the endpoint slope is zero; v14 fills that gap."""
    for fn in functions:
        f = fn['expr']
        fprime = fn['deriv']
        color = fn['color']
        for x_edge, direction in ((_STANDARD_MIN, -1), (_STANDARD_MAX, 1)):
            try:
                y_edge = float(f(np.array([x_edge]))[0])
                slope = float(fprime(x_edge))
            except Exception:
                continue
            if not np.isfinite(y_edge) or not (_STANDARD_MIN <= y_edge <= _STANDARD_MAX):
                continue
            if abs(slope) >= 0.001:
                continue
            ax.annotate(
                '',
                xy=(x_edge + direction * _STANDARD_ARROW_LENGTH, y_edge),
                xytext=(x_edge, y_edge),
                arrowprops=dict(
                    arrowstyle='-|>', color=color,
                    lw=_STANDARD_ARROW_WIDTH, mutation_scale=12
                ),
                annotation_clip=False,
            )


def make_window_graph(ax, functions, xmin, xmax, ymin, ymax, title='', xlabel='x', ylabel='y'):
    """Standard Cartesian appearance on a custom two-sided window."""
    x_range = xmax - xmin
    y_range = ymax - ymin
    x_step = _nice_grid_step(x_range, max_lines=16)
    y_step = _nice_grid_step(y_range, max_lines=16)
    x_ticks = np.arange(np.ceil(xmin/x_step)*x_step, xmax + x_step*0.01, x_step)
    y_ticks = np.arange(np.ceil(ymin/y_step)*y_step, ymax + y_step*0.01, y_step)
    x_pad = x_step * 0.55
    y_pad = y_step * 0.55
    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_pad, ymax + y_pad)

    for xt in x_ticks:
        ax.plot([xt, xt], [ymin, ymax], color='#aaaaaa', linewidth=0.4, zorder=0)
    for yt in y_ticks:
        ax.plot([xmin, xmax], [yt, yt], color='#aaaaaa', linewidth=0.4, zorder=0)

    arrow_L = min(x_step, y_step) * 0.45
    for fn in functions:
        f = fn['expr']; fprime = fn['deriv']; color = fn['color']; label = fn.get('label')
        x = np.linspace(xmin, xmax, 2200)
        y = f(x)
        finite = np.isfinite(y)
        mask = finite & (y >= ymin) & (y <= ymax)
        inds = np.where(mask)[0]
        if len(inds):
            segments = np.split(inds, np.where(np.diff(inds) > 5)[0] + 1)
            for seg in segments:
                if len(seg) > 1:
                    ax.plot(x[seg], y[seg], color=color, linewidth=1.5, label=label, zorder=3)
                    label = None

        for x_edge, direction in ((xmin, -1.0), (xmax, 1.0)):
            try:
                y_edge = float(f(np.array([x_edge]))[0])
                slope = float(fprime(x_edge))
            except Exception:
                continue
            if not np.isfinite(y_edge) or not (ymin <= y_edge <= ymax):
                continue
            dx_d = direction; dy_d = slope * direction
            mag = np.hypot(dx_d, dy_d)
            if mag < 1e-9:
                dx_d, dy_d, mag = direction, 0.0, 1.0
            dx_d /= mag; dy_d /= mag
            ax.annotate('', xy=(x_edge + dx_d*arrow_L, y_edge + dy_d*arrow_L), xytext=(x_edge, y_edge),
                        arrowprops=dict(arrowstyle='-|>', color=color, lw=1.2, mutation_scale=12),
                        annotation_clip=False)

        for edge_y, direction in ((ymin, -1.0), (ymax, 1.0)):
            vals = y - edge_y
            good = np.isfinite(vals[:-1]) & np.isfinite(vals[1:])
            cross = np.where(good & (np.sign(vals[:-1]) != np.sign(vals[1:])))[0]
            for idx in cross:
                xr = np.interp(0, [vals[idx], vals[idx+1]], [x[idx], x[idx+1]])
                if xr <= xmin + 1e-4 or xr >= xmax - 1e-4:
                    continue
                try:
                    slope = float(fprime(xr))
                except Exception:
                    continue
                if abs(slope) < 1e-9:
                    continue
                dy_d = direction; dx_d = dy_d / slope
                mag = np.hypot(dx_d, dy_d); dx_d /= mag; dy_d /= mag
                ax.annotate('', xy=(xr + dx_d*arrow_L, edge_y + dy_d*arrow_L), xytext=(xr, edge_y),
                            arrowprops=dict(arrowstyle='-|>', color=color, lw=1.2, mutation_scale=12),
                            annotation_clip=False)

    # Axes and arrows on both ends whenever that axis is in the window.
    tri = dict(arrowstyle='-|>', color='#222222', lw=1.2, mutation_scale=13)
    if ymin <= 0 <= ymax:
        ax.plot([xmin, xmax], [0, 0], color='#222222', linewidth=1.2, zorder=2)
        ax.annotate('', xy=(xmax + x_pad*0.8, 0), xytext=(xmax, 0), arrowprops=tri, annotation_clip=False)
        ax.annotate('', xy=(xmin - x_pad*0.8, 0), xytext=(xmin, 0), arrowprops=tri, annotation_clip=False)
        ax.text(xmax + x_pad*0.62, y_step*0.22, xlabel, fontsize=12, fontweight='bold',
                fontfamily='Times New Roman', ha='center', va='bottom')
    if xmin <= 0 <= xmax:
        ax.plot([0, 0], [ymin, ymax], color='#222222', linewidth=1.2, zorder=2)
        ax.annotate('', xy=(0, ymax + y_pad*0.8), xytext=(0, ymax), arrowprops=tri, annotation_clip=False)
        ax.annotate('', xy=(0, ymin - y_pad*0.8), xytext=(0, ymin), arrowprops=tri, annotation_clip=False)
        ax.text(x_step*0.20, ymax + y_pad*0.62, ylabel, fontsize=12, fontweight='bold',
                fontfamily='Times New Roman', ha='left', va='center')

    # Keep numeric labels light and sparse.
    x_lab_every = _label_every(len(x_ticks)); y_lab_every = _label_every(len(y_ticks))
    ax.set_xticks([t for i,t in enumerate(x_ticks) if i % x_lab_every == 0])
    ax.set_yticks([t for i,t in enumerate(y_ticks) if i % y_lab_every == 0])
    ax.set_xticklabels([_fmt(t) for i,t in enumerate(x_ticks) if i % x_lab_every == 0],
                       fontfamily='Times New Roman', fontsize=9)
    ax.set_yticklabels([_fmt(t) for i,t in enumerate(y_ticks) if i % y_lab_every == 0],
                       fontfamily='Times New Roman', fontsize=9)
    ax.tick_params(which='major', length=3, width=0.8, color='#444444')
    for spine in ax.spines.values():
        spine.set_visible(False)
    key_points = _find_key_points(functions, xmin, xmax, ymin, ymax)
    _draw_legend(ax, functions, key_points, xmin, xmax, ymin, ymax)
    ax.set_title(title, fontfamily='Times New Roman', fontsize=12, pad=8)
    return ax


_v13_make_standard_graph = _v13_module.make_standard_graph


def make_standard_graph(ax, functions, title=''):
    result = _v13_make_standard_graph(ax, functions, title=title)
    _add_horizontal_curve_exit_arrows(ax, functions)
    _reveal_standard_exit_arrows(ax)
    return result


if hasattr(_v13_module, 'make_piecewise_graph'):
    _v13_make_piecewise_graph = _v13_module.make_piecewise_graph

    def make_piecewise_graph(ax, pieces, title='', dot_scale=0.65):
        result = _v13_make_piecewise_graph(ax, pieces, title=title, dot_scale=dot_scale)
        _reveal_standard_exit_arrows(ax)
        return result


# All other graph types and helpers are inherited unchanged from v13.
# =============================================================================

# PASTE YOUR GRAPH CODE BELOW THIS LINE

def _save(fig, filename):
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)

fig, axs = plt.subplots(1,3,figsize=(12,3.8))
# A: removable discontinuity
fA=lambda x:.15*np.asarray(x)**3-.15*np.asarray(x)+1
fpA=lambda x:.45*np.asarray(x)**2-.15
make_window_graph(axs[0],[{'expr':fA,'deriv':fpA,'color':'steelblue','label':None}],-3,3,-3,3)
xv=.9; yv=float(fA(np.array([xv]))[0]); axs[0].plot(xv,yv,'o',mfc='white',mec='steelblue',mew=1.6,ms=7,zorder=8); axs[0].set_title('A')
# B: vertical asymptote
fB=lambda x:-1/(np.asarray(x)-1)
fpB=lambda x:1/(np.asarray(x)-1)**2
make_window_graph(axs[1],[{'expr':fB,'deriv':fpB,'color':'steelblue','label':None}],-3,3,-3,3); axs[1].set_title('B')
# C: jump discontinuity
left=lambda x:np.where(np.asarray(x)<1,-.18*(np.asarray(x)+1)**2+1.5,np.nan)
leftp=lambda x:np.where(np.asarray(x)<1,-.36*(np.asarray(x)+1),0.0)
right=lambda x:np.where(np.asarray(x)>=1,.55*(np.asarray(x)-1.8)**2-1.4,np.nan)
rightp=lambda x:np.where(np.asarray(x)>=1,1.1*(np.asarray(x)-1.8),0.0)
make_window_graph(axs[2],[{'expr':left,'deriv':leftp,'color':'steelblue','label':None},{'expr':right,'deriv':rightp,'color':'steelblue','label':None}],-3,3,-3,3)
yL=float(-.18*(1+1)**2+1.5); yR=float(.55*(1-1.8)**2-1.4)
axs[2].plot(1,yL,'o',mfc='white',mec='steelblue',mew=1.6,ms=7,zorder=8); axs[2].plot(1,yR,'o',mfc='steelblue',mec='steelblue',ms=7,zorder=8); axs[2].set_title('C')
fig.tight_layout(); _save(fig,'discontinuity_types.png')

fig, ax = plt.subplots(figsize=(6.7,4.2)); make_window_graph(ax,[],-.5,4.6,-.5,2.7)
ax.plot([0,1],[1,0],color='steelblue',linewidth=1.5); ax.plot(0,1,'o',color='steelblue',ms=6); ax.plot(1,0,'o',mfc='white',mec='steelblue',mew=1.6,ms=7)
ax.plot([1,2],[1,1],color='steelblue',linewidth=1.5); ax.plot(1,1,'o',color='steelblue',ms=6); ax.plot(2,1,'o',mfc='white',mec='steelblue',mew=1.6,ms=7)
ax.plot(2,2,'o',color='steelblue',ms=6); ax.plot([2,3,4],[1,2,1],color='steelblue',linewidth=1.5); ax.plot(3,2,'o',color='steelblue',ms=6); ax.plot(4,1,'o',color='steelblue',ms=6)
_save(fig,'continuity_reference.png')

fig, ax = plt.subplots(figsize=(7.0,4.2)); make_window_graph(ax,[],0,10,0,7)
a,b=2,8; fa,fb,k=5.8,2.0,3.6
xs=np.array([1.2,2,3.2,4.0,5.2,6.0,7.0,8,9.1]); ys=np.array([6.5,5.8,4.1,4.7,2.1,4.6,3.0,2.0,4.4])
xd=np.linspace(xs.min(),xs.max(),800); yd=np.interp(xd,xs,ys); kernel=np.exp(-.5*(np.linspace(-3,3,41))**2); kernel/=kernel.sum(); yd=np.convolve(yd,kernel,mode='same')
ax.axvspan(a,b,0,1,color='#d9dde1',alpha=.35,zorder=0); ax.plot(xd,yd,color='steelblue',linewidth=1.5,zorder=3)
for yv,lab in [(fa,'f(a)'),(k,'k'),(fb,'f(b)')]:
    ax.hlines(yv,0,a if lab!='k' else 10,color='#777',linewidth=.8,linestyle='--' if lab=='k' else ':'); ax.text(.12,yv,lab,ha='left',va='bottom',fontsize=9)
for xv,lab in [(a,'a'),(b,'b')]: ax.vlines(xv,0,fa if lab=='a' else fb,color='#777',linewidth=.8,linestyle=':'); ax.text(xv,.12,lab,ha='center',va='bottom',fontsize=9)
_save(fig,'ivt_reference.png')
