"""Manuscript figures for the risk-calibrated replanning trigger.

    python figures.py    ->  figures/fig1_risk_map.{pdf,png} ... fig4_forward_progress.{pdf,png}

Terminology (used in every label and caption):
    computed  -- evaluated from the leaf-energy distribution of the Markov task-chain
                 model; exact under the model's assumptions, not a measurement.
    r_D(B)    -- brownout risk P(E_D > B) over a D-task horizon with budget B.
    U_D, L_D  -- Cantelli upper and lower bounds on r_D(B), from mean and variance only.
    B_p(D)    -- quantile budget: smallest B with r_D(B) <= p (computed).
    B+_p, B-_p -- Cantelli safe and failure budgets; B-_p <= B_p <= B+_p.
    N(B)      -- number of tasks completed before brownout.
"""
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker
import matplotlib.text
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from sim.sweep_trigger import (P, e, depth_profile, exact_tail, cantelli_bound,
                           cantelli_lower, exact_required_budget, expected_completed,
                           progress_steps, min_trajectory_energy)

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.4, "grid.linewidth": 0.4,
    "grid.alpha": 0.35, "savefig.dpi": 300, "pdf.fonttype": 42,
})
# Okabe-Ito palette (colour-blind safe), one meaning per colour across all figures.
C_UP, C_LO, C_MEAN, C_BAND, C_COMP = "#0072B2", "#D55E00", "#009E73", "0.87", "black"

V, V_MIN = 3.2, 0.1
B_MAX = 600e-3
LO, HI = 1e-3, 1 - 1e-3
RISK_TICKS = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]

to_mF = (lambda B: 2 * B / (V**2 - V_MIN**2), lambda C: C * (V**2 - V_MIN**2) / 2)  # mJ <-> mF


def matched_rc(fig_width, reference_width=4.6):
    """rcParams scaled by fig_width / reference_width, so a figure made wider than the
    reference looks the same as the others when all are shown at a common width."""
    k = fig_width / reference_width
    keys = ["font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize", "ytick.labelsize",
            "legend.fontsize", "axes.linewidth", "xtick.major.size", "ytick.major.size",
            "xtick.major.width", "ytick.major.width", "xtick.major.pad", "ytick.major.pad"]
    return {key: float(plt.rcParams[key]) * k for key in keys}


def logit(x):
    x = np.clip(x, LO, HI)
    return np.log(x / (1 - x))






def logit_risk_axis(ax):
    ax.set_yscale("logit")
    ax.set_ylim(LO, HI)
    ax.set_yticks(RISK_TICKS, [f"{t:g}" for t in RISK_TICKS])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def risk_steps(marginal):
    """Computed r_D(B) as an exact step function on [0, B_MAX]."""
    x = np.array(sorted(marginal))
    tail = 1 - np.cumsum([marginal[v] for v in x])
    keep = x <= B_MAX
    return np.r_[0, x[keep], B_MAX] * 1e3, np.r_[1, tail[keep], tail[keep][-1] if keep.any() else 1]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


# ---------------------------------------------------------------- data
PROF = depth_profile(P, e, start=0, max_depth=40)
PROF20 = PROF[:21]
DEPTHS = np.array([r[0] for r in PROF20])
MU = np.array([r[1] for r in PROF20])
SD = np.sqrt([r[2] for r in PROF20])
E_MIN_D = np.array(min_trajectory_energy(P, e, start=0, max_depth=40))  # cheapest trajectory


# ---------------------------------------------------------------- Figure 1
def risk_plane(figsize=(4.6, 3.5)):
    """Shared planning-budget plane: computed risk r_D(B) on a logit colour scale."""
    p_levels = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    cmap = ListedColormap(plt.cm.plasma(np.linspace(0, 0.9, 256)))
    norm = Normalize(logit(LO), logit(HI))
    budgets = np.linspace(0.5e-3, B_MAX, 1200)
    risk = np.array([[exact_tail(r[3], B) for r in PROF20] for B in budgets])

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    mesh = ax.pcolormesh(DEPTHS, budgets * 1e3, logit(risk), shading="nearest",
                         cmap=cmap, norm=norm, rasterized=True)
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(0, B_MAX * 1e3)
    ax.set_xticks(range(0, 21, 2))
    ax.set_xlabel(r"Planning Horizon $D$ (Tasks after the Replan)")
    ax.set_ylabel(r"Energy Budget $B$ (mJ)")
    cb = fig.colorbar(mesh, ax=ax, ticks=[logit(p) for p in p_levels], pad=0.02)
    cb.ax.set_yticklabels([f"{p:g}" for p in p_levels])
    cb.set_label(r"Computed Brownout Risk $r_D(B)$ (Logit Scale)")
    return fig, ax


def vertex_counts(start=0, max_depth=20, ndigits=9, through=True):
    """Trajectory counts per vertex (D, E_D), keyed by (D, E in mJ).

    through=True: number of the full-length trajectories (start to max_depth) that
    pass through the vertex = (histories reaching it) x (continuations from its
    state to max_depth). Every column then sums to the total trajectory count.
    through=False: number of distinct histories reaching the vertex only."""
    n = len(e)
    # continuations[L][s]: number of length-L paths starting from state s
    cont = [[1] * n]
    for _ in range(max_depth):
        cont.append([sum(cont[-1][j] for j in range(n) if P[s][j] > 0) for s in range(n)])
    layer = {(start, round(e[start], ndigits)): 1}
    out = {}
    for D in range(max_depth + 1):
        if D > 0:
            nxt = {}
            for (s, E), c in layer.items():
                for j in range(n):
                    if P[s][j] > 0:
                        key = (j, round(E + e[j], ndigits))
                        nxt[key] = nxt.get(key, 0) + c
            layer = nxt
        for (s, E), c in layer.items():
            k = (D, round(E * 1e3, 6))
            out[k] = out.get(k, 0) + c * (cont[max_depth - D][s] if through else 1)
    return out


STATE_NAMES = ["Measure", "Analyze", "Transmit", "Receive"]
# Colours from the state machine in the teaser figure (shiny-train-vis-abstract.pdf).
STATE_COLORS = ["#FFE800", "#00D700", "#6EDFFF", "#E30000"]


def vertex_states(start=0, max_depth=20, ndigits=9):
    """Set of state indices in which trajectories reach each vertex (D, E in mJ).
    Most vertices have one state. At odd depths a trajectory has just completed a
    loop, and because loop order does not change energy, a vertex can be reached
    ending in Analyze (last loop a measure loop) or in Receive (last loop a
    communication loop)."""
    layer, out = {(start, round(e[start], ndigits))}, {}
    for D in range(max_depth + 1):
        if D > 0:
            layer = {(j, round(E + e[j], ndigits)) for s, E in layer
                     for j in range(len(e)) if P[s][j] > 0}
        for s, E in layer:
            out.setdefault((D, round(E * 1e3, 6)), set()).add(s)
    return out


def vertex_probabilities(max_depth=20):
    """Probability that a trajectory passes through each vertex (D, E in mJ): the mass
    of the leaf distribution at energy E at depth D (summed over states). Unlike the
    trajectory counts it does not depend on the horizon, and each column sums to 1."""
    return {(D, round(E * 1e3, 6)): p for D, _, _, mg in PROF[:max_depth + 1] for E, p in mg.items()}


def fmt_prob(p):
    """Two significant figures. From 0.01 up: decimals without the leading zero (.90,
    .048), the convention for quantities bounded by 1, which keeps labels narrow.
    Below 0.01: scientific notation."""
    if p >= 0.995:
        return "1"
    if p >= 0.1:
        return f"{p:.2f}".lstrip("0")
    if p >= 0.01:
        return f"{p:.3f}".lstrip("0")
    mant, exp = f"{p:.1e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    return rf"${mant}{{\times}}10^{{{int(exp)}}}$"


def fmt_pct(p):
    """Probability as a percentage without the % sign. Two significant figures from 1 %
    up (100, 90, 86, 9.8, 1.2); one below, where finer precision is not meaningful at
    this scale and narrow labels matter (0.8, 0.09, 0.02); below 0.01 % in scientific
    notation (4x10^-5)."""
    v = 100 * p
    if v >= 99.5:
        return "100"
    if v >= 9.95:
        return f"{v:.0f}"
    if v >= 0.995:
        return f"{v:.1f}"
    if v >= 0.0095:
        d = -math.floor(math.log10(v))                # one significant figure
        if float(f"{v:.{d}f}") >= 10.0 ** (1 - d):   # rounded up a decade: 0.0998 -> 0.1
            d -= 1
        s = f"{v:.{d}f}"
        return s if float(s) < 1 else f"{v:.1f}"      # 0.995 -> "1.0" stays two s.f.
    mant, exp = f"{v:.0e}".split("e")
    return rf"${mant}{{\times}}10^{{{int(exp)}}}$"


# Figure 6 labels vertices whose displayed value is at least 0.1 (%), so a vertex shown
# as 0.1 in Figure 1c is never missing from Figure 6.
LABEL_CUTOFF = 0.00095


def trajectory_segments(start=0, max_depth=20, ndigits=9):
    """Distinct (D, E) -> (D+1, E') segments over every trajectory from `start`.
    Trajectories share prefixes and energies, so each segment is drawn once."""
    nodes, segs = {(start, round(e[start], ndigits))}, set()
    for D in range(max_depth):
        nxt = set()
        for s, E in nodes:
            for j, pij in enumerate(P[s]):
                if pij > 0:
                    E2 = round(E + e[j], ndigits)
                    segs.add((D, E, E2))
                    nxt.add((j, E2))
        nodes = nxt
    return [[(D, E * 1e3), (D + 1, E2 * 1e3)] for D, E, E2 in sorted(segs)]


C_TREE = "0.5"  # trajectory-tree underlay


def tree_underlay(ax, max_depth=20, transpose=False):
    """Draw the candidate-trajectory tree unobtrusively beneath other content.
    transpose=False: (D, E_D) as in Figure 1b. transpose=True: (E_D, D + 1), i.e.
    cumulative energy against tasks completed, the axes of Figure 4."""
    from matplotlib.collections import LineCollection
    segs = trajectory_segments(max_depth=max_depth)
    if transpose:
        segs = [[(E0, D0 + 1), (E1, D1 + 1)] for (D0, E0), (D1, E1) in segs]
    ax.add_collection(LineCollection(segs, colors=C_TREE, linewidths=0.45, alpha=0.6, zorder=1))
    verts = np.array(sorted({pt for seg in segs for pt in seg}))
    ax.scatter(verts[:, 0], verts[:, 1], s=2.5, color=C_TREE, alpha=0.8, edgecolors="none",
               zorder=1, clip_on=True)


def tree_handle():
    return Line2D([], [], color=C_TREE, lw=0.6, marker="o", ms=1.8, alpha=0.8,
                  label="Candidate Trajectories")


def fig1_risk_map():
    fig, ax = risk_plane()
    # Risk is piecewise constant in D, so each level's boundary is drawn per column
    # (the quantile budget B_p(D)) rather than as an interpolated contour.
    halo = [pe.withStroke(linewidth=2.4, foreground="black")]
    levels = [(0.01, "0.01"), (0.05, "0.05"), (0.25, "0.25"), (0.5, "≥0.5")]
    bps = {p: np.array([exact_required_budget(r[3], p) for r in PROF20]) * 1e3 for p, _ in levels}
    for p, _ in levels:
        ax.step(DEPTHS, bps[p], where="mid", color="white", lw=1.0, path_effects=halo)
    ax.plot(DEPTHS, MU * 1e3, color=C_MEAN, lw=1.8)
    handles = [Line2D([], [], color="white", lw=1.0, path_effects=halo,
                      label=r"Risk Boundary $B_p(D)$, Labelled by $p$"),
               Line2D([], [], color=C_MEAN, lw=1.8, label=r"Mean Demand $\mu_D$")]
    leg = ax.legend(handles=handles, loc="lower right", facecolor="white", framealpha=1)

    # Boundaries can sit closer together than a label is tall, so each label is a
    # callout: it points to one step of its boundary that no other boundary shares,
    # and sits at the nearest position whose box touches no boundary, the mean line,
    # the legend or another label. The leader line may cross other lines.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_data = ax.transData.inverted()
    placed = [leg.get_window_extent(renderer).expanded(1.05, 1.1).transformed(to_data)]
    mean_mJ = MU * 1e3

    def hits_lines(x0, x1, y0, y1):
        for bp in bps.values():
            for c in range(len(bp)):
                if c + 0.5 > x0 and c - 0.5 < x1 and y0 <= bp[c] <= y1:
                    return True
                if c + 1 < len(bp) and x0 < c + 0.5 < x1 and \
                        min(bp[c], bp[c + 1]) <= y1 and max(bp[c], bp[c + 1]) >= y0:
                    return True
        ys = np.interp(np.linspace(x0, x1, 60), DEPTHS, mean_mJ)
        return bool(((ys >= y0) & (ys <= y1)).any())

    def overlaps(box, others):
        return any(box[0] < o.x1 and box[1] > o.x0 and box[2] < o.y1 and box[3] > o.y0 for o in others)

    def data_per_point():
        a, b = ax.transData.transform([(0, 0), (1, 1)])
        return 72 / fig.dpi / (b[0] - a[0]) * fig.dpi / 72, 1 / ((b[1] - a[1]) * 72 / fig.dpi)

    for p, lab in levels:
        bp = bps[p]
        others = [bps[q] for q, _ in levels if q != p]
        # target: a step not shared with another boundary, with room around it, near the middle
        cands = [(min(min(abs(o[D] - bp[D]) for o in others), 60.0), -abs(D - 10), D)
                 for D in range(2, 18) if bp[D] < 0.93 * B_MAX * 1e3
                 and all(abs(o[D] - bp[D]) > 1e-6 for o in others)]
        D = max(cands)[2]
        target = (D, bp[D])
        probe = ax.text(0, 0, lab, fontsize=7, path_effects=halo)
        bb = probe.get_window_extent(renderer)
        probe.remove()
        w, h = (bb.width * 1.25, bb.height * 1.35)
        inv = ax.transData.inverted()
        best = None
        for cx in np.arange(0.6, 20.0, 0.25):
            for cy in np.arange(20, B_MAX * 1e3 - 15, 5):
                px, py = ax.transData.transform((cx, cy))
                (x0, y0), (x1, y1) = inv.transform([(px - w / 2, py - h / 2), (px + w / 2, py + h / 2)])
                if x0 < -0.4 or x1 > 20.4 or y0 < 5 or y1 > B_MAX * 1e3 - 5:
                    continue
                if hits_lines(x0, x1, y0, y1) or overlaps((x0, x1, y0, y1), placed):
                    continue
                tx, ty = ax.transData.transform(target)
                dist = np.hypot(px - tx, py - ty)
                if dist >= 0.12 * fig.dpi and (best is None or dist < best[0]):
                    best = (dist, cx, cy, (x0, x1, y0, y1))
        if best is None:
            print(f"warning: no clear position for the p = {lab} label")
            continue
        _, cx, cy, box = best
        ann = ax.annotate(lab, xy=target, xytext=(cx, cy), color="white", fontsize=7,
                          ha="center", va="center", path_effects=halo,
                          arrowprops=dict(arrowstyle="-", color="white", lw=0.7,
                                          shrinkA=1.5, shrinkB=0))
        ann.arrow_patch.set_path_effects([pe.withStroke(linewidth=1.8, foreground="black")])
        ax.plot(*target, "o", ms=2.6, color="white", mec="black", mew=0.6, zorder=5)  # leader endpoint
        from matplotlib.transforms import Bbox
        placed.append(Bbox.from_extents(box[0], box[2], box[1], box[3]))
    save(fig, "fig1_risk_map")


PAGE_W, COL_W = 7.0, 3.33  # ACM sigconf \textwidth and \columnwidth, in inches
SWEEP_CARS = Path.home() / "catch" / "sweep_cars"
SWEEP_ARMS = (("thimble_noreplan", "No replan"), ("thimble_depth", "Depth"), ("thimble_risk", "Risk"))
SWEEP_WAKES = (("vmax", r"$V_{\max}$"), ("energy", "Energy"))


def label_vertex_counts(fig, ax, fontsize=7.5, color="white", halo_color="black", gap=3.0,
                        dot_r=1.5, avoid=(), max_depth=20, texts=None, own_margin=0.6,
                        leaders=False, constrained_first=False):
    """Label each vertex with the number of full-length trajectories passing through it.

    Each label prefers the spot close, below and right of its vertex. A spot is rejected
    if it comes within 3 pt of another label or of another vertex's dot, or if it lies
    nearer another vertex than its own (so a moved label cannot be misattributed).
    Labels are placed right to left, so in a crowded pair the left one moves.
    gap: minimum clearance in points between a label and other labels or dots.
    dot_r: vertex marker radius in points; offsets grow with it so labels clear the dot.
    avoid: artists (e.g. a legend) that labels must not overlap.
    texts: optional {vertex: label text}; defaults to the pass-through trajectory counts.
    own_margin: a label's distance to its own dot must be below own_margin times its
    distance to any other dot, so near-ties (ambiguous attribution) are rejected.
    leaders: if no spot beside the dot is clear, place the label in the nearest clear
    spot farther out and join it to its dot with a thin leader line, which makes the
    attribution explicit. Returns the number of labels that still found no clear spot."""
    halo = [pe.withStroke(linewidth=2.0, foreground=halo_color)]
    fig.canvas.draw()                       # fix the layout before measuring text
    renderer = fig.canvas.get_renderer()
    pt = fig.dpi / 72
    top = ax.get_ylim()[1]
    if texts is None:
        texts = {k: str(n) for k, n in vertex_counts(max_depth=max_depth).items()}
    counts = {k: t for k, t in texts.items() if k[1] <= top}
    # Every drawn vertex is an obstacle and an ownership competitor, labelled or not.
    all_keys = [k for k in vertex_counts(max_depth=max_depth) if k[1] <= top]
    vpix = {k: ax.transData.transform(k) for k in all_keys}
    dx, dy, side = dot_r + 0.1, dot_r - 0.5, dot_r + 1.5
    placements = [((dx, -dy), "left", "top"), ((-dx, dy), "right", "bottom"),
                  ((-dx, -dy), "right", "top"), ((dx, dy), "left", "bottom"),
                  ((side, 0.0), "left", "center"), ((-side, 0.0), "right", "center"),
                  ((0.0, -(dot_r + 1.1)), "center", "top"), ((0.0, dot_r + 1.1), "center", "bottom")]

    def box_dist(bb, p):
        dx = max(bb.x0 - p[0], 0, p[0] - bb.x1)
        dy = max(bb.y0 - p[1], 0, p[1] - bb.y1)
        return np.hypot(dx, dy)

    def seg_dist(a, b, p):
        """Distance from point p to the segment a-b (display units)."""
        ab, ap = b - a, p - a
        t = np.clip(ap @ ab / max(ab @ ab, 1e-9), 0, 1)
        return np.hypot(*(ap - t * ab))

    def put(key, off, ha, va, leader=False):
        kw = {}
        if leader:
            # Dotted and lighter than the solid trajectory edges, so a leader cannot be
            # mistaken for part of the tree.
            kw["arrowprops"] = dict(arrowstyle="-", color="0.45", lw=0.7, linestyle=(0, (1, 1.2)),
                                    shrinkA=0.5, shrinkB=(dot_r + 0.4) * pt * 72 / fig.dpi)
        return ax.annotate(counts[key], key, xytext=off, textcoords="offset points",
                           ha=ha, va=va, fontsize=fontsize, color=color, path_effects=halo,
                           zorder=4, annotation_clip=True, **kw)

    frame = ax.get_window_extent(renderer)
    inside = lambda bb: (bb.x0 >= frame.x0 + pt and bb.x1 <= frame.x1 - pt
                         and bb.y0 >= frame.y0 + pt and bb.y1 <= frame.y1 - pt)
    placed, crowded = [a.get_window_extent(renderer) for a in avoid], 0

    def n_options(key):
        """Spots beside the dot that are valid ignoring other labels (frame, dots, ownership)."""
        others = [p for k, p in vpix.items() if k != key]
        n = 0
        for off, ha, va in placements:
            t = put(key, off, ha, va)
            bb = t.get_window_extent(renderer)
            t.remove()
            n += (inside(bb) and all(box_dist(bb, p) > (gap + dot_r - 1.0) * pt for p in others)
                  and box_dist(bb, vpix[key]) < own_margin * min((box_dist(bb, p) for p in others),
                                                                 default=np.inf))
        return n

    # Most constrained labels first (fewest valid spots), then right to left.
    order = sorted(counts, key=lambda k: (n_options(k), -k[0], -k[1])) if constrained_first else \
        sorted(counts, key=lambda k: (-k[0], -k[1]))
    for key in order:
        others = [p for k, p in vpix.items() if k != key]   # all other dots
        chosen = None
        for off, ha, va in placements:
            t = put(key, off, ha, va)
            bb = t.get_window_extent(renderer)
            if (inside(bb) and not any(bb.padded(gap * pt).overlaps(o) for o in placed)
                    and all(box_dist(bb, p) > (gap + dot_r - 1.0) * pt for p in others)
                    and box_dist(bb, vpix[key]) < own_margin * min((box_dist(bb, p) for p in others),
                                                                   default=np.inf)):
                chosen = bb
                break
            t.remove()
        if chosen is None and leaders:      # callout: nearest clear spot farther out
            for radius in (10, 13, 16, 20, 24, 28, 34, 40):
                for deg in (-35, 35, 0, -70, 70, 145, -145, 180, 110, -110):
                    off = (radius * np.cos(np.radians(deg)), radius * np.sin(np.radians(deg)))
                    t = put(key, off, "center", "center", leader=True)
                    t.update_positions(renderer)                              # place the offset text first
                    bb = matplotlib.text.Text.get_window_extent(t, renderer)   # text only, not the leader
                    # A callout is identified only by its leader, so it must sit well clear of
                    # every other dot, and the leader must not pass near another dot.
                    c = np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])
                    if (inside(bb) and not any(bb.padded(gap * pt).overlaps(o) for o in placed)
                            and all(box_dist(bb, p) > (gap + dot_r + 3.0) * pt for p in others)
                            and all(seg_dist(c, vpix[key], p) > (dot_r + 2.5) * pt for p in others)):
                        chosen = bb
                        break
                    t.remove()
                if chosen is not None:
                    break
        if chosen is None:                  # nothing clear: fall back to the default spot
            chosen = put(key, *placements[0]).get_window_extent(renderer)
            crowded += 1
        placed.append(chosen)
    return crowded


def fig1b_risk_map_trajectories():
    """Planning-budget plane with every candidate trajectory overlaid."""
    # Full page width, so axis text, ticks and frame are scaled to look the same as
    # the 4.6-in figures when all are shown at a common width.
    with plt.rc_context(matched_rc(7.0)):
        from matplotlib.collections import LineCollection
        fig, ax = risk_plane(figsize=(7.0, 6.2))  # full width and tall: room for every label
        segs = trajectory_segments()
        ax.add_collection(LineCollection(segs, colors="white", linewidths=0.6,
                                         alpha=0.85, capstyle="round"))
        # Distinct vertices (D, E_D): the atoms of the leaf distribution at each horizon.
        verts = np.array(sorted({pt for seg in segs for pt in seg}))
        ax.scatter(verts[:, 0], verts[:, 1], s=5, color="white", edgecolors="none", zorder=3,
                   clip_on=True)
        label_vertex_counts(fig, ax)
        save(fig, "fig1b_risk_map_trajectories")


PAPER_DEPTH = 10  # execution depth shown in the paper figures


def tree_top(max_depth):
    """Upper energy limit (mJ) that fits every vertex up to max_depth, rounded up to 50 mJ."""
    top = max(max(r[3]) for r in PROF[:max_depth + 1]) * 1e3
    return 50 * np.ceil(top * 1.03 / 50)


def draw_tree_panel(ax, max_depth=PAPER_DEPTH, xtick_step=1,
                    xlabel=r"Execution Depth $D$ (Tasks after the Replan)", ms=4.4, xpad=0.5):
    """Candidate-trajectory tree on the energy / execution-depth plane, vertices coloured
    by state, region below the cheapest trajectory shaded. No risk colouring."""
    from matplotlib.collections import LineCollection
    Ds = np.arange(max_depth + 1)
    top = tree_top(max_depth)
    ax.fill_between(Ds, 0, E_MIN_D[:max_depth + 1] * 1e3, step="mid", color=C_BAND, lw=0, zorder=0)
    segs = trajectory_segments(max_depth=max_depth)
    ax.add_collection(LineCollection(segs, colors="0.3", linewidths=0.7, capstyle="round", zorder=2))
    # Vertices coloured by the state the trajectory is in, using the teaser figure's
    # palette. A vertex reached in two states (Analyze and Receive, at odd depths)
    # gets a split marker: left half the lower-index state, right half the other.
    mec, mew = "0.15", 0.45
    for (D, E), states in vertex_states(max_depth=max_depth).items():
        s = sorted(states)
        if len(s) == 1:
            ax.plot(D, E, "o", ms=ms, mfc=STATE_COLORS[s[0]], mec=mec, mew=mew, zorder=3)
        else:
            ax.plot(D, E, "o", ms=ms, fillstyle="left", mfc=STATE_COLORS[s[0]],
                    markerfacecoloralt=STATE_COLORS[s[-1]], mec=mec, mew=mew, zorder=3)
    ax.set_xlim(-0.5, max_depth + xpad)
    ax.set_ylim(0, top)
    ax.set_xticks(range(0, max_depth + 1, xtick_step))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Cumulative Energy $E_D$ (mJ)")
    handles = [Line2D([], [], ls="", marker="o", ms=ms + 0.6, mfc=c, mec=mec, mew=mew, label=n)
               for n, c in zip(STATE_NAMES, STATE_COLORS)]
    handles.append(Line2D([], [], ls="", marker="o", ms=ms + 0.6, fillstyle="left", mfc=STATE_COLORS[1],
                          markerfacecoloralt=STATE_COLORS[3], mec=mec, mew=mew,
                          label="Analyze or Receive"))
    return ax.legend(handles=handles, loc="upper left", framealpha=0.95, handletextpad=0.3,
                     borderpad=0.4, labelspacing=0.3)


def draw_variance_panel(ax, max_depth=PAPER_DEPTH, sigmas=(15e-3, 25e-3, 35e-3),
                        styles=(":", "--", "-"), spread=False, xtick_step=2, xpad=0.5):
    """Variance-trigger replan boundaries over the tree underlay, region below the
    cheapest trajectory shaded. Line styles run from least to most continuous as the
    allowed variance grows. spread=True adds the aggregate view of the same fan-out:
    mean cumulative energy mu_D with +/- sigma_D and +/- 2 sigma_D bands."""
    Ds = np.arange(max_depth + 1)
    ax.fill_between(Ds, 0, E_MIN_D[:max_depth + 1] * 1e3, step="mid", color=C_BAND, lw=0, zorder=0)
    tree_underlay(ax, max_depth=max_depth)
    handles = []
    if spread:
        mu, sd = MU[:max_depth + 1] * 1e3, SD[:max_depth + 1] * 1e3
        ax.fill_between(Ds, mu - 2 * sd, mu + 2 * sd, color=C_MEAN, alpha=0.12, lw=0, zorder=1.4)
        ax.fill_between(Ds, mu - sd, mu + sd, color=C_MEAN, alpha=0.25, lw=0, zorder=1.5)
        ax.plot(Ds, mu, color=C_MEAN, lw=1.6, zorder=1.6)
        handles = [Line2D([], [], color=C_MEAN, lw=1.6, label=r"Mean Energy $\mu_D$"),
                   Patch(color=C_MEAN, alpha=0.25 + 0.12, lw=0, label=r"$\mu_D \pm \sigma_D$"),
                   Patch(color=C_MEAN, alpha=0.12, lw=0, label=r"$\mu_D \pm 2\sigma_D$")]
    for s, ls in zip(sigmas, styles):
        ax.axvline(plan_horizon_variance(s) + 0.5, color=C_VAR, ls=ls, lw=1.6, zorder=2)
    ax.set_xlim(-0.5, max_depth + xpad)
    ax.set_ylim(0, tree_top(max_depth))
    ax.set_xticks(range(0, max_depth + 1, xtick_step))
    ax.set_xlabel(r"Execution Depth $D$")
    trig = [Line2D([], [], color=C_VAR, ls=ls, lw=1.6, label=rf"Replan at $\sigma_D \geq$ {s*1e3:.0f} mJ")
            for s, ls in zip(sigmas, styles)]
    return trig + [tree_handle(), Patch(color=C_BAND, label="Below Cheapest Trajectory")] + handles


def fig1c_trajectory_tree():
    """Paper figure (figure*, full page width), drawn at print size."""
    fig, ax = plt.subplots(figsize=(PAGE_W, 3.9), layout="constrained")
    leg = draw_tree_panel(ax, xpad=0.9)
    probs = {k: fmt_pct(p) for k, p in vertex_probabilities(PAPER_DEPTH).items()}   # every vertex
    crowded = label_vertex_counts(fig, ax, fontsize=7, color="black", halo_color="white", gap=2.0,
                                  dot_r=2.2, avoid=[leg], max_depth=PAPER_DEPTH, texts=probs)
    if crowded:
        print(f"note: {crowded} labels in fig1c kept at default despite crowding")
    save(fig, "fig1c_trajectory_tree")



# ---------------------------------------------------------------- Figure 2
def fig2_cantelli_bracket(slices=(2, 4, 10)):
    budgets = np.linspace(0, B_MAX, 3000)
    fig, axs = plt.subplots(1, len(slices), figsize=(7.0, 2.7), sharey=True, layout="constrained")
    for ax, D, tag in zip(axs, slices, "abc"):
        _, m, v, mg = PROF[D]
        U = np.array([cantelli_bound(m, v, B) for B in budgets])
        L = np.array([cantelli_lower(m, v, B, floor=E_MIN_D[D]) for B in budgets])
        x, r = risk_steps(mg)
        ax.fill_between(budgets * 1e3, np.clip(L, LO, HI), np.clip(U, LO, HI), color=C_BAND, lw=0)
        ax.plot(budgets * 1e3, np.clip(U, LO, HI), color=C_UP)
        ax.plot(budgets * 1e3, np.clip(L, LO, HI), color=C_LO)
        ax.step(x, np.clip(r, LO, HI), where="post", color=C_COMP, lw=1.2)
        ax.axvline(m * 1e3, color=C_MEAN, lw=1.6)
        logit_risk_axis(ax)
        ax.set_xlim(0, B_MAX * 1e3)
        ax.set_title(f"({tag}) $D$ = {D}")
        ax.set_xlabel(r"Energy Budget $B$ (mJ)")
        ax.grid(True)
    axs[0].set_ylabel("Brownout Risk (Logit Scale)")
    handles = [Line2D([], [], color=C_COMP, lw=1.2, label=r"Computed Risk $r_D(B)$"),
               Line2D([], [], color=C_UP, label=r"Cantelli Upper Bound $U_D(B)$"),
               Line2D([], [], color=C_LO, label=r"Lower Bound $L_D(B)$: 1 below Cheapest Trajectory, Else Cantelli"),
               Patch(color=C_BAND, label="Not Resolved by the Bounds"),
               Line2D([], [], color=C_MEAN, lw=1.6, label=r"Mean Demand $\mu_D$")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    save(fig, "fig2_cantelli_bracket")


# ---------------------------------------------------------------- Figure 3
def fig3_budget_vs_horizon(p_grid=(0.01, 0.05, 0.25, 0.75, 0.95, 0.99)):
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.9), sharex=True, sharey=True, layout="constrained")
    for ax, p, tag in zip(axs.flat, p_grid, "abcdef"):
        b_plus = (MU + SD * np.sqrt((1 - p) / p)) * 1e3
        b_minus = np.maximum(MU - SD * np.sqrt(p / (1 - p)), E_MIN_D[:len(MU)]) * 1e3
        b_p = np.array([exact_required_budget(r[3], p) for r in PROF20]) * 1e3
        ax.fill_between(DEPTHS, b_minus, b_plus, color=C_BAND, lw=0)
        ax.plot(DEPTHS, b_plus, color=C_UP)
        ax.plot(DEPTHS, b_minus, color=C_LO, lw=2.6, solid_capstyle="butt")  # wide: shows where it meets B_p
        ax.plot(DEPTHS, MU * 1e3, color=C_MEAN, lw=1.4)
        ax.plot(DEPTHS, b_p, color=C_COMP, lw=1.1, marker="o", ms=2.4)
        ax.set_title(f"({tag}) $p$ = {p:g}")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, B_MAX * 1e3)
        ax.set_xticks(range(0, 21, 4))
        ax.grid(True)
    for ax in axs[:, 0]:
        ax.set_ylabel(r"Energy Budget $B$ (mJ)")
    for ax in axs[-1, :]:
        ax.set_xlabel(r"Planning Horizon $D$")
    for ax in axs[:, -1]:
        ax.secondary_yaxis("right", functions=to_mF).set_ylabel("Capacitance at 3.2 V (mF)")
    handles = [Line2D([], [], color=C_COMP, lw=1.1, marker="o", ms=2.4, label=r"Quantile Budget $B_p(D)$ (Computed)"),
               Line2D([], [], color=C_UP, label=r"Cantelli Safe Budget $B^{+}_{p}(D)$"),
               Line2D([], [], color=C_LO, lw=2.6, label=r"Failure Budget $B^{-}_{p}(D)$"),
               Line2D([], [], color=C_MEAN, lw=1.4, label=r"Mean Demand $\mu_D$")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    save(fig, "fig3_budget_vs_horizon")


# ---------------------------------------------------------------- Figure 4
def fig4_forward_progress(ps=(0.01, 0.05, 0.25)):
    """Tasks completed against budget, over the transposed trajectory tree.
    Black: computed E[N(B)]. Blue: with probability >= 1 - p at least N+_p(B)
    tasks complete, where N+_p(B) = (risk-trigger plan length) + 1 (Cantelli)."""
    styles = [":", "-", "--"]
    budgets = np.linspace(0, B_MAX, 2400)
    xs, ys = progress_steps(PROF)
    keep = xs <= B_MAX

    fig, ax = plt.subplots(figsize=(4.6, 3.6), layout="constrained")
    tree_underlay(ax, max_depth=30, transpose=True)
    for p, ls in zip(ps, styles):
        n_plus = [plan_horizon_risk(p, B) + 1 if B >= E_MIN_D[0] else 0 for B in budgets]
        ax.plot(budgets * 1e3, n_plus, color=C_UP, ls=ls, lw=1.5, drawstyle="steps-post", zorder=2)
    ax.step(np.r_[0, xs[keep] * 1e3, B_MAX * 1e3], np.r_[0, ys[keep], ys[keep][-1]],
            where="post", color=C_COMP, lw=1.3, zorder=3)
    ax.set_xlim(0, B_MAX * 1e3)
    ax.set_ylim(0, ys[keep][-1] * 1.1)
    ax.set_xlabel(r"Energy Budget $B$ (mJ)")
    ax.set_ylabel(r"Tasks Completed before Brownout, $N$")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(True, zorder=0)
    handles = ([Line2D([], [], color=C_COMP, lw=1.3, label=r"Computed $E[N(B)]$")]
               + [Line2D([], [], color=C_UP, ls=ls, lw=1.5,
                         label=rf"$N \geq N^{{+}}_{{p}}(B)$ with Prob. $\geq 1-p$, $p$ = {p:g}")
                  for p, ls in zip(ps, styles)]
               + [tree_handle()])
    ax.legend(handles=handles, loc="upper left", framealpha=0.95)
    save(fig, "fig4_forward_progress")


# ---------------------------------------------------------------- Figure 5
C_VAR = "#E69F00"  # variance trigger family (Okabe-Ito orange); risk trigger family uses C_UP


def plan_horizon_variance(sigma_th):
    """Variance trigger: replan at the first depth with sigma_D >= sigma_th.
    Returns the plan length H (last depth below threshold); independent of B."""
    sd = np.sqrt([r[2] for r in PROF])
    return next(D for D in range(len(sd)) if sd[D] >= sigma_th) - 1


def plan_horizon_risk(p, B):
    """Risk trigger: replan at the first depth with U_D(B) >= p (Cantelli).
    Returns the plan length H (last depth below threshold)."""
    return next((D for D, m, v, _ in PROF if cantelli_bound(m, v, B) >= p), len(PROF)) - 1


def fig5_trigger_comparison_cantelli(sigmas=(15e-3, 25e-3, 35e-3), ps=(0.01, 0.05, 0.25)):
    """Replan regions of both trigger families over the candidate-trajectory tree."""
    styles = [":", "-", "--"]
    fig, ax = plt.subplots(figsize=(4.6, 3.9), layout="constrained")
    D20 = np.arange(21)
    ax.fill_between(D20, 0, E_MIN_D[:21] * 1e3, step="mid", color=C_BAND, lw=0, zorder=0)
    tree_underlay(ax, max_depth=20)
    for s, ls in zip(sigmas, styles):
        ax.axvline(plan_horizon_variance(s) + 0.5, color=C_VAR, ls=ls, lw=1.6, zorder=2)
    for p, ls in zip(ps, styles):
        b_plus = (MU + SD * np.sqrt((1 - p) / p)) * 1e3
        ax.step(D20, b_plus, where="mid", color=C_UP, ls=ls, lw=1.6, zorder=2)
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(0, B_MAX * 1e3)
    ax.set_xticks(range(0, 21, 2))
    ax.set_xlabel(r"Planning Horizon $D$ (Tasks after the Replan)")
    ax.set_ylabel(r"Energy Budget $B$ (mJ)")
    # One family per legend column (matplotlib fills columns top to bottom).
    handles = ([Line2D([], [], color=C_VAR, ls=ls, lw=1.6, label=rf"Variance Trigger, $\sigma_D \geq$ {s*1e3:.0f} mJ")
                for s, ls in zip(sigmas, styles)]
               + [tree_handle()]
               + [Line2D([], [], color=C_UP, ls=ls, lw=1.6, label=rf"Risk Trigger, $U_D(B) \geq$ {p:g}")
                  for p, ls in zip(ps, styles)]
               + [Patch(color=C_BAND, label="Below Cheapest Trajectory")])
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    save(fig, "fig5_trigger_comparison_cantelli")


def fig5b_variance_trigger():
    """Paper figure (single column), drawn at print size."""
    fig, ax = plt.subplots(figsize=(COL_W, 3.15), layout="constrained")
    handles = draw_variance_panel(ax)
    ax.set_ylabel(r"Energy (mJ)")
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False,
               handlelength=1.8, columnspacing=1.0)
    save(fig, "fig5b_variance_trigger")


def fig5c_variance_trigger_spread():
    """Paper figure (single column): fig5b plus the aggregate view of the fan-out,
    mean cumulative energy with sigma bands. Drawn at print size."""
    fig, ax = plt.subplots(figsize=(COL_W, 3.55), layout="constrained")
    h = draw_variance_panel(ax, spread=True)   # r15, r25, r35, tree, below, mean, 1sd, 2sd
    handles = h[0:4] + h[5:8] + [h[4]]
    ax.set_ylabel(r"Energy (mJ)")
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False,
               handlelength=1.8, columnspacing=1.0)
    save(fig, "fig5c_variance_trigger_spread")


def fig6_tree_and_trigger(height=3.6, count_fontsize=7.0, ratio=(1.0, 1.0), gap=2.0, ms=4.4,
                          label_cutoff=LABEL_CUTOFF):
    """Paper figure (figure*, full page width): (a) the trajectory tree with counts,
    (b) variance-trigger boundaries over the same tree with the mean / sigma-band layer.
    Equal panel widths and a shared energy axis. Drawn at print size."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(PAGE_W, height), sharey=True, layout="constrained",
                                   gridspec_kw={"width_ratios": list(ratio)})
    legL = draw_tree_panel(axL, ms=ms, xpad=0.9)
    h = draw_variance_panel(axR, spread=True, xtick_step=1, xpad=0.9)   # r15, r25, r35, tree, below, mean, 1sd, 2sd
    handles = [h[0], h[5], h[1], h[6], h[2], h[7], h[3], h[4]]  # column-major fill gives two clean rows
    axR.tick_params(labelleft=False)
    axL.set_title("(a) Candidate Trajectories")
    axR.set_title("(b) Variance-Triggered Replanning")
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False,
               handlelength=1.8, columnspacing=1.2)
    # Vertices with pass-through probability >= label_cutoff are labelled, in percent.
    probs = {k: fmt_pct(p) for k, p in vertex_probabilities(PAPER_DEPTH).items() if p >= label_cutoff}
    crowded = label_vertex_counts(fig, axL, fontsize=count_fontsize, color="black", halo_color="white",
                                  gap=gap, dot_r=ms / 2, avoid=[legL], max_depth=PAPER_DEPTH, texts=probs)
    if crowded:
        print(f"note: {crowded} labels in fig6 kept at default despite crowding")
    save(fig, "fig6_tree_and_trigger")
    return crowded


def draw_trigger_panel(ax, sigmas=(15e-3, 25e-3, 35e-3), styles=(":", "--", "-"),
                       eps=(0.0, 0.01, 0.05, 0.1, 0.2), max_depth=PAPER_DEPTH, xtick_step=2):
    """Replanning depth chosen by the variance trigger and by the exact risk trigger of the
    forecast, over the candidate trajectories. Returns (variance+context handles, risk handles).

    Risk trigger: fires at depth D when r_D(B) > eps, with r_D the exact overdraw
    probability from the leaf distribution (no Cantelli bound). Its boundary in column D
    is B_eps(D), the smallest budget with r_D(B) <= eps; budgets below it fire. eps = 0
    fires whenever any reachable trajectory overdraws."""
    Ds = np.arange(max_depth + 1)
    ax.fill_between(Ds, 0, E_MIN_D[:max_depth + 1] * 1e3, step="mid", color=C_BAND, lw=0, zorder=0)
    tree_underlay(ax, max_depth=max_depth)
    # Risk boundaries: darker for smaller eps. Larger eps are drawn first, wider, so where
    # two boundaries coincide the lighter one shows as an outline around the darker one.
    shades = plt.cm.Blues(np.linspace(0.97, 0.45, len(eps)))
    widths = np.linspace(1.3, 3.1, len(eps))
    for x, c, w in sorted(zip(eps, shades, widths), key=lambda t: -t[0]):
        b_eps = [exact_required_budget(PROF[D][3], x) * 1e3 for D in Ds]
        ax.step(Ds, b_eps, where="mid", color=c, lw=w, zorder=2, solid_capstyle="butt")
    for s, ls in zip(sigmas, styles):
        ax.axvline(plan_horizon_variance(s) + 0.5, color=C_VAR, ls=ls, lw=1.6, zorder=3)
    ax.set_xlim(-0.5, max_depth + 0.5)
    ax.set_ylim(0, tree_top(max_depth))
    ax.set_xticks(range(0, max_depth + 1, xtick_step))
    ax.set_xlabel(r"Execution Depth $D$")
    ax.set_ylabel(r"Energy Budget $B$ (mJ)")
    var_handles = [Line2D([], [], color=C_VAR, ls=ls, lw=1.6, label=rf"Replan at $\sigma_D \geq$ {s*1e3:.0f} mJ")
                   for s, ls in zip(sigmas, styles)]
    context = [tree_handle(), Patch(color=C_BAND, label="Below Cheapest Trajectory")]
    risk_handles = [Line2D([], [], color=c, lw=1.8, label=rf"Replan at $r_D(B) > {x:g}$")
                    for x, c in zip(eps, shades)]
    return var_handles + context, risk_handles


def fig5_trigger_comparison():
    """Paper figure (single column), drawn at print size."""
    fig, ax = plt.subplots(figsize=(COL_W, 3.95), layout="constrained")
    left, right = draw_trigger_panel(ax)
    # Column-major fill: left column variance trigger + context, right column risk trigger.
    fig.legend(handles=left + right, loc="outside lower center", ncol=2, frameon=False,
               handlelength=1.8, columnspacing=0.9, fontsize=7)
    save(fig, "fig5_trigger_comparison")


_FIG7_PANEL_B = {}   # plot-area size of Figure 7's panel (b), recorded when Figure 7 is built


def fig7_tree_and_triggers(height=4.4, label_fontsize=7.0, ratio=(2.0, 1.0), gap=2.0, ms=4.4,
                           xpad=1.5):
    """Paper figure (figure*, full page width): (a) the trajectory tree with every vertex
    labelled by its pass-through probability (percent), 2/3 of the width; (b) the exact
    risk trigger and the variance trigger over the same trajectories, 1/3 of the width.
    Shared energy scale. Drawn at print size."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(PAGE_W, height), sharey=True, layout="constrained",
                                   gridspec_kw={"width_ratios": list(ratio)})
    legL = draw_tree_panel(axL, ms=ms, xpad=xpad)
    left, right = draw_trigger_panel(axR)
    # No duplicated axis text: panel (b) reads energy off panel (a)'s axis, and a single
    # depth label spans both panels (each keeps its own tick numbers, the widths differ).
    axR.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    axR.set_ylabel("")
    axL.set_ylabel(r"Cumulative Energy $E_D$ or Budget $B$ (mJ)")
    axL.set_xlabel(r"Execution Depth $D$ (Tasks after the Replan)")   # reserves room above the legend
    axR.set_xlabel("")
    axL.set_title("(a) Candidate Trajectories")
    axR.set_title("(b) Replanning Triggers")
    # Two rows: variance trigger + context, then the risk thresholds (column-major fill).
    rows = [left, right]
    handles = [h for pair in zip(*rows) for h in pair]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False,
               handlelength=1.8, columnspacing=1.1)
    probs = {k: fmt_pct(p) for k, p in vertex_probabilities(PAPER_DEPTH).items()}   # every vertex
    crowded = label_vertex_counts(fig, axL, fontsize=label_fontsize, color="black", halo_color="white",
                                  gap=gap, dot_r=ms / 2, avoid=[legL], max_depth=PAPER_DEPTH, texts=probs,
                                  leaders=True, constrained_first=True)
    if crowded:
        print(f"note: {crowded} labels in fig7 kept at default despite crowding")
    # Freeze the layout, then centre the single depth label under both panels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    posL, posR = axL.get_position(), axR.get_position()
    fig.set_layout_engine("none")
    top = axL.xaxis.label.get_window_extent(renderer).y1
    y_axes = axL.transAxes.inverted().transform((0, top))[1]
    x_axes = ((posL.x0 + posR.x1) / 2 - posL.x0) / posL.width
    axL.xaxis.set_label_coords(x_axes, y_axes)
    _FIG7_PANEL_B["size"] = (posR.width * PAGE_W, posR.height * height)   # inches
    save(fig, "fig7_tree_and_triggers")
    return crowded


def fig5_trigger_panel():
    """Standalone copy of Figure 7's panel (b): identical plot-area size and styling, no
    legend or panel title. Being standalone, it carries its own axis ticks and labels."""
    if "size" not in _FIG7_PANEL_B:
        fig7_tree_and_triggers()
    w, h = _FIG7_PANEL_B["size"]
    left, bottom, right, top = 0.52, 0.42, 0.06, 0.06              # margins for tick labels and axis labels
    W, H = w + left + right, h + bottom + top
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bottom / H, w / W, h / H])          # exact plot-area size, no layout engine
    draw_trigger_panel(ax)
    save(fig, "fig5_trigger_panel")


def fig1c_trajectory_tree_column(height=4.6, label_fontsize=7.0, xpad=1.5, ms=4.4, gap=2.0):
    """Single-column version of Figure 1c (\\columnwidth), drawn at print size so every
    font prints at the same size as the other figures. All vertex probabilities are
    labelled; the state legend sits below the plot to leave the tree its full width."""
    fig, ax = plt.subplots(figsize=(COL_W, height), layout="constrained")
    leg = draw_tree_panel(ax, ms=ms, xpad=xpad, xtick_step=2,
                          xlabel=r"Execution Depth $D$ (Tasks after the Replan)")
    handles = leg.legend_handles
    leg.remove()
    fig.legend(handles=handles, labels=[t.get_text() for t in leg.get_texts()], loc="outside lower center",
               ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.9)
    probs = {k: fmt_pct(p) for k, p in vertex_probabilities(PAPER_DEPTH).items()}   # every vertex
    crowded = label_vertex_counts(fig, ax, fontsize=label_fontsize, color="black", halo_color="white",
                                  gap=gap, dot_r=ms / 2, max_depth=PAPER_DEPTH, texts=probs,
                                  leaders=True, constrained_first=True)
    if crowded:
        print(f"note: {crowded} labels in fig1c (column) kept at default despite crowding")
    save(fig, "fig1c_trajectory_tree_column")
    return crowded


# ---------------------------------------------------------------- Figure: cars dose-response
def load_sweep(summary):
    """(A, B, tph, zero) from a dose-response summary.csv. tph[(wake, arm)] is the arm's mean
    tasks per hour pooled over nodes and seeds (every arm runs the same node/seed set, so ratios
    of these are summary.csv's vs_leacs); zero[wake] the number of runs per cell in which LEACS
    completed no task; both indexed [b, a]. Missing cells stay NaN."""
    with open(summary, newline="") as f:
        rows = list(csv.DictReader(f))
    A = sorted({float(r["a"]) for r in rows})
    B = sorted({float(r["b"]) for r in rows})
    tph, zero = {}, {}
    for r in rows:
        i, j = B.index(float(r["b"])), A.index(float(r["a"]))
        tph.setdefault((r["wake"], r["arm"]), np.full((len(B), len(A)), np.nan))[i, j] = float(r["tasks_per_h"])
        if r["arm"] == "leacs":
            zero.setdefault(r["wake"], np.full((len(B), len(A)), np.nan))[i, j] = int(r["leacs_zero_runs"])
    return A, B, tph, zero


def cars_heatmaps(summary=SWEEP_CARS / "summary.csv", energy=False, row=False, baseline="leacs", height=None):
    """Thimble throughput relative to a baseline over the (a, b) plane on cars, pooled over
    held-out traces and seeds, for the Vmax wake rule (plus energy if energy=True).
    baseline="leacs": every Thimble arm against LEACS. baseline="thimble_noreplan": the
    replanning arms (depth, risk) against Thimble without replanning; saved with "_replan".
    row=False: one \\columnwidth column, rows = arm, columns = wake rule; with the LEACS
    baseline, hatched cells: LEACS completed no task on some runs. row=True: \\textwidth, rows =
    wake rule, columns = arm, no hatching (the caption carries that note); saved with "_row".
    The diverging scale is centred on parity and sized to the plotted frames; the colorbar
    spans only their range (and 0)."""
    wakes = SWEEP_WAKES if energy else tuple(w for w in SWEEP_WAKES if w[0] != "energy")
    arms = tuple(a for a in SWEEP_ARMS if a[0] != baseline)
    base_label = "LEACS" if baseline == "leacs" else dict(SWEEP_ARMS)[baseline]
    if row:
        panels = [[(arm, wake) for arm in arms] for wake in wakes]
    else:
        panels = [[(arm, wake) for wake in wakes] for arm in arms]
    shape = (len(panels), len(panels[0]))
    if height is None:
        height = {(False, (3, 1)): 5.0, (False, (3, 2)): 5.6, (True, (1, 3)): 2.4, (True, (2, 3)): 4.2,
                  (False, (2, 1)): 4.0, (False, (2, 2)): 4.0, (True, (1, 2)): 2.4, (True, (2, 2)): 4.2}[(row, shape)]
    hatch = baseline == "leacs" and not row
    A, B, tph, zero = load_sweep(summary)
    rel = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for w, _ in wakes:
            for arm, _ in arms:
                if (w, arm) in tph and (w, baseline) in tph:
                    r = 100 * (tph[(w, arm)] / tph[(w, baseline)] - 1)
                    rel[(w, arm)] = np.where(np.isfinite(r), r, np.nan)   # zero baseline mean: no ratio
    vals = np.concatenate([g.ravel() for g in rel.values()])
    lo = 10 * math.floor(min(np.nanmin(vals), 0) / 10)
    hi = 10 * math.ceil(max(np.nanmax(vals), 0) / 10)
    cmap = plt.get_cmap("RdBu").with_extremes(bad="0.85")
    norm = Normalize(-max(-lo, hi), max(-lo, hi))
    width = (PAGE_W if row else COL_W) - 2 * plt.rcParams["savefig.pad_inches"]   # save() pads each side
    fig, axs = plt.subplots(len(panels), len(panels[0]), figsize=(width, height), sharex=True, sharey=True,
                            squeeze=False, layout="constrained")
    for ax, letter, ((arm, arm_label), (wake, wake_label)) in zip(axs.flat, "abcdef", sum(panels, [])):
        grid = rel.get((wake, arm), np.full((len(B), len(A)), np.nan))
        im = ax.imshow(grid, cmap=cmap, norm=norm, origin="lower", aspect="equal", interpolation="nearest")
        for (i, j), v in np.ndenumerate(grid):
            if np.isfinite(v):
                s = "0" if round(v) == 0 else f"{v:+.0f}".replace("-", "\u2212")
                r, g, b_, _ = cmap(norm(v))
                ax.text(j, i, s, ha="center", va="center", fontsize=6.5, zorder=3,
                        color="white" if 0.2126 * r + 0.7152 * g + 0.0722 * b_ < 0.45 else "black")
        zw = zero.get(wake)
        if hatch and zw is not None:
            for i, j in zip(*np.nonzero(zw > 0)):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=0, hatch="////",
                                       hatchcolor="0.35", hatch_linewidth=0.4, zorder=2))
        ax.set_title(f"({letter}) {arm_label}, {wake_label}", fontsize=7.5)
        ax.set_xticks([0, 2, 4], [f"{A[k]:g}" for k in (0, 2, 4)])
        ax.set_xticks([1, 3], minor=True)
        ax.set_yticks([0, 2, 4], [f"{B[k]:g}" for k in (0, 2, 4)])
        ax.set_yticks([1, 3], minor=True)
        ax.tick_params(labelsize=6.5)
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$b$ = P(R$\rightarrow$T)")
    for ax in axs[-1]:
        ax.set_xlabel(r"$a$ = P(A$\rightarrow$M)")
    cb = fig.colorbar(im, ax=axs, location="bottom" if not row and len(wakes) > 1 else "right", aspect=35,
                      shrink=0.9, boundaries=np.linspace(lo, hi, 257),
                      ticks=np.arange(lo, hi + 1, 10 if hi - lo <= 60 else 30))
    sep = "\n" if axs.shape[0] == 1 or baseline != "leacs" else " "   # else the label outruns the colorbar
    num = "Thimble" if baseline == "leacs" else "Replanning"
    cb.set_label(f"({num} \u2212 {base_label}) / {base_label},{sep}tasks per hour (%)", fontsize=7.5)
    if hatch:
        fig.legend(handles=[Patch(facecolor="white", edgecolor="0.35", hatch="////", lw=0.4,
                                  label="LEACS completed no task on some runs")],
                   loc="outside upper center", frameon=False, fontsize=7)
    save(fig, "cars_heatmaps" + ("_replan" if baseline != "leacs" else "") + ("_row" if row else ""))


if __name__ == "__main__":
    fig1_risk_map()
    fig1b_risk_map_trajectories()
    fig1c_trajectory_tree()
    fig2_cantelli_bracket()
    fig3_budget_vs_horizon()
    fig4_forward_progress()
    fig5_trigger_comparison()
    fig5_trigger_comparison_cantelli()
    fig5b_variance_trigger()
    fig5c_variance_trigger_spread()
    fig6_tree_and_trigger()
    fig7_tree_and_triggers()
    fig5_trigger_panel()
    fig1c_trajectory_tree_column()
    cars_heatmaps()
    cars_heatmaps(row=True)
    cars_heatmaps(baseline="thimble_noreplan")
