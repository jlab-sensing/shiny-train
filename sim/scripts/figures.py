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

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sweep_trigger import (
    P,
    cantelli_bound,
    cantelli_lower,
    depth_profile,
    e,
    exact_required_budget,
    exact_tail,
    min_trajectory_energy,
    progress_steps,
)

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- style
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
)
# Okabe-Ito palette (colour-blind safe), one meaning per colour across all figures.
C_UP, C_LO, C_MEAN, C_BAND, C_COMP = "#0072B2", "#D55E00", "#009E73", "0.87", "black"

V, V_MIN = 3.2, 0.1
B_MAX = 600e-3
LO, HI = 1e-3, 1 - 1e-3
RISK_TICKS = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]

to_mF = (
    lambda B: 2 * B / (V**2 - V_MIN**2),
    lambda C: C * (V**2 - V_MIN**2) / 2,
)  # mJ <-> mF


def matched_rc(fig_width, reference_width=4.6):
    """rcParams scaled by fig_width / reference_width, so a figure made wider than the
    reference looks the same as the others when all are shown at a common width."""
    k = fig_width / reference_width
    keys = [
        "font.size",
        "axes.labelsize",
        "axes.titlesize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "axes.linewidth",
        "xtick.major.size",
        "ytick.major.size",
        "xtick.major.width",
        "ytick.major.width",
        "xtick.major.pad",
        "ytick.major.pad",
    ]
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
    return np.r_[0, x[keep], B_MAX] * 1e3, np.r_[
        1, tail[keep], tail[keep][-1] if keep.any() else 1
    ]


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
E_MIN_D = np.array(
    min_trajectory_energy(P, e, start=0, max_depth=40)
)  # cheapest trajectory


# ---------------------------------------------------------------- Figure 1
def risk_plane(figsize=(4.6, 3.5)):
    """Shared planning-budget plane: computed risk r_D(B) on a logit colour scale."""
    p_levels = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    cmap = ListedColormap(plt.cm.plasma(np.linspace(0, 0.9, 256)))
    norm = Normalize(logit(LO), logit(HI))
    budgets = np.linspace(0.5e-3, B_MAX, 1200)
    risk = np.array([[exact_tail(r[3], B) for r in PROF20] for B in budgets])

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    mesh = ax.pcolormesh(
        DEPTHS,
        budgets * 1e3,
        logit(risk),
        shading="nearest",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
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
        cont.append(
            [sum(cont[-1][j] for j in range(n) if P[s][j] > 0) for s in range(n)]
        )
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
            layer = {
                (j, round(E + e[j], ndigits))
                for s, E in layer
                for j in range(len(e))
                if P[s][j] > 0
            }
        for s, E in layer:
            out.setdefault((D, round(E * 1e3, 6)), set()).add(s)
    return out


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
    ax.add_collection(
        LineCollection(segs, colors=C_TREE, linewidths=0.45, alpha=0.6, zorder=1)
    )
    verts = np.array(sorted({pt for seg in segs for pt in seg}))
    ax.scatter(
        verts[:, 0],
        verts[:, 1],
        s=2.5,
        color=C_TREE,
        alpha=0.8,
        edgecolors="none",
        zorder=1,
        clip_on=True,
    )


def tree_handle():
    return Line2D(
        [],
        [],
        color=C_TREE,
        lw=0.6,
        marker="o",
        ms=1.8,
        alpha=0.8,
        label="Candidate Trajectories",
    )


def fig1_risk_map():
    fig, ax = risk_plane()
    # Risk is piecewise constant in D, so each level's boundary is drawn per column
    # (the quantile budget B_p(D)) rather than as an interpolated contour.
    halo = [pe.withStroke(linewidth=2.4, foreground="black")]
    levels = [(0.01, "0.01"), (0.05, "0.05"), (0.25, "0.25"), (0.5, "≥0.5")]
    bps = {
        p: np.array([exact_required_budget(r[3], p) for r in PROF20]) * 1e3
        for p, _ in levels
    }
    for p, _ in levels:
        ax.step(DEPTHS, bps[p], where="mid", color="white", lw=1.0, path_effects=halo)
    ax.plot(DEPTHS, MU * 1e3, color=C_MEAN, lw=1.8)
    handles = [
        Line2D(
            [],
            [],
            color="white",
            lw=1.0,
            path_effects=halo,
            label=r"Risk Boundary $B_p(D)$, Labelled by $p$",
        ),
        Line2D([], [], color=C_MEAN, lw=1.8, label=r"Mean Demand $\mu_D$"),
    ]
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
                if (
                    c + 1 < len(bp)
                    and x0 < c + 0.5 < x1
                    and min(bp[c], bp[c + 1]) <= y1
                    and max(bp[c], bp[c + 1]) >= y0
                ):
                    return True
        ys = np.interp(np.linspace(x0, x1, 60), DEPTHS, mean_mJ)
        return bool(((ys >= y0) & (ys <= y1)).any())

    def overlaps(box, others):
        return any(
            box[0] < o.x1 and box[1] > o.x0 and box[2] < o.y1 and box[3] > o.y0
            for o in others
        )

    def data_per_point():
        a, b = ax.transData.transform([(0, 0), (1, 1)])
        return 72 / fig.dpi / (b[0] - a[0]) * fig.dpi / 72, 1 / (
            (b[1] - a[1]) * 72 / fig.dpi
        )

    for p, lab in levels:
        bp = bps[p]
        others = [bps[q] for q, _ in levels if q != p]
        # target: a step not shared with another boundary, with room around it, near the middle
        cands = [
            (min(min(abs(o[D] - bp[D]) for o in others), 60.0), -abs(D - 10), D)
            for D in range(2, 18)
            if bp[D] < 0.93 * B_MAX * 1e3
            and all(abs(o[D] - bp[D]) > 1e-6 for o in others)
        ]
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
                (x0, y0), (x1, y1) = inv.transform(
                    [(px - w / 2, py - h / 2), (px + w / 2, py + h / 2)]
                )
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
        ann = ax.annotate(
            lab,
            xy=target,
            xytext=(cx, cy),
            color="white",
            fontsize=7,
            ha="center",
            va="center",
            path_effects=halo,
            arrowprops=dict(
                arrowstyle="-", color="white", lw=0.7, shrinkA=1.5, shrinkB=0
            ),
        )
        ann.arrow_patch.set_path_effects(
            [pe.withStroke(linewidth=1.8, foreground="black")]
        )
        ax.plot(
            *target, "o", ms=2.6, color="white", mec="black", mew=0.6, zorder=5
        )  # leader endpoint
        from matplotlib.transforms import Bbox

        placed.append(Bbox.from_extents(box[0], box[2], box[1], box[3]))
    save(fig, "fig1_risk_map")


PAGE_W, COL_W = 7.0, 3.33  # ACM sigconf \textwidth and \columnwidth, in inches


def label_vertex_counts(
    fig,
    ax,
    fontsize=7.5,
    color="white",
    halo_color="black",
    gap=3.0,
    dot_r=1.5,
    avoid=(),
    max_depth=20,
):
    """Label each vertex with the number of full-length trajectories passing through it.

    Each label prefers the spot close, below and right of its vertex. A spot is rejected
    if it comes within 3 pt of another label or of another vertex's dot, or if it lies
    nearer another vertex than its own (so a moved label cannot be misattributed).
    Labels are placed right to left, so in a crowded pair the left one moves.
    gap: minimum clearance in points between a label and other labels or dots.
    dot_r: vertex marker radius in points; offsets grow with it so labels clear the dot.
    avoid: artists (e.g. a legend) that labels must not overlap."""
    halo = [pe.withStroke(linewidth=2.0, foreground=halo_color)]
    fig.canvas.draw()  # fix the layout before measuring text
    renderer = fig.canvas.get_renderer()
    pt = fig.dpi / 72
    top = ax.get_ylim()[1]
    counts = {
        k: n for k, n in vertex_counts(max_depth=max_depth).items() if k[1] <= top
    }
    vpix = {k: ax.transData.transform(k) for k in counts}
    dx, dy, side = dot_r + 0.1, dot_r - 0.5, dot_r + 1.5
    placements = [
        ((dx, -dy), "left", "top"),
        ((-dx, dy), "right", "bottom"),
        ((-dx, -dy), "right", "top"),
        ((dx, dy), "left", "bottom"),
        ((side, 0.0), "left", "center"),
        ((-side, 0.0), "right", "center"),
        ((0.0, -(dot_r + 1.1)), "center", "top"),
        ((0.0, dot_r + 1.1), "center", "bottom"),
    ]

    def box_dist(bb, p):
        dx = max(bb.x0 - p[0], 0, p[0] - bb.x1)
        dy = max(bb.y0 - p[1], 0, p[1] - bb.y1)
        return np.hypot(dx, dy)

    def put(key, off, ha, va):
        return ax.annotate(
            str(counts[key]),
            key,
            xytext=off,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color,
            path_effects=halo,
            zorder=4,
            annotation_clip=True,
        )

    frame = ax.get_window_extent(renderer)
    inside = lambda bb: (
        bb.x0 >= frame.x0 + pt
        and bb.x1 <= frame.x1 - pt
        and bb.y0 >= frame.y0 + pt
        and bb.y1 <= frame.y1 - pt
    )
    placed, crowded = [a.get_window_extent(renderer) for a in avoid], 0
    for key in sorted(counts, key=lambda k: (-k[0], -k[1])):
        others = [p for k, p in vpix.items() if k != key]
        chosen = None
        for off, ha, va in placements:
            t = put(key, off, ha, va)
            bb = t.get_window_extent(renderer)
            if (
                inside(bb)
                and not any(bb.padded(gap * pt).overlaps(o) for o in placed)
                and all(box_dist(bb, p) > (gap + dot_r - 1.0) * pt for p in others)
                and box_dist(bb, vpix[key])
                < min((box_dist(bb, p) for p in others), default=np.inf)
            ):
                chosen = bb
                break
            t.remove()
        if chosen is None:  # nothing clear: fall back to the default spot
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

        fig, ax = risk_plane(
            figsize=(7.0, 6.2)
        )  # full width and tall: room for every label
        segs = trajectory_segments()
        ax.add_collection(
            LineCollection(
                segs, colors="white", linewidths=0.6, alpha=0.85, capstyle="round"
            )
        )
        # Distinct vertices (D, E_D): the atoms of the leaf distribution at each horizon.
        verts = np.array(sorted({pt for seg in segs for pt in seg}))
        ax.scatter(
            verts[:, 0],
            verts[:, 1],
            s=5,
            color="white",
            edgecolors="none",
            zorder=3,
            clip_on=True,
        )
        label_vertex_counts(fig, ax)
        save(fig, "fig1b_risk_map_trajectories")


PAPER_DEPTH = 10  # execution depth shown in the paper figures


def tree_top(max_depth):
    """Upper energy limit (mJ) that fits every vertex up to max_depth, rounded up to 50 mJ."""
    top = max(max(r[3]) for r in PROF[: max_depth + 1]) * 1e3
    return 50 * np.ceil(top * 1.03 / 50)


def draw_tree_panel(
    ax,
    max_depth=PAPER_DEPTH,
    xtick_step=1,
    xlabel=r"Execution Depth $D$ (Tasks after the Replan)",
    ms=4.4,
):
    """Candidate-trajectory tree on the energy / execution-depth plane, vertices coloured
    by state, region below the cheapest trajectory shaded. No risk colouring."""
    from matplotlib.collections import LineCollection

    Ds = np.arange(max_depth + 1)
    top = tree_top(max_depth)
    ax.fill_between(
        Ds, 0, E_MIN_D[: max_depth + 1] * 1e3, step="mid", color=C_BAND, lw=0, zorder=0
    )
    segs = trajectory_segments(max_depth=max_depth)
    ax.add_collection(
        LineCollection(segs, colors="0.3", linewidths=0.7, capstyle="round", zorder=2)
    )
    # Vertices coloured by the state the trajectory is in, using the teaser figure's
    # palette. A vertex reached in two states (Analyze and Receive, at odd depths)
    # gets a split marker: left half the lower-index state, right half the other.
    mec, mew = "0.15", 0.45
    for (D, E), states in vertex_states(max_depth=max_depth).items():
        s = sorted(states)
        if len(s) == 1:
            ax.plot(
                D, E, "o", ms=ms, mfc=STATE_COLORS[s[0]], mec=mec, mew=mew, zorder=3
            )
        else:
            ax.plot(
                D,
                E,
                "o",
                ms=ms,
                fillstyle="left",
                mfc=STATE_COLORS[s[0]],
                markerfacecoloralt=STATE_COLORS[s[-1]],
                mec=mec,
                mew=mew,
                zorder=3,
            )
    ax.set_xlim(-0.5, max_depth + 0.5)
    ax.set_ylim(0, top)
    ax.set_xticks(range(0, max_depth + 1, xtick_step))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Cumulative Energy $E_D$ (mJ)")
    handles = [
        Line2D([], [], ls="", marker="o", ms=ms + 0.6, mfc=c, mec=mec, mew=mew, label=n)
        for n, c in zip(STATE_NAMES, STATE_COLORS)
    ]
    handles.append(
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=ms + 0.6,
            fillstyle="left",
            mfc=STATE_COLORS[1],
            markerfacecoloralt=STATE_COLORS[3],
            mec=mec,
            mew=mew,
            label="Analyze or Receive",
        )
    )
    return ax.legend(
        handles=handles,
        loc="upper left",
        framealpha=0.95,
        handletextpad=0.3,
        borderpad=0.4,
        labelspacing=0.3,
    )


def draw_variance_panel(
    ax,
    max_depth=PAPER_DEPTH,
    sigmas=(15e-3, 25e-3, 35e-3),
    styles=(":", "--", "-"),
    spread=False,
    xtick_step=2,
):
    """Variance-trigger replan boundaries over the tree underlay, region below the
    cheapest trajectory shaded. Line styles run from least to most continuous as the
    allowed variance grows. spread=True adds the aggregate view of the same fan-out:
    mean cumulative energy mu_D with +/- sigma_D and +/- 2 sigma_D bands."""
    Ds = np.arange(max_depth + 1)
    ax.fill_between(
        Ds, 0, E_MIN_D[: max_depth + 1] * 1e3, step="mid", color=C_BAND, lw=0, zorder=0
    )
    tree_underlay(ax, max_depth=max_depth)
    handles = []
    if spread:
        mu, sd = MU[: max_depth + 1] * 1e3, SD[: max_depth + 1] * 1e3
        ax.fill_between(
            Ds, mu - 2 * sd, mu + 2 * sd, color=C_MEAN, alpha=0.12, lw=0, zorder=1.4
        )
        ax.fill_between(
            Ds, mu - sd, mu + sd, color=C_MEAN, alpha=0.25, lw=0, zorder=1.5
        )
        ax.plot(Ds, mu, color=C_MEAN, lw=1.6, zorder=1.6)
        handles = [
            Line2D([], [], color=C_MEAN, lw=1.6, label=r"Mean Energy $\mu_D$"),
            Patch(color=C_MEAN, alpha=0.25 + 0.12, lw=0, label=r"$\mu_D \pm \sigma_D$"),
            Patch(color=C_MEAN, alpha=0.12, lw=0, label=r"$\mu_D \pm 2\sigma_D$"),
        ]
    for s, ls in zip(sigmas, styles):
        ax.axvline(plan_horizon_variance(s) + 0.5, color=C_VAR, ls=ls, lw=1.6, zorder=2)
    ax.set_xlim(-0.5, max_depth + 0.5)
    ax.set_ylim(0, tree_top(max_depth))
    ax.set_xticks(range(0, max_depth + 1, xtick_step))
    ax.set_xlabel(r"Execution Depth $D$")
    trig = [
        Line2D(
            [],
            [],
            color=C_VAR,
            ls=ls,
            lw=1.6,
            label=rf"Replan at $\sigma_D \geq$ {s * 1e3:.0f} mJ",
        )
        for s, ls in zip(sigmas, styles)
    ]
    return (
        trig
        + [tree_handle(), Patch(color=C_BAND, label="Below Cheapest Trajectory")]
        + handles
    )


def fig1c_trajectory_tree():
    """Paper figure (figure*, full page width), drawn at print size."""
    fig, ax = plt.subplots(figsize=(PAGE_W, 3.9), layout="constrained")
    leg = draw_tree_panel(ax)
    crowded = label_vertex_counts(
        fig,
        ax,
        fontsize=7,
        color="black",
        halo_color="white",
        dot_r=2.2,
        avoid=[leg],
        max_depth=PAPER_DEPTH,
    )
    if crowded:
        print(f"note: {crowded} labels in fig1c kept at default despite crowding")
    save(fig, "fig1c_trajectory_tree")


# ---------------------------------------------------------------- Figure 2
def fig2_cantelli_bracket(slices=(2, 4, 10)):
    budgets = np.linspace(0, B_MAX, 3000)
    fig, axs = plt.subplots(
        1, len(slices), figsize=(7.0, 2.7), sharey=True, layout="constrained"
    )
    for ax, D, tag in zip(axs, slices, "abc"):
        _, m, v, mg = PROF[D]
        U = np.array([cantelli_bound(m, v, B) for B in budgets])
        L = np.array([cantelli_lower(m, v, B, floor=E_MIN_D[D]) for B in budgets])
        x, r = risk_steps(mg)
        ax.fill_between(
            budgets * 1e3, np.clip(L, LO, HI), np.clip(U, LO, HI), color=C_BAND, lw=0
        )
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
    handles = [
        Line2D([], [], color=C_COMP, lw=1.2, label=r"Computed Risk $r_D(B)$"),
        Line2D([], [], color=C_UP, label=r"Cantelli Upper Bound $U_D(B)$"),
        Line2D(
            [],
            [],
            color=C_LO,
            label=r"Lower Bound $L_D(B)$: 1 below Cheapest Trajectory, Else Cantelli",
        ),
        Patch(color=C_BAND, label="Not Resolved by the Bounds"),
        Line2D([], [], color=C_MEAN, lw=1.6, label=r"Mean Demand $\mu_D$"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    save(fig, "fig2_cantelli_bracket")


# ---------------------------------------------------------------- Figure 3
def fig3_budget_vs_horizon(p_grid=(0.01, 0.05, 0.25, 0.75, 0.95, 0.99)):
    fig, axs = plt.subplots(
        2, 3, figsize=(7.0, 4.9), sharex=True, sharey=True, layout="constrained"
    )
    for ax, p, tag in zip(axs.flat, p_grid, "abcdef"):
        b_plus = (MU + SD * np.sqrt((1 - p) / p)) * 1e3
        b_minus = np.maximum(MU - SD * np.sqrt(p / (1 - p)), E_MIN_D[: len(MU)]) * 1e3
        b_p = np.array([exact_required_budget(r[3], p) for r in PROF20]) * 1e3
        ax.fill_between(DEPTHS, b_minus, b_plus, color=C_BAND, lw=0)
        ax.plot(DEPTHS, b_plus, color=C_UP)
        ax.plot(
            DEPTHS, b_minus, color=C_LO, lw=2.6, solid_capstyle="butt"
        )  # wide: shows where it meets B_p
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
        ax.secondary_yaxis("right", functions=to_mF).set_ylabel(
            "Capacitance at 3.2 V (mF)"
        )
    handles = [
        Line2D(
            [],
            [],
            color=C_COMP,
            lw=1.1,
            marker="o",
            ms=2.4,
            label=r"Quantile Budget $B_p(D)$ (Computed)",
        ),
        Line2D([], [], color=C_UP, label=r"Cantelli Safe Budget $B^{+}_{p}(D)$"),
        Line2D([], [], color=C_LO, lw=2.6, label=r"Failure Budget $B^{-}_{p}(D)$"),
        Line2D([], [], color=C_MEAN, lw=1.4, label=r"Mean Demand $\mu_D$"),
    ]
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
        n_plus = [
            plan_horizon_risk(p, B) + 1 if B >= E_MIN_D[0] else 0 for B in budgets
        ]
        ax.plot(
            budgets * 1e3,
            n_plus,
            color=C_UP,
            ls=ls,
            lw=1.5,
            drawstyle="steps-post",
            zorder=2,
        )
    ax.step(
        np.r_[0, xs[keep] * 1e3, B_MAX * 1e3],
        np.r_[0, ys[keep], ys[keep][-1]],
        where="post",
        color=C_COMP,
        lw=1.3,
        zorder=3,
    )
    ax.set_xlim(0, B_MAX * 1e3)
    ax.set_ylim(0, ys[keep][-1] * 1.1)
    ax.set_xlabel(r"Energy Budget $B$ (mJ)")
    ax.set_ylabel(r"Tasks Completed before Brownout, $N$")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(True, zorder=0)
    handles = (
        [Line2D([], [], color=C_COMP, lw=1.3, label=r"Computed $E[N(B)]$")]
        + [
            Line2D(
                [],
                [],
                color=C_UP,
                ls=ls,
                lw=1.5,
                label=rf"$N \geq N^{{+}}_{{p}}(B)$ with Prob. $\geq 1-p$, $p$ = {p:g}",
            )
            for p, ls in zip(ps, styles)
        ]
        + [tree_handle()]
    )
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
    return (
        next((D for D, m, v, _ in PROF if cantelli_bound(m, v, B) >= p), len(PROF)) - 1
    )


def fig5_trigger_comparison(sigmas=(15e-3, 25e-3, 35e-3), ps=(0.01, 0.05, 0.25)):
    """Replan regions of both trigger families over the candidate-trajectory tree."""
    styles = [":", "-", "--"]
    fig, ax = plt.subplots(figsize=(4.6, 3.9), layout="constrained")
    D20 = np.arange(21)
    ax.fill_between(
        D20, 0, E_MIN_D[:21] * 1e3, step="mid", color=C_BAND, lw=0, zorder=0
    )
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
    handles = (
        [
            Line2D(
                [],
                [],
                color=C_VAR,
                ls=ls,
                lw=1.6,
                label=rf"Variance Trigger, $\sigma_D \geq$ {s * 1e3:.0f} mJ",
            )
            for s, ls in zip(sigmas, styles)
        ]
        + [tree_handle()]
        + [
            Line2D(
                [],
                [],
                color=C_UP,
                ls=ls,
                lw=1.6,
                label=rf"Risk Trigger, $U_D(B) \geq$ {p:g}",
            )
            for p, ls in zip(ps, styles)
        ]
        + [Patch(color=C_BAND, label="Below Cheapest Trajectory")]
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    save(fig, "fig5_trigger_comparison")


def fig5b_variance_trigger():
    """Paper figure (single column), drawn at print size."""
    fig, ax = plt.subplots(figsize=(COL_W, 3.15), layout="constrained")
    handles = draw_variance_panel(ax)
    ax.set_ylabel(r"Energy (mJ)")
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.0,
    )
    save(fig, "fig5b_variance_trigger")


def fig5c_variance_trigger_spread():
    """Paper figure (single column): fig5b plus the aggregate view of the fan-out,
    mean cumulative energy with sigma bands. Drawn at print size."""
    fig, ax = plt.subplots(figsize=(COL_W, 3.55), layout="constrained")
    h = draw_variance_panel(
        ax, spread=True
    )  # r15, r25, r35, tree, below, mean, 1sd, 2sd
    handles = h[0:4] + h[5:8] + [h[4]]
    ax.set_ylabel(r"Energy (mJ)")
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.0,
    )
    save(fig, "fig5c_variance_trigger_spread")


def fig6_tree_and_trigger(
    height=3.6, count_fontsize=7.0, ratio=(1.0, 1.0), gap=2.0, ms=4.4
):
    """Paper figure (figure*, full page width): (a) the trajectory tree with counts,
    (b) variance-trigger boundaries over the same tree with the mean / sigma-band layer.
    Equal panel widths and a shared energy axis. Drawn at print size."""
    fig, (axL, axR) = plt.subplots(
        1,
        2,
        figsize=(PAGE_W, height),
        sharey=True,
        layout="constrained",
        gridspec_kw={"width_ratios": list(ratio)},
    )
    legL = draw_tree_panel(axL, ms=ms)
    h = draw_variance_panel(
        axR, spread=True, xtick_step=1
    )  # r15, r25, r35, tree, below, mean, 1sd, 2sd
    handles = [
        h[0],
        h[5],
        h[1],
        h[6],
        h[2],
        h[7],
        h[3],
        h[4],
    ]  # column-major fill gives two clean rows
    axR.tick_params(labelleft=False)
    axL.set_title("(a) Candidate Trajectories")
    axR.set_title("(b) Variance-Triggered Replanning")
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
    )
    crowded = label_vertex_counts(
        fig,
        axL,
        fontsize=count_fontsize,
        color="black",
        halo_color="white",
        gap=gap,
        dot_r=ms / 2,
        avoid=[legL],
        max_depth=PAPER_DEPTH,
    )
    if crowded:
        print(f"note: {crowded} labels in fig6 kept at default despite crowding")
    save(fig, "fig6_tree_and_trigger")
    return crowded


if __name__ == "__main__":
    fig1_risk_map()
    fig1b_risk_map_trajectories()
    fig1c_trajectory_tree()
    fig2_cantelli_bracket()
    fig3_budget_vs_horizon()
    fig4_forward_progress()
    fig5_trigger_comparison()
    fig5b_variance_trigger()
    fig5c_variance_trigger_spread()
    fig6_tree_and_trigger()
