"""Shared style for every plot of the paper: the mplhep ROOT style at print size, with an automatic overlap check.

With mplhep installed (pip install mplhep) its ROOT style is used: axis labels at the right and top ends of the
axes, ticks inside on all four sides with minor ticks, TeX Gyre Heros type. Without mplhep the same settings are
applied directly. mplhep styles are made for 10 x 10 inch plots with 26 pt type; every length and font size is
scaled here to the size at which a plot is printed (one column, COL, or a third of the page, THIRD), keeping
the proportions of the style.

save() checks every plot before writing it: no two texts may overlap (tick labels, axis labels, legend entries,
annotations), no text may stick out of the plot, and no legend or annotation may cover data. Problems are
printed with the plot's name.
"""
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

COL, WIDE = 3.375, 7.0                                  # inches: one column / two columns of a revtex page (printed at 100%)
THIRD = (WIDE - 0.25) / 3                               # a plot printed three across the page
COLORS = dict(data="black", comb="#0b3c8c", hmm="#2e8b3a", deph="#d45500", device="#c0001a", tree="#7a1fa2",
              tree_d="#b77fd1", comb_d="#8c8c8c", coh0="#5b8fd6", ideal="#0b3c8c")
HEP = None
PROBLEMS = []


def _register_heros():
    for f in glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-*.otf") + \
             glob.glob("/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre/texgyreheros-*.otf"):
        try:
            font_manager.fontManager.addfont(f)
        except Exception:
            pass


def use(style="ROOT", base=9):
    """Apply the mplhep style (ROOT by default) or its emulation, scaled to print size."""
    global HEP
    try:
        import mplhep as hep
        plt.style.use(getattr(hep.style, style))
        HEP = hep
    except Exception:
        HEP = None
        _register_heros()
        plt.rcParams.update({
            "font.family": "sans-serif", "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "custom", "mathtext.rm": "TeX Gyre Heros", "mathtext.it": "TeX Gyre Heros:italic",
            "mathtext.bf": "TeX Gyre Heros:bold", "mathtext.sf": "TeX Gyre Heros", "mathtext.cal": "TeX Gyre Heros",
            "mathtext.tt": "TeX Gyre Heros", "mathtext.default": "regular",
            "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
            "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.minor.top": True, "ytick.minor.right": True,
            "xaxis.labellocation": "right", "yaxis.labellocation": "top", "axes.unicode_minus": False})
    k = base / 26.0 * 10.0 / COL                                              # scale of the 10-inch style to print size
    plt.rcParams.update({
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base, "legend.fontsize": base - 1.5,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1, "legend.frameon": False, "legend.handlelength": 1.5,
        "legend.borderpad": 0.3, "legend.labelspacing": 0.3, "legend.borderaxespad": 0.4,
        "axes.linewidth": 2 * k * 0.42, "xtick.major.size": 12 * k * 0.33, "ytick.major.size": 12 * k * 0.33,
        "xtick.minor.size": 6 * k * 0.33, "ytick.minor.size": 6 * k * 0.33, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5, "xtick.major.pad": 3, "ytick.major.pad": 3, "lines.linewidth": 1.3,
        "lines.markersize": 4, "savefig.bbox": "standard", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "figure.dpi": 150,
        "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
        "xaxis.labellocation": "right", "yaxis.labellocation": "top"})
    return HEP is not None


# ----------------------------------------------------------------------------------------------- overlap check
def _texts(fig):
    out = []
    for ax in fig.axes:
        (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
        for t in ax.get_xticklabels():                                        # only tick labels that are drawn
            if t.get_visible() and t.get_text().strip() and x0 - 1e-9 <= t.get_position()[0] <= x1 + 1e-9:
                out.append(t)
        for t in ax.get_yticklabels():
            if t.get_visible() and t.get_text().strip() and y0 - 1e-9 <= t.get_position()[1] <= y1 + 1e-9:
                out.append(t)
        for t in (ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title):
            if t.get_visible() and t.get_text().strip():
                out.append(t)
        out += [t for t in ax.texts if t.get_visible() and t.get_text().strip()]
        if ax.get_legend():
            out += [t for t in ax.get_legend().get_texts() if t.get_text().strip()]
    for lg in fig.legends:
        out += [t for t in lg.get_texts() if t.get_text().strip()]
    return out


def _data_points(ax):
    pts = []
    for ln in ax.get_lines():
        xy = np.asarray(ln.get_xydata(), float)
        if ln.get_visible() and len(xy):
            pts.append(ln.get_transform().transform(xy))                       # each line in its own coordinates
    for c in ax.collections:
        try:
            off = c.get_offsets()
            if len(off):
                pts.append(c.get_offset_transform().transform(np.asarray(off, float)))
        except Exception:
            pass
    for p in ax.patches:
        try:
            v = p.get_path().vertices
            if len(v) < 400:
                pts.append(p.get_transform().transform(v))
        except Exception:
            pass
    pts = [q for q in pts if q.size]
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def check(fig, name):
    """Report overlapping texts, texts outside the plot, and legends or annotations on top of data."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    problems = []
    T = _texts(fig)
    B = [t.get_window_extent(r) for t in T]
    fb = fig.bbox
    for i in range(len(T)):
        bi = B[i]
        if bi.x0 < fb.x0 - 1 or bi.x1 > fb.x1 + 1 or bi.y0 < fb.y0 - 1 or bi.y1 > fb.y1 + 1:
            problems.append(f"text outside the plot: '{T[i].get_text()}' (text x {bi.x0:.0f}..{bi.x1:.0f}, y {bi.y0:.0f}..{bi.y1:.0f}; plot {fb.x1:.0f} x {fb.y1:.0f})")
        for j in range(i + 1, len(T)):
            bj = B[j]
            if bi.x0 < bj.x1 - 0.5 and bj.x0 < bi.x1 - 0.5 and bi.y0 < bj.y1 - 0.5 and bj.y0 < bi.y1 - 0.5:
                problems.append(f"texts overlap: '{T[i].get_text()}' / '{T[j].get_text()}'")
    for ax in fig.axes:
        pts = _data_points(ax)
        if not len(pts):
            continue
        boxes = []
        if ax.get_legend():
            boxes.append(("legend", ax.get_legend().get_window_extent(r)))
        boxes += [(f"annotation '{t.get_text()}'", t.get_window_extent(r)) for t in ax.texts if t.get_visible() and t.get_text().strip()]
        for what, bb in boxes:
            inside = (pts[:, 0] > bb.x0 + 1) & (pts[:, 0] < bb.x1 - 1) & (pts[:, 1] > bb.y0 + 1) & (pts[:, 1] < bb.y1 - 1)
            if inside.any():
                problems.append(f"{what} covers {int(inside.sum())} data point(s)")
    for p in problems:
        PROBLEMS.append((name, p))
        print(f"    OVERLAP in {name}: {p}")
    return problems


def save(fig, path, tight=False):
    """Check the plot for overlaps, then write it at its printed size."""
    check(fig, getattr(path, "name", str(path)))
    fig.savefig(path, bbox_inches="tight" if tight else None)
    plt.close(fig)


def panel_label(ax, text, x=0.03, y=0.95):
    ax.text(x, y, text, transform=ax.transAxes, va="top", ha="left", fontweight="bold")


# ----------------------------------------------------------------------------------------------- histograms with ratio
def hist_with_ratio(ax, axr, x_data, models, bins, density=True, ratio_ylim=(0.5, 1.5), logy=False):
    """Data as points with statistical errors, models as step lines; ratio model / data below.
    models: list of (values, label, color, linestyle)."""
    c = 0.5 * (bins[1:] + bins[:-1])
    w = np.diff(bins)
    nd, _ = np.histogram(x_data, bins)
    norm = (nd.sum() * w) if density else 1.0
    hd = nd / norm
    ed = np.sqrt(nd) / norm
    for x, lab, col, ls in models:
        nm, _ = np.histogram(x, bins)
        hm = nm / ((nm.sum() * w) if density else 1.0)
        ax.stairs(hm, bins, color=col, ls=ls, lw=1.2, label=lab)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(hd > 0, hm / hd, np.nan)
        axr.stairs(r, bins, color=col, ls=ls, lw=1.0)
    ax.errorbar(c, hd, yerr=ed, fmt="o", ms=2.6, color=COLORS["data"], elinewidth=0.7, capsize=0, label="Geant4", zorder=5)
    with np.errstate(divide="ignore", invalid="ignore"):
        axr.errorbar(c, np.ones_like(c), yerr=np.where(hd > 0, ed / hd, np.nan), fmt="none", ecolor="0.55", elinewidth=0.6)
    axr.axhline(1, color="k", lw=0.6)
    axr.set_ylim(*ratio_ylim)
    if logy:
        ax.set_yscale("log")
    ax.tick_params(labelbottom=False)
    return hd


def ratio_plot(width, height, ratio=(3, 1)):
    """One plot with its ratio panel underneath (a single HEP-style plot)."""
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2, 1, height_ratios=ratio)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)
    fig.get_layout_engine().set(hspace=0.0, h_pad=0.01)
    return fig, ax, axr
