"""Figures for Dataset 1, one function per figure.

Every function draws one figure and returns it (saving if `out` is given). All cosmetic choices
are keyword arguments with the same names everywhere:
    figsize, fontsize, title_fontsize, label_fontsize, tick_fontsize,
    legend (bool), legend_fontsize, legend_loc, legend_ncol,
    colors / styles per model, cmap, vmin/vmax, ratio_ylim, bins, logy, dpi.
`models` is a list of dicts {"label": str, "S": (n, 368) showers in MeV, "color": str, "ls": str}.
Call `use_style("ATLAS", fontsize=11)` once for mplhep styling (plain matplotlib if not installed).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

ANG, RAD = [1, 10, 10, 1, 1], [8, 16, 19, 5, 5]                                # Dataset 1 photon geometry
LAYERS = [8, 160, 190, 5, 5]
EDGES = np.concatenate([[0], np.cumsum(LAYERS)])
LNAMES = ["layer 0", "layer 1", "layer 2", "layer 3", "layer 12"]
HEP = None

def use_style(name="ATLAS", fontsize=11, dpi=150):
    """mplhep style (ATLAS, CMS, LHCb2, ALICE) if installed, else matplotlib defaults; sets the base font size."""
    global HEP
    try:
        import mplhep as hep
        plt.style.use(getattr(hep.style, name))
        HEP = hep
    except Exception: HEP = None
    plt.rcParams.update({"font.size": fontsize, "figure.dpi": dpi})
    return HEP is not None

def _sci_ylabel(ax, label):
    """put the axis' power of ten into the label instead of the floating offset text (which overlaps legends and titles)"""
    from matplotlib.ticker import FuncFormatter
    top = ax.get_ylim()[1]
    if top <= 0 or 1e-2 <= top < 1e3: ax.set_ylabel(label); return
    k = int(np.floor(np.log10(top)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 10 ** k:g}"))
    ax.set_ylabel(f"{label} [$\\times 10^{{{k}}}$]")

def _finish(fig, ax, out, title=None, title_fontsize=None, xlabel=None, ylabel=None, label_fontsize=None, tick_fontsize=None,
            legend=True, legend_fontsize=None, legend_loc="best", legend_ncol=1):
    if title: ax.set_title(title, fontsize=title_fontsize)
    if xlabel: ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel: ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if tick_fontsize: ax.tick_params(labelsize=tick_fontsize)
    if legend and ax.get_legend_handles_labels()[0]:
        if legend_loc == "above":      # outside the frame, above the axes: never overlaps the data
            ax.legend(fontsize=legend_fontsize, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=legend_ncol, frameon=False, borderaxespad=0.0)
        else: ax.legend(fontsize=legend_fontsize, loc=legend_loc, ncol=legend_ncol, frameon=False)
    if out: fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return fig

# ------------------------------------------------------------------------------------------------ correlation matrix
def plot_corr(S, title, out=None, figsize=(5.2, 4.6), cmap="RdBu_r", vmin=-1, vmax=1, layer_lines=True, line_color="k", line_width=0.4,
              cbar_label="correlation", method="pearson", **kw):
    from scipy.stats import spearmanr
    C = np.nan_to_num(np.corrcoef(S.T) if method == "pearson" else spearmanr(S).correlation)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(C, vmin=vmin, vmax=vmax, cmap=cmap)
    if layer_lines:
        for e in EDGES[1:-1]: ax.axhline(e - 0.5, color=line_color, lw=line_width); ax.axvline(e - 0.5, color=line_color, lw=line_width)
    fig.colorbar(im, ax=ax, fraction=0.046, label=cbar_label)
    return _finish(fig, ax, out, title=title, xlabel="voxel", ylabel="voxel", legend=False, **kw)

# ------------------------------------------------------------------------------------------------ histogram + ratio panel
def plot_hist_ratio(x_data, models, xlabel, out=None, bins=40, range_pct=(0.5, 99.5), figsize=(5.2, 4.6), height_ratios=(3, 1),
                    data_label="Geant4", data_fmt="k.", data_ms=5, ratio_ylim=(0.5, 1.5), ratio_label="model/data", ylabel="density", logy=False,
                    headroom=1.15, ratio_min_count=10, legend=True, legend_fontsize=None, legend_loc="above", legend_ncol=1,
                    title=None, title_fontsize=None, label_fontsize=None, tick_fontsize=None):
    """data as points with Poisson errors, models as step histograms, ratio panel below. models: [{"label","x","color","ls"}] with x the values."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=height_ratios, hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)
    lo, hi = np.percentile(np.concatenate([x_data] + [m["x"] for m in models]), range_pct)
    b = np.linspace(lo, hi, bins + 1)
    c = 0.5 * (b[1:] + b[:-1])
    hd, _ = np.histogram(x_data, b, density=True)
    nd, _ = np.histogram(x_data, b)
    ax.errorbar(c, hd, yerr=hd / np.sqrt(np.maximum(nd, 1)), fmt=data_fmt, ms=data_ms, label=data_label)
    for m in models:
        hm, _ = np.histogram(m["x"], b, density=True)
        if HEP is not None: HEP.histplot(hm, b, ax=ax, label=m["label"], color=m.get("color"), linestyle=m.get("ls", "-"), histtype="step")
        else: ax.step(c, hm, where="mid", label=m["label"], color=m.get("color"), ls=m.get("ls", "-"))
        with np.errstate(divide="ignore", invalid="ignore"): r = np.where(nd >= ratio_min_count, hm / hd, np.nan)
        axr.step(c, r, where="mid", color=m.get("color"), ls=m.get("ls", "-"))
    if logy: ax.set_yscale("log"); ax.set_ylim(top=ax.get_ylim()[1] * 10 ** (headroom - 1))
    else: ax.set_ylim(0, ax.get_ylim()[1] * headroom)
    axr.axhline(1, color="k", lw=0.6)
    axr.set_ylim(*ratio_ylim)
    axr.set_xlabel(xlabel, fontsize=label_fontsize)
    axr.set_ylabel(ratio_label, fontsize=label_fontsize)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_xlabel("")
    if logy: ax.set_ylabel(ylabel, fontsize=label_fontsize)
    else: _sci_ylabel(ax, ylabel); ax.yaxis.label.set_size(label_fontsize or plt.rcParams["axes.labelsize"])
    if tick_fontsize: ax.tick_params(labelsize=tick_fontsize); axr.tick_params(labelsize=tick_fontsize)
    return _finish(fig, ax, out, title=title, title_fontsize=title_fontsize, legend=legend, legend_fontsize=legend_fontsize, legend_loc=legend_loc, legend_ncol=legend_ncol)

def plot_energy(S_data, models, which="total", out=None, **kw):
    """which: "total" or a layer index 0..4. models: [{"label","S","color","ls"}]."""
    f = (lambda S: S.sum(1)) if which == "total" else (lambda S, i=which: S[:, EDGES[i]:EDGES[i + 1]].sum(1))
    xl = "total energy [MeV]" if which == "total" else f"{LNAMES[which]} energy [MeV]"
    return plot_hist_ratio(f(S_data), [dict(m, x=f(m["S"])) for m in models], xl, out=out, **kw)

def plot_voxel_marginal(S_data, models, voxel, out=None, bins=30, **kw):
    li = int(np.searchsorted(EDGES, voxel, side="right") - 1)
    return plot_hist_ratio(S_data[:, voxel], [dict(m, x=m["S"][:, voxel]) for m in models], f"{LNAMES[li]}, voxel {voxel} [MeV]", out=out, bins=bins, **kw)

def representative_voxels(S_data, per_layer=(0.3, 0.6, 0.9), n_max=12):
    """low-, middle- and high-energy voxel of every layer (by mean energy)."""
    means = S_data.mean(0)
    pick = []
    for i in range(len(LAYERS)):
        idx = np.arange(EDGES[i], EDGES[i + 1])
        order = idx[np.argsort(means[idx])]
        pick += [int(order[int(q * (len(order) - 1))]) for q in (per_layer if len(order) >= 3 else (0.5,))]
    return pick[:n_max]

# ------------------------------------------------------------------------------------------------ mean shower, radial profiles
def plot_mean_shower(S_data, models, out=None, figsize=(11, 4.6), height_ratios=(3, 1.3), ms=3, ratio_ylim=(0, 2), data_label="Geant4",
                     legend_fontsize=None, legend_loc="lower left", legend_ncol=1, label_fontsize=None, tick_fontsize=None):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=height_ratios, hspace=0.05)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)
    means = S_data.mean(0)
    x = np.arange(len(means))
    ax.semilogy(x, np.maximum(means, 1e-9), "k.", ms=ms, label=data_label)
    for m in models:
        mm = m["S"].mean(0)
        ax.semilogy(x, np.maximum(mm, 1e-9), ".", ms=ms, color=m.get("color"), label=m["label"])
        with np.errstate(divide="ignore", invalid="ignore"): axr.plot(x, mm / means, ".", ms=ms, color=m.get("color"))
    for e in EDGES[1:-1]: ax.axvline(e - 0.5, color="k", lw=0.4); axr.axvline(e - 0.5, color="k", lw=0.4)
    axr.axhline(1, color="k", lw=0.6)
    axr.set_ylim(*ratio_ylim)
    axr.set_xlabel("voxel (layer by layer, ring by ring)", fontsize=label_fontsize)
    axr.set_ylabel("model/data", fontsize=label_fontsize)
    ax.set_ylabel("mean energy per voxel [MeV]", fontsize=label_fontsize)
    plt.setp(ax.get_xticklabels(), visible=False)
    if tick_fontsize: ax.tick_params(labelsize=tick_fontsize); axr.tick_params(labelsize=tick_fontsize)
    return _finish(fig, ax, out, legend_fontsize=legend_fontsize, legend_loc=legend_loc, legend_ncol=legend_ncol)

def plot_radial(S_data, models, layer, out=None, figsize=(5.2, 4.2), data_label="Geant4", data_ms=6, legend_loc="upper right", **kw):
    L = S_data[:, EDGES[layer]:EDGES[layer + 1]]
    nr = L.shape[1] // ANG[layer]
    prof = lambda M: M.reshape(len(M), nr, ANG[layer]).sum(2).mean(0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(np.arange(nr), prof(L), "k.", ms=data_ms, label=data_label)
    for m in models: ax.semilogy(np.arange(nr), prof(m["S"][:, EDGES[layer]:EDGES[layer + 1]]), label=m["label"], color=m.get("color"), ls=m.get("ls", "-"))
    return _finish(fig, ax, out, xlabel=f"{LNAMES[layer]}: ring (radial bin)", ylabel="mean energy per ring [MeV]", legend_loc=legend_loc, **kw)

# ------------------------------------------------------------------------------------------------ polar maps
def _layer_map(S, li): return S[:, EDGES[li]:EDGES[li + 1]].mean(0).reshape(RAD[li], ANG[li])
def _polar_axes(ax, Z, li, norm, cmap, edgecolors="face"):
    na = ANG[li]
    if na == 1: Z, na = np.repeat(Z, 36, 1), 36
    th = np.linspace(0, 2 * np.pi, na + 1)
    rr = np.arange(RAD[li] + 1)
    TH, RR = np.meshgrid(th, rr)
    pc = ax.pcolormesh(TH, RR, Z, norm=norm, cmap=cmap, shading="flat", edgecolors=edgecolors)
    ax.set_xticks([])
    ax.set_yticks([])
    return pc

def plot_polar_layer(S, layer, out=None, ref=None, ratio=False, figsize=(4.2, 4.6), cmap=None, title=None, title_fontsize=10, cbar_label=None,
                     ratio_range=(0.6, 1.4), cbar_fontsize=7):
    """One layer. ratio=False: mean energy (log colour scale, shared with `ref` if given); ratio=True: S / ref."""
    Z = _layer_map(S, layer)
    if ratio:
        Zr = _layer_map(ref, layer)
        with np.errstate(divide="ignore", invalid="ignore"): Z = np.where(Zr > 0, Z / Zr, np.nan)
        norm, cmap = Normalize(*ratio_range), cmap or "coolwarm"
    else:
        Zs = [Z] + ([_layer_map(ref, layer)] if ref is not None else [])
        pos = np.concatenate([z[z > 0] for z in Zs])
        lo = max(pos.min(), 1e-6 * pos.max())
        norm, cmap = LogNorm(lo, pos.max()), cmap or "viridis"
        Z = np.maximum(Z, lo)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))
    pc = _polar_axes(ax, Z, layer, norm, cmap)
    cb = fig.colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05, label=cbar_label or ("ratio" if ratio else "mean energy [MeV]"))
    cb.ax.tick_params(labelsize=cbar_fontsize)
    return _finish(fig, ax, out, title=title, title_fontsize=title_fontsize, legend=False)

def plot_polar_grid(S_data, S_model, model_label, out=None, figsize=None, title_fontsize=10, row_fontsize=9, cbar_fontsize=6, ratio_range=(0.6, 1.4),
                    cmap="viridis", ratio_cmap="coolwarm", data_label="Geant4"):
    """QFAN Fig. 10 layout: one column per layer, rows Geant4 / model / ratio."""
    nL = len(LAYERS)
    fig, axes = plt.subplots(3, nL, figsize=figsize or (3.0 * nL, 8.6), subplot_kw=dict(projection="polar"))
    for li in range(nL):
        Zd, Zm = _layer_map(S_data, li), _layer_map(S_model, li)
        pos = np.concatenate([Zd[Zd > 0], Zm[Zm > 0]])
        lo = max(pos.min(), 1e-6 * pos.max())
        norm = LogNorm(lo, pos.max())
        for row, Z, lab in ((0, Zd, data_label), (1, Zm, model_label)):
            ax = axes[row, li]
            pc = _polar_axes(ax, np.maximum(Z, lo), li, norm, cmap)
            ax.set_title(LNAMES[li] if row == 0 else "", fontsize=title_fontsize)
            ax.set_ylabel(lab if li == 0 else "", labelpad=25, fontsize=row_fontsize)
            cb = fig.colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05, label="mean energy [MeV]" if li == nL // 2 else None)
            cb.ax.tick_params(labelsize=cbar_fontsize)
        with np.errstate(divide="ignore", invalid="ignore"): R = np.where(Zd > 0, Zm / Zd, np.nan)
        ax = axes[2, li]
        pc = _polar_axes(ax, R, li, Normalize(*ratio_range), ratio_cmap)
        ax.set_ylabel(f"{model_label} / {data_label}" if li == 0 else "", labelpad=25, fontsize=row_fontsize)
        cb = fig.colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05, label="ratio" if li == nL // 2 else None)
        cb.ax.tick_params(labelsize=cbar_fontsize)
    if out: fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return fig

# ------------------------------------------------------------------------------------------------ training, hardware
def plot_training(histories, out=None, figsize=(5.2, 4.2), **kw):
    """histories: [{"label","step","val","color","ls"}]."""
    fig, ax = plt.subplots(figsize=figsize)
    for h in histories: ax.semilogy(h["step"][1:], h["val"][1:], label=h["label"], color=h.get("color"), ls=h.get("ls", "-"))
    return _finish(fig, ax, out, xlabel="optimizer step", ylabel="validation MMD$^2$ (exact)", **kw)

def plot_hw_tv(tv_device, tv_noiseless, floor, device_label="ibm_kingston", out=None, figsize=(6, 3.8), **kw):
    """total-variation distance of the count law per measure-and-reset round: device and a noiseless sample, with the multinomial floor."""
    k = len(tv_device)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, k + 1), tv_device, "o-", color="tab:red", label=f"{device_label} vs exact")
    ax.plot(range(1, k + 1), tv_noiseless, "s--", color="tab:blue", label="noiseless sample vs exact (shot noise)")
    ax.axhline(floor, color="k", lw=0.6, ls=":", label="multinomial floor")
    ax.set_ylim(0, None)
    return _finish(fig, ax, out, xlabel="coordinate (measure-and-reset round)", ylabel="total-variation distance of the count law", **kw)

def plot_hw_counts(C_device, q_exact, coordinate, out=None, figsize=(4.2, 3.4), tv=None, device_label="device", **kw):
    """count histogram of one coordinate: device bars against the exact law."""
    K = q_exact.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(K) - 0.2, np.bincount(C_device, minlength=K) / len(C_device), width=0.4, color="tab:red", label=device_label)
    ax.bar(np.arange(K) + 0.2, q_exact, width=0.4, color="tab:blue", label="exact")
    ax.set_xticks(range(K))
    ttl = f"coordinate {coordinate}" + (f"  (TV {tv:.2f})" if tv is not None else "")
    return _finish(fig, ax, out, title=ttl, xlabel="count $c$", ylabel="probability", **kw)

def plot_hw_corr(C_a, C_b=None, title="", out=None, figsize=(4.2, 3.8), cmap="RdBu_r", vmin=-1, vmax=1, difference=False, **kw):
    """Spearman correlations between the coordinates' counts (one matrix; difference=True draws corr(C_a) - corr(C_b))."""
    from scipy.stats import spearmanr
    M = np.nan_to_num(spearmanr(C_a).correlation)
    if difference: M = M - np.nan_to_num(spearmanr(C_b).correlation)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _finish(fig, ax, out, title=title, xlabel="coordinate", ylabel="coordinate", legend=False, **kw)

def plot_polar_grid_multi(S_data, models, out=None, order="grouped", figsize=None, title_fontsize=10, row_fontsize=9, cbar_fontsize=6,
                          ratio_range=(0.6, 1.4), cmap="viridis", ratio_cmap="coolwarm", data_label="Geant4",
                          cbar_label="mean energy [MeV]", ratio_cbar_label="ratio to Geant4"):
    """Average shower per layer for several models in one grid (one column per layer).
    models: [{"label": str, "S": (n, 368)}].  order="grouped": data, every model, then every model/data ratio
    (energy rows share one log color scale per layer, ratio rows one linear scale); order="paired": data,
    model 1, model 1/data, model 2, model 2/data, ..."""
    nL = len(LAYERS)
    rows = [("energy", None, data_label)]
    rl = lambda m: m.get("ratio_label") or f"{m['label']} / {data_label}"
    if order == "grouped": rows += [("energy", m, m["label"]) for m in models] + [("ratio", m, rl(m)) for m in models]
    else:
        for m in models: rows += [("energy", m, m["label"]), ("ratio", m, rl(m))]
    nR = len(rows)
    fig, axes = plt.subplots(nR, nL, figsize=figsize or (3.0 * nL, 2.75 * nR + 0.6), subplot_kw=dict(projection="polar"), squeeze=False)
    for li in range(nL):
        Zd = _layer_map(S_data, li)
        maps = [Zd] + [_layer_map(m["S"], li) for m in models]
        pos = np.concatenate([z[z > 0] for z in maps])
        lo = max(pos.min(), 1e-6 * pos.max())
        norm = LogNorm(lo, pos.max())
        last_e = max(r for r, (k, _, _) in enumerate(rows) if k == "energy")
        last_r = max((r for r, (k, _, _) in enumerate(rows) if k == "ratio"), default=None)
        for r, (kind, m, lab) in enumerate(rows):
            ax = axes[r, li]
            if kind == "energy":
                Z = Zd if m is None else _layer_map(m["S"], li)
                pc = _polar_axes(ax, np.maximum(Z, lo), li, norm, cmap)
            else:
                with np.errstate(divide="ignore", invalid="ignore"): Z = np.where(Zd > 0, _layer_map(m["S"], li) / Zd, np.nan)
                pc = _polar_axes(ax, Z, li, Normalize(*ratio_range), ratio_cmap)
            if r == 0: ax.set_title(LNAMES[li], fontsize=title_fontsize)
            if li == 0: ax.set_ylabel(lab, labelpad=25, fontsize=row_fontsize)
            show_cb = (order != "grouped") or r in (last_e, last_r)
            if show_cb:
                group = [axes[q, li] for q, (k, _, _) in enumerate(rows) if k == kind] if order == "grouped" else [ax]
                cb = fig.colorbar(pc, ax=group, orientation="horizontal", pad=0.03, fraction=0.035 if order == "grouped" else 0.05,
                                  label=(cbar_label if kind == "energy" else ratio_cbar_label) if li == nL // 2 else None)
                cb.ax.tick_params(labelsize=cbar_fontsize)
    if out: fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return fig
