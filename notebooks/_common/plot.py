import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 6.5, "axes.linewidth": .6, "xtick.major.width": .6,
                     "ytick.major.width": .6, "pdf.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]})

BLUE, LBLUE, ORANGE, GREY, DARK = "#2a78d6", "#8bbbe8", "#eb6834", "#bdbdbd", "#6f6f6f"


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def title(ax, letter, text, pad=4):
    ax.set_title(f"{letter}   {text}", loc="left", fontsize=7, fontweight="bold", pad=pad,
                 color="black")


def dapi_crop(img, box_um, scale, channel=None, down=2):
    bx0, by0, bx1, by1 = box_um
    px = [int(round(v * scale)) for v in (bx0, by0, bx1, by1)]
    d = img
    if not hasattr(d, "shape"):
        d = list(d.values())[0]
        d = d[list(d.data_vars)[0]] if hasattr(d, "data_vars") else d
    d = d.sel(c=channel) if channel is not None else (d.isel(c=0) if "c" in d.dims else d)
    return np.asarray(d[px[1]:px[3]:down, px[0]:px[2]:down]), (bx0, bx1, by1, by0)


def dapi_axes(ax, dapi, ext, polys=None, colour=BLUE, head="", bar_um=None):
    lo, hi = np.percentile(dapi[dapi > 0], [1, 99.5]) if (dapi > 0).any() else (0, 1)
    ax.imshow(dapi, cmap="gray", vmin=lo, vmax=hi, extent=ext, interpolation="bilinear")
    if polys is not None and len(polys):
        polys.set_crs(None, allow_override=True).boundary.plot(ax=ax, color=colour, linewidth=.45,
                                                               aspect=None)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.axis("off")
    if head:
        ax.text(.025, .975, head, transform=ax.transAxes, fontsize=6, color=colour,
                fontweight="bold", va="top", ha="left", zorder=10,
                bbox=dict(facecolor="black", edgecolor="none", alpha=.65, pad=1.2))
    if bar_um:
        w = ext[1] - ext[0]
        y = ext[2] - .06 * w
        ax.plot([ext[1] - .06 * w - bar_um, ext[1] - .06 * w], [y, y], color="w", lw=1.3)
        ax.text(ext[1] - .06 * w - bar_um / 2, y - .015 * w, f"{bar_um:.0f} µm", ha="center",
                va="bottom", fontsize=5.5, color="w")


def save(fig, out_dir, letter, name):
    out = out_dir / f"panel{letter}_{name}.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("->", out.name, flush=True)
    return out
