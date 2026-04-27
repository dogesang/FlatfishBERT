from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_W, FIG_H = 7.2, 6.2
DPI = 600

FONT_FAMILY = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
PANEL_LABEL_SIZE = 11
AXIS_LABEL_SIZE = 9
TICK_LABEL_SIZE = 8
LEGEND_FONT_SIZE = 8
ANNOTATION_FONT_SIZE = 7.5
SPINE_WIDTH = 0.8
BAR_EDGE_WIDTH = 0.6
GRID_COLOR = "#666666"
GRID_ALPHA = 0.25

TEXT_GRAY = "#666666"
BLUE = "#0072B2"
ORANGE_RED = "#D55E00"
BASELINE_GRAY = "#666666"
CDS_GREEN_FILL = "#DDE9D8"
CDS_GREEN_EDGE = "#5B7F5A"
INTRON_ORANGE_FILL = "#F2E2D4"
INTRON_ORANGE_EDGE = "#B67A4B"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGION_STATS_PATH = PROJECT_ROOT / "analysis" / "sequence_stats" / "outputs" / "region_stats_sampled.csv"
PERIODICITY_PATH = PROJECT_ROOT / "analysis" / "sequence_stats" / "outputs" / "periodicity_spectrum.csv"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILY,
        "font.size": TICK_LABEL_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "axes.linewidth": SPINE_WIDTH,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def style_axes(ax, y_grid=True):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
    ax.tick_params(width=SPINE_WIDTH * 0.9, length=3)
    if y_grid:
        ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.7, alpha=GRID_ALPHA, color=GRID_COLOR)
    else:
        ax.grid(False)



def add_panel_heading(ax, label, title):
    ax.text(
        -0.14,
        1.08,
        f"{label}  {title}",
        transform=ax.transAxes,
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )



def load_panel_b_data():
    df = pd.read_csv(REGION_STATS_PATH)
    df = df[df["label"].isin(["CDS", "intron"])].copy()
    df["log10_length"] = np.log10(df["length"].clip(lower=1))

    rng = np.random.default_rng(42)
    sampled = []
    for label in ["CDS", "intron"]:
        part = df[df["label"] == label]
        n = min(12000, len(part))
        idx = rng.choice(part.index.to_numpy(), size=n, replace=False)
        sampled.append(part.loc[idx])
    sampled_df = pd.concat(sampled, ignore_index=True)

    return sampled_df



def normalize_periodicity(curve_period, curve_power):
    bg_mask = (curve_period >= 2.5) & (curve_period <= 4.0) & ~(
        (curve_period >= 2.9) & (curve_period <= 3.1)
    )
    background = curve_power[bg_mask].mean()
    if background <= 1e-10:
        background = 1.0
    return curve_power / background



def load_panel_c_data():
    df = pd.read_csv(PERIODICITY_PATH)
    period = df["period"].to_numpy()
    cds = df["CDS_macro"].to_numpy()
    intron = df["intron_macro"].to_numpy()

    mask = (period >= 2.0) & (period <= 5.0)
    period = period[mask]
    cds_norm = normalize_periodicity(period, cds[mask])
    intron_norm = normalize_periodicity(period, intron[mask])
    return period, cds_norm, intron_norm



def draw_violin_with_box(ax, values_a, values_b, ylabel, violin_fills, violin_edges, box_fills, box_edges, ylim=None, yticks=None):
    positions = [0, 1]
    vp = ax.violinplot(
        [values_a, values_b],
        positions=positions,
        widths=0.88,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, fill_color, edge_color in zip(vp["bodies"], violin_fills, violin_edges):
        body.set_facecolor(fill_color)
        body.set_edgecolor(edge_color)
        body.set_alpha(0.15)
        body.set_linewidth(1.1)

    bp = ax.boxplot(
        [values_a, values_b],
        positions=positions,
        widths=0.24,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#2A2A2A", linewidth=1.2),
        whiskerprops=dict(color="#666666", linewidth=0.8),
        capprops=dict(color="#666666", linewidth=0.8),
    )
    for patch, fill_color, edge_color in zip(bp["boxes"], box_fills, box_edges):
        patch.set_facecolor(fill_color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(edge_color)
        patch.set_linewidth(1.1)

    ax.set_xticks(positions)
    ax.set_xticklabels(["CDS", "intron"])
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    style_axes(ax, y_grid=True)



def draw_panel_a(ax):
    labels = ["Length-only\nbaseline", "FlatfishBert\nExp4", "DNABERT-2\nExp4"]
    values = [71.60, 97.57, 98.67]
    colors = [BASELINE_GRAY, BLUE, ORANGE_RED]
    x = np.arange(len(labels))

    bars = ax.bar(
        x,
        values,
        width=0.62,
        color=colors,
        edgecolor="black",
        linewidth=BAR_EDGE_WIDTH,
    )

    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    style_axes(ax, y_grid=True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.4,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
        )

    ax.text(
        0.02,
        0.94,
        "Threshold = 236 bp",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        color=TEXT_GRAY,
    )



def draw_panel_b(fig, subspec):
    sampled_df = load_panel_b_data()
    subgs = subspec.subgridspec(1, 2, wspace=0.34)
    ax_len = fig.add_subplot(subgs[0, 0])
    ax_gc = fig.add_subplot(subgs[0, 1])

    cds_len = sampled_df.loc[sampled_df["label"] == "CDS", "log10_length"].to_numpy()
    intron_len = sampled_df.loc[sampled_df["label"] == "intron", "log10_length"].to_numpy()
    cds_gc = sampled_df.loc[sampled_df["label"] == "CDS", "gc"].to_numpy()
    intron_gc = sampled_df.loc[sampled_df["label"] == "intron", "gc"].to_numpy()

    draw_violin_with_box(
        ax_len,
        cds_len,
        intron_len,
        "log10 length (bp)",
        [CDS_GREEN_FILL, INTRON_ORANGE_FILL],
        [CDS_GREEN_EDGE, INTRON_ORANGE_EDGE],
        [CDS_GREEN_FILL, INTRON_ORANGE_FILL],
        [CDS_GREEN_EDGE, INTRON_ORANGE_EDGE],
    )
    draw_violin_with_box(
        ax_gc,
        cds_gc,
        intron_gc,
        "GC fraction",
        [CDS_GREEN_FILL, INTRON_ORANGE_FILL],
        [CDS_GREEN_EDGE, INTRON_ORANGE_EDGE],
        [CDS_GREEN_FILL, INTRON_ORANGE_FILL],
        [CDS_GREEN_EDGE, INTRON_ORANGE_EDGE],
        ylim=(0.2, 0.7),
        yticks=np.arange(0.2, 0.71, 0.1),
    )

    return ax_len, ax_gc



def draw_panel_c(ax):
    period, cds_norm, intron_norm = load_panel_c_data()

    ax.plot(period, cds_norm, color=CDS_GREEN_EDGE, linewidth=1.2, label="CDS")
    ax.plot(period, intron_norm, color=INTRON_ORANGE_EDGE, linewidth=1.2, label="intron")
    ax.axvline(3.0, color=TEXT_GRAY, linewidth=0.9, linestyle=(0, (3, 3)), alpha=0.7)

    ax.set_xlabel("Period (bp)")
    ax.set_ylabel("Normalized spectral power")
    ax.set_xlim(2.0, 5.0)
    ymax = max(cds_norm.max(), intron_norm.max()) * 1.10
    ax.set_ylim(0.85, ymax)
    style_axes(ax, y_grid=True)
    ax.legend(frameon=False, loc="upper right")

    peak_idx = np.argmin(np.abs(period - 3.0))
    ax.annotate(
        "3-bp periodicity",
        xy=(period[peak_idx], cds_norm[peak_idx]),
        xytext=(3.55, cds_norm[peak_idx] + 0.45),
        fontsize=ANNOTATION_FONT_SIZE,
        color=TEXT_GRAY,
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color=TEXT_GRAY, linewidth=0.8, shrinkA=2, shrinkB=2),
    )



def draw_panel_d(ax):
    categories = ["high-sim removed", "≥95% removed"]
    x = np.arange(len(categories))
    flatfish = np.array([0.000118, -0.000401])
    dnabert2 = np.array([0.000073, -0.000284])

    ax.axhline(0, color=TEXT_GRAY, linewidth=0.9, linestyle=(0, (3, 3)), alpha=0.85)
    ax.plot(x, flatfish, marker="o", markersize=4.8, color=BLUE, linewidth=1.2, label="FlatfishBert")
    ax.plot(x, dnabert2, marker="o", markersize=4.8, color=ORANGE_RED, linewidth=1.2, label="DNABERT-2")

    ax.set_ylabel("Δ Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(-0.0005, 0.0002)
    ax.set_yticks([-0.0005, -0.0004, -0.0003, -0.0002, -0.0001, 0.0, 0.0001, 0.0002])
    style_axes(ax, y_grid=True)
    ax.legend(frameon=False, loc="lower left")



def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, constrained_layout=False, facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.18])

    ax_a = fig.add_subplot(gs[0, 0])
    draw_panel_a(ax_a)

    ax_b_left, ax_b_right = draw_panel_b(fig, gs[0, 1])

    ax_c = fig.add_subplot(gs[1, 0])
    draw_panel_c(ax_c)

    ax_d = fig.add_subplot(gs[1, 1])
    draw_panel_d(ax_d)

    add_panel_heading(ax_a, "A", "Length-only baseline")
    add_panel_heading(ax_b_left, "B", "Sequence features")
    add_panel_heading(ax_c, "C", "3-bp periodicity")
    add_panel_heading(ax_d, "D", "Similarity sensitivity")

    fig.subplots_adjust(left=0.08, right=0.985, top=0.95, bottom=0.10, hspace=0.48, wspace=0.34)

    out_png = OUTPUT_DIR / "Figure3.png"
    out_pdf = OUTPUT_DIR / "Figure3.pdf"
    out_svg = OUTPUT_DIR / "Figure3.svg"

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_svg}")


if __name__ == "__main__":
    main()
