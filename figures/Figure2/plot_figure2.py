from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_W, FIG_H = 7.2, 5.6
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

EXP_COLORS = {
    "Exp1": "#D55E00",
    "Exp2": "#E69F00",
    "Exp3": "#009E73",
    "Exp4": "#0072B2",
    "Exp5": "#CC79A7",
}
MODEL_COLORS = {
    "FlatfishBert": "#0072B2",
    "DNABERT-2": "#D55E00",
}
TEXT_GRAY = "#666666"
CONNECTOR_GRAY = "#D3D3D3"

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



def add_panel_label(ax, label):
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )



def add_panel_heading(ax, label, title):
    ax.text(
        -0.12,
        1.08,
        f"{label}  {title}",
        transform=ax.transAxes,
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def draw_panel_a(ax):
    labels = [
        "Exp1\nconcat",
        "Exp2\nunconcat",
        "Exp3\nseq_dedup",
        "Exp4\ncross_dedup",
        "Exp5\nunconcat_gene",
    ]
    values = [99.63, 97.81, 97.78, 97.57, 97.60]
    colors = [EXP_COLORS[f"Exp{i}"] for i in range(1, 6)]
    x = np.arange(len(labels))

    bars = ax.bar(
        x,
        values,
        width=0.68,
        color=colors,
        edgecolor="black",
        linewidth=BAR_EDGE_WIDTH,
    )

    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(97.0, 100.0)
    ax.set_yticks(np.arange(97.0, 100.1, 0.5))
    style_axes(ax, y_grid=True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.07,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
        )

    exp4_bar = bars[3]
    exp4_x = exp4_bar.get_x() + exp4_bar.get_width() / 2
    exp4_y = values[3]
    ax.annotate(
        "principal governed regime",
        xy=(exp4_x, exp4_y + 0.03),
        xytext=(exp4_x + 0.45, exp4_y + 0.62),
        textcoords="data",
        fontsize=ANNOTATION_FONT_SIZE,
        color=TEXT_GRAY,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color=TEXT_GRAY, linewidth=0.8, shrinkA=2, shrinkB=2),
    )


def draw_panel_b(ax):
    row_labels = ["Exp1", "Exp2", "Exp3", "Exp4", "Exp5"]
    col_labels = ["0–100", "100–200", "200–300", "300–500", "500–1000", "1000–2000", "2000+"]
    data = np.array(
        [
            [100.00, 93.75, 95.61, 98.09, 99.41, 99.75, 99.89],
            [96.65, 96.93, 97.50, 97.53, 98.42, 97.89, 97.68],
            [96.36, 96.94, 97.68, 98.04, 98.03, 97.81, 97.62],
            [96.20, 96.60, 97.33, 97.51, 97.62, 97.88, 97.24],
            [96.27, 96.75, 97.35, 97.52, 98.11, 97.76, 95.17],
        ]
    )

    im = ax.imshow(data, cmap="viridis", vmin=93.5, vmax=100.0, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Length bin (bp)")
    ax.set_ylabel("Governance regime")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text_color = "white" if val < 97.3 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=ANNOTATION_FONT_SIZE, color=text_color)

    style_axes(ax, y_grid=False)
    ax.tick_params(length=0)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Macro-F1 (%)", fontsize=AXIS_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE, width=SPINE_WIDTH * 0.8, length=2.5)
    cbar.outline.set_linewidth(SPINE_WIDTH)


def draw_panel_c(ax):
    categories = ["Overall", "0–100", "500–1000", "1000–2000", "2000+"]
    flatfish = np.array([97.57, 96.20, 97.62, 97.88, 97.24])
    dnabert2 = np.array([98.72, 97.45, 99.58, 99.73, 99.23])
    gains = dnabert2 - flatfish

    y = np.arange(len(categories))

    for yi, flat_val, dna_val, gain in zip(y, flatfish, dnabert2, gains):
        x_min, x_max = sorted([flat_val, dna_val])
        ax.hlines(yi, x_min, x_max, color=CONNECTOR_GRAY, linewidth=1.0, zorder=1)
        ax.scatter(flat_val, yi, s=30, color=MODEL_COLORS["FlatfishBert"], marker="o", zorder=3, label="FlatfishBert" if yi == 0 else None)
        ax.scatter(dna_val, yi, s=30, color=MODEL_COLORS["DNABERT-2"], marker="o", zorder=3, label="DNABERT-2" if yi == 0 else None)

        text_x = min(max((flat_val + dna_val) / 2, 96.18), 99.82)
        ax.text(
            text_x,
            yi - 0.18,
            f"+{gain:.2f}",
            ha="center",
            va="center",
            fontsize=ANNOTATION_FONT_SIZE,
            color=TEXT_GRAY,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1 (%)")
    ax.set_xlim(96.0, 100.0)
    ax.set_xticks(np.arange(96.0, 100.1, 0.5))
    style_axes(ax, y_grid=False)
    ax.grid(axis="x", linestyle=(0, (3, 3)), linewidth=0.7, alpha=GRID_ALPHA, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, constrained_layout=False, facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.18], width_ratios=[1.25, 1.0])

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)

    add_panel_heading(ax_a, "A", "Governance ladder")
    add_panel_heading(ax_b, "B", "Length-binned performance")
    add_panel_heading(ax_c, "C", "Baseline calibration")

    fig.subplots_adjust(left=0.08, right=0.985, top=0.95, bottom=0.14, hspace=0.42, wspace=0.42)

    out_png = OUTPUT_DIR / "Figure2.png"
    out_pdf = OUTPUT_DIR / "Figure2.pdf"
    out_svg = OUTPUT_DIR / "Figure2.svg"

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_svg}")


if __name__ == "__main__":
    main()
