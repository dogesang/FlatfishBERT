from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_W, FIG_H = 7.2, 7.1
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

BLUE = "#0072B2"
BLUE_LIGHT = "#8DBFE5"
ORANGE_RED = "#D55E00"
ORANGE_LIGHT = "#F3B38A"
AUG_POS = "#8A8A8A"
AUG_NEG = "#8A8A8A"
TEXT_GRAY = "#666666"
CONNECTOR_GRAY = "#D3D3D3"
HEATMAP_SPINE_WIDTH = 0.6

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



def add_minor_title(ax, title):
    ax.set_title(title, fontsize=7.2, color=TEXT_GRAY, pad=4)



def draw_panel_a(fig, subspec):
    subgs = subspec.subgridspec(1, 2, width_ratios=[1.28, 0.72], wspace=0.42)
    ax_a1 = fig.add_subplot(subgs[0, 0])
    ax_a2 = fig.add_subplot(subgs[0, 1])

    exp2_macro = 98.53
    exp4_macro = 98.72

    ax_a1.hlines(0, exp2_macro, exp4_macro, color=CONNECTOR_GRAY, linewidth=1.1, zorder=1)
    ax_a1.scatter(exp2_macro, 0, s=42, color=ORANGE_RED, edgecolor="#333333", linewidth=0.45, zorder=3)
    ax_a1.scatter(exp4_macro, 0, s=42, color=BLUE, edgecolor="#333333", linewidth=0.45, zorder=3)
    ax_a1.text(exp2_macro, 0.16, "98.53", ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE)
    ax_a1.text(exp4_macro, 0.16, "98.72", ha="center", va="bottom", fontsize=ANNOTATION_FONT_SIZE)

    ax_a1.set_xlim(98.42, 98.78)
    ax_a1.set_xticks([98.45, 98.55, 98.65, 98.75])
    ax_a1.set_yticks([0])
    ax_a1.set_yticklabels(["Macro-F1"])
    ax_a1.set_xlabel("Macro-F1 (%)")
    style_axes(ax_a1, y_grid=False)
    ax_a1.grid(axis="x", linestyle=(0, (3, 3)), linewidth=0.7, alpha=GRID_ALPHA, color=GRID_COLOR)
    ax_a1.spines["top"].set_visible(False)
    ax_a1.spines["right"].set_visible(False)

    labels = ["Exp4", "Exp2"]
    times = [17.5, 8.27]
    epoch_text = ["2.51\nepochs", "0.90\nepochs"]
    colors = [BLUE, ORANGE_RED]
    x = np.arange(len(labels))

    bars = ax_a2.bar(x, times, width=0.52, color=colors, edgecolor="black", linewidth=BAR_EDGE_WIDTH)
    ax_a2.set_ylabel("Training time (h)")
    ax_a2.set_xticks(x)
    ax_a2.set_xticklabels(labels)
    ax_a2.set_ylim(0, 20)
    ax_a2.set_yticks(np.arange(0, 21, 5))
    style_axes(ax_a2, y_grid=True)
    ax_a2.spines["top"].set_visible(False)
    ax_a2.spines["right"].set_visible(False)

    for bar, epoch in zip(bars, epoch_text):
        ax_a2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 0.7,
            epoch,
            ha="center",
            va="top",
            fontsize=5.9,
            linespacing=0.92,
            color=TEXT_GRAY,
        )

    return ax_a1, ax_a2



def draw_panel_b(ax):
    labels = ["H. hippoglossus", "S. senegalensis", "S. solea", "Mean"]
    flatfish = np.array([96.83, 97.91, 97.68, 97.47])
    dnabert2 = np.array([98.02, 98.98, 99.10, 98.70])
    x = np.arange(len(labels))
    width = 0.34

    bars1 = ax.bar(x - width / 2, flatfish, width=width, color=BLUE, edgecolor="black", linewidth=BAR_EDGE_WIDTH, label="FlatfishBert")
    bars2 = ax.bar(x + width / 2, dnabert2, width=width, color=ORANGE_RED, edgecolor="black", linewidth=BAR_EDGE_WIDTH, label="DNABERT-2")

    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=16, ha="right")
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_fontstyle("italic" if i < 3 else "normal")
    ax.set_ylim(96.0, 99.5)
    ax.set_yticks(np.arange(96.0, 99.6, 0.5))
    style_axes(ax, y_grid=True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.01))

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_FONT_SIZE,
            )

    ax.text(
        0.5,
        -0.27,
        "Δ from in-domain mean: −0.10 (FlatfishBert), −0.02 (DNABERT-2)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        color=TEXT_GRAY,
        clip_on=False,
    )



def draw_panel_c(ax):
    labels = ["Task G", "Task N", "Task N\n(+RC)", "Task S", "Task S\n(+RC)"]
    flatfish = np.array([93.70, 83.09, 86.66, 82.22, 80.57])
    dnabert2 = np.array([95.49, 91.48, 92.63, 86.82, 86.28])
    x = np.arange(len(labels))

    for xi, flat_val, dna_val in zip(x, flatfish, dnabert2):
        ax.vlines(xi, min(flat_val, dna_val), max(flat_val, dna_val), color=CONNECTOR_GRAY, linewidth=1.0, zorder=1)
        ax.scatter(xi, flat_val, s=58, color=BLUE, edgecolor="#333333", linewidth=0.45, zorder=3, label="FlatfishBert" if xi == 0 else None)
        ax.scatter(xi, dna_val, s=58, color=ORANGE_RED, edgecolor="#333333", linewidth=0.45, zorder=3, label="DNABERT-2" if xi == 0 else None)

    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(80, 100)
    ax.set_yticks(np.arange(80, 100.1, 2))
    style_axes(ax, y_grid=True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.995), borderaxespad=0.0)

    red_left_edge_task_n_rc = x[2]
    red_left_edge_task_s_rc = x[4]

    ax.annotate(
        "RC improves\nTask N",
        xy=(red_left_edge_task_n_rc, dnabert2[2] + 0.06),
        xytext=(red_left_edge_task_n_rc - 0.9, dnabert2[2] + 1.5),
        fontsize=6.1,
        color=AUG_POS,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color=AUG_POS, linewidth=0.55, shrinkA=0, shrinkB=0),
    )
    ax.annotate(
        "RC slightly\nreduces Task S",
        xy=(red_left_edge_task_s_rc, dnabert2[4] + 0.06),
        xytext=(red_left_edge_task_s_rc - 1.25, dnabert2[4] + 1.6),
        fontsize=6.1,
        color=AUG_NEG,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color=AUG_NEG, linewidth=0.55, shrinkA=0, shrinkB=0),
    )



def draw_heatmap(ax, data, row_labels, col_labels, cmap, vmin, vmax, cbar_label, annotate_fmt=".1f", cbar_pad=0.07, cbar_fraction=0.055):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.tick_params(length=0)
    style_axes(ax, y_grid=False)
    for spine in ax.spines.values():
        spine.set_linewidth(HEATMAP_SPINE_WIDTH)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text_color = "white" if val > (vmin + vmax) / 2 else "black"
            ax.text(j, i, format(val, annotate_fmt), ha="center", va="center", fontsize=ANNOTATION_FONT_SIZE, color=text_color)

    cbar = plt.colorbar(im, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
    cbar.set_label(cbar_label, fontsize=AXIS_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE, width=SPINE_WIDTH * 0.8, length=2.5)
    cbar.outline.set_linewidth(HEATMAP_SPINE_WIDTH)



def draw_panel_d(fig, subspec):
    subgs = subspec.subgridspec(1, 2, width_ratios=[1.05, 1.2], wspace=1.2)
    ax_d1 = fig.add_subplot(subgs[0, 0])
    ax_d2 = fig.add_subplot(subgs[0, 1])

    d1 = np.array(
        [
            [72.22, 56.55],
            [82.54, 61.64],
            [94.74, 68.42],
            [92.86, 75.00],
        ]
    )
    d1_rows = ["FlatfishBert", "FlatfishBert (+RC)", "DNABERT-2", "DNABERT-2 (+RC)"]
    d1_cols = ["snRNA", "snoRNA"]
    draw_heatmap(ax_d1, d1, d1_rows, d1_cols, "YlGnBu", 50, 100, "F1 (%)", cbar_pad=0.05, cbar_fraction=0.07)
    add_minor_title(ax_d1, "Task N minority subtypes")
    for tick in ax_d1.get_yticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
    ax_d1.tick_params(axis="y", pad=1)

    labels = ["FlatfishBert", "FlatfishBert\n(+RC)", "DNABERT-2", "DNABERT-2\n(+RC)"]
    values = np.array([49.7, 45.6, 62.5, 61.3])
    y = np.arange(len(labels))
    stem_colors = [BLUE, BLUE_LIGHT, ORANGE_RED, ORANGE_LIGHT]
    marker_faces = [BLUE, "white", ORANGE_RED, "white"]
    marker_edges = [BLUE, BLUE_LIGHT, ORANGE_RED, ORANGE_LIGHT]

    style_axes(ax_d2, y_grid=False)
    ax_d2.spines["top"].set_visible(False)
    ax_d2.spines["right"].set_visible(False)
    ax_d2.spines["left"].set_visible(False)

    for yi, val, stem_color, face_color, edge_color in zip(y, values, stem_colors, marker_faces, marker_edges):
        ax_d2.hlines(yi, 40, val, color=stem_color, linewidth=1.2)
        ax_d2.plot(val, yi, "o", markersize=5.2, markerfacecolor=face_color, markeredgecolor=edge_color, markeredgewidth=1.1)
        ax_d2.text(val + 0.7, yi, f"{val:.1f}", va="center", ha="left", fontsize=ANNOTATION_FONT_SIZE, color=TEXT_GRAY)

    ax_d2.set_yticks(y)
    ax_d2.set_yticklabels(labels)
    for tick in ax_d2.get_yticklabels():
        tick.set_rotation(28)
        tick.set_ha("right")
        tick.set_va("center")
    ax_d2.invert_yaxis()
    ax_d2.set_xlim(40, 65)
    ax_d2.set_xticks([40, 45, 50, 55, 60, 65])
    ax_d2.set_xlabel("F1 (%)")
    add_minor_title(ax_d2, "Task S minority class")
    ax_d2.tick_params(axis="y", length=0, pad=1)
    ax_d2.grid(False)

    return ax_d1, ax_d2



def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, constrained_layout=False, facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.26], width_ratios=[0.84, 1.16])

    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])

    ax_a1, ax_a2 = draw_panel_a(fig, gs[0, 0])
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)
    ax_d1, ax_d2 = draw_panel_d(fig, gs[1, 1])

    add_panel_heading(ax_a1, "A", "Split sensitivity")
    add_panel_heading(ax_b, "B", "Cross-species transfer")
    add_panel_heading(ax_c, "C", "Diagnostic task overview")
    add_panel_heading(ax_d1, "D", "Class-level diagnostics")

    pos_b = ax_b.get_position()
    ax_b.set_position([pos_b.x0, pos_b.y0 - 0.015, pos_b.width, pos_b.height])

    pos_d1 = ax_d1.get_position()
    pos_d2 = ax_d2.get_position()
    d_shift_y = 0.145
    d2_shift_x = 0.065
    ax_d1.set_position([pos_d1.x0, pos_d1.y0 - d_shift_y, pos_d1.width, pos_d1.height])
    ax_d2.set_position([pos_d2.x0 + d2_shift_x, pos_d2.y0 - d_shift_y, pos_d2.width - 0.015, pos_d2.height])

    fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.10, hspace=0.88, wspace=0.52)

    out_png = OUTPUT_DIR / "Figure4.png"
    out_pdf = OUTPUT_DIR / "Figure4.pdf"
    out_svg = OUTPUT_DIR / "Figure4.svg"

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_svg}")


if __name__ == "__main__":
    main()
