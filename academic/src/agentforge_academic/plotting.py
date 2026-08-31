from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}
DEFAULT_COLORS = [
    PALETTE["blue_main"],
    PALETTE["green_3"],
    PALETTE["red_strong"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["neutral"],
]
AGENT_COLORS = {
    "react": PALETTE["red_strong"],
    "preact": PALETTE["red_2"],
    "rap_single": PALETTE["teal"],
    "single_selective": PALETTE["violet"],
    "routed_always": PALETTE["blue_secondary"],
    "routed_selective": PALETTE["blue_main"],
}
DOMAIN_MARKERS = {
    "navigation": "o",
    "manipulation": "D",
    "retrieval": "^",
    "arithmetic": "s",
    "household": "P",
    "pooled": "*",
}


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    use_tex: bool = False
    font_family: tuple[str, ...] = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> None:
    style = style or FigureStyle()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(style.font_family),
            "font.size": style.font_size,
            "axes.linewidth": style.axes_linewidth,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": style.use_tex,
        }
    )


def finalize_figure(
    fig: plt.Figure,
    out_path: Path,
    formats: list[str] | None = None,
    dpi: int = 300,
    close: bool = True,
    pad: float = 0.05,
) -> list[Path]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chosen = formats or ([out_path.suffix.lstrip(".")] if out_path.suffix else ["pdf", "png"])
    saved: list[Path] = []
    for fmt in chosen:
        target = out_path.with_suffix(f".{fmt}")
        fig.savefig(target, dpi=dpi, bbox_inches="tight", pad_inches=pad)
        saved.append(target)
    if close:
        plt.close(fig)
    return saved


def annotate_bars(
    ax: plt.Axes, bars: BarContainer, fmt: str = "{:.2f}", fontsize: int = 10, padding: int = 3
) -> None:
    for patch in bars:
        height = patch.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, padding),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def make_grouped_bar(
    ax: plt.Axes,
    categories: list[str],
    series: list[list[float]],
    labels: list[str],
    ylabel: str = "Value",
    colors: list[str] | None = None,
    annotate: bool = False,
) -> BarContainer:
    categories_arr = np.asarray(categories)
    widths = np.asarray(series, dtype=float)
    if widths.ndim != 2 or widths.shape[1] != len(categories_arr):
        raise ValueError("series_must_align_with_categories")
    n_series = widths.shape[0]
    palette = colors or DEFAULT_COLORS[:n_series]
    x = np.arange(len(categories_arr))
    width = min(0.8 / n_series, 0.22)
    last: BarContainer | None = None
    for index, (values, label) in enumerate(zip(widths, labels, strict=True)):
        offset = (index - (n_series - 1) / 2) * width
        last = ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=palette[index % len(palette)],
            edgecolor="black",
            linewidth=1.2,
            hatch=["", "//", "\\\\", "..", "xx", ""][index % 6],
        )
        if annotate:
            annotate_bars(ax, last, fmt="{:.2f}", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    assert last is not None
    return last


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_cost_quality(summary_csv: Path, output: Path) -> None:
    apply_publication_style(FigureStyle(font_size=14, axes_linewidth=2.0))
    rows = [row for row in _read_summary(summary_csv) if row["domain"] != "pooled"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for row in rows:
        ax.scatter(
            float(row["mean_cost"]),
            float(row["success_rate"]),
            s=70,
            marker=DOMAIN_MARKERS.get(row["domain"], "o"),
            color=AGENT_COLORS.get(row["agent"], PALETTE["neutral"]),
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
    ax.set_xlabel("Normalized compute cost")
    ax.set_ylabel("Task success rate")
    ax.set_ylim(-0.03, 1.08)
    ax.grid(True, linewidth=0.6, alpha=0.28)
    agent_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="black",
            label=name,
            markersize=8,
        )
        for name, color in AGENT_COLORS.items()
    ]
    domain_handles = [
        Line2D([0], [0], marker=marker, color="#333333", linestyle="", label=domain, markersize=7)
        for domain, marker in DOMAIN_MARKERS.items()
        if domain != "pooled"
    ]
    ax.legend(handles=agent_handles + domain_handles, fontsize=8, loc="lower right", ncol=2)
    finalize_figure(fig, output, formats=["png", "pdf"])


def plot_ablation(summary_csv: Path, output: Path) -> None:
    apply_publication_style(FigureStyle(font_size=13, axes_linewidth=2.0))
    default_order = [
        "react",
        "preact",
        "rap_single",
        "single_selective",
        "routed_always",
        "routed_selective",
    ]
    labels_map = {
        "react": "ReAct\n(never)",
        "preact": "PreAct",
        "rap_single": "RAP",
        "single_selective": "Single+\ngate",
        "routed_always": "Routed\nalways",
        "routed_selective": "Routed+\ngate",
    }
    rows = _read_summary(summary_csv)
    lookup = {(row["agent"], row["domain"]): row for row in rows if row["domain"] != "pooled"}
    domains = sorted({domain for _, domain in lookup})
    order = [
        agent for agent in default_order if all((agent, domain) in lookup for domain in domains)
    ]
    if not order or not domains:
        return
    labels = [labels_map[agent] for agent in order]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    series_success = [
        [float(lookup[(agent, domain)]["success_rate"]) for agent in order] for domain in domains
    ]
    series_cost = [
        [float(lookup[(agent, domain)]["mean_cost"]) for agent in order] for domain in domains
    ]
    colors = [PALETTE["blue_main"], PALETTE["teal"], PALETTE["green_3"], PALETTE["violet"]]
    make_grouped_bar(axes[0], labels, series_success, domains, ylabel="Success rate", colors=colors)
    make_grouped_bar(axes[1], labels, series_cost, domains, ylabel="Mean cost", colors=colors)
    axes[0].set_ylim(0.0, 1.05)
    axes[1].legend(fontsize=8, loc="upper left")
    finalize_figure(fig, output, formats=["png", "pdf"])


def plot_expert_count(summary_csv: Path, output: Path) -> None:
    apply_publication_style(FigureStyle(font_size=14, axes_linewidth=2.0))
    rows = [
        row
        for row in _read_summary(summary_csv)
        if row["domain"] == "pooled" and row["agent"].startswith("routed_k")
    ]
    rows.sort(key=lambda row: int(row["agent"].removeprefix("routed_k")))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ks = [int(row["agent"].removeprefix("routed_k")) for row in rows]
    success = [float(row["success_rate"]) for row in rows]
    cost = [float(row["mean_cost"]) for row in rows]
    ax.plot(ks, success, color=PALETTE["blue_main"], marker="o", linewidth=2.4, label="Success")
    ax.set_xlabel("Experts in the routed mixture (top-k)")
    ax.set_ylabel("Pooled success rate")
    ax.set_xticks(ks)
    ax.set_ylim(0.0, 1.05)
    twin = ax.twinx()
    twin.spines["top"].set_visible(False)
    twin.plot(
        ks,
        cost,
        color=PALETTE["red_strong"],
        marker="s",
        linewidth=2.0,
        linestyle="--",
        label="Cost",
    )
    twin.set_ylabel("Pooled mean cost")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="center right")
    finalize_figure(fig, output, formats=["png", "pdf"])


def write_figures(output: Path) -> None:
    summary = output / "summary.csv"
    plot_cost_quality(summary, output / "cost_quality.png")
    plot_ablation(summary, output / "ablation.png")
    plot_expert_count(summary, output / "expert_count.png")
