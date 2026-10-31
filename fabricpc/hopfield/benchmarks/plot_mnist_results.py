"""Generate Task 13 result diagrams from JSON report files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "baseline": "#4C78A8",
    "hopfield": "#F58518",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def seed_curve(
    rows: list[dict], kind: str, levels: list[float], variant: str
) -> np.ndarray:
    seeds = sorted({row["seed"] for row in rows})
    curve = np.empty((len(seeds), len(levels)), dtype=np.float64)
    for row_index, seed in enumerate(seeds):
        for level_index, level in enumerate(levels):
            matches = [
                row["head_acc"]
                for row in rows
                if row["seed"] == seed
                and row["variant"] == variant
                and row["kind"] == kind
                and abs(float(row["level"]) - level) < 1e-12
            ]
            if not matches:
                raise ValueError(
                    f"Missing result for seed={seed}, variant={variant}, "
                    f"kind={kind}, level={level}"
                )
            curve[row_index, level_index] = matches[0]
    return curve


def plot_curve(
    ax, rows: list[dict], kind: str, levels: list[float], title: str, xlabel: str
) -> None:
    for variant in ("baseline", "hopfield"):
        values = seed_curve(rows, kind, levels, variant) * 100.0
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        ax.plot(
            levels,
            mean,
            marker="o",
            linewidth=2,
            label="Identity (baseline)" if variant == "baseline" else "Hopfield",
            color=COLORS[variant],
        )
        ax.fill_between(
            levels, mean - std, mean + std, alpha=0.16, color=COLORS[variant]
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Head accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)


def seed_auc(rows: list[dict], kind: str, variant: str) -> dict[int, float]:
    seeds = sorted({row["seed"] for row in rows if row["kind"] == kind})
    result = {}
    for seed in seeds:
        seed_rows = [
            row
            for row in rows
            if row["seed"] == seed and row["kind"] == kind and row["variant"] == variant
        ]
        levels = sorted({float(row["level"]) for row in seed_rows})
        if len(levels) < 2:
            raise ValueError(
                f"Need at least two {kind} levels to calculate robustness AUC."
            )
        values = seed_curve(rows, kind, levels, variant)[seeds.index(seed)]
        area = sum(
            (levels[index + 1] - levels[index])
            * (values[index + 1] + values[index])
            / 2.0
            for index in range(len(levels) - 1)
        )
        result[seed] = float(area / (levels[-1] - levels[0]))
    return result


def mean_auc(rows: list[dict], kind: str) -> tuple[float, float, int]:
    baseline = seed_auc(rows, kind, "baseline")
    hopfield = seed_auc(rows, kind, "hopfield")
    paired_seeds = sorted(set(baseline) & set(hopfield))
    baseline_mean = float(np.mean([baseline[seed] for seed in paired_seeds]))
    hopfield_mean = float(np.mean([hopfield[seed] for seed in paired_seeds]))
    wins = sum(hopfield[seed] > baseline[seed] for seed in paired_seeds)
    return baseline_mean, hopfield_mean, wins


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MNIST Task 13 JSON results")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("reports/mnist_prototype_retrieval/results.json"),
        help="Full Task 13 results JSON",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        help="Optional baseline-only JSON for exact baseline-row verification",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/mnist_prototype_retrieval/diagrams"),
        help="Directory for generated plots",
    )
    args = parser.parse_args()

    result = load_json(args.results)
    rows = result["rows"]
    seeds = result["seeds"]
    config = result.get("config", {})
    algorithm = config.get("algorithm", "backprop")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sigma_base = np.array([seed["E3a_0.5_baseline"] for seed in seeds]) * 100.0
    sigma_hop = np.array([seed["E3a_0.5_hopfield"] for seed in seeds]) * 100.0
    sigma_delta = sigma_hop - sigma_base
    clean_base = np.array([seed["E1_clean_baseline"] for seed in seeds]) * 100.0
    clean_hop = np.array([seed["E1_clean_hopfield"] for seed in seeds]) * 100.0
    clean_delta = clean_hop - clean_base

    baseline_matches = None
    baseline_row_count = 0
    if args.baseline_results is not None:
        baseline_rows = load_json(args.baseline_results)["rows"]
        baseline_row_count = len(baseline_rows)
        baseline_matches = 0
        for baseline_row in baseline_rows:
            matches = [
                row["head_acc"]
                for row in rows
                if row["seed"] == baseline_row["seed"]
                and row["variant"] == "baseline"
                and row["kind"] == baseline_row["kind"]
                and abs(float(row["level"]) - float(baseline_row["level"])) < 1e-12
            ]
            if matches and abs(matches[0] - baseline_row["head_acc"]) < 1e-12:
                baseline_matches += 1

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(
        f"MNIST: Identity vs Hopfield ({algorithm.upper()}, "
        f"mean \u00b1 std over {len(seeds)} seeds)",
        fontsize=16,
        fontweight="bold",
    )

    levels_by_kind = {
        kind: sorted({float(row["level"]) for row in rows if row["kind"] == kind})
        for kind in ("gaussian", "salt_pepper", "occlusion")
    }
    plot_curve(
        axes[0, 0],
        rows,
        "gaussian",
        levels_by_kind["gaussian"],
        "Gaussian noise",
        "Gaussian sigma",
    )
    plot_curve(
        axes[0, 1],
        rows,
        "salt_pepper",
        levels_by_kind["salt_pepper"],
        "Salt-and-pepper noise",
        "Corruption probability",
    )
    plot_curve(
        axes[0, 2],
        rows,
        "occlusion",
        levels_by_kind["occlusion"],
        "Occlusion",
        "Occlusion size (pixels)",
    )

    beta_counts = Counter(seed["beta_star"] for seed in seeds)
    betas = sorted(beta_counts)
    axes[1, 0].bar(
        [str(beta) for beta in betas],
        [beta_counts[beta] for beta in betas],
        color="#7B7B7B",
    )
    axes[1, 0].set_title("Selected beta by seed")
    axes[1, 0].set_xlabel("Beta*")
    axes[1, 0].set_ylabel("Seed count")
    axes[1, 0].set_ylim(0, max(beta_counts.values()) + 1)
    axes[1, 0].grid(axis="y", alpha=0.25)

    seed_ids = np.arange(len(seeds))
    width = 0.38
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].bar(
        seed_ids - width / 2,
        clean_delta,
        width,
        label="Clean",
        color=COLORS["baseline"],
    )
    axes[1, 1].bar(
        seed_ids + width / 2,
        sigma_delta,
        width,
        label="Gaussian sigma=0.5",
        color=COLORS["hopfield"],
    )
    axes[1, 1].set_title("Hopfield minus baseline by seed")
    axes[1, 1].set_xlabel("Seed")
    axes[1, 1].set_ylabel("Accuracy difference (pp)")
    axes[1, 1].set_xticks(seed_ids, seed_ids)
    axes[1, 1].legend(frameon=False, fontsize=9)
    axes[1, 1].grid(axis="y", alpha=0.25)

    axes[1, 2].axis("off")
    auc_results = {
        kind: mean_auc(rows, kind) for kind in ("gaussian", "salt_pepper", "occlusion")
    }
    summary = (
        "SUMMARY\n\n"
        f"Clean: {np.mean(clean_base):.2f}% -> {np.mean(clean_hop):.2f}% "
        f"({np.mean(clean_delta):+.2f} pp)\n\n"
        f"Gaussian \u03c3=0.5: {np.mean(sigma_base):.2f}% -> "
        f"{np.mean(sigma_hop):.2f}% ({np.mean(sigma_delta):+.2f} pp)\n"
        f"Hopfield wins: {int(np.sum(sigma_delta > 0))}/{len(seeds)} seeds\n\n"
        "Robustness AUC (identity \u2192 Hopfield):\n"
    )
    for kind, title in (
        ("gaussian", "Gaussian"),
        ("salt_pepper", "Salt/pepper"),
        ("occlusion", "Occlusion"),
    ):
        base_auc, hop_auc, wins = auc_results[kind]
        summary += (
            f"{title}: {base_auc * 100:.2f}% \u2192 {hop_auc * 100:.2f}% "
            f"({(hop_auc - base_auc) * 100:+.2f} pp; {wins}/{len(seeds)} wins)\n"
        )
    if baseline_matches is not None:
        summary += f"\nBaseline rows matched: {baseline_matches}/{baseline_row_count}"
    axes[1, 2].text(
        0.08,
        0.52,
        summary,
        va="center",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.8",
            "facecolor": "#F2F5F7",
            "edgecolor": "#D5DBDE",
        },
    )

    output_png = args.output_dir / "mnist_task13_results.png"
    output_pdf = args.output_dir / "mnist_task13_results.pdf"
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_png}")
    print(f"Saved {output_pdf}")


if __name__ == "__main__":
    main()
