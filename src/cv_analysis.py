"""CV (Cross-Validation) analysis script for finding optimal seed.

This script explores multiple seeds for StratifiedGroupKFold to find
the seed that produces the most uniform distribution of Dry_Total_g
across folds.
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from tqdm import tqdm

# Configuration
DATA_PATH = Path("data/input/csiro-biomass/train_wide.csv")
OUTPUT_DIR = Path("data/output/cv_analysis")
TARGET_COL = "Dry_Total_g"
GROUP_COL = "site"
STRATIFY_COL = "State"
N_FOLDS = 5
SEED_RANGE = range(1000)  # 0-999


def load_data() -> pd.DataFrame:
    """Load train_wide.csv and create site column."""
    df = pd.read_csv(DATA_PATH)
    df["site"] = df["State"] + "_" + df["Sampling_Date"]
    # 一貫した分割のためにソート（これがないとseedの再現性が担保されない）
    df = df.sort_values(by=["image_id"]).reset_index(drop=True)
    return df


def create_folds_stratified_group(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    seed: int = 42,
) -> pd.DataFrame:
    """Create fold column using StratifiedGroupKFold."""
    df = df.copy()
    df["fold"] = -1

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(
        sgkf.split(df, y=df[STRATIFY_COL], groups=df[GROUP_COL])
    ):
        df.loc[val_idx, "fold"] = fold

    return df


def create_folds_group(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
) -> pd.DataFrame:
    """Create fold column using GroupKFold (baseline)."""
    df = df.copy()
    df["fold"] = -1

    gkf = GroupKFold(n_splits=n_folds)
    for fold, (_, val_idx) in enumerate(gkf.split(df, groups=df[GROUP_COL])):
        df.loc[val_idx, "fold"] = fold

    return df


def evaluate_fold_distribution(df: pd.DataFrame, target_col: str = TARGET_COL) -> dict:
    """Evaluate distribution uniformity across folds using KS test.

    Returns:
        dict with:
        - ks_stat_max: Maximum KS statistic across all fold pairs (lower is better)
        - ks_stat_mean: Mean KS statistic across all fold pairs
        - ks_pvalue_min: Minimum p-value across all fold pairs
        - fold_stats: Per-fold statistics (mean, median, std, count)
    """
    folds = sorted(df["fold"].unique())
    fold_values = {fold: df[df["fold"] == fold][target_col].values for fold in folds}

    # Calculate KS statistics for all fold pairs
    ks_stats = []
    ks_pvalues = []
    for fold_i, fold_j in combinations(folds, 2):
        stat, pvalue = ks_2samp(fold_values[fold_i], fold_values[fold_j])
        ks_stats.append(stat)
        ks_pvalues.append(pvalue)

    # Per-fold statistics
    fold_stats = {}
    for fold in folds:
        values = fold_values[fold]
        fold_stats[fold] = {
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "count": len(values),
        }

    return {
        "ks_stat_max": max(ks_stats),
        "ks_stat_mean": np.mean(ks_stats),
        "ks_pvalue_min": min(ks_pvalues),
        "fold_stats": fold_stats,
    }


def run_seed_search(
    df: pd.DataFrame,
    seeds: range = SEED_RANGE,
    n_folds: int = N_FOLDS,
) -> pd.DataFrame:
    """Search for optimal seed across given range.

    Returns:
        DataFrame with columns: seed, ks_stat_max, ks_stat_mean, ks_pvalue_min,
        fold0_mean, fold0_median, ..., fold4_mean, fold4_median, etc.
    """
    results = []

    for seed in tqdm(seeds, desc="Searching seeds"):
        df_folded = create_folds_stratified_group(df, n_folds=n_folds, seed=seed)
        eval_result = evaluate_fold_distribution(df_folded)

        row = {
            "seed": seed,
            "ks_stat_max": eval_result["ks_stat_max"],
            "ks_stat_mean": eval_result["ks_stat_mean"],
            "ks_pvalue_min": eval_result["ks_pvalue_min"],
        }

        # Add per-fold statistics
        for fold, stats in eval_result["fold_stats"].items():
            row[f"fold{fold}_mean"] = stats["mean"]
            row[f"fold{fold}_median"] = stats["median"]
            row[f"fold{fold}_std"] = stats["std"]
            row[f"fold{fold}_count"] = stats["count"]

        results.append(row)

    return pd.DataFrame(results)


def visualize_results(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    best_seed: int,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Generate visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Best seed boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    df_best = create_folds_stratified_group(df, seed=best_seed)
    fold_data = [df_best[df_best["fold"] == f][TARGET_COL].values for f in range(N_FOLDS)]
    bp = ax.boxplot(fold_data, labels=[f"Fold {i}" for i in range(N_FOLDS)], patch_artist=True)

    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, N_FOLDS))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_xlabel("Fold")
    ax.set_ylabel(TARGET_COL)
    ax.set_title(f"Best Seed ({best_seed}): {TARGET_COL} Distribution per Fold\n(StratifiedGroupKFold)")

    # Add mean markers
    means = [np.mean(d) for d in fold_data]
    ax.scatter(range(1, N_FOLDS + 1), means, color="red", marker="D", s=50, zorder=3, label="Mean")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "best_seed_boxplot.png", dpi=150)
    plt.close()

    # 2. Comparison: GroupKFold vs StratifiedGroupKFold
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # GroupKFold (baseline)
    df_group = create_folds_group(df)
    fold_data_group = [df_group[df_group["fold"] == f][TARGET_COL].values for f in range(N_FOLDS)]
    bp1 = axes[0].boxplot(fold_data_group, labels=[f"Fold {i}" for i in range(N_FOLDS)], patch_artist=True)
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
    means_group = [np.mean(d) for d in fold_data_group]
    axes[0].scatter(range(1, N_FOLDS + 1), means_group, color="red", marker="D", s=50, zorder=3, label="Mean")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel(TARGET_COL)
    axes[0].set_title("GroupKFold (Baseline)")
    axes[0].legend()

    # Calculate KS stat for GroupKFold
    eval_group = evaluate_fold_distribution(df_group)
    axes[0].text(
        0.02, 0.98,
        f"KS max: {eval_group['ks_stat_max']:.4f}\nKS mean: {eval_group['ks_stat_mean']:.4f}",
        transform=axes[0].transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # StratifiedGroupKFold (best seed)
    bp2 = axes[1].boxplot(fold_data, labels=[f"Fold {i}" for i in range(N_FOLDS)], patch_artist=True)
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
    axes[1].scatter(range(1, N_FOLDS + 1), means, color="red", marker="D", s=50, zorder=3, label="Mean")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel(TARGET_COL)
    axes[1].set_title(f"StratifiedGroupKFold (seed={best_seed})")
    axes[1].legend()

    # Calculate KS stat for best seed
    eval_best = evaluate_fold_distribution(df_best)
    axes[1].text(
        0.02, 0.98,
        f"KS max: {eval_best['ks_stat_max']:.4f}\nKS mean: {eval_best['ks_stat_mean']:.4f}",
        transform=axes[1].transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Set same y-axis limits
    y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=150)
    plt.close()

    # 3. Fold histograms comparison
    fig, axes = plt.subplots(2, N_FOLDS, figsize=(18, 8))

    # Determine common x-axis range
    all_values = df[TARGET_COL].values
    x_min, x_max = all_values.min(), all_values.max()
    bins = np.linspace(x_min, x_max, 20)

    # Upper row: GroupKFold (baseline)
    for fold in range(N_FOLDS):
        ax = axes[0, fold]
        fold_vals = fold_data_group[fold]
        ax.hist(fold_vals, bins=bins, edgecolor="black", alpha=0.7, color=colors[fold])
        ax.axvline(np.mean(fold_vals), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(fold_vals):.1f}")
        ax.set_title(f"Fold {fold} (n={len(fold_vals)})")
        ax.set_xlim(x_min, x_max)
        ax.legend(fontsize=8)
        if fold == 0:
            ax.set_ylabel("GroupKFold\n(Baseline)\nCount")

    # Lower row: StratifiedGroupKFold (best seed)
    for fold in range(N_FOLDS):
        ax = axes[1, fold]
        fold_vals = fold_data[fold]
        ax.hist(fold_vals, bins=bins, edgecolor="black", alpha=0.7, color=colors[fold])
        ax.axvline(np.mean(fold_vals), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(fold_vals):.1f}")
        ax.set_xlabel(TARGET_COL)
        ax.set_xlim(x_min, x_max)
        ax.legend(fontsize=8)
        if fold == 0:
            ax.set_ylabel(f"StratifiedGroupKFold\n(seed={best_seed})\nCount")

    fig.suptitle(f"{TARGET_COL} Distribution per Fold: GroupKFold vs StratifiedGroupKFold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "fold_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Overlaid fold histograms comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: GroupKFold (baseline) - all folds overlaid
    for fold in range(N_FOLDS):
        fold_vals = fold_data_group[fold]
        axes[0].hist(
            fold_vals,
            bins=bins,
            alpha=0.4,
            color=colors[fold],
            edgecolor=colors[fold],
            linewidth=1.5,
            label=f"Fold {fold} (n={len(fold_vals)}, mean={np.mean(fold_vals):.1f})",
        )
    axes[0].set_xlabel(TARGET_COL)
    axes[0].set_ylabel("Count")
    axes[0].set_title("GroupKFold (Baseline)\nAll Folds Overlaid")
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(x_min, x_max)

    # Right: StratifiedGroupKFold (best seed) - all folds overlaid
    for fold in range(N_FOLDS):
        fold_vals = fold_data[fold]
        axes[1].hist(
            fold_vals,
            bins=bins,
            alpha=0.4,
            color=colors[fold],
            edgecolor=colors[fold],
            linewidth=1.5,
            label=f"Fold {fold} (n={len(fold_vals)}, mean={np.mean(fold_vals):.1f})",
        )
    axes[1].set_xlabel(TARGET_COL)
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"StratifiedGroupKFold (seed={best_seed})\nAll Folds Overlaid")
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(x_min, x_max)

    # Set same y-axis limits
    y_max_hist = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(0, y_max_hist)
    axes[1].set_ylim(0, y_max_hist)

    plt.tight_layout()
    plt.savefig(output_dir / "fold_histograms_overlaid.png", dpi=150)
    plt.close()

    # 5. Score distribution across seeds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # KS stat max distribution
    axes[0].hist(results_df["ks_stat_max"], bins=20, edgecolor="black", alpha=0.7)
    axes[0].axvline(
        results_df.loc[results_df["seed"] == best_seed, "ks_stat_max"].values[0],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Best seed ({best_seed})",
    )
    axes[0].set_xlabel("KS Statistic (Max)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of KS Stat Max Across Seeds")
    axes[0].legend()

    # KS stat mean distribution
    axes[1].hist(results_df["ks_stat_mean"], bins=20, edgecolor="black", alpha=0.7)
    axes[1].axvline(
        results_df.loc[results_df["seed"] == best_seed, "ks_stat_mean"].values[0],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Best seed ({best_seed})",
    )
    axes[1].set_xlabel("KS Statistic (Mean)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of KS Stat Mean Across Seeds")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=150)
    plt.close()


def print_summary(results_df: pd.DataFrame, best_seed: int, df: pd.DataFrame) -> None:
    """Print summary of results."""
    print("\n" + "=" * 60)
    print("CV ANALYSIS SUMMARY")
    print("=" * 60)

    # Best seed info
    best_row = results_df[results_df["seed"] == best_seed].iloc[0]
    print(f"\nBest Seed: {best_seed}")
    print(f"  KS Stat Max:  {best_row['ks_stat_max']:.4f}")
    print(f"  KS Stat Mean: {best_row['ks_stat_mean']:.4f}")
    print(f"  KS P-value Min: {best_row['ks_pvalue_min']:.4f}")

    # Per-fold statistics for best seed
    print(f"\nPer-Fold Statistics (seed={best_seed}):")
    print("-" * 50)
    print(f"{'Fold':<6} {'Count':<8} {'Mean':<12} {'Median':<12} {'Std':<12}")
    print("-" * 50)
    for fold in range(N_FOLDS):
        count = int(best_row[f"fold{fold}_count"])
        mean = best_row[f"fold{fold}_mean"]
        median = best_row[f"fold{fold}_median"]
        std = best_row[f"fold{fold}_std"]
        print(f"{fold:<6} {count:<8} {mean:<12.2f} {median:<12.2f} {std:<12.2f}")

    # Comparison with GroupKFold
    print("\n" + "-" * 50)
    print("Comparison with GroupKFold (baseline):")
    df_group = create_folds_group(df)
    eval_group = evaluate_fold_distribution(df_group)
    print(f"  GroupKFold KS Stat Max:  {eval_group['ks_stat_max']:.4f}")
    print(f"  GroupKFold KS Stat Mean: {eval_group['ks_stat_mean']:.4f}")

    improvement_max = eval_group["ks_stat_max"] - best_row["ks_stat_max"]
    improvement_mean = eval_group["ks_stat_mean"] - best_row["ks_stat_mean"]
    print(f"\n  Improvement (KS Max):  {improvement_max:+.4f} ({'better' if improvement_max > 0 else 'worse'})")
    print(f"  Improvement (KS Mean): {improvement_mean:+.4f} ({'better' if improvement_mean > 0 else 'worse'})")

    # Top 5 seeds
    print("\n" + "-" * 50)
    print("Top 5 Seeds (by KS Stat Max):")
    top5 = results_df.nsmallest(5, "ks_stat_max")[["seed", "ks_stat_max", "ks_stat_mean"]]
    print(top5.to_string(index=False))

    print("\n" + "=" * 60)


def main() -> None:
    """Main function."""
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} samples")
    print(f"  Unique sites: {df[GROUP_COL].nunique()}")
    print(f"  States: {df[STRATIFY_COL].unique().tolist()}")

    print(f"\nSearching optimal seed (range: {SEED_RANGE.start}-{SEED_RANGE.stop - 1})...")
    results_df = run_seed_search(df, seeds=SEED_RANGE)

    # Find best seed (minimum KS stat max)
    best_seed = results_df.loc[results_df["ks_stat_max"].idxmin(), "seed"]

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.csv'}")

    # Visualize
    print("Generating visualizations...")
    visualize_results(df, results_df, best_seed)
    print(f"Visualizations saved to {OUTPUT_DIR}")

    # Print summary
    print_summary(results_df, best_seed, df)


if __name__ == "__main__":
    main()
