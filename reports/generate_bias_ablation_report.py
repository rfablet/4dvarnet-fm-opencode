#!/usr/bin/env python3
"""
Generate a synthesis PDF for the H1 bias ablation (param_bias / forcing_state_bias sweep).

Unlike generate_baseline_report.py (which expects a {"cs1": ..., "cs2": ...} dict
plus a trajectory .npz), this reads the JSON produced by eval_bias_ablation.py:
{"config": {...}, "cases": [{"param_bias": ..., "forcing_state_bias": ...,
"Weak-4DVar": {...}, "Strong-4DVar": {...}, "EnKF": {...}, "ETKF": {...}}, ...]}

Usage:
    python reports/generate_bias_ablation_report.py \\
        --json experiments/H1_bias_ablation/results.json \\
        --output reports/outputs/H1_bias_ablation.pdf
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

METHODS = ["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"]
METHOD_COLORS = {
    "Weak-4DVar": "#ff7f0e",
    "Strong-4DVar": "#2ca02c",
    "EnKF": "#d62728",
    "ETKF": "#9467bd",
}


def load_results(json_path):
    with open(json_path) as f:
        return json.load(f)


def split_sweeps(results):
    """Group cases by which bias dimension varies while the other is held at 0."""
    param_sweep = sorted(
        (r for r in results if r["forcing_state_bias"] == 0.0),
        key=lambda r: r["param_bias"],
    )
    forcing_sweep = sorted(
        (r for r in results if r["param_bias"] == 0.0),
        key=lambda r: r["forcing_state_bias"],
    )
    joint = sorted(
        (r for r in results if r["param_bias"] != 0.0 and r["forcing_state_bias"] != 0.0),
        key=lambda r: r["param_bias"],
    )
    return param_sweep, forcing_sweep, joint


COMPONENT_PANELS = [("X", "X"), ("Y", "Y"), ("Z", "Z"), (None, "Mean")]


def _component_rmse(row, method, comp_key):
    entry = row.get(method, {})
    if comp_key is None:
        return entry.get("mean", float("nan"))
    return entry.get(comp_key, {}).get("mean", float("nan"))


def draw_rmse_grid(fig, rows, x_key, xlabel, title):
    xs = [r[x_key] for r in rows]
    axes = fig.subplots(2, 2)
    for ax, (comp_key, label) in zip(axes.flat, COMPONENT_PANELS):
        for method in METHODS:
            ys = [_component_rmse(r, method, comp_key) for r in rows]
            ax.plot(xs, ys, "o-", color=METHOD_COLORS[method], label=method, lw=1.6, ms=4)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(f"RMSE ({label})", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3, ls="--")
    axes.flat[0].legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=12, fontweight="bold")


def make_metrics_table_lines(results):
    lines = [
        "Bias Ablation -- Mean RMSE per Case",
        "=" * 92,
        f"{'param_bias':<12}{'forcing_state_bias':<20}" + "".join(f"{m:<16}" for m in METHODS),
        "-" * 92,
    ]
    for r in results:
        row = f"{r['param_bias']:<12.2f}{r['forcing_state_bias']:<20.2f}"
        for m in METHODS:
            val = r.get(m, {}).get("mean", float("nan"))
            row += f"{val:<16.4f}"
        lines.append(row)
    lines.append("-" * 92)
    return lines


def make_test_data_lines(results):
    num_windows = {r.get("num_windows", "?") for r in results}
    nw_str = str(num_windows.pop()) if len(num_windows) == 1 else "/".join(str(n) for n in sorted(num_windows))
    return [
        "Test data:",
        f"  {nw_str} trajectories per case, generated with seed=200+i (i = case index)",
        "  Forcing corrupted by Ornstein-Uhlenbeck noise (tau_eta=5.0, sigma_eta=0.707), quartic coupling",
    ]


def make_model_parameters_lines(config):
    if not config:
        return []
    weak = config.get("weak4dvar", {})
    strong = config.get("strong4dvar", {})
    enkf = config.get("enkf", {})
    etkf = config.get("etkf", {})
    return [
        "Common parameters:",
        f"  T_max = {config.get('T_max', '?')}s    dt = {config.get('dt', '?')}"
        f"    obs_interval = {config.get('obs_interval', '?')}",
        f"  R_var = {config.get('R_var', '?')}    B_var = {config.get('B_var', '?')}"
        f"    forcing_coupling = {config.get('forcing_coupling', '?')}",
        "",
        "DA Baseline parameters:",
        f"  DA window steps (DWS) = {config.get('da_window_steps', '?')}"
        f"    batch_size = {config.get('batch_size', '?')}",
        f"  Weak-4DVar  : opt_steps={weak.get('opt_steps', '?')}, lr={weak.get('lr', '?')}",
        f"  Strong-4DVar: max_iter={strong.get('max_iter', '?')}, lr={strong.get('lr', '?')}",
        f"  EnKF        : inflation={enkf.get('inflation', '?')}",
        f"  ETKF        : inflation={etkf.get('inflation', '?')}",
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate H1 bias-ablation synthesis PDF")
    parser.add_argument("--json", required=True, help="Path to eval_bias_ablation.py results JSON")
    parser.add_argument("--output", default="reports/outputs/H1_bias_ablation.pdf")
    args = parser.parse_args()

    data = load_results(args.json)
    config = data.get("config", {}) if isinstance(data, dict) else {}
    results = data.get("cases", []) if isinstance(data, dict) else data
    param_sweep, forcing_sweep, joint = split_sweeps(results)

    print(f"Generating PDF: {args.output}")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    with PdfPages(args.output) as pdf:
        # Page 1: title + metrics table
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        lines = [
            "4DVarNet-FM: H1 Bias Ablation Report",
            "=" * 92,
            "",
            "Sweeps param_bias and forcing_state_bias independently on a case=2",
            "(corrupted, quartic-coupled forcing) dataset and evaluates the 4",
            "baseline DA methods. CS1-CS4 are untouched by this ablation.",
            "",
        ] + make_model_parameters_lines(config) + [
            "",
        ] + make_test_data_lines(results) + [
            "",
        ] + make_metrics_table_lines(results) + [
            "",
            "Config file: " + args.json,
        ]
        ax.text(0.03, 0.98, "\n".join(lines), transform=ax.transAxes,
                fontsize=8.5, fontfamily="monospace", verticalalignment="top")
        fig.suptitle("Bias Ablation -- Synthesis Report", fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close()
        print("  Page 1: Title + metrics table")

        if param_sweep:
            fig = plt.figure(figsize=(11, 8.5))
            draw_rmse_grid(fig, param_sweep, "param_bias", "param_bias",
                           "RMSE vs. param_bias  (forcing_state_bias = 0)")
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            plt.close(fig)
            print("  Page 2: RMSE vs param_bias (X/Y/Z/Mean)")

        if forcing_sweep:
            fig = plt.figure(figsize=(11, 8.5))
            draw_rmse_grid(fig, forcing_sweep, "forcing_state_bias", "forcing_state_bias",
                           "RMSE vs. forcing_state_bias  (param_bias = 0)")
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            plt.close(fig)
            print("  Page 3: RMSE vs forcing_state_bias (X/Y/Z/Mean)")

        if joint:
            fig = plt.figure(figsize=(11, 8.5))
            draw_rmse_grid(fig, joint, "param_bias", "param_bias (= forcing_state_bias)",
                           "RMSE vs. joint bias  (param_bias = forcing_state_bias)")
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            plt.close(fig)
            print("  Page 4: RMSE vs joint bias (X/Y/Z/Mean)")

    print(f"\nDone: {args.output}")


if __name__ == "__main__":
    main()
