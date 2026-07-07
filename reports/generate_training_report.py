#!/usr/bin/env python3
"""Generate training synthesis PDF for all E* (DirectUNet) / F* (VanillaCFM) / G* (Tweedie) experiments."""
import os, sys, re, json, csv, argparse
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(BASE, "experiments")
CFG_DIR = os.path.join(BASE, "config", "experiment")
OUTPUT_DIR = os.path.join(BASE, "reports", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMPONENTS = ["X", "Y", "Z"]

def discover_exp_ids():
    """Find every config/experiment/{E,F,G}<n>_*.yaml, sorted E1,E2,...,F1,F2,...,G1,G2,...
    Tie-broken alphabetically by full id so ordering (and therefore color
    assignment) is deterministic regardless of filesystem listing order."""
    found = []
    for fname in os.listdir(CFG_DIR):
        m = re.match(r"^([EFG])(\d+)_.*\.yaml$", fname)
        if m:
            found.append((m.group(1), int(m.group(2)), fname[:-len(".yaml")]))
    found.sort(key=lambda t: (t[0], t[1], t[2]))
    return [eid for _, _, eid in found]

EXP_IDS = discover_exp_ids()
MODEL_LABEL = {"direct_unet": "DirectUNet", "vanilla_cfm": "VanillaCFM", "tweedie": "Tweedie"}

# Validated CVD-safe categorical palette (fixed hue order, never cycled by
# rank/value — see reports/README or the dataviz skill's palette reference).
# 8 slots covers today's E1/E2/F1/F2 x {cs1,cs2}; a 9th+ experiment repeats
# the cycle rather than inventing an unvalidated hue.
CATEGORICAL_PALETTE = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
EXP_COLOR = {eid: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i, eid in enumerate(EXP_IDS)}
VARIANT_STYLE = {"default": "-", "small": "--", "rand": ":", "other": "-."}

def variant_of(eid):
    if "_rand" in eid:
        return "rand"
    if "_small" in eid:
        return "small"
    if "_default" in eid:
        return "default"
    return "other"

def fmt(val):
    return f"{val:.4f}" if isinstance(val, (int, float)) and np.isfinite(val) else "  n/a "

def load_config(eid):
    path = os.path.join(CFG_DIR, f"{eid}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)

def trained_on(eid, configs):
    return configs.get(eid, {}).get("data", {}).get("train_mix", "?")

def display_label(eid, configs):
    """Short plot label that disambiguates same-numbered cs1/cs2 variants, e.g. 'E1 (cs2)'."""
    return f"{eid.split('_')[0]} ({trained_on(eid, configs)})"

def load_results():
    data = {}
    for eid in EXP_IDS:
        rpath = os.path.join(EXP_DIR, eid, "results.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                data[eid] = json.load(f)
    return data

def latest_version_dir(eid):
    stage_dir = os.path.join(EXP_DIR, eid, "outputs", "stage1")
    if not os.path.isdir(stage_dir):
        return None
    versions = sorted(
        (v for v in os.listdir(stage_dir) if v.startswith("version_")),
        key=lambda v: int(v.split("_")[1]),
    )
    return os.path.join(stage_dir, versions[-1]) if versions else None

def load_loss_curve(eid):
    """Return (epochs, train_loss, val_loss) from the latest CSV logger run, deduped by epoch."""
    vdir = latest_version_dir(eid)
    if vdir is None:
        return None
    csv_path = os.path.join(vdir, "metrics.csv")
    if not os.path.exists(csv_path):
        return None
    train, val = {}, {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ep = int(row["epoch"])
            if row.get("train_loss"):
                train[ep] = float(row["train_loss"])
            if row.get("val_loss"):
                val[ep] = float(row["val_loss"])
    epochs = sorted(set(train) | set(val))
    train_arr = np.array([train.get(e, np.nan) for e in epochs])
    val_arr = np.array([val.get(e, np.nan) for e in epochs])
    return np.array(epochs), train_arr, val_arr

def hyperparameter_table(configs):
    lines = [
        "Experiment Hyperparameters",
        "=" * 146,
        f"{'ID':<32} {'Model':<12} {'Channels':<16} {'S1Ep':<6} {'S2Ep':<6} {'LR':<8} "
        f"{'GradClip':<9} {'TimeEmb':<8} {'N_outer':<8} {'SigPrior':<9} "
        f"{'TrainMix':<16} {'Rand':<6} {'ParamNoise':<10}",
        "-" * 146,
    ]
    for eid in EXP_IDS:
        cfg = configs[eid]
        m = cfg["model"]
        mt = m["model_type"]
        # tweedie's fields sit directly under `model:`; direct_unet/vanilla_cfm
        # nest theirs under `model.<model_type>:`.
        mcfg = m if mt == "tweedie" else m.get(mt, {})
        ch = str(mcfg.get("hidden_channels", "?"))
        s1 = cfg["training"]["stage1"]
        # train.py only ever executes stage 2 for model_type == "tweedie" (see
        # train.py's `if model_type == "tweedie" and epochs_s2 > 0`), so a
        # nonzero stage2.epochs in a direct_unet/vanilla_cfm config is inert —
        # report the *effective* stage2 epoch count, not the declared one.
        s2_epochs = cfg["training"]["stage2"]["epochs"] if mt == "tweedie" else 0
        d = cfg["data"]
        rand = d.get("randomize_params", False)
        lines.append(
            f"{eid:<32} {MODEL_LABEL.get(mt, mt):<12} {ch:<16} {s1['epochs']:<6} {s2_epochs:<6} {s1['lr']:<8} "
            f"{s1['gradient_clip_val']:<9} {mcfg.get('time_emb_dim', '-'):<8} "
            f"{mcfg.get('N_outer', '-'):<8} {mcfg.get('sigma_prior', '-'):<9} "
            f"{d.get('train_mix', '?'):<16} {'yes' if rand else 'no':<6} "
            f"{d.get('param_noise', '-') if rand else '-':<10}"
        )
    lines += [
        "-" * 146,
        "",
        "Shared settings (config/lorenz63_default.yaml):",
        "  dt=0.01, T_max=3.0s, obs_interval=20, R_var=0.5, B_var=2.0",
        "  num_windows=2000, spinup_steps=10000, batch_size=32, dropout=0.1",
        "  optimizer=Adam, gradient_weight=0.1 (grad-loss term)",
        "  Stage 2 only ever runs for Tweedie (S2Ep>0 above); DirectUNet/VanillaCFM are stage-1-only.",
        "",
        "CS1: param_bias=0.0, forcing_state_bias=0.0, coupling=linear (ideal conditions)",
        "CS2: param_bias=0.15, forcing_state_bias=0.15, coupling=quartic (corrupted)",
        "See TrainMix column above for what each experiment was actually trained on.",
        "",
        "DirectUNet: single UNet1D pass, obs -> state (MSE + gradient loss), no flow matching.",
        "VanillaCFM: conditional flow matching, LinearInterpolant(nu=1.0), Euler sampling",
        "            over N_outer=10 steps, x0 ~ N(0, sigma_prior).",
        "Tweedie: 2-stage solver -- stage1 fits a Gaussian mean estimator, stage2 fits a",
        "         non-Gaussian residual correction on top of the frozen stage1 estimator.",
    ]
    return lines

def status_table(exp_data):
    lines = ["Run Status", "=" * 60]
    for eid in EXP_IDS:
        has_results = eid in exp_data
        has_loss = latest_version_dir(eid) is not None
        status = "trained (results + loss log)" if has_results and has_loss else \
                 "loss log only" if has_loss else "NOT YET TRAINED"
        lines.append(f"  {eid:<28} {status}")
    return lines

def make_loss_curves_page(pdf, curves, configs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Training & Validation Loss", fontsize=14, fontweight="bold")

    for ax, key, title in zip(axes, ("train", "val"), ("Train Loss", "Validation Loss")):
        for eid in EXP_IDS:
            if eid not in curves:
                continue
            epochs, train_arr, val_arr = curves[eid]
            y = train_arr if key == "train" else val_arr
            color = EXP_COLOR[eid]
            style = VARIANT_STYLE[variant_of(eid)]
            label = display_label(eid, configs)  # e.g. "E1 (cs2)"
            ax.plot(epochs, y, style, color=color, lw=1.8, alpha=0.9, label=label)
        # ax.axvline(100, color="black", ls="--", lw=1.2, alpha=0.6, label="Stage 1 / Stage 2")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(f"{title} (log scale)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, ls="--", which="both")
        ax.legend(fontsize=7.5, loc="upper right", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close()
    print("  Page: Loss curves (train/val)")

def metrics_table(exp_data, configs):
    lines = [
        "Per-Variable Mean RMSE Summary (CS1 / CS2)",
        "=" * 104,
    ]
    for cs_label, cs_key in [("CS1", "fm_cs1"), ("CS2", "fm_cs2")]:
        lines += [
            f"--- {cs_label} ---",
            f"{'ID':<32} {'Model':<12} {'Trained On':<20} {'X':<10} {'Y':<10} {'Z':<10} {'Mean':<10}",
            "-" * 104,
        ]
        for eid in EXP_IDS:
            if eid not in exp_data or cs_key not in exp_data[eid]:
                continue
            r = exp_data[eid]
            cs = r[cs_key]
            mt = MODEL_LABEL.get(r.get("model_type", ""), "?")
            train_mix = configs.get(eid, {}).get("data", {}).get("train_mix", "?")
            lines.append(
                f"{eid:<32} {mt:<12} {train_mix:<20} {fmt(cs['X']['mean']):<10} {fmt(cs['Y']['mean']):<10} "
                f"{fmt(cs['Z']['mean']):<10} {fmt(cs['mean']):<10}"
            )
        lines.append("")
    header = f"{'ID':<32} {'Model':<12} {'Trained On':<20} {'CS1 mu':<9} {'CS2 mu':<9} {'Deg':<8} {'Time(s)':<9}"
    lines += [
        "-" * len(header),
        "Degradation (CS2 mean / CS1 mean) and training time:",
        "-" * len(header),
        header,
        "-" * len(header),
    ]
    for eid in EXP_IDS:
        if eid not in exp_data:
            continue
        r = exp_data[eid]
        mt = MODEL_LABEL.get(r.get("model_type", ""), "?")
        train_mix = configs.get(eid, {}).get("data", {}).get("train_mix", "?")
        c1 = r.get("fm_cs1", {}).get("mean", float("nan"))
        c2 = r.get("fm_cs2", {}).get("mean", float("nan"))
        deg = r.get("fm_degradation", float("nan"))
        t = r.get("total_time_seconds", 0)
        lines.append(f"{eid:<32} {mt:<12} {train_mix:<20} {fmt(c1):<9} {fmt(c2):<9} {deg:<7.2f}x {t:<9.0f}")
    return lines

def make_cs_bar_charts(fig, exp_data, configs):
    present = [eid for eid in EXP_IDS if eid in exp_data]
    display = [display_label(eid, configs) for eid in present]
    colors = [EXP_COLOR[eid] for eid in present]

    cs1_vals = [exp_data[eid].get("fm_cs1", {}).get("mean", float("nan")) for eid in present]
    cs2_vals = [exp_data[eid].get("fm_cs2", {}).get("mean", float("nan")) for eid in present]
    deg_vals = [exp_data[eid].get("fm_degradation", float("nan")) for eid in present]

    axes = fig.subplots(1, 3)
    fig.suptitle("CS1 / CS2 Mean RMSE & Robustness", fontsize=14, fontweight="bold", y=1.03)
    titles = ["CS1 Mean RMSE (ideal)", "CS2 Mean RMSE (corrupted)", "Degradation (CS2/CS1)"]
    datasets = [cs1_vals, cs2_vals, deg_vals]

    for col, (ax, title, vals) in enumerate(zip(axes, titles, datasets)):
        x = np.arange(len(vals))
        bars = ax.bar(x, vals, color=colors, width=0.55, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            fmt_str = f"{val:.3f}" if col < 2 else f"{val:.2f}x"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt_str,
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        if col == 2:
            ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(display, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("Mean RMSE" if col < 2 else "Ratio", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, ls="--")
        clean = [v for v in vals if np.isfinite(v)]
        if clean:
            ax.set_ylim(0, max(clean) * 1.3)

def make_component_breakdown(pdf, exp_data, configs):
    present = [eid for eid in EXP_IDS if eid in exp_data]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Per-Component RMSE Breakdown (CS1, CS2)", fontsize=14, fontweight="bold")

    for row, (cs_label, cs_key) in enumerate([("CS1", "fm_cs1"), ("CS2", "fm_cs2")]):
        for ci, comp in enumerate(COMPONENTS):
            ax = axes[row, ci]
            vals, labels, bar_colors = [], [], []
            for eid in present:
                r = exp_data[eid]
                if cs_key not in r:
                    continue
                vals.append(r[cs_key][comp]["mean"])
                labels.append(display_label(eid, configs))
                bar_colors.append(EXP_COLOR[eid])
            x = np.arange(len(vals))
            bars = ax.bar(x, vals, color=bar_colors, width=0.55, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
            ax.set_ylabel("RMSE", fontsize=9)
            ax.set_title(f"{cs_label} — {comp}", fontsize=10)
            ax.grid(True, axis="y", alpha=0.3, ls="--")
            if vals:
                ax.set_ylim(0, max(vals) * 1.35)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close()
    print("  Page: Per-component RMSE breakdown")

def load_trajectories(eid, case):
    tpath = os.path.join(EXP_DIR, eid, f"trajectories_{case}.npz")
    if not os.path.exists(tpath):
        return None
    d = np.load(tpath)
    return d["trajectories"], d["truths"]

def best_model_for_case(exp_data, cs_key):
    """Experiment id with the lowest mean RMSE on this case, or None."""
    scored = [(exp_data[eid][cs_key]["mean"], eid) for eid in EXP_IDS
              if eid in exp_data and np.isfinite(exp_data[eid].get(cs_key, {}).get("mean", float("nan")))]
    return min(scored)[1] if scored else None

def make_example_trajectory_page(pdf, exp_data, configs):
    """One row per case (CS1/CS2): the lowest-RMSE model's reconstruction vs truth
    for X/Y/Z on a representative (median-RMSE) test trajectory."""
    rows = []
    for case in ("cs1", "cs2"):
        eid = best_model_for_case(exp_data, f"fm_{case}")
        if eid is None:
            continue
        loaded = load_trajectories(eid, case)
        if loaded is None:
            continue
        rows.append((case, eid, *loaded))
    if not rows:
        print("  Skipped example trajectories (no trajectory files)")
        return

    fig, axes = plt.subplots(len(rows), 3, figsize=(14, 3.6 * len(rows)), squeeze=False)
    fig.suptitle("Example Reconstructions — Best Model per Case (median-RMSE trajectory)",
                 fontsize=14, fontweight="bold")

    for row, (case, eid, traj_arr, truths) in enumerate(rows):
        rmses = np.sqrt(np.mean((traj_arr - truths) ** 2, axis=(1, 2)))
        idx = int(np.argsort(rmses)[len(rmses) // 2])  # representative, not cherry-picked
        mean_rmse = exp_data[eid][f"fm_{case}"]["mean"]
        color = EXP_COLOR[eid]
        time = np.linspace(0, 3.0, truths.shape[1])  # T_max=3.0s (lorenz63_default)
        obs_mask = np.zeros(truths.shape[1], dtype=bool)
        obs_mask[np.arange(20, truths.shape[1], 20)] = True  # obs_interval=20

        for ci, comp in enumerate(COMPONENTS):
            ax = axes[row, ci]
            ax.plot(time, truths[idx, :, ci], "-", color="black", lw=1.5, alpha=0.8, label="Truth")
            ax.plot(time, traj_arr[idx, :, ci], "--", color=color, lw=1.8, alpha=0.9,
                    label=display_label(eid, configs))
            ax.scatter(time[obs_mask], truths[idx, obs_mask, ci],
                       c="gray", s=9, alpha=0.5, zorder=3, label="Obs" if ci == 0 else None)
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel(comp, fontsize=9)
            ax.grid(True, alpha=0.3, ls="--")
            ax.legend(fontsize=7, loc="upper right")
            if ci == 1:
                ax.set_title(
                    f"{case.upper()} best: {eid} (case mean RMSE={mean_rmse:.3f}) — "
                    f"traj #{idx}, RMSE={rmses[idx]:.3f}", fontsize=9.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close()
    print("  Page: Example trajectories (best model per case)")

def main():
    parser = argparse.ArgumentParser(description="Generate E*/F*/G* training synthesis report")
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "synthesis_training_E_F.pdf"))
    args = parser.parse_args()

    if not EXP_IDS:
        print(f"No E*/F*/G* experiment configs found in {CFG_DIR}")
        sys.exit(1)

    configs = {eid: load_config(eid) for eid in EXP_IDS}
    exp_data = load_results()
    curves = {}
    for eid in EXP_IDS:
        c = load_loss_curve(eid)
        if c is not None:
            curves[eid] = c

    print(f"Configs loaded: {len(configs)} / {len(EXP_IDS)}")
    print(f"Results loaded: {len(exp_data)} / {len(EXP_IDS)}")
    print(f"Loss curves loaded: {len(curves)} / {len(EXP_IDS)}")
    for eid in EXP_IDS:
        r = "results" if eid in exp_data else "no-results"
        l = "loss-log" if eid in curves else "no-loss-log"
        print(f"  {eid}: {r}, {l}")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    with PdfPages(args.output) as pdf:
        # Page 1: Title + hyperparameters + status
        fig1, ax1 = plt.subplots(figsize=(12, 9))
        ax1.axis("off")
        lines = hyperparameter_table(configs) + [""] + status_table(exp_data)
        ax1.text(0.02, 0.98, "\n".join(lines), transform=ax1.transAxes,
                 fontsize=7.3, fontfamily="monospace", verticalalignment="top")
        fig1.suptitle(f"4DVarNet-FM: Training Experiments ({len(EXP_IDS)} runs, E*/F*/G*)", fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig1)
        plt.close()
        print("  Page 1: Title + hyperparameters + run status")

        # Page 2: Loss curves
        if curves:
            make_loss_curves_page(pdf, curves, configs)
        else:
            print("  Skipped loss curves (no data)")

        # Page 3: Metrics table
        fig3, ax3 = plt.subplots(figsize=(11, 8))
        ax3.axis("off")
        ax3.text(0.03, 0.97, "\n".join(metrics_table(exp_data, configs)), transform=ax3.transAxes,
                 fontsize=8.5, fontfamily="monospace", verticalalignment="top")
        fig3.suptitle("Results Summary (CS1 / CS2)", fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig3)
        plt.close()
        print("  Page 3: Metrics table")

        # Page 4: CS1/CS2 bar charts
        if exp_data:
            fig4 = plt.figure(figsize=(13, 5))
            make_cs_bar_charts(fig4, exp_data, configs)
            plt.tight_layout(rect=[0, 0, 1, 0.90])
            pdf.savefig(fig4)
            plt.close()
            print("  Page 4: CS1/CS2 bar charts")

            # Page 5: Per-component breakdown
            make_component_breakdown(pdf, exp_data, configs)

            # Page 6: Example trajectories for the best model per case
            make_example_trajectory_page(pdf, exp_data, configs)

        # Final page: notes
        fig_last, ax_last = plt.subplots(figsize=(11, 6))
        ax_last.axis("off")
        missing = [eid for eid in EXP_IDS if eid not in exp_data]
        notes = ["Notes", "=" * 70, ""]
        if missing:
            notes += ["Not yet trained (config exists, no results.json/loss log found):"]
            notes += [f"  - {eid}" for eid in missing]
            notes += ["", "Re-run this report after training to include them:"]
            for eid in missing:
                notes.append(f"  python train.py --config-name experiment/{eid}")
            notes.append("  python reports/generate_training_report.py")
        else:
            notes += [f"All {len(EXP_IDS)} experiments have results."]
        ax_last.text(0.03, 0.95, "\n".join(notes), transform=ax_last.transAxes,
                     fontsize=10, fontfamily="monospace", verticalalignment="top")
        fig_last.suptitle("Notes", fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig_last)
        plt.close()
        print("  Final page: Notes")

    print(f"\nReport saved: {args.output}")

if __name__ == "__main__":
    main()
