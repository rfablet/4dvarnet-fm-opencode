#!/usr/bin/env python3
"""
4DVarNet-FM: Bias ablation — sweep param_bias / forcing_state_bias for the
baseline DA methods on a case=2-style (corrupted forcing) dataset.

This never touches CS1-CS4 (see data.lorenz63.make_mixed_datasets) or any
other experiment config; it only builds extra datasets via
data.lorenz63.make_bias_ablation_dataset for the (param_bias, forcing_state_bias)
pairs listed under `bias_ablation.cases`.

Usage:
    python eval_bias_ablation.py --config-name experiment/H1_bias_ablation
"""
import os
import random
import sys
import json
import time
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.lorenz63 import Lorenz63Config, make_bias_ablation_dataset
from evaluation.baselines import Weak4DVar, Strong4DVar, EnKF, ETKF
from evaluation.run import evaluate_baseline, fmt_rmse, EXP_DIR

_BASELINE_METHODS = ["Weak-4DVar", "Strong-4DVar", "EnKF", "ETKF"]


@hydra.main(config_path="config", config_name="lorenz63_default", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    seed = cfg.data.get("seed", 123)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if "bias_ablation" not in cfg:
        raise ValueError(
            "No 'bias_ablation' section found. Run with "
            "--config-name experiment/H1_bias_ablation (or another config "
            "that defines bias_ablation.cases)."
        )

    dc = cfg.data
    ac = cfg.bias_ablation
    bc = cfg.baselines
    coupling = ac.get("forcing_coupling", "quartic")
    num_windows = ac.get("num_windows", 200)
    da_window_steps = bc.da_window_steps
    batch_size = bc.get("batch_size", 1)
    enkf_inflation = bc.enkf.inflation if "enkf" in bc else 1.0
    etkf_inflation = bc.etkf.inflation if "etkf" in bc else 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_cfg = Lorenz63Config(
        dt=dc.dt, T_max=dc.T_max, obs_interval=dc.obs_interval,
        R_var=dc.R_var, B_var=dc.B_var, spinup_steps=dc.spinup_steps,
        window_spacing=dc.window_spacing,
        sigma_true=dc.sigma_true, rho_true=dc.rho_true, beta_true=dc.beta_true,
        gamma=dc.gamma, W_L_bar=dc.W_L_bar, c1=dc.c1, c2=dc.c2,
        sigma_0=dc.sigma_0, sigma_L=dc.sigma_L, tau_eta=dc.tau_eta,
        sigma_eta=dc.sigma_eta,
    )

    method_pool = {
        "Weak-4DVar": Weak4DVar(dt=dc.dt, da_window_steps=da_window_steps,
                                 device=device, coupling_type=coupling),
        "Strong-4DVar": Strong4DVar(dt=dc.dt, da_window_steps=da_window_steps,
                                     device=device, coupling_type=coupling),
        "EnKF": EnKF(dt=dc.dt, device=device, coupling_type=coupling, inflation=enkf_inflation),
        "ETKF": ETKF(dt=dc.dt, device=device, coupling_type=coupling, inflation=etkf_inflation),
    }

    out_dir = os.path.join(EXP_DIR, "H1_bias_ablation")
    os.makedirs(out_dir, exist_ok=True)

    # Generate or load cached datasets (one per bias_ablation case)
    datasets_cache = os.path.join(out_dir, "datasets.pt")
    if os.path.exists(datasets_cache):
        t0 = time.time()
        datasets = torch.load(datasets_cache, weights_only=False)
        print(f"Loaded cached datasets in {time.time()-t0:.1f}s")
    else:
        t0 = time.time()
        datasets = {}
        for i, val in enumerate(ac.cases):
            pb, fsb = float(val.param_bias), float(val.forcing_state_bias)
            datasets[i] = make_bias_ablation_dataset(
                base_cfg, param_bias=pb, forcing_state_bias=fsb,
                forcing_coupling=coupling, num_windows=num_windows, seed=1234 + i,
            )
        print(f"Datasets generated in {time.time()-t0:.1f}s")
        torch.save(datasets, datasets_cache)

    config = {
        "T_max": dc.T_max, "dt": dc.dt, "obs_interval": dc.obs_interval,
        "R_var": dc.R_var, "B_var": dc.B_var,
        "da_window_steps": da_window_steps, "batch_size": batch_size,
        "forcing_coupling": coupling, "num_windows": num_windows,
        "weak4dvar": {"opt_steps": bc.weak4dvar.opt_steps, "lr": bc.weak4dvar.lr},
        "strong4dvar": {"max_iter": bc.strong4dvar.max_iter, "lr": bc.strong4dvar.lr},
        "enkf": {"inflation": enkf_inflation},
        "etkf": {"inflation": etkf_inflation},
    }

    results = []
    for i, val in enumerate(ac.cases):
        pb, fsb = float(val.param_bias), float(val.forcing_state_bias)
        print(f"\n[param_bias={pb}, forcing_state_bias={fsb}]")
        ds = datasets[i]
        row = {"param_bias": pb, "forcing_state_bias": fsb, "num_windows": num_windows}
        for name in _BASELINE_METHODS:
            m, s = evaluate_baseline(method_pool[name], ds, ds.cfg, device, batch_size=batch_size)
            row[name] = fmt_rmse(m, s)
            print(f"  {name:<15} mean={np.mean(m):.4f}")
        results.append(row)

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump({"config": config, "cases": results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
