"""
Tests for the param_bias / forcing_state_bias ablation helper.

make_bias_ablation_dataset() must let CS1/CS2/CS3/CS4 (built by make_mixed_datasets)
stay exactly as they are today, while still producing datasets for arbitrary
bias values requested by an ablation experiment config.
"""
import pytest
from data.lorenz63 import Lorenz63Config, make_bias_ablation_dataset, make_mixed_datasets


@pytest.fixture
def base_cfg():
    return Lorenz63Config(seed=1, num_windows=3, T_max=1.0, dt=0.01, obs_interval=10,
                          spinup_steps=50, window_spacing=50)


def test_ablation_dataset_uses_requested_bias_values(base_cfg):
    ds = make_bias_ablation_dataset(base_cfg, param_bias=0.3, forcing_state_bias=0.4,
                               forcing_coupling="quartic", num_windows=3)
    assert ds.cfg.case == 2
    assert ds.cfg.param_bias == 0.3
    assert ds.cfg.forcing_state_bias == 0.4
    assert ds.cfg.forcing_coupling == "quartic"
    assert len(ds) == 3


def test_ablation_dataset_da_params_reflect_bias(base_cfg):
    unbiased = make_bias_ablation_dataset(base_cfg, param_bias=0.0, forcing_state_bias=0.0, num_windows=1)
    biased = make_bias_ablation_dataset(base_cfg, param_bias=0.2, forcing_state_bias=0.2, num_windows=1)
    assert unbiased.cfg.da_params == (unbiased.cfg.sigma_true, unbiased.cfg.rho_true, unbiased.cfg.beta_true)
    assert biased.cfg.da_params != (biased.cfg.sigma_true, biased.cfg.rho_true, biased.cfg.beta_true)
    assert biased.cfg.da_params[0] < biased.cfg.sigma_true


def test_ablation_helper_does_not_change_mixed_datasets(base_cfg):
    """Calling make_bias_ablation_dataset must not perturb CS1-CS4 bias values."""
    make_bias_ablation_dataset(base_cfg, param_bias=0.9, forcing_state_bias=0.9, num_windows=1)
    mixed = make_mixed_datasets(base_cfg, num_train_windows=2, num_val_windows=1,
                                num_test_windows=1, include_randparam_test=False)
    assert mixed["test_cs1"].cfg.param_bias == 0.0
    assert mixed["test_cs1"].cfg.case == 1
    assert mixed["test_cs2"].cfg.param_bias == 0.15
    assert mixed["test_cs2"].cfg.forcing_state_bias == 0.15
    assert mixed["test_cs2"].cfg.case == 2


def test_mixed_datasets_ignore_base_cfg_bias_fields(base_cfg):
    """CS1/CS2 bias values are fixed regardless of base_cfg.param_bias/forcing_state_bias."""
    weird_base = Lorenz63Config(**{**base_cfg.__dict__, "param_bias": 0.5, "forcing_state_bias": 0.5})
    mixed = make_mixed_datasets(weird_base, num_train_windows=1, num_val_windows=1,
                                num_test_windows=1, include_randparam_test=False)
    assert mixed["test_cs1"].cfg.param_bias == 0.0
    assert mixed["test_cs2"].cfg.param_bias == 0.15
    assert mixed["test_cs2"].cfg.forcing_state_bias == 0.15
