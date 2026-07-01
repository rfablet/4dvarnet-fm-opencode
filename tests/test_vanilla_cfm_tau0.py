import torch
from models.vanilla_cfm import VanillaCFM


class _MockBatch:
    def __init__(self, B=2, T=50, D=3):
        self.states = torch.randn(B, T, D)
        self.obs = torch.randn(B, T, D)
        self.obs_mask = torch.ones(B, T, dtype=torch.bool)
        self.batch_size = B


class TestVanillaCFMTau0:
    def test_tau0_loss_uses_zero_tau(self):
        model = VanillaCFM(state_dim=3, hidden_channels=[4, 8], train_tau_0_only=True)
        batch = _MockBatch(B=2, T=50, D=3)
        model.train()
        loss = model.compute_cfm_loss(batch)
        assert torch.isfinite(loss), "τ=0 loss is not finite"

    def test_tau0_sample_is_single_step(self):
        model = VanillaCFM(state_dim=3, hidden_channels=[4, 8], train_tau_0_only=True)
        model.eval()
        obs = torch.randn(1, 50, 3)
        with torch.no_grad():
            samples = model.sample(obs)
        assert samples.shape == (1, 50, 3)
        assert torch.isfinite(samples).all()

    def test_tau0_flag_gives_different_loss_than_default(self):
        torch.manual_seed(42)
        model_t0 = VanillaCFM(state_dim=3, hidden_channels=[4, 8], train_tau_0_only=True)
        model_default = VanillaCFM(state_dim=3, hidden_channels=[4, 8], train_tau_0_only=False)
        batch = _MockBatch(B=2, T=50, D=3)
        model_t0.train()
        model_default.train()
        loss_t0 = model_t0.compute_cfm_loss(batch)
        loss_default = model_default.compute_cfm_loss(batch)
        assert loss_t0.shape == loss_default.shape

    def test_tau0_model_flag_stored(self):
        model = VanillaCFM(state_dim=3, hidden_channels=[4, 8], train_tau_0_only=True)
        assert model.train_tau_0_only is True
        model2 = VanillaCFM(state_dim=3, hidden_channels=[4, 8], train_tau_0_only=False)
        assert model2.train_tau_0_only is False
