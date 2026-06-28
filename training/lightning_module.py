import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.solver import TweedieSolver
from training.losses import StateMSELoss


class Lit4DVarNetFM(pl.LightningModule):
    def __init__(
        self,
        model: TweedieSolver,
        stage: int = 1,
        lr: float = 1e-3,
        gradient_clip_val: float = 10.0,
        use_gradient_loss: bool = True,
        gradient_weight: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.stage = stage
        self.lr = lr
        self.gradient_clip_val = gradient_clip_val
        self.loss_fn = StateMSELoss(
            use_gradient_loss=use_gradient_loss,
            gradient_weight=gradient_weight,
        )
        self._frozen = False

    def configure_optimizers(self):
        if self.stage == 1:
            params = self.model.mean_estimator.parameters()
        else:
            params = self.model.non_gaussian.parameters()
        return torch.optim.Adam(params, lr=self.lr)

    def on_train_start(self):
        if self._frozen:
            return
        if self.stage == 1:
            for p in self.model.non_gaussian.parameters():
                p.requires_grad = False
            for p in self.model.mean_estimator.parameters():
                p.requires_grad = True
        else:
            for p in self.model.mean_estimator.parameters():
                p.requires_grad = False
            for p in self.model.non_gaussian.parameters():
                p.requires_grad = True
        self._frozen = True

    def training_step(self, batch, batch_idx):
        if self.stage == 1:
            pred = self.model.estimate_mean(batch.obs)
        else:
            pred = self.model(batch.obs)
        loss = self.loss_fn(pred, batch.states)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.stage == 1:
            pred = self.model.estimate_mean(batch.obs)
        else:
            pred = self.model(batch.obs)
        loss = self.loss_fn(pred, batch.states)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch.batch_size)
        return loss

    def load_legacy_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        err = self.model.load_state_dict(state_dict, strict=False)
        if err.missing_keys:
            self.model.mean_estimator.load_state_dict(state_dict, strict=False)

    def forward(self, obs, **kwargs):
        return self.model(obs, **kwargs)
