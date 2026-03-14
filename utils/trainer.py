"""
ESMM Trainer.

Loss minimized (symbolic):
  L = λ_ctr * BCE(pCTR, y_click) + λ_ctcvr * BCE(pCTR × pCVR, y_purchase)

  No direct BCE(pCVR, y_purchase) — CVR learns only through CTCVR gradient.
  L2 regularization via AdamW weight_decay (λ = 1e-6 default).

Early stopping: monitors AUC_CVR (on clicked val samples).
  Rationale: CVR AUC is the primary business metric (ranking quality given click).
  AUC_CTCVR would also be acceptable; AUC_CTR alone is not sufficient.
"""

import logging
import os
import time
from typing import Dict, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import compute_esmm_metrics

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 3, mode: str = "max", min_delta: float = 1e-4):
        self.patience    = patience
        self.mode        = mode
        self.min_delta   = min_delta
        self.best        = float("-inf") if mode == "max" else float("inf")
        self.counter     = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = (value > self.best + self.min_delta if self.mode == "max"
                    else value < self.best - self.min_delta)
        if improved:
            self.best    = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class ESMMTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        patience: int = 3,
        use_amp: bool = True,
        log_dir: str = "runs/exp",
        device: str = "cuda",
        ctr_loss_weight: float = 1.0,
        target_cpa: float = 50000.0,     # li, used to compute synthetic bid in cost monitor
    ):
        self.model         = model.to(device)
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.device        = device
        self.log_dir       = log_dir
        self.use_amp       = use_amp and device == "cuda"
        self.ctr_loss_weight = ctr_loss_weight
        self.target_cpa    = target_cpa

        os.makedirs(log_dir, exist_ok=True)

        # AdamW: Adam + decoupled L2 weight decay (better for embedding models)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # Cosine annealing: smooth LR decay, avoids discrete drops
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=30, eta_min=lr * 0.01
        )
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        # Monitor AUC_CVR (primary business metric)
        self.early_stop = EarlyStopping(patience=patience, mode="max")

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = total_ctr = total_ctcvr = 0.0
        n = 0

        for batch in tqdm(self.train_loader, desc="  train", leave=False):
            batch   = {k: v.to(self.device) for k, v in batch.items()}
            click   = batch.pop("click")
            purchase= batch.pop("purchase")

            self.optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=self.use_amp):
                p_ctr, p_cvr, p_ctcvr = self.model(batch)
                loss, l_ctr, l_ctcvr  = self.model.compute_loss(
                    p_ctr, p_ctcvr, click, purchase
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            # Gradient clipping: prevents exploding gradients in attention layers
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss   += loss.item()
            total_ctr    += l_ctr.item()
            total_ctcvr  += l_ctcvr.item()
            n += 1

        return {"loss": total_loss/n, "loss_ctr": total_ctr/n, "loss_ctcvr": total_ctcvr/n}

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_p_ctr, all_p_cvr, all_p_ctcvr = [], [], []
        all_click, all_purchase, all_bid   = [], [], []

        for batch in tqdm(loader, desc="  eval ", leave=False):
            batch    = {k: v.to(self.device) for k, v in batch.items()}
            click    = batch.pop("click").cpu().numpy()
            purchase = batch.pop("purchase").cpu().numpy()

            with autocast(enabled=self.use_amp):
                p_ctr, p_cvr, p_ctcvr = self.model(batch)

            all_p_ctr.append(p_ctr.cpu().numpy())
            all_p_cvr.append(p_cvr.cpu().numpy())
            all_p_ctcvr.append(p_ctcvr.cpu().numpy())
            all_click.append(click)
            all_purchase.append(purchase)
            # Synthetic uniform bid for cost ratio monitoring
            all_bid.append(np.full(len(click), self.target_cpa, dtype=np.float32))

        p_ctr    = np.concatenate(all_p_ctr)
        p_cvr    = np.concatenate(all_p_cvr)
        p_ctcvr  = np.concatenate(all_p_ctcvr)
        click    = np.concatenate(all_click)
        purchase = np.concatenate(all_purchase)
        bid      = np.concatenate(all_bid)

        return compute_esmm_metrics(p_ctr, p_cvr, p_ctcvr, click, purchase, bid)

    def train(self, epochs: int) -> str:
        best_ckpt = os.path.join(self.log_dir, "best_model.pt")

        with mlflow.start_run():
            mlflow.log_params({
                "epochs":          epochs,
                "lr":              self.optimizer.param_groups[0]["lr"],
                "ctr_loss_weight": self.ctr_loss_weight,
                "target_cpa":      self.target_cpa,
                "patience":        self.early_stop.patience,
            })

            for epoch in range(1, epochs + 1):
                t0 = time.time()
                tr  = self._train_epoch()
                val = self._evaluate(self.val_loader)
                self.scheduler.step()
                elapsed = time.time() - t0

                # Primary monitoring metric: CVR AUC on clicked val samples
                monitor_val = val.get("auc_cvr", 0.0)
                if np.isnan(monitor_val):
                    monitor_val = val.get("auc_ctcvr", 0.0)

                cost_flag = (
                    " ⚠UNDERDELIVERY" if val.get("cost_status") == "underdelivery" else
                    " ↑OVERDELIVERY"  if val.get("cost_status") == "overdelivery"  else ""
                )

                mlflow.log_metrics({
                    "train_loss":    tr["loss"],
                    "train_ctr":     tr["loss_ctr"],
                    "train_ctcvr":   tr["loss_ctcvr"],
                    "val_auc_ctr":   val.get("auc_ctr",   0),
                    "val_auc_cvr":   val.get("auc_cvr",   0),
                    "val_auc_ctcvr": val.get("auc_ctcvr", 0),
                    "val_cost_ratio":val.get("cost_ratio", -1),
                    "lr":            self.scheduler.get_last_lr()[0],
                }, step=epoch)

                improved = self.early_stop.step(monitor_val)

                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Loss={tr['loss']:.4f} "
                    f"(ctr={tr['loss_ctr']:.4f}, ctcvr={tr['loss_ctcvr']:.4f}) | "
                    f"AUC_CTR={val.get('auc_ctr',0):.4f} | "
                    f"AUC_CVR={val.get('auc_cvr',0):.4f} | "
                    f"AUC_CTCVR={val.get('auc_ctcvr',0):.4f} | "
                    f"CostRatio={val.get('cost_ratio',-1):.3f}{cost_flag} | "
                    f"{elapsed:.1f}s" + (" <- best" if improved else "")
                )

                if improved:
                    torch.save({
                        "epoch":       epoch,
                        "model_state": self.model.state_dict(),
                        "optim_state": self.optimizer.state_dict(),
                        "val_metrics": {k: float(v) if isinstance(v, float) else v
                                        for k, v in val.items()},
                    }, best_ckpt)
                    mlflow.log_artifact(best_ckpt)

                if self.early_stop.should_stop:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(patience={self.early_stop.patience}, "
                        f"best AUC_CVR={self.early_stop.best:.4f})"
                    )
                    break

        return best_ckpt
