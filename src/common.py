"""
Shared utilities for the DAFD replication + extension study.

Target paper (replicated):
  Lashkaripour A. et al., "Machine learning enables design automation of microfluidic
  flow-focusing droplet generation", Nature Communications 12, 25 (2021).
  doi:10.1038/s41467-020-20284-z
Follow-up paper (extension baseline):
  Lashkaripour A. et al., "Design automation of microfluidic single and double emulsion
  droplets with machine learning", Nature Communications 15, 83 (2024).
  doi:10.1038/s41467-023-44068-3

Data: experimental CSVs released by the authors at https://github.com/CIDARLAB/DAFD
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RES_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

torch.set_num_threads(1)

# ----------------------------------------------------------------------------------------
# Column definitions (identical to the authors' ModelHelper.py)
# ----------------------------------------------------------------------------------------
FEATURES_2021 = [
    "orifice_size", "aspect_ratio", "expansion_ratio", "normalized_orifice_length",
    "normalized_water_inlet", "normalized_oil_inlet", "flow_rate_ratio", "capillary_number",
]
# 2024 paper: orifice width + Ca + flow-rate ratio + normalised depth, dispersed inlet,
# continuous inlet, outlet width (orifice length is dropped; viscosity ratio is constant here).
FEATURES_2024 = [
    "orifice_size", "capillary_number", "flow_rate_ratio", "aspect_ratio",
    "normalized_water_inlet", "normalized_oil_inlet", "expansion_ratio",
]
GEOM = FEATURES_2021[:6]

# Fluid properties used by the authors (ModelHelper.calculate_formulaic_relations)
SIGMA = 0.005     # N/m, water / NF350 mineral oil + 5% Span 80
MU_OIL = 0.0572   # Pa s

# Plot palette (validated categorical slots 1-3 + neutrals)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load(which: int = 888) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(DATA_DIR, f"ExperimentalResults_{which}.csv"))
    # Device ID = unique as-fabricated geometry (rounded to remove float noise)
    d["device"] = d[GEOM].round(2).astype(str).agg("|".join, axis=1)
    d["device"] = d["device"].astype("category").cat.codes
    return d


# ----------------------------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------------------------
def flow_rates(d: pd.DataFrame):
    """Continuous (oil) and dispersed (water) flow rates in m^3/s from Ca and phi,
    using exactly the authors' definition of the capillary number."""
    W = d["orifice_size"].values * 1e-6
    H = W * d["aspect_ratio"].values
    Wo = W * d["normalized_oil_inlet"].values
    Ww = W * d["normalized_water_inlet"].values
    Ca = d["capillary_number"].values
    Qc = Ca * SIGMA * H * Wo / (MU_OIL * Ww * (1.0 / W - 1.0 / (2.0 * Wo)))
    Qd = Qc / d["flow_rate_ratio"].values
    return Qc, Qd


def rate_from_diameter(d: pd.DataFrame, D_um: np.ndarray) -> np.ndarray:
    """Generation rate (Hz) from droplet diameter via conservation of mass, f = 6 Qd / (pi D^3)."""
    _, Qd = flow_rates(d)
    D = np.asarray(D_um) * 1e-6
    return 6.0 * Qd / (np.pi * D ** 3)


def hydraulic_diameter(d: pd.DataFrame) -> np.ndarray:
    W = d["orifice_size"].values
    H = W * d["aspect_ratio"].values
    return 2 * W * H / (W + H)


# ----------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------
def metrics(y, p) -> dict:
    y = np.asarray(y, float); p = np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return dict(
        R2=1 - ss_res / ss_tot,
        RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
        MAE=float(np.mean(np.abs(y - p))),
        MAPE=float(100 * np.mean(np.abs((y - p) / y))),
    )


# ----------------------------------------------------------------------------------------
# Neural networks (PyTorch ports of the authors' Keras models)
# ----------------------------------------------------------------------------------------
def _glorot(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)   # Keras default kernel init
        nn.init.zeros_(m.bias)


class MLP(nn.Module):
    def __init__(self, n_in, hidden, out_act=None):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        if out_act == "sigmoid":
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self.apply(_glorot)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class NNConfig:
    hidden: tuple
    batch: int
    epochs: int
    patience: int
    opt: str = "nadam"          # 'nadam' (Keras default lr=0.002), 'adam', 'rmsprop'
    lr: float = 0.002
    weight_decay: float = 0.0
    task: str = "reg"           # 'reg' or 'clf'


# Architectures exactly as in the authors' repository (models/forward_models/*.py)
CFG_SIZE = NNConfig(hidden=(16, 16, 16, 8), batch=20, epochs=400, patience=30)
CFG_RATE1 = NNConfig(hidden=(32, 16, 16, 8), batch=10, epochs=400, patience=30)
CFG_RATE2 = NNConfig(hidden=(16, 16, 16, 8), batch=20, epochs=400, patience=30)
CFG_REGIME = NNConfig(hidden=(32, 16, 16), batch=30, epochs=300, patience=20,
                      opt="rmsprop", lr=0.001, task="clf")
# 2024 paper: 2 hidden layers (512, 16), Adam lr 3e-4, L2 1e-3, MSE, batch 32
CFG_2024 = NNConfig(hidden=(512, 16), batch=32, epochs=400, patience=30,
                    opt="adam", lr=3e-4, weight_decay=1e-3)


class NNModel:
    """Feature standardisation (fit on training data only) + MLP trained like Keras."""

    def __init__(self, cfg: NNConfig, seed: int = 0):
        self.cfg, self.seed = cfg, seed

    def fit(self, X, y):
        cfg = self.cfg
        set_seed(self.seed)
        X = np.asarray(X, np.float32); y = np.asarray(y, np.float32)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xt = torch.tensor((X - self.mu) / self.sd); yt = torch.tensor(y)
        self.model = MLP(X.shape[1], cfg.hidden, "sigmoid" if cfg.task == "clf" else None)
        params = self.model.parameters()
        if cfg.opt == "nadam":
            opt = torch.optim.NAdam(params, lr=cfg.lr)
        elif cfg.opt == "rmsprop":
            opt = torch.optim.RMSprop(params, lr=cfg.lr, alpha=0.9, eps=1e-7)
        else:
            opt = torch.optim.Adam(params, lr=cfg.lr)
        loss_fn = nn.BCELoss() if cfg.task == "clf" else nn.MSELoss()
        n = len(yt); best, wait = np.inf, 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(n); tot = 0.0
            self.model.train()
            for i in range(0, n, cfg.batch):
                idx = perm[i:i + cfg.batch]
                opt.zero_grad()
                out = self.model(Xt[idx])
                loss = loss_fn(out, yt[idx])
                if cfg.weight_decay:   # L2 penalty on kernels (Keras-style regulariser)
                    loss = loss + cfg.weight_decay * sum(
                        (p ** 2).sum() for nme, p in self.model.named_parameters() if "weight" in nme)
                loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            ep_loss = tot / n
            # Keras EarlyStopping on the training loss, min_delta = 0, no weight restore
            if ep_loss < best:
                best, wait = ep_loss, 0
            else:
                wait += 1
                if wait >= cfg.patience:
                    break
        return self

    def predict(self, X):
        X = (np.asarray(X, np.float32) - self.mu) / self.sd
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X)).numpy().astype(float)


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
