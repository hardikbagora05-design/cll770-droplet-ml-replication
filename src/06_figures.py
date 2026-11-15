"""Step 6 - All figures for the report and slides (reads results/*.csv, writes figures/*.png)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Circle
from common import RES_DIR, FIG_DIR, C1, C2, C3, INK, INK2, GRID, style_axes, load, flow_rates

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.labelcolor": INK, "text.color": INK,
                     "savefig.dpi": 220, "savefig.bbox": "tight"})
C4 = "#eda100"
REG = {1: "Dripping", 2: "Jetting"}
TGT = {"droplet_size": "Diameter", "generation_rate": "Generation rate"}
PAPER = {  # Lashkaripour et al. 2021, Table (test set, mean +/- SD, 10 sessions)
    ("droplet_size", 1): dict(R2=(0.893, 0.029), MAPE=(11.2, 1.3), MAE=(9.9, 1.2), RMSE=(13.1, 1.6)),
    ("droplet_size", 2): dict(R2=(0.966, 0.010), MAPE=(4.8, 0.5), MAE=(5.9, 0.8), RMSE=(8.2, 1.3)),
    ("generation_rate", 1): dict(R2=(0.889, 0.026), MAPE=(33.5, 4.2), MAE=(19.6, 2.7), RMSE=(31.7, 5.8)),
    ("generation_rate", 2): dict(R2=(0.956, 0.009), MAPE=(15.8, 2.9), MAE=(15.4, 2.1), RMSE=(21.9, 2.8)),
}
save = lambda f, n: (f.savefig(os.path.join(FIG_DIR, n)), plt.close(f))

# ---------------------------------------------------------------- Fig 1: device schematic
f, ax = plt.subplots(figsize=(6.4, 2.6))
W = 1.0
def rect(x0, x1, y0, y1, c="#dbe8f8"):
    ax.add_patch(Polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], closed=True, fc=c, ec=INK2, lw=0.8))
rect(-4, -0.6, -1.0, 1.0, "#cfe3f7")             # water inlet (dispersed)
rect(-0.6, 0.0, -0.5, 0.5, "#cfe3f7")             # orifice approach
rect(0.0, 1.5, -0.5, 0.5, "#f3f2ee")              # orifice
rect(1.5, 6.0, -1.8, 1.8, "#f3f2ee")              # outlet (expansion)
rect(-1.8, -0.6, 1.0, 3.2, "#fbe3d6"); rect(-1.8, -0.6, -3.2, -1.0, "#fbe3d6")  # oil inlets
rect(-0.6, 0.0, 0.5, 1.0, "#fbe3d6"); rect(-0.6, 0.0, -1.0, -0.5, "#fbe3d6")
for x, r in ((2.5, 0.42), (3.7, 0.42), (4.9, 0.42)):
    ax.add_patch(Circle((x, 0), r, fc="#8fbbec", ec=C1, lw=1))
ax.add_patch(Polygon([[-0.6, -0.35], [0.9, -0.12], [1.3, 0], [0.9, 0.12], [-0.6, 0.35]], fc="#8fbbec", ec=C1, lw=1))
kw = dict(arrowstyle="-|>", mutation_scale=10, color=INK2, lw=1)
ax.add_patch(FancyArrowPatch((-3.8, 0), (-2.3, 0), **kw)); ax.text(-3.9, 0.25, "Water $Q_d$", fontsize=8)
ax.add_patch(FancyArrowPatch((-1.2, 3.1), (-1.2, 1.9), **kw)); ax.text(-1.05, 2.6, "Oil $Q_c$", fontsize=8)
ax.add_patch(FancyArrowPatch((-1.2, -3.1), (-1.2, -1.9), **kw))
ax.annotate("", (0, -0.75), (1.5, -0.75), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
ax.text(0.75, -1.15, "$L_{or}$", ha="center", fontsize=8)
ax.annotate("", (0.15, -0.5), (0.15, 0.5), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
ax.text(0.27, 0.62, "$W_{or}$", fontsize=8)
ax.annotate("", (5.8, -1.8), (5.8, 1.8), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
ax.text(5.9, 1.2, "$W_{out}$", fontsize=8)
ax.text(3.7, -0.85, "droplet $D$, rate $f$", ha="center", fontsize=8, color=C1)
ax.text(-4, -4.1, "depth $H$ (into page);  inputs: $W_{or}$, $H/W_{or}$, $W_{out}/W_{or}$, $L_{or}/W_{or}$, "
        "$W_{w}/W_{or}$, $W_{o}/W_{or}$, $\\phi=Q_c/Q_d$, Ca", fontsize=7.5, color=INK2)
ax.set_xlim(-4.2, 6.6); ax.set_ylim(-4.4, 3.4); ax.set_aspect("equal"); ax.axis("off")
save(f, "fig1_schematic.png")

# ---------------------------------------------------------------- Fig 2: replication parity
pr = pd.read_csv(os.path.join(RES_DIR, "replication_predictions.csv"))
rm = pd.read_csv(os.path.join(RES_DIR, "replication_metrics.csv"))
f, axs = plt.subplots(2, 2, figsize=(6.6, 6.2))
for i, tgt in enumerate(TGT):
    for j, reg in enumerate((1, 2)):
        ax = axs[i, j]; s = pr[(pr.target == tgt) & (pr.regime == reg)]
        ax.scatter(s.y, s.yhat, s=7, alpha=0.35, color=C1 if reg == 1 else C2, lw=0)
        lo, hi = min(s.y.min(), s.yhat.min()), max(s.y.max(), s.yhat.max())
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
        m = rm[(rm.target == tgt) & (rm.regime == reg)]
        p = PAPER[(tgt, reg)]
        ax.text(0.04, 0.96, f"This work: R² {m.R2.mean():.3f}, MAPE {m.MAPE.mean():.1f}%\n"
                            f"Paper:       R² {p['R2'][0]:.3f}, MAPE {p['MAPE'][0]:.1f}%",
                transform=ax.transAxes, va="top", fontsize=7.5, family="DejaVu Sans Mono")
        u = "µm" if tgt == "droplet_size" else "Hz"
        ax.set_xlabel(f"Measured {TGT[tgt].lower()} ({u})"); ax.set_ylabel(f"Predicted ({u})")
        ax.set_title(f"{TGT[tgt]} – {REG[reg]}"); style_axes(ax)
f.tight_layout(); save(f, "fig2_replication_parity.png")

# ---------------------------------------------------------------- Fig 3: replication bars
f, axs = plt.subplots(1, 2, figsize=(7, 2.6))
labels = ["D drip", "D jet", "f drip", "f jet"]
keys = [("droplet_size", 1), ("droplet_size", 2), ("generation_rate", 1), ("generation_rate", 2)]
x = np.arange(4)
for ax, met in zip(axs, ("R2", "MAPE")):
    pm = [PAPER[k][met][0] for k in keys]; ps = [PAPER[k][met][1] for k in keys]
    om = [rm[(rm.target == k[0]) & (rm.regime == k[1])][met].mean() for k in keys]
    os_ = [rm[(rm.target == k[0]) & (rm.regime == k[1])][met].std() for k in keys]
    ax.bar(x - 0.19, pm, 0.36, yerr=ps, color=C1, label="Paper (2021)", capsize=2, ecolor=INK2)
    ax.bar(x + 0.19, om, 0.36, yerr=os_, color=C2, label="This replication", capsize=2, ecolor=INK2)
    ax.set_xticks(x, labels); ax.set_title("R² (test)" if met == "R2" else "MAPE (%) (test)"); style_axes(ax)
    if met == "R2": ax.set_ylim(0.8, 1.0)
h, l = axs[0].get_legend_handles_labels()
f.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
f.tight_layout(); save(f, "fig3_replication_bars.png")

# ---------------------------------------------------------------- Fig 4: mass conservation
mc = pd.read_csv(os.path.join(RES_DIR, "mass_conservation.csv"))
f, axs = plt.subplots(1, 2, figsize=(7, 2.8))
ax = axs[0]
for ds, c, lab in ((998, C2, "998-pt file (nominal geometry)"), (888, C1, "888-pt regression file")):
    s = mc[mc.dataset == ds]
    ax.hist(s.R, bins=np.linspace(0.9, 1.12, 89), color=c, alpha=0.75, label=lab)
ax.set_yscale("log"); ax.set_xlabel(r"$R = f_{reported}\,\pi D^3/6\;/\;Q_d$"); ax.set_ylabel("Count")
ax.set_title("Mass-conservation ratio"); ax.legend(frameon=False, fontsize=7); style_axes(ax)
ax = axs[1]; s = mc[mc.dataset == 888]
f_mc = 6 * s.Qd_uL_min / 6e10 / (np.pi * (s.droplet_size * 1e-6) ** 3)
ax.scatter(f_mc, s.generation_rate, s=6, color=C1, alpha=0.5, lw=0)
ax.plot([1, 600], [1, 600], color=INK2, lw=1, ls="--")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$6Q_d/(\pi D^3)$ from Ca, $\phi$, D (Hz)"); ax.set_ylabel("Reported rate f (Hz)")
ax.set_title("888 points: f is fully determined by D"); style_axes(ax)
f.tight_layout(); save(f, "fig4_mass_conservation.png")

# ---------------------------------------------------------------- Fig 5: random vs device split
cm = pd.read_csv(os.path.join(RES_DIR, "comparison_metrics.csv"))
order = ["PL-2", "PL-7", "NN21", "NN24-MC", "XGB24-MC", "CONS-MC", "HYB-MC"]
nice = ["PL-2", "PL-7", "NN\n2021", "NN\n2024", "XGB\n2024", "Cons.\n2024", "Hybrid\n(ours)"]
f, axs = plt.subplots(2, 2, figsize=(7.4, 5.2))
for i, tgt in enumerate(TGT):
    for j, reg in enumerate((1, 2)):
        ax = axs[i, j]; x = np.arange(len(order))
        for k, (sch, c, lab) in enumerate((("random", C1, "Random 80/20 split"), ("device", C2, "Unseen device (GroupKFold)"))):
            g = cm[(cm.scheme == sch) & (cm.target == tgt) & (cm.regime == reg)].groupby("model").MAPE
            ax.bar(x + (k - 0.5) * 0.38, g.mean()[order], 0.36, yerr=g.std()[order], color=c, label=lab,
                   capsize=1.5, ecolor=INK2, error_kw=dict(lw=0.7))
        ax.set_xticks(x, nice, fontsize=7); ax.set_ylabel("MAPE (%)")
        ax.set_title(f"{TGT[tgt]} – {REG[reg]}"); style_axes(ax)
axs[0, 0].legend(frameon=False, fontsize=7)
f.tight_layout(); save(f, "fig5_random_vs_device.png")

# ---------------------------------------------------------------- Fig 6: conformal coverage
cf = pd.read_csv(os.path.join(RES_DIR, "conformal.csv"))
f, axs = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
mods = ["PL-7", "XGB24", "HYB"]; nm = ["PL-7", "XGB-2024", "Hybrid"]
for ax, reg in zip(axs, (1, 2)):
    x = np.arange(3)
    for k, (cal, c, lab) in enumerate((("naive", C1, "Random-row calibration"), ("device", C2, "Device-wise calibration"))):
        g = cf[(cf.regime == reg) & (cf.calibration == cal)].groupby("model").coverage
        ax.bar(x + (k - 0.5) * 0.38, g.mean()[mods], 0.36, yerr=g.std()[mods], color=c, label=lab,
               capsize=2, ecolor=INK2, error_kw=dict(lw=0.7))
    ax.axhline(90, color=INK, lw=1, ls="--"); ax.text(2.45, 91, "target 90%", fontsize=7, ha="right")
    ax.set_xticks(x, nm); ax.set_ylim(50, 100); ax.set_title(f"Coverage on unseen devices – {REG[reg]}")
    style_axes(ax)
axs[0].set_ylabel("Empirical coverage (%)")
h, l = axs[0].get_legend_handles_labels()
f.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
f.tight_layout(); save(f, "fig6_conformal.png")

# ---------------------------------------------------------------- Fig 7: SHAP
sh = pd.read_csv(os.path.join(RES_DIR, "shap_importance.csv"))
lab = {"flow_rate_ratio": "Flow-rate ratio φ", "capillary_number": "Capillary no. Ca",
       "orifice_size": "Orifice width", "aspect_ratio": "Depth H/W", "expansion_ratio": "Outlet W_out/W",
       "normalized_water_inlet": "Water inlet W_w/W", "normalized_oil_inlet": "Oil inlet W_o/W"}
f, axs = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
feat_order = sh[sh.regime == 1].sort_values("share_pct").feature.tolist()
for ax, reg, c in zip(axs, (1, 2), (C1, C2)):
    s = sh[sh.regime == reg].set_index("feature").loc[feat_order]
    ax.barh([lab[k] for k in feat_order], s.share_pct, color=c, height=0.6)
    for yv, v in enumerate(s.share_pct):
        ax.text(v + 0.8, yv, f"{v:.0f}%", va="center", fontsize=7, color=INK2)
    ax.set_title(f"Mean |SHAP| share – {REG[reg]}"); ax.set_xlim(0, 55); style_axes(ax)
f.tight_layout(); save(f, "fig7_shap.png")

# ---------------------------------------------------------------- Fig 8: learning curve
lc = pd.read_csv(os.path.join(RES_DIR, "learning_curve.csv"))
f, axs = plt.subplots(1, 2, figsize=(7, 2.7))
for ax, reg, band in zip(axs, (1, 2), ((250, 300), (200, 250))):
    ax.axvspan(*band, color=GRID, alpha=0.8, lw=0)
    ax.text(np.mean(band), 0.97, "paper's\n'enough data'", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=6.5, color=INK2)
    for m, c, nmm in (("NN21", C1, "NN-2021"), ("XGB24", C2, "XGB-2024"), ("PL-7", C3, "PL-7"), ("HYB", C4, "Hybrid")):
        g = lc[(lc.regime == reg) & (lc.model == m)].groupby("n_train").MAPE.mean()
        ax.plot(g.index, g.values, "-o", color=c, ms=3.5, lw=1.8, label=nmm)
    ax.set_xlabel("Training points"); ax.set_ylabel("Diameter MAPE (%)"); ax.set_title(REG[reg]); style_axes(ax)
axs[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper right", bbox_to_anchor=(1, 0.82))
f.tight_layout(); save(f, "fig8_learning_curve.png")

# ---------------------------------------------------------------- Fig 9: data overview
d = load(888)
Qc, Qd = flow_rates(d)
f, axs = plt.subplots(1, 2, figsize=(7, 2.7))
for reg, c in ((1, C1), (2, C2)):
    s = d[d.regime == reg]
    axs[0].scatter(s.capillary_number, s.flow_rate_ratio, s=8, color=c, alpha=0.55, lw=0, label=REG[reg])
    axs[1].scatter(s.droplet_size, s.generation_rate, s=8, color=c, alpha=0.55, lw=0, label=REG[reg])
axs[0].set_xscale("log"); axs[0].set_xlabel("Capillary number Ca"); axs[0].set_ylabel("Flow-rate ratio φ = Qc/Qd")
axs[0].set_title("Operating map (888 pts, 42 devices)")
axs[1].set_xlabel("Droplet diameter D (µm)"); axs[1].set_ylabel("Generation rate f (Hz)"); axs[1].set_yscale("log")
axs[1].set_title("Outputs")
for ax in axs: style_axes(ax)
axs[0].legend(frameon=False, fontsize=7)
f.tight_layout(); save(f, "fig9_data_overview.png")
print("figures written:", sorted(os.listdir(FIG_DIR)))

# ---------------------------------------------------------------- Fig 10 (slides): diameter only, wide
f, axs = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, reg in zip(axs, (1, 2)):
    x = np.arange(len(order))
    for k, (sch, c, lab) in enumerate((("random", C1, "Random 80/20 split (paper's test)"), ("device", C2, "Unseen device (leave-device-out)"))):
        g = cm[(cm.scheme == sch) & (cm.target == "droplet_size") & (cm.regime == reg)].groupby("model").MAPE
        ax.bar(x + (k - 0.5) * 0.38, g.mean()[order], 0.36, yerr=g.std()[order], color=c, label=lab,
               capsize=1.5, ecolor=INK2, error_kw=dict(lw=0.7))
    ax.set_xticks(x, nice, fontsize=9); ax.set_ylabel("Diameter MAPE (%)", fontsize=10)
    ax.set_title(f"Droplet diameter – {REG[reg]}", fontsize=12); style_axes(ax); ax.tick_params(labelsize=9)
h, l = axs[0].get_legend_handles_labels()
f.legend(h, l, frameon=False, fontsize=10, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07))
f.tight_layout(); save(f, "fig10_slide_diameter_split.png")
