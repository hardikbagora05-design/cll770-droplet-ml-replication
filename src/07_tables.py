"""Step 7 - Summary tables (markdown) used in the report: results/summary_tables.md"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from common import RES_DIR

L = []
pm = lambda g, c, n=1: f"{g[c]['mean']:.{n}f} ± {g[c]['std']:.{n}f}"

rm = pd.read_csv(os.path.join(RES_DIR, "replication_metrics.csv"))
L.append("## T1 Replication (10 sessions, 80/20)\n\n| Target | Regime | R² | RMSE | MAE | MAPE (%) |\n|---|---|---|---|---|---|")
for (t, r), g in rm[rm.target != "regime"].groupby(["target", "regime"]):
    g = g[["R2", "RMSE", "MAE", "MAPE"]].agg(["mean", "std"])
    L.append(f"| {t} | {r} | {pm(g,'R2',3)} | {pm(g,'RMSE')} | {pm(g,'MAE')} | {pm(g,'MAPE')} |")
a = rm[rm.target == "regime"].Accuracy
L.append(f"\nRegime classifier accuracy: {a.mean():.1f} ± {a.std():.1f} %\n")

cm = pd.read_csv(os.path.join(RES_DIR, "comparison_metrics.csv"))
order = ["PL-2", "PL-7", "NN21", "NN21-MC", "NN24-MC", "XGB24-MC", "CONS-MC", "HYB-MC"]
for tgt in ("droplet_size", "generation_rate"):
    L.append(f"## T2 {tgt}: MAPE (%) and R², random vs unseen-device\n")
    L.append("| Model | Drip random | Drip device | Jet random | Jet device | R² drip dev | R² jet dev |\n|---|---|---|---|---|---|---|")
    for m in order:
        cells = []
        for reg in (1, 2):
            for sch in ("random", "device"):
                g = cm[(cm.model == m) & (cm.target == tgt) & (cm.regime == reg) & (cm.scheme == sch)][["MAPE"]].agg(["mean", "std"])
                cells.append(pm(g, "MAPE"))
        r2 = [cm[(cm.model == m) & (cm.target == tgt) & (cm.regime == reg) & (cm.scheme == "device")].R2.mean() for reg in (1, 2)]
        L.append(f"| {m} | " + " | ".join(cells) + f" | {r2[0]:.3f} | {r2[1]:.3f} |")
    L.append("")

cc = pd.read_csv(os.path.join(RES_DIR, "classifier_cv.csv"))
L.append("## T3 Regime classifier accuracy (%)\n\n| Model | Random | Unseen device |\n|---|---|---|")
for m, g in cc.groupby("model"):
    L.append(f"| {m} | " + " | ".join(pm(g[g.scheme == s][['Accuracy']].agg(['mean', 'std']), 'Accuracy')
                                     for s in ("random", "device")) + " |")

cf = pd.read_csv(os.path.join(RES_DIR, "conformal.csv"))
L.append("\n## T4 Conformal 90% intervals on unseen devices\n\n| Regime | Calibration | Model | Coverage (%) | ±D (%) | ±f (%) |\n|---|---|---|---|---|---|")
for (r, c, m), g in cf.groupby(["regime", "calibration", "model"]):
    L.append(f"| {r} | {c} | {m} | {g.coverage.mean():.1f} ± {g.coverage.std():.1f} | {g.D_halfwidth_pct.mean():.0f} | {g.f_halfwidth_pct.mean():.0f} |")

lc = pd.read_csv(os.path.join(RES_DIR, "learning_curve.csv"))
L.append("\n## T5 Learning curve (diameter MAPE %, mean of 3 seeds)\n")
t = lc.groupby(["regime", "model", "n_train"]).MAPE.mean().round(1).reset_index()
for r in (1, 2):
    p = t[t.regime == r].pivot(index="n_train", columns="model", values="MAPE")
    L.append(f"Regime {r}\n\n" + p.to_markdown() + "\n")

sh = pd.read_csv(os.path.join(RES_DIR, "shap_importance.csv"))
L.append("## T6 SHAP share (%)\n\n" + sh.pivot(index="feature", columns="regime", values="share_pct").round(1).to_markdown())

open(os.path.join(RES_DIR, "summary_tables.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

# ---------------- T7 fitted PL-7 exponents (all data, 1000 bootstrap) ----------------
import numpy as np
from sklearn.linear_model import LinearRegression
from common import load, FEATURES_2024
d = load(888); rows = []
L.append("\n## T7 Fitted scaling law  ln(D/W_or) = c0 + sum b_i ln(x_i)   (95% bootstrap CI)\n")
L.append("| Input | Dripping exponent | Jetting exponent |\n|---|---|---|")
coef = {}
rng = np.random.default_rng(0)
for reg in (1, 2):
    r = d[d.regime == reg]; X = np.log(r[FEATURES_2024].values); y = np.log(r.droplet_size / r.orifice_size).values
    b = LinearRegression().fit(X, y).coef_
    bs = np.array([LinearRegression().fit(X[i], y[i]).coef_ for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))])
    coef[reg] = [f"{b[k]:+.3f} [{np.percentile(bs[:,k],2.5):+.3f}, {np.percentile(bs[:,k],97.5):+.3f}]" for k in range(len(b))]
for k, f in enumerate(FEATURES_2024):
    L.append(f"| {f} | {coef[1][k]} | {coef[2][k]} |")
for reg in (1, 2):
    r = d[d.regime == reg]; X = np.log(r[["flow_rate_ratio", "capillary_number"]].values)
    lr = LinearRegression().fit(X, np.log(r.droplet_size / r.orifice_size))
    L.append(f"\nPL-2 regime {reg}: D/W = {np.exp(lr.intercept_):.3f} * phi^{lr.coef_[0]:+.3f} * Ca^{lr.coef_[1]:+.3f}")
open(os.path.join(RES_DIR, "summary_tables.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L[-14:]))

# ---------------- T8 physical inconsistency of separate D and f networks (NN21) ----------------
oof = pd.read_csv(os.path.join(RES_DIR, "comparison_oof.csv"))
a = oof[oof.model == "NN21"].reset_index(drop=True); b = oof[oof.model == "NN21-MC"].reset_index(drop=True)
gap = 100 * np.abs(a.fhat / b.fhat - 1)
L.append("\n## T8 |f_rate-net / f_mass-conservation(D_size-net) - 1| (%) for the 2021 networks\n\n| Scheme | Regime | Median | 90th pct | Share > 20 % |\n|---|---|---|---|---|")
for s in ("random", "device"):
    for reg in (1, 2):
        m = (a.scheme == s) & (a.regime == reg)
        L.append(f"| {s} | {reg} | {np.median(gap[m]):.1f} | {np.percentile(gap[m], 90):.1f} | {100*(gap[m] > 20).mean():.0f} |")
open(os.path.join(RES_DIR, "summary_tables.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L[-6:]))
