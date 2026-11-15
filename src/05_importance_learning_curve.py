"""
Step 5 (Extensions E6-E7)
  E6  Feature attribution - TreeSHAP (XGBoost pred_contribs) for the 2024-style diameter model,
      trained on all data of each regime. Mean |SHAP| per input on log-free D/Dh scale.
  E7  Learning curve - re-test the 2021 claim that ~250-300 dripping / 200-250 jetting points
      are enough. Train on 20-100 % of the 80 % training set (random split, 3 seeds),
      report diameter MAPE on the fixed 20 % test set.
Outputs: results/shap_importance.csv, results/learning_curve.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from common import load, FEATURES_2021, FEATURES_2024, CFG_SIZE, NNModel, metrics, hydraulic_diameter, RES_DIR
import importlib
mc = importlib.import_module("03_model_comparison")

d = load(888)
# ---------------- SHAP ----------------
rows = []
for reg in (1, 2):
    r = d[d.regime == reg]
    m = xgb.XGBRegressor(**mc.XGB_PARAMS).fit(r[FEATURES_2024].values, r.droplet_size / hydraulic_diameter(r))
    contrib = m.get_booster().predict(xgb.DMatrix(r[FEATURES_2024].values), pred_contribs=True)[:, :-1]
    imp = np.abs(contrib).mean(0)
    for f, v in zip(FEATURES_2024, imp):
        rows.append(dict(regime=reg, feature=f, mean_abs_shap=v, share_pct=100 * v / imp.sum()))
shap = pd.DataFrame(rows)
shap.to_csv(os.path.join(RES_DIR, "shap_importance.csv"), index=False)
print(shap.pivot(index="feature", columns="regime", values="share_pct").round(1))


# ---------------- learning curve ----------------
def lc_job(reg, seed, frac):
    r = d[d.regime == reg]
    tr, te = train_test_split(r, test_size=0.2, random_state=seed)
    sub = tr.sample(frac=frac, random_state=seed)
    out = []
    nn = NNModel(CFG_SIZE, seed).fit(sub[FEATURES_2021].values, sub.droplet_size.values)
    out.append(("NN21", nn.predict(te[FEATURES_2021].values)))
    x = xgb.XGBRegressor(**mc.XGB_PARAMS).fit(sub[FEATURES_2024].values, sub.droplet_size / hydraulic_diameter(sub))
    out.append(("XGB24", x.predict(te[FEATURES_2024].values) * hydraulic_diameter(te)))
    out.append(("HYB", mc.hybrid(sub, te, seed)))
    out.append(("PL-7", mc.power_law(sub, te, FEATURES_2024)[0]))
    return [dict(regime=reg, seed=seed, frac=frac, n_train=len(sub), model=m,
                 MAPE=metrics(te.droplet_size, p)["MAPE"]) for m, p in out]


jobs = [(reg, s, f) for reg in (1, 2) for s in range(3) for f in (0.2, 0.35, 0.5, 0.65, 0.8, 1.0)]
lc = pd.DataFrame([r for o in Parallel(n_jobs=2)(delayed(lc_job)(*j) for j in jobs) for r in o])
lc.to_csv(os.path.join(RES_DIR, "learning_curve.csv"), index=False)
print(lc.groupby(["regime", "model", "frac"]).agg(n=("n_train", "mean"), MAPE=("MAPE", "mean")).round(1).unstack(0).to_string())
