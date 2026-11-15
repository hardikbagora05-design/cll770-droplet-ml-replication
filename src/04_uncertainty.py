"""
Step 4 (Extension E5) - Calibrated uncertainty for new devices via split-conformal prediction.

For each outer leave-device-out fold (GroupKFold(5) x 2 repeats) the training devices are
split again into a proper-training part (75 %) and a calibration part (25 %). The 90 %
quantile q of |log(D/Dhat)| on the calibration part gives the interval
        D in [Dhat * exp(-q), Dhat * exp(+q)],
and, through mass conservation (f ~ D^-3), f in [fhat * exp(-3q), fhat * exp(+3q)].
Two calibration strategies are compared:
  naive  - calibration rows drawn at random (devices leak between train and calibration)
  device - calibration rows are whole held-out devices (matches the deployment situation)
Target coverage on the unseen test devices = 90 %.
Output: results/conformal.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from xgboost import XGBRegressor
from common import load, FEATURES_2024, hydraulic_diameter, RES_DIR
import importlib
mc = importlib.import_module("03_model_comparison")

ALPHA = 0.10


def predict(model, tr, te):
    if model == "HYB":
        return mc.hybrid(tr, te)
    if model == "PL-7":
        return mc.power_law(tr, te, FEATURES_2024)[0]
    x = XGBRegressor(**mc.XGB_PARAMS).fit(tr[FEATURES_2024].values, tr.droplet_size / hydraulic_diameter(tr))
    return x.predict(te[FEATURES_2024].values) * hydraulic_diameter(te)


rows = []
d = load(888)
for reg in (1, 2):
    r = d[d.regime == reg]
    for rep in range(2):
        for k, (a, b) in enumerate(GroupKFold(5, shuffle=True, random_state=rep).split(r, groups=r.device)):
            trall, te = r.iloc[a], r.iloc[b]
            for cal_mode in ("naive", "device"):
                if cal_mode == "naive":
                    ptr, cal = train_test_split(trall, test_size=0.25, random_state=k)
                else:
                    gi, ci = next(GroupShuffleSplit(1, test_size=0.25, random_state=k)
                                  .split(trall, groups=trall.device))
                    ptr, cal = trall.iloc[gi], trall.iloc[ci]
                for model in ("PL-7", "XGB24", "HYB"):
                    q = np.quantile(np.abs(np.log(cal.droplet_size / predict(model, ptr, cal))),
                                    np.ceil((len(cal) + 1) * (1 - ALPHA)) / len(cal))
                    Dhat = predict(model, ptr, te)
                    inside = np.abs(np.log(te.droplet_size / Dhat)) <= q
                    rows.append(dict(regime=reg, rep=rep, fold=k, calibration=cal_mode, model=model,
                                     q=q, coverage=100 * inside.mean(),
                                     D_halfwidth_pct=100 * (np.exp(q) - 1),
                                     f_halfwidth_pct=100 * (np.exp(3 * q) - 1)))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(RES_DIR, "conformal.csv"), index=False)
print(res.groupby(["regime", "calibration", "model"])[["coverage", "D_halfwidth_pct", "f_halfwidth_pct"]]
      .agg(["mean", "std"]).round(1).to_string())
