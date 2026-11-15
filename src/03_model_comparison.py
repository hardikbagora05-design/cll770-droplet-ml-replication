"""
Step 3 (Extensions E2-E4) - Physics-consistent prediction, scaling laws, 2024-style models,
and *leave-device-out* validation.

Models (one per regime):
  PL-2     classical power law   D/W = k * phi^a * Ca^b                      (+ mass conservation)
  PL-7     log-linear power law in all 7 dimensionless/2024 inputs           (+ mass conservation)
  NN21     paper's networks: separate diameter and rate networks (direct rate)
  NN21-MC  paper's diameter network, rate from conservation of mass
  NN24-MC  2024 network (512,16; Adam 3e-4; L2 1e-3) on D/Dh, rate from mass conservation
  XGB24-MC 2024 boosted trees (100 trees, lr 0.3, depth 6, lambda 1) on D/Dh, + mass cons.
  CONS-MC  2024 consensus = mean(NN24, XGB24) diameter, + mass conservation
  HYB-MC   (this work) PL-7 backbone + shallow XGB residual correction, + mass conservation

Validation schemes:
  random   80/20 random split x 5 sessions          (what both papers report)
  device   GroupKFold(5) by device geometry x 2 repeats  (unseen device = real design use-case)
Also: regime classifier (NN21, XGB) under both schemes.
Outputs: results/comparison_metrics.csv, results/comparison_oof.csv, results/classifier_cv.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GroupKFold
from xgboost import XGBRegressor, XGBClassifier
from common import (load, FEATURES_2021, FEATURES_2024, CFG_SIZE, CFG_RATE1, CFG_RATE2, CFG_2024,
                    CFG_REGIME, NNModel, metrics, rate_from_diameter, hydraulic_diameter, RES_DIR)

XGB_PARAMS = dict(n_estimators=100, learning_rate=0.3, max_depth=6, reg_lambda=1.0,
                  min_child_weight=1, n_jobs=1)


def power_law(tr, te, cols):
    X = lambda d: np.log(d[cols].values)
    y = np.log(tr.droplet_size.values / tr.orifice_size.values)
    lr = LinearRegression().fit(X(tr), y)
    return np.exp(lr.predict(X(te))) * te.orifice_size.values, lr


def fit_predict(tr, te, reg, seed):
    """Return dict model -> (D_pred, f_pred) on te."""
    out = {}
    out["PL-2"] = power_law(tr, te, ["flow_rate_ratio", "capillary_number"])[0]
    out["PL-7"] = power_law(tr, te, FEATURES_2024)[0]
    nnD = NNModel(CFG_SIZE, seed).fit(tr[FEATURES_2021].values, tr.droplet_size.values)
    Dnn = nnD.predict(te[FEATURES_2021].values)
    nnF = NNModel(CFG_RATE1 if reg == 1 else CFG_RATE2, seed).fit(tr[FEATURES_2021].values,
                                                                  tr.generation_rate.values)
    res = {k: (v, rate_from_diameter(te, v)) for k, v in out.items()}
    res["NN21"] = (Dnn, nnF.predict(te[FEATURES_2021].values))
    res["NN21-MC"] = (Dnn, rate_from_diameter(te, Dnn))
    Dh_tr, Dh_te = hydraulic_diameter(tr), hydraulic_diameter(te)
    nn24 = NNModel(CFG_2024, seed).fit(tr[FEATURES_2024].values, tr.droplet_size.values / Dh_tr)
    D24 = nn24.predict(te[FEATURES_2024].values) * Dh_te
    xgb = XGBRegressor(random_state=seed, **XGB_PARAMS).fit(tr[FEATURES_2024].values,
                                                            tr.droplet_size.values / Dh_tr)
    Dx = xgb.predict(te[FEATURES_2024].values) * Dh_te
    for k, v in (("NN24-MC", D24), ("XGB24-MC", Dx), ("CONS-MC", 0.5 * (D24 + Dx))):
        res[k] = (v, rate_from_diameter(te, v))
    res["HYB-MC"] = (hybrid(tr, te, seed), None)
    res["HYB-MC"] = (res["HYB-MC"][0], rate_from_diameter(te, res["HYB-MC"][0]))
    return res


HYB_PARAMS = dict(n_estimators=300, learning_rate=0.1, max_depth=3, reg_lambda=1.0, n_jobs=1)


def hybrid(tr, te, seed=0):
    """Proposed physics-informed hybrid: PL-7 log-linear scaling law as backbone, plus a
    shallow boosted-tree correction learned on the log-residuals. Always positive D."""
    X = lambda d: np.log(d[FEATURES_2024].values)
    y = np.log(tr.droplet_size.values / tr.orifice_size.values)
    lr = LinearRegression().fit(X(tr), y)
    xg = XGBRegressor(random_state=seed, **HYB_PARAMS).fit(tr[FEATURES_2024].values, y - lr.predict(X(tr)))
    return np.exp(lr.predict(X(te)) + xg.predict(te[FEATURES_2024].values)) * te.orifice_size.values


def job(scheme, rep, fold, reg, tr_idx, te_idx):
    d = load(888); r = d[d.regime == reg]
    tr, te = r.loc[tr_idx], r.loc[te_idx]
    res = fit_predict(tr, te, reg, seed=100 * rep + fold)
    rows, oof = [], []
    for mdl, (D, f) in res.items():
        for tgt, y, p in (("droplet_size", te.droplet_size.values, D),
                          ("generation_rate", te.generation_rate.values, f)):
            rows.append(dict(scheme=scheme, rep=rep, fold=fold, regime=reg, model=mdl, target=tgt,
                             **metrics(y, p)))
        oof.append(pd.DataFrame(dict(scheme=scheme, rep=rep, fold=fold, regime=reg, model=mdl, idx=te.index,
                                     device=te.device.values, D=te.droplet_size.values, Dhat=D,
                                     f=te.generation_rate.values, fhat=f)))
    return rows, oof


def clf_job(scheme, rep, fold, tr_idx, te_idx):
    a = load(998); tr, te = a.loc[tr_idx], a.loc[te_idx]
    ytr, yte = (tr.regime == 2).astype(float).values, (te.regime == 2).values
    nn = NNModel(CFG_REGIME, seed=100 * rep + fold).fit(tr[FEATURES_2021].values, ytr)
    xg = XGBClassifier(random_state=rep, **XGB_PARAMS).fit(tr[FEATURES_2021].values, ytr)
    return [dict(scheme=scheme, rep=rep, fold=fold, model=m,
                 Accuracy=100 * np.mean((p > 0.5) == yte))
            for m, p in (("NN21", nn.predict(te[FEATURES_2021].values)),
                         ("XGB", xg.predict_proba(te[FEATURES_2021].values)[:, 1]))]


if __name__ == "__main__":
    d = load(888)
    jobs = []
    for reg in (1, 2):
        r = d[d.regime == reg]
        for s in range(5):
            tr, te = train_test_split(r.index, test_size=0.2, random_state=s)
            jobs.append(("random", s, 0, reg, tr, te))
        for rep in range(2):
            gkf = GroupKFold(n_splits=5, shuffle=True, random_state=rep)
            for k, (tr, te) in enumerate(gkf.split(r, groups=r.device)):
                jobs.append(("device", rep, k, reg, r.index[tr], r.index[te]))
    a = load(998)
    cjobs = []
    for s in range(5):
        tr, te = train_test_split(a.index, test_size=0.2, random_state=s)
        cjobs.append(("random", s, 0, tr, te))
    for rep in range(2):
        gkf = GroupKFold(n_splits=5, shuffle=True, random_state=rep)
        for k, (tr, te) in enumerate(gkf.split(a, groups=a.device)):
            cjobs.append(("device", rep, k, a.index[tr], a.index[te]))

    out = Parallel(n_jobs=2, verbose=5)(delayed(job)(*j) for j in jobs)
    cout = Parallel(n_jobs=2)(delayed(clf_job)(*j) for j in cjobs)
    res = pd.DataFrame([r for o in out for r in o[0]])
    oof = pd.concat([p for o in out for p in o[1]])
    cres = pd.DataFrame([r for o in cout for r in o])
    res.to_csv(os.path.join(RES_DIR, "comparison_metrics.csv"), index=False)
    oof.to_csv(os.path.join(RES_DIR, "comparison_oof.csv"), index=False)
    cres.to_csv(os.path.join(RES_DIR, "classifier_cv.csv"), index=False)
    pd.set_option("display.width", 200)
    print(res.groupby(["scheme", "regime", "target", "model"])[["R2", "MAPE", "MAE"]].mean().round(3).to_string())
    print(cres.groupby(["scheme", "model"]).Accuracy.agg(["mean", "std"]).round(2))
