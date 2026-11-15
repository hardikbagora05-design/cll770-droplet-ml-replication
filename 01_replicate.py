"""
Step 1 - Replication of Lashkaripour et al., Nat. Commun. 12, 25 (2021).

Protocol (as reported in the paper):
  * Regime classifier (dripping vs jetting): trained/tested on all 998 points.
  * Four regressors (diameter & rate, for each regime): trained/tested on the 888 bounded
    points (25-250 um, 5-500 Hz): 474 dripping, 414 jetting.
  * 80/20 random train/test split, repeated over 10 sessions, mean +/- SD reported.
  * Architectures / optimisers / batch sizes / early stopping copied from the authors' code.
Outputs: results/replication_metrics.csv, results/replication_predictions.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from common import (load, FEATURES_2021, CFG_SIZE, CFG_RATE1, CFG_RATE2, CFG_REGIME,
                    NNModel, metrics, RES_DIR)

N_SESSIONS = 10
CFG = {("droplet_size", 1): CFG_SIZE, ("droplet_size", 2): CFG_SIZE,
       ("generation_rate", 1): CFG_RATE1, ("generation_rate", 2): CFG_RATE2}


def run_session(s):
    rows, preds = [], []
    # --- regime classifier --------------------------------------------------------
    a = load(998)
    tr, te = train_test_split(a, test_size=0.2, random_state=s)
    clf = NNModel(CFG_REGIME, seed=s).fit(tr[FEATURES_2021].values, (tr.regime == 2).astype(float).values)
    p = clf.predict(te[FEATURES_2021].values) > 0.5
    rows.append(dict(session=s, target="regime", regime=0,
                     Accuracy=100 * np.mean(p == (te.regime == 2).values)))
    # --- regressors -----------------------------------------------------------------
    d = load(888)
    for reg in (1, 2):
        r = d[d.regime == reg]
        tr, te = train_test_split(r, test_size=0.2, random_state=s)
        for tgt in ("droplet_size", "generation_rate"):
            m = NNModel(CFG[(tgt, reg)], seed=s).fit(tr[FEATURES_2021].values, tr[tgt].values)
            yp = m.predict(te[FEATURES_2021].values)
            rows.append(dict(session=s, target=tgt, regime=reg, **metrics(te[tgt].values, yp)))
            preds.append(pd.DataFrame(dict(session=s, target=tgt, regime=reg, idx=te.index,
                                           y=te[tgt].values, yhat=yp)))
    return rows, preds


if __name__ == "__main__":
    out = Parallel(n_jobs=2)(delayed(run_session)(s) for s in range(N_SESSIONS))
    rows = [r for o in out for r in o[0]]
    preds = pd.concat([p for o in out for p in o[1]])
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RES_DIR, "replication_metrics.csv"), index=False)
    preds.to_csv(os.path.join(RES_DIR, "replication_predictions.csv"), index=False)
    summ = res.groupby(["target", "regime"]).agg(["mean", "std"]).drop(columns="session")
    print(summ.round(3).to_string())
