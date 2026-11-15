"""
Step 2 (Extension E1) - Is the reported generation rate an independent measurement?

For every point we compute the dispersed flow rate Qd from (Ca, phi, geometry) with the
authors' own Ca definition, then the mass-conservation ratio
        R = f_reported * (pi D^3 / 6) / Qd .
If f were measured independently (e.g. by high-speed imaging), R would scatter by
~5-20 % (diameter error enters as D^3).  R ~ 1 to within 0.1 % means f was *computed*
from D, so f carries no information beyond D, and a separate rate network is redundant.
Outputs: results/mass_conservation.csv, results/mass_conservation_summary.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from common import load, flow_rates, RES_DIR

rows = []
for which in (888, 998):
    d = load(which)
    Qc, Qd = flow_rates(d)
    R = d.generation_rate.values * np.pi * (d.droplet_size.values * 1e-6) ** 3 / 6 / Qd
    out = d[["regime", "droplet_size", "generation_rate", "capillary_number", "flow_rate_ratio"]].copy()
    out["Qc_mL_h"] = Qc * 3.6e9
    out["Qd_uL_min"] = Qd * 6e10
    out["R"] = R
    out["dataset"] = which
    rows.append(out)
    dev = np.abs(R - 1)
    print(f"{which}: median R={np.median(R):.4f}, 1-99 pct = {np.percentile(R,1):.4f}-{np.percentile(R,99):.4f}, "
          f"|R-1|<0.2%: {100*np.mean(dev<0.002):.1f}%, <1%: {100*np.mean(dev<0.01):.1f}%, <5%: {100*np.mean(dev<0.05):.1f}%")
mc = pd.concat(rows)
mc.to_csv(os.path.join(RES_DIR, "mass_conservation.csv"), index=False)
summ = mc.groupby("dataset").R.describe(percentiles=[.01, .25, .5, .75, .99])
summ["pct_within_0.2pct"] = mc.groupby("dataset").R.apply(lambda r: 100 * np.mean(np.abs(r - 1) < 0.002))
summ["pct_within_1pct"] = mc.groupby("dataset").R.apply(lambda r: 100 * np.mean(np.abs(r - 1) < 0.01))
summ.to_csv(os.path.join(RES_DIR, "mass_conservation_summary.csv"))
print(summ.round(4).to_string())
# Flow-rate ranges (useful for the report)
m = mc[mc.dataset == 888]
print("Qc range (mL/h):", m.Qc_mL_h.min().round(3), m.Qc_mL_h.max().round(2),
      "| Qd range (uL/min):", m.Qd_uL_min.min().round(3), m.Qd_uL_min.max().round(2))
