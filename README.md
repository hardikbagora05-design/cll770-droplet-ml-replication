# Machine-learning prediction of droplet size in flow-focusing microfluidics — replication & extension

**CLL770 – Introduction to Microfluidics and Applications (IIT Delhi) · Course project**

This repository reproduces the predictive models of

> A. Lashkaripour *et al.*, "Machine learning enables design automation of microfluidic flow-focusing droplet generation", *Nature Communications* **12**, 25 (2021). https://doi.org/10.1038/s41467-020-20284-z

and extends them using ideas from the authors' follow-up paper

> A. Lashkaripour *et al.*, "Design automation of microfluidic single and double emulsion droplets with machine learning", *Nature Communications* **15**, 83 (2024). https://doi.org/10.1038/s41467-023-44068-3

All experimental data are the authors' open data (998 experiments on 43 micromilled devices; DI water in NF350 mineral oil + 5 % Span 80), from https://github.com/CIDARLAB/DAFD.

## Headline results

| | Paper (2021) | This replication |
|---|---|---|
| Diameter, dripping – R² / MAPE | 0.893 / 11.2 % | 0.900 / 10.1 % |
| Diameter, jetting – R² / MAPE | 0.966 / 4.8 % | 0.965 / 4.8 % |
| Rate, dripping – R² / MAPE | 0.889 / 33.5 % | 0.898 / 38.4 % |
| Rate, jetting – R² / MAPE | 0.956 / 15.8 % | 0.959 / 14.1 % |
| Regime classifier accuracy | 95.1 % | 95.4 % |

Extensions (see `results/summary_tables.md`):

1. **Mass conservation audit.** In the 888-point regression set, 99.9 % of reported generation rates satisfy f = 6Q<sub>d</sub>/(πD³) to within 0.2 %. The rate is computed from the measured diameter, not measured separately, so a separate rate network is redundant, and rate error ≈ 3 × diameter error.
2. **Unseen-device validation (GroupKFold by device).** The random 80/20 split leaks device information. On unseen devices, the 2021 network's diameter MAPE rises from 10.6 % to 17.0 % (dripping) and from 5.3 % to 10.9 % (jetting). A 7-term log-linear scaling law (12.9 % / 5.4 %) and the 2024-style XGBoost and consensus models generalise better.
3. **Physics-informed hybrid model (this work)**, a scaling-law backbone plus a shallow boosted-tree residual. It has the best interpolation accuracy (7.1 % / 3.7 % diameter MAPE) and beats the 2021 network even when trained on only 20 % of the data.
4. **Calibrated uncertainty** (split-conformal). Calibrating on random rows under-covers new devices (69–79 % instead of 90 %). Device-wise calibration restores 85–89 %.
5. **Physics from data.** Fitted exponents: dripping D/W ∝ φ<sup>−0.36</sup>Ca<sup>−0.21</sup>; jetting is Ca-independent with D ∝ φ<sup>−0.41</sup>(W<sub>out</sub>/W)<sup>+0.41</sup>. SHAP analysis agrees.

## Repository layout

```
data/        ExperimentalResults_998.csv (all points), ExperimentalResults_888.csv (regression subset)
src/         common.py (data, physics, PyTorch port of the authors' Keras networks)
             01_replicate.py            replication of the 2021 paper (Table 1 metrics)
             02_mass_conservation.py    E1 mass-conservation audit
             03_model_comparison.py     E2–E4 scaling laws, 2024 models, hybrid, random vs unseen-device
             04_uncertainty.py          E5 split-conformal prediction intervals
             05_importance_learning_curve.py  E6 SHAP, E7 learning curve
             06_figures.py, 07_tables.py, 08_equations.py (report equations as images)
results/     all metrics as CSV + summary_tables.md
figures/     all report / slide figures (PNG)
```

## How to run

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
bash run_all.sh            # ~12 min on a 2-core laptop CPU
```

Each script can also be run on its own, e.g. `python src/01_replicate.py`.

## Notes on fidelity

* The authors' networks were written in Keras 2.2 / TensorFlow 1.12. They are ported to PyTorch with identical layer sizes, activations, Glorot initialisation, optimiser (Nadam, lr = 0.002; RMSprop for the classifier), batch sizes, epochs and early-stopping rule (see `src/common.py`).
* Ca → flow-rate conversion uses the authors' own formula (`ModelHelper.calculate_formulaic_relations`, σ = 5 mN/m, μ<sub>oil</sub> = 57.2 mPa s).
* "Device" = unique as-fabricated geometry (42 groups in the 888-point file).
* The 2024 paper's double-emulsion data (on OSF) are not used.

## Data licence / attribution

Experimental data © the original authors (Lashkaripour *et al.*, CIDAR Lab), redistributed here unchanged for academic reproducibility. Please cite the papers above if you use them. Code in `src/` is released under the MIT licence.
