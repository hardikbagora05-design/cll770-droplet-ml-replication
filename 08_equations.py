"""Render report equations as PNG images (matplotlib mathtext) -> figures/eq*.png"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import FIG_DIR

EQ = {
 "eq1_navier_stokes": r"$\rho\left(\dfrac{\partial \mathbf{u}}{\partial t}+\mathbf{u}\cdot\nabla\mathbf{u}\right)=-\nabla p+\mu\nabla^{2}\mathbf{u},\qquad \nabla\cdot\mathbf{u}=0,\qquad \Delta p=\sigma\left(\dfrac{1}{R_1}+\dfrac{1}{R_2}\right)$",
 "eq2_buckingham": r"$\dfrac{D}{W_{or}}=F\left(\mathrm{Ca},\ \phi,\ \lambda,\ \dfrac{H}{W_{or}},\ \dfrac{W_{out}}{W_{or}},\ \dfrac{L_{or}}{W_{or}},\ \dfrac{W_w}{W_{or}},\ \dfrac{W_o}{W_{or}},\ \mathrm{Re},\ \mathrm{We},\ \mathrm{Bo}\right)$",
 "eq3_capillary": r"$\mathrm{Ca}=\dfrac{\mu_c\,Q_c\,W_w}{\sigma\,H\,W_o}\left(\dfrac{1}{W_{or}}-\dfrac{1}{2W_o}\right),\qquad Q_d=\dfrac{Q_c}{\phi}$",
 "eq4_powerlaw": r"$\ln\dfrac{D}{W_{or}}=c_0+\sum_{i=1}^{7}\,b_i\ln x_i\quad\Leftrightarrow\quad\dfrac{D}{W_{or}}=e^{c_0}\prod_{i=1}^{7}x_i^{\,b_i}$",
 "eq5_mass": r"$f=\dfrac{Q_d}{V_d}=\dfrac{6\,Q_d}{\pi D^{3}},\qquad \dfrac{\delta f}{f}\approx-3\,\dfrac{\delta D}{D}+\dfrac{\delta Q_d}{Q_d}$",
 "eq6_hybrid": r"$\ln\dfrac{\hat{D}}{W_{or}}=P_{7}(\mathbf{x})+g_{\mathrm{XGB}}(\mathbf{x}),\qquad P_{7}(\mathbf{x})=c_0+\sum_{i}b_i\ln x_i,\qquad \hat{f}=\dfrac{6\,Q_d}{\pi\hat{D}^{3}}$",
 "eq7_qc": r"$Q_c=\dfrac{\mathrm{Ca}\,\sigma\,H\,W_o}{\mu_c\,W_w\left(\dfrac{1}{W_{or}}-\dfrac{1}{2W_o}\right)},\qquad \sigma=5\ \mathrm{mN\,m^{-1}},\quad \mu_c=57.2\ \mathrm{mPa\,s}$",
 "eq8_ratio": r"$R=\dfrac{f_{\mathrm{reported}}\ \pi D^{3}/6}{Q_d};\qquad R\equiv1\ \Leftrightarrow\ f\ \mathrm{was\ computed\ from}\ D$",
}
for name, tex in EQ.items():
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=15, color="#0b0b0b")
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=200, bbox_inches="tight", pad_inches=0.06,
                facecolor="white")
    plt.close(fig)
print("equations written")

# Pad every equation to a common canvas width so all render at the same font size in the doc
from PIL import Image
import glob
W = 1500
for f in sorted(glob.glob(os.path.join(FIG_DIR, "eq*.png"))):
    im = Image.open(f).convert("RGB")
    c = Image.new("RGB", (W, im.height + 16), "white")
    c.paste(im, ((W - im.width) // 2, 8))
    c.save(f)
