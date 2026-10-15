"""Plot every predeclared Gaussian width; baby x2, one seed only."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=ROOT/"runs/gaussian_baby_x2_v1/summary.json")
    result=json.loads(parser.parse_args().input.read_text())
    assert result["passed"] and result["complete"] and len(result["per_width"])==2
    plt.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman","DejaVu Serif"],
        "font.size":10, "legend.fontsize":9, "legend.frameon":False,
        "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.25))
    for ax,row in zip(axes,result["per_width"]):
        for key,label,color,style,marker in [
            ("gaussian_consistent","Ours: Gaussian variance law","#D55E00","-","o"),
            ("gaussian_aa","Gaussian AA stretch","#0072B2","-.","^")]:
            y=np.array(row["psnr_y"][key]); assert y.shape==(10,) and np.isfinite(y).all()
            ax.plot(result["m"],y,label=label,color=color,linestyle=style,marker=marker,
                    markersize=2.4,linewidth=1.55)
        ax.axhline(row["baseline_y"],color=".55",ls="--",lw=1.,label="Bicubic interpolation")
        ax.set_title(r"Anchor $\sigma="+str(row["anchor_sigma"])+r"$ pixels",fontsize=10)
        ax.set_xlabel("Steps"); ax.set_xticks([1,2,4,6,8,10]); ax.set_xlim(.8,10.2)
        ax.grid(axis="y",alpha=.2); ax.set_ylabel("Y-PSNR (dB)")
    fig.suptitle("Baby ×2 — Gaussian pilot, one image only",fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2,
               bbox_to_anchor=(.5,.005))
    fig.subplots_adjust(left=.085,right=.98,bottom=.3,top=.78,wspace=.32)
    for suffix in ("png","pdf"):
        fig.savefig(ROOT/f"figures/fig_gaussian_baby_x2_pilot.{suffix}",dpi=300,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__": main()
