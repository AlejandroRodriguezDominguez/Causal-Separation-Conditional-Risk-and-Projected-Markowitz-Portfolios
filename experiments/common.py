import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "font.family": "serif",
                     "mathtext.fontset": "cm"})
OUT_FIG = os.path.join(os.path.dirname(__file__), "..", "out", "figures")
OUT_TAB = os.path.join(os.path.dirname(__file__), "..", "out", "tables")
os.makedirs(OUT_FIG, exist_ok=True); os.makedirs(OUT_TAB, exist_ok=True)

def savefig(fig, name):
    p = os.path.join(OUT_FIG, name)
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("wrote", p)

def savetab(text, name):
    p = os.path.join(OUT_TAB, name)
    open(p, "w").write(text); print("wrote", p)
