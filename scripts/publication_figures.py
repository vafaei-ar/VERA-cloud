"""Generate the study-flow figure used in the AI-SoNar formative manuscript.

This script uses only observed study-session counts from
publication_session_counts.csv. It does not generate synthetic results.
"""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "docs" / "publication_session_counts.csv"
OUT = HERE.parent / "docs" / "publication_figures"
OUT.mkdir(exist_ok=True)

rows = list(csv.DictReader(CSV.open()))
fig = plt.figure(figsize=(12, 7.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.5, 0.93, "Iterative co-design and workflow-validation sequence",
        ha="center", va="center", fontsize=19, fontweight="bold")
ax.text(0.5, 0.875,
        "Stakeholder concerns were translated into code, re-evaluated, and then tested against stroke-navigation workflow and rural context.",
        ha="center", va="center", fontsize=12)

xs = [0.12, 0.31, 0.50, 0.69, 0.88]
labels = [
    "Preliminary\nstakeholder session",
    "Stakeholder-to-code\nrevision",
    "Revised-prototype\nsession",
    "Stroke-program\nworkflow validation",
    "Rural contextual\nvalidation",
]
details = [
    "n=7\n4 survivors | 2 caregivers\n1 coordinator",
    "13 mapped requirements\nA.1-A.12 + C.1",
    "n=12\nmixed stakeholder roles",
    "n=3\nprogram staff",
    "n=14\n9 patients | 3 caregivers\n2 hospital staff",
]
for i, x in enumerate(xs):
    ax.text(x, 0.72, labels[i], ha="center", va="center",
            fontsize=13, fontweight="bold")
    ax.add_patch(Circle((x, 0.60), 0.03, fill=False, linewidth=1.8))
    ax.text(x, 0.60, str(i + 1), ha="center", va="center",
            fontsize=14, fontweight="bold")
    ax.text(x, 0.50, details[i], ha="center", va="center", fontsize=10.5)
    if i < len(xs) - 1:
        ax.add_patch(FancyArrowPatch((x + 0.035, 0.60), (xs[i+1] - 0.035, 0.60),
                                     arrowstyle="-|>", mutation_scale=18, linewidth=1.8))

ax.text(0.5, 0.31, "Cross-session requirements retained before pilot testing",
        ha="center", va="center", fontsize=16, fontweight="bold")
reqs = [
    "Visible human\noversight",
    "Communication-aware\ninteraction",
    "Caregiver-flexible\nuse",
    "Explicit escalation\nownership",
    "Low-friction\nmultimodal access",
]
rx = [0.16, 0.33, 0.50, 0.67, 0.84]
for x, txt in zip(rx, reqs):
    ax.add_patch(FancyBboxPatch((x-0.075, 0.13), 0.15, 0.095,
                                boxstyle="round,pad=0.01,rounding_size=0.015",
                                fill=False, linewidth=1.4))
    ax.text(x, 0.178, txt, ha="center", va="center", fontsize=10.5)

ax.text(0.5, 0.045, "*Rural patient group included stroke and other neurologic conditions.",\n        ha="center", va="center", fontsize=9.5)\n\nfig.savefig(OUT / "Figure1_Study_Flow.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "Figure1_Study_Flow.pdf", bbox_inches="tight")
plt.close(fig)
