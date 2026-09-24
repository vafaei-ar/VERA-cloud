from pathlib import Path
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "docs" / "publication_graphical_abstract_findings.csv"
OUT = HERE.parent / "docs" / "publication_figures" / "Stroke_Graphical_Abstract.jpg"
OUT.parent.mkdir(exist_ok=True)

rows = list(csv.DictReader(CSV.open()))
by = {}
for row in rows:
    by.setdefault(row["section"], []).append(row["item"])

# Stroke allows no more than 7 inches square. Saving without bbox_inches keeps
# the output at exactly 7 x 7 inches (2100 x 2100 pixels at 300 dpi).
fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.text(0.5, 0.955, "Human-Supervised AI for Post-Stroke Navigation",
        ha="center", va="center", fontsize=17, fontweight="bold")

def box(x, y, w, h, title, items):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.012,rounding_size=0.018",
                               fill=False, linewidth=1.6))
    ax.text(x + w / 2, y + h - 0.035, title,
            ha="center", va="top", fontsize=13, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.45, "\n".join(items),
            ha="center", va="center", fontsize=12, linespacing=1.3)

box(0.07, 0.64, 0.37, 0.22, "Post-stroke transition", by["poststroke_context"])
box(0.07, 0.36, 0.37, 0.22, "Iterative co-design", by["codesign"])
box(0.07, 0.08, 0.37, 0.22, "Bounded clinical role", by["clinical_role"])
ax.add_patch(FancyArrowPatch((0.255, 0.64), (0.255, 0.585),
                             arrowstyle="-|>", mutation_scale=18, linewidth=1.6))
ax.add_patch(FancyArrowPatch((0.255, 0.36), (0.255, 0.305),
                             arrowstyle="-|>", mutation_scale=18, linewidth=1.6))

ax.text(0.73, 0.865, "Requirements before pilot",
        ha="center", va="center", fontsize=14, fontweight="bold")
for y, item in zip([0.70, 0.57, 0.44, 0.31, 0.18], by["requirements"]):
    ax.add_patch(FancyBboxPatch((0.52, y), 0.42, 0.085,
                               boxstyle="round,pad=0.008,rounding_size=0.014",
                               fill=False, linewidth=1.3))
    ax.text(0.73, y + 0.0425, item,
            ha="center", va="center", fontsize=12, wrap=True)
ax.add_patch(FancyArrowPatch((0.45, 0.47), (0.515, 0.47),
                             arrowstyle="-|>", mutation_scale=18, linewidth=1.6))

ax.text(0.5, 0.025, by["key_message"][0],
        ha="center", va="center", fontsize=12.5, fontweight="bold")
fig.savefig(OUT, dpi=300, facecolor="white")
plt.close(fig)
