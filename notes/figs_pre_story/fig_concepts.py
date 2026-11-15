"""Figure 1: two parts, and two ways of combining them that keep different amounts of the parts."""
import matplotlib.pyplot as plt
from diagram import *

plt.rcParams["savefig.dpi"] = 400
A = "assets/"
avocado, armchair, on, blend = [plt.imread(A + f"im{k}.png")[..., :3] for k in (13, 14, 15, 16)]

F = Flow(6.5, 1.5)
S, s = 1.3, 0.63  # composite and part image heights
y = 0.02

F.head(0.03, 1.39, "a", "Two parts")
F.img(0.05, y + s + 0.04, avocado, s, edge=DGREY, lw=0.6)
F.img(0.05, y, armchair, s, edge=DGREY, lw=0.6)
F.text(0.9, y + 0.72, "combine", fs=5.8, c=DGREY)
F.arrow((0.72, y + 0.65), (1.08, y + 0.65), lw=0.9)

x = 1.12
F.head(x, 1.39, "b", "Role–filler composition")
F.img(x, y, on, S, edge=BLUE, lw=0.9)
tx = x + S + 0.07
F.text(tx, y + 1.17, "on(avocado, armchair)", fs=6.2, ha="left", fontweight="bold", c=BLUE)
F.text(tx, y + 0.97, "each part stays intact", fs=5.9, ha="left", c=DGREY)
F.text(tx, y + 0.85, "in its role", fs=5.9, ha="left", c=DGREY)
F.text(tx, y + 0.63, "both parts return", fs=5.9, ha="left", fontweight="bold")
F.text(tx, y + 0.31, r"$x\otimes\rho_1 + y\otimes\rho_2$", fs=6.6, ha="left", c=BLUE)
F.text(tx, y + 0.17, "one decomposition", fs=5.9, ha="left", c=DGREY)

x = 3.72
F.head(x, 1.39, "c", "Blending")
F.img(x, y, blend, S, edge=RED, lw=0.9)
tx = x + S + 0.07
F.text(tx, y + 1.17, "an armchair in the shape", fs=6.2, ha="left", fontweight="bold", c=RED)
F.text(tx, y + 1.06, "of an avocado", fs=6.2, ha="left", fontweight="bold", c=RED)
F.text(tx, y + 0.85, "the parts merge", fs=5.9, ha="left", c=DGREY)
F.text(tx, y + 0.63, "neither part returns", fs=5.9, ha="left", fontweight="bold")
F.text(tx, y + 0.31, r"$x + y$", fs=6.6, ha="left", c=RED)
F.text(tx, y + 0.17, "a line of decompositions", fs=5.9, ha="left", c=DGREY)

F.save("fig_parts")
print("ok")
