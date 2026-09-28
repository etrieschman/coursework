# %%
import numpy as np
import cvxpy as cp

# %%
# ===================
# Problem 6
# ===================
# Part 2
# -------------------
Y = np.array([[-8.5, 2.5, 5], [2.5, -8.75, 5], [5, 5, -35]])
Y_block = np.concatenate(
    [
        np.concatenate([np.zeros_like(Y), -Y], axis=1),
        np.concatenate([Y, np.zeros_like(Y)], axis=1),
    ],
    axis=0,
)
I_r = np.array([0, 1, 0])
I_j = np.array([2, 0, 0])
I = np.hstack([I_r, I_j])
V = np.linalg.solve(Y_block, I)
V_r = V[:3]
V_j = V[3:]
print("Voltages in Cartesian:")
for i in range(3):
    print(f"V_{i + 1} = {np.round(V_r[i], 4) + 1j * np.round(V_j[i], 4)}")
print("\nVoltages in polar:")
for i in range(3):
    magnitude = np.sqrt(V_r[i] ** 2 + V_j[i] ** 2)
    angle = np.arctan2(V_j[i], V_r[i])
    print(f"V_{i + 1} = {np.round(magnitude, 4)} ∠ {np.round(angle, 4)} rad")
# %%
