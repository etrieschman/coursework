# %%
import numpy as np
import cvxpy as cp
import polars as pl
import pandas as pd

import pandapower as pp
import pandapower.networks as pn

pd.set_option("display.precision", 3)  # digits after the decimal
pd.set_option("display.width", 200)  # don't wrap columns
pd.set_option("display.max_columns", None)


def polar_str(x, round=4):
    magnitude = np.sqrt(x.real**2 + x.imag**2)
    angle = np.arctan2(x.imag, x.real)
    return f"{np.round(magnitude, round)} ∠ {np.round(angle, round)} rad"


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
print("Voltages (Cartesian):")
for i in range(3):
    print(f"V_{i + 1} = {np.round(V_r[i], 4) + 1j * np.round(V_j[i], 4)}kV")
print("\nVoltages (polar):")
for i in range(3):
    magnitude = np.sqrt(V_r[i] ** 2 + V_j[i] ** 2)
    angle = np.arctan2(V_j[i], V_r[i])
    print(f"V_{i + 1} = {polar_str(V_r[i] + 1j * V_j[i])} kV")

# %%
# ===================
# Part 3
# -------------------
Y_3 = -25j
I_3 = (V_r[-1] + 1j * V_j[-1]) * Y_3
print(f"I_3 (Cartesian) = {np.round(I_3.real, 4) + 1j * np.round(I_3.imag, 4)} kV")
print(f"I_3 (polar) = {polar_str(I_3)} kV")

# %%
# ===================
# Part 4
# -------------------
V = V_r + 1j * V_j
I = -(I_r + 1j * I_j)
I[-1] = I_3

S = V * I.conj()
print("Complex power (Cartesian):")
for i in range(3):
    print(
        f"S_{i + 1} = {np.round(S[i].real, 4) + 1j * np.round(S[i].imag, 4)} MVA absorbed"
    )
print("\nComplex power (polar):")
for i in range(3):
    print(f"S_{i + 1} = {polar_str(S[i])} MVA absorbed")


# %%
# ===================
# Problem 7
# ===================
baseMVA = 100
# bus_i type   Pd      Qd  Gs Bs area Vm Va baseKV zone Vmax Vmin
bus = np.array(
    [
        [1, 2, 0, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [2, 1, 300, 98.61, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [3, 2, 300, 98.61, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [4, 3, 400, 131.47, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [5, 2, 0, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
    ]
)

# bus     Pg  Qg  Qmax    Qmin   Vg mBase status Pmax Pmin Pc1 Pc2 Qc1min Qc1max Qc2min Qc2max ramp_agc ramp_10 ramp_30 ramp_q apf
gen = np.array(
    [
        [1, 40, 0, 30, -30, 1, 100, 1, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 170, 0, 127.5, -127.5, 1, 100, 1, 170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 323.49, 0, 390, -390, 1, 100, 1, 520, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 150, -150, 1, 100, 1, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 466.51, 0, 450, -450, 1, 100, 1, 600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

# fbus tbus  r        x       b       rateA rateB rateC ratio angle status angmin angmax
branch = np.array(
    [
        [1, 2, 0.00281, 0.0281, 0.00712, 400, 400, 400, 0, 0, 1, -360, 360],
        [1, 4, 0.00304, 0.0304, 0.00658, 0, 0, 0, 0, 0, 1, -360, 360],
        [1, 5, 0.00064, 0.0064, 0.03126, 0, 0, 0, 0, 0, 1, -360, 360],
        [2, 3, 0.00108, 0.0108, 0.01852, 0, 0, 0, 0, 0, 1, -360, 360],
        [3, 4, 0.00297, 0.0297, 0.00674, 0, 0, 0, 0, 0, 1, -360, 360],
        [4, 5, 0.00297, 0.0297, 0.00674, 240, 240, 240, 0, 0, 1, -360, 360],
    ]
)

# ===================
# Part 2
# -------------------
n_l, n_b = len(branch), len(bus)
# line rows: column 0 = ground, columns 1..n_b = buses
A_line = np.zeros((n_l, n_b + 1))
for ell, (i, o) in enumerate(branch[:, :2]):
    A_line[ell, int(i)] = 1
    A_line[ell, int(o)] = -1

# shunt rows: bus i -> ground
A_shunt = np.hstack([-np.ones((n_b, 1)), np.eye(n_b)])
A = np.vstack([A_line, A_shunt])

Z_line = branch[:, 2] + 1j * branch[:, 3]
y_line = 1 / Z_line
y_shunt = np.abs(A_line[:, 1:].T) @ (1j * branch[:, 4] / 2)
y = np.concatenate([y_line, y_shunt])

Y = (A.T @ np.diag(y) @ A)[1:, 1:]
labels = [f"bus_{i}" for i in range(1, n_b + 1)]
df = pd.DataFrame(Y, index=labels, columns=labels)
display(df)

# %%
# ===================
# Part 6
# -------------------
# load network
net = pn.case5()
# solve network with Newton-Raphson
pp.runpp(net, algorithm="nr", numba=False)

# bus results
bus = net.res_bus.copy().reset_index(names="bus")
# update bus name and flip signs on power for convention
bus["bus"] += 1
bus["p_mw"] *= -1
bus["q_mvar"] *= -1
display(bus)

# line results
# pull in line mapping and keep specific vars
line = (
    net.line[["from_bus", "to_bus"]]
    .rename(columns={"from_bus": "from", "to_bus": "to"})
    .join(
        net.res_line.copy()[
            [
                "p_from_mw",
                "q_from_mvar",
                "p_to_mw",
                "q_to_mvar",
                "pl_mw",
                "ql_mvar",
                "i_ka",
                "loading_percent",
            ]
        ]
    )
)
line["from"] += 1
line["to"] += 1
display(line)

# line losses
print(f"Total line losses: {bus['p_mw'].sum():0.2f}MW")

# %%
