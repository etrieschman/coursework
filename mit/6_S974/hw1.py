# %%
import numpy as np
import cvxpy as cp

# %%
# =======================
# Problem 1 Part (a)
# =======================
# constants
nodes = np.array(["C1", "C2", "W", "D1", "D2", "S"])
arcs = np.array(
    [
        "D1->C1",
        "D1->C2",
        "D1->W",
        "D2->C1",
        "D2->C2",
        "D2->W",
        "W->C1",
        "W->C2",
        "D1->S",
        "D2->S",
    ]
)
b = np.array([-200, -700, 0, 400, 600, -100])
c = np.array([30, 30, 15, 50, 23, 15, 11, 14, 0, 0])
A = np.array(
    [
        [-1, 0, 0, 1, 0, 0],
        [0, -1, 0, 1, 0, 0],
        [0, 0, -1, 1, 0, 0],
        [-1, 0, 0, 0, 1, 0],
        [0, -1, 0, 0, 1, 0],
        [0, 0, -1, 0, 1, 0],
        [-1, 0, 1, 0, 0, 0],
        [0, -1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, -1],
        [0, 0, 0, 0, 1, -1],
    ]
)
# variable
f = cp.Variable(len(arcs))
# problem
objective = cp.Minimize(c @ f)
constraints = [A.T @ f == b, f >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()
# solution
print("~~~ PROBLEM 1 PART (A) ~~~")
print("Optimal shipping plan:")
for arc, flow in zip(arcs, np.round(f.value, 2)):
    print(f"  {arc:>6}: {flow:8.2f}")
print("Optimal cost:", np.round(problem.value, 2))


# =======================
# Problem 1 Part (c)
# =======================
# constants
nodes_c = nodes[:-1]
arcs_c = arcs[:-2]
b_c = np.array([-500, -700, 0, 400, 600])
c_c = c[:-2]
A_c = A[:-2, :-1]
budget = 23_000
# variables
f_c = cp.Variable(len(arcs_c))
x = cp.Variable(2)
z = cp.Variable()
# problem
objective = cp.Minimize(z)
inj = A_c.T @ f_c
constraints = [
    inj[0:2] + x == 0,  # LHS is outflow - inflow
    inj[2] == b_c[2],
    inj[3:] <= b_c[3:],
    f_c >= 0,
    x >= 0,
    x <= -b_c[0:2],
    f_c[6] <= 200,
    c_c @ f_c <= budget,
    1 - x / (-b_c[0:2]) <= z,
]
problem = cp.Problem(objective, constraints)
problem.solve()
print("\n~~~ PROBLEM 1 PART (C) ~~~")
print("Optimal shipping plan:")
for arc, flow in zip(arcs_c, np.round(f_c.value, 2)):
    print(f"  {arc:>6}: {flow:8.2f}")
print("Shipping cost:", np.round(c_c @ f_c.value, 2))
print("Unmet demand (C1, C2):", np.round(-b_c[:2] - x.value, 2))
print("Unmet fractions (C1, C2):", np.round(1 - x.value / (-b_c[0:2]), 2))
print("Unused supply (D1, D2):", np.round(b_c[3:] - inj[3:].value, 2))


# %%
# =======================
# Problem 2 Part (d)
# =======================
# Constants
U = np.array([[[1, 1], [6, 3]], [[3, 6], [5, 5]]])
A, B = 0, 1
O, L = 0, 1

# variable
pi = cp.Variable((2, 2))

# constraints
constraints = [pi >= 0, cp.sum(pi) == 1]
constraints += [
    cp.sum([pi[s_i, s_mi] * (U[s_i, s_mi, A] - U[sp_i, s_mi, A]) for s_mi in [O, L]])
    >= 0
    for sp_i in [O, L]
    for s_i in [O, L]
]
constraints += [
    cp.sum([pi[s_mi, s_i] * (U[s_mi, s_i, B] - U[s_mi, sp_i, B]) for s_mi in [O, L]])
    >= 0
    for sp_i in [O, L]
    for s_i in [O, L]
]

# solve problem
objective = cp.Maximize(cp.sum(cp.multiply(pi, U[:, :, A] + U[:, :, B])))
problem = cp.Problem(objective, constraints)
problem.solve()

# report
print("\n~~~ PROBLEM 2 PART (D) ~~~")
print("Joint distribution (pi):\n", np.round(pi.value, 2))
ep_A = cp.sum(cp.multiply(pi, U[:, :, A])).value
ep_B = cp.sum(cp.multiply(pi, U[:, :, B])).value
print(f"Expected payoffs (A, B): ({ep_A:0.2f}, {ep_B:0.2f})")
print("Total expected welfare:", np.round(ep_A + ep_B, 2))

# %%


# %%
