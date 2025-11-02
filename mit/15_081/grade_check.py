# %%
import numpy as np
import cvxpy as cp

# %%
w_h = cp.Variable()
w_m = cp.Variable()
w_f = cp.Variable()
g_h, g_m, g_f = 100, 100, 100

# constraints
constraints = [
    w_h >= 0.15, w_m >= 0.15, w_f >= w_m,
    w_h + w_m + w_f <= 1,
    w_h + w_m + w_f >= 0.9,
    w_m + w_f <= 0.8,
    w_m + w_f >= 0.5,
]

# objective
obj = cp.Maximize(w_h*g_h + w_m*g_m + w_f*g_f + (1 - w_h - w_m - w_f)*100)

# solve
problem = cp.Problem(obj, constraints)
problem.solve()

print('grade', problem.value)
print('weights (w_h, w_m, w_f):', w_h.value, w_m.value, w_f.value)
# %%
