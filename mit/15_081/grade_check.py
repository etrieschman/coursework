# %%
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# %%
w_h = cp.Variable()
w_m = cp.Variable()
w_f = cp.Variable()
g_h, g_m = 98, 69
target = 90

def get_lp_final_grade(g_f):
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
    return problem.value

g_fs = np.linspace(60, 100, 100)
final_grades = []
for g_f in g_fs:
    final_grades.append(get_lp_final_grade(g_f))
final_grades = np.array(final_grades)
plt.plot(final_grades, g_fs)
plt.xlabel('Course grade')
plt.ylabel('grade on final')
plt.axvline(x=target, color='C1', linestyle='--')
plt.axhline(y=g_fs[np.argmin(np.abs(final_grades - target))], color='C2', linestyle='--')
plt.show()


# %%
g_h, g_l, g_n, g_m, __, g_p = 9.8, 9.9, 9.8, 7.92, 0.0, 9.9
w_h, w_l, w_n, w_m, w_f, w_p = 0.15, .15, .05, .23, .37, .05
def get_g_f(target):
    return (target - g_h*w_h - g_l*w_l - g_n*w_n - g_m*w_m - g_p*w_p) / w_f

target = 9
targets = np.linspace(7, 10, 100)
g_fs = [get_g_f(t) for t in targets]

plt.plot(targets, g_fs)
plt.axvline(x=target, color='C1', linestyle='--')
plt.axhline(
    y=get_g_f(target), 
    color='C2', linestyle='--')
plt.xlabel('Course grade')
plt.ylabel('grade on final')
plt.show()

# %%
