# %%
import timeit
import cvxpy as cp
import numpy as np
import polars as pl
import pandapower as pp
import pandapower.networks as pn


# %%
# ========================
# Problem 1
# ------------------------
# Givens
# ------------------------
def c1(p):
    return 2000 + 6 * p + 0.15 * cp.square(p)


def c2(p):
    return 7000 + 8 * p + 0.12 * cp.square(p)


def c3(p):
    return 6000 + 7 * p + 0.08 * cp.square(p)


def mc1(p):
    return 6 + 0.30 * p


def mc2(p):
    return 8 + 0.24 * p


def mc3(p):
    return 7 + 0.16 * p


def total_cost(p):
    return c1(p[P1]) + c2(p[P2]) + c3(p[P3])


def marginal_cost(p):
    return [mc1(p[P1]), mc2(p[P2]), mc3(p[P3])]


# ------------------------
# Part 1 - Economic dispatch
# ------------------------
# Dispatch when marginal costs are equal and energy balanced
# Set up as linear system
P1, P2, P3, lam = 0, 1, 2, 3
b = np.array([-6.0, -8.0, -7.0, 900.0])
A = np.array([[0.30, 0, 0, -1], [0, 0.24, 0, -1], [0, 0, 0.16, -1], [1, 1, 1, 0]])
x = np.linalg.solve(A, b)
print("(P1, P2, P3):\t\t", np.round(x[:3], 3))
print("system marginal cost:\t", np.round(x[-1], 3))
print("gen marginal cost:\t", np.round(marginal_cost(x[:3]), 3))
print("total cost:\t\t", np.round(total_cost(x[:3]).value, 3))

# (P1, P2, P3):		     [221.717 268.813 409.47 ]
# system marginal cost:	 72.515
# gen marginal cost:	 [72.515 72.515 72.515]
# total cost:		     50805.366


# %%
# ------------------------
# Part 2 - With limits
# ------------------------
# Solve with optimization
# constants
pmin = [100.0, 200.0, 100.0]
pmax = [300.0, 400.0, 350.0]
load = 900.0
# variable
p = cp.Variable(3)
# constraints
const = [pmin <= p, p <= pmax, cp.sum(p) == load]
# objective
obj = c1(p[P1]) + c2(p[P2]) + c3(p[P3])
# problem
prob = cp.Problem(cp.Minimize(total_cost(p)), const)
prob.solve()

print("(P1, P2, P3):\t\t", np.round(p.value, 3))
print("system marginal cost:\t", -np.round(const[-1].dual_value, 3))
print("gen marginal cost:\t", np.round(marginal_cost(p.value), 3))
print("total cost:\t\t", np.round(total_cost(p).value, 3))

# (P1, P2, P3):		     [248.148 301.852 350.   ]
# system marginal cost:	 80.444
# gen marginal cost:	 [80.444 80.444 63.   ]
# total cost:		     51324.074


# %%
# ------------------------
# Part 3 - Load value
# ------------------------
# P3 fixed at 350, lambda at 85:
target_lam = 85.0
b_pt3 = b.copy()[:2]
A_pt3 = A.copy()[:2, :2]
b_pt3 += target_lam

x_pt3 = np.linalg.solve(A_pt3, b_pt3)
# check
assert np.abs(mc1(x_pt3[P1]) - target_lam) < 1e-6
assert np.abs(mc2(x_pt3[P2]) - target_lam) < 1e-6
p_pt3 = np.append(x_pt3, pmax[P3])
print("(P1, P2, P3):\t", np.round(p_pt3, 3))
print("load:\t\t", np.round(p_pt3.sum(), 3))

# (P1, P2, P3):	 [263.333 320.833 350.   ]
# load:		     934.167


# %%
# ------------------------
# Part 4 - Renewables
# ------------------------
load_pt4 = load * (1 - 0.333)
const = [pmin <= p, p <= pmax, cp.sum(p) == load_pt4]
prob = cp.Problem(cp.Minimize(total_cost(p)), const)
prob.solve()

print("(P1, P2, P3):\t\t", np.round(p.value, 3))
print("system marginal cost:\t", -np.round(const[-1].dual_value, 3))
print("gen marginal cost:\t", np.round(marginal_cost(p.value), 3))
print("total cost:\t\t", np.round(total_cost(p).value, 3))

# (P1, P2, P3):		     [141.409 200.    258.891]
# system marginal cost:	 48.423
# gen marginal cost:	 [48.423 56.    48.423]
# total cost:		     32422.131


# %%
# ========================
# Problem 2
# ------------------------
# Part 1 - Explain the code
# ------------------------
# The supplied code PS10.2-6.220-FA2026.ipynb imports Case 9
# of the Pandapower networks and runs it under various solver
# strategies. It first solves it with ACOPF, then with DCOPF,
# and then with ACOPF initialized to the power values solved
# for with DCOPF.


# ------------------------
# Part 2 - Complete the table
# ------------------------
def dc_total_cost(net_dcopf):
    # Slack bus generation cost
    ext_poly_cost = net_dcopf.poly_cost.loc[net_dcopf.poly_cost["et"] == "ext_grid"]
    ext_pgen = net_dcopf.res_ext_grid["p_mw"].iloc[0]
    ext_total_cost = (
        ext_poly_cost["cp0_eur"].iloc[0]
        + ext_poly_cost["cp1_eur_per_mw"].iloc[0] * ext_pgen
        + ext_poly_cost["cp2_eur_per_mw2"].iloc[0] * ext_pgen**2
    )

    # PV buses generation cost
    #   Note: set_index aligns the naming of the rows of poly_cost and res_gen,
    #   to ensure the right elements are multiplied together
    pv_poly_cost = net_dcopf.poly_cost[net_dcopf.poly_cost["et"] == "gen"].set_index(
        "element"
    )
    pv_pgen = net_dcopf.res_gen["p_mw"]
    pv_total_cost = (
        pv_poly_cost["cp0_eur"]
        + pv_poly_cost["cp1_eur_per_mw"] * pv_pgen
        + pv_poly_cost["cp2_eur_per_mw2"] * pv_pgen**2
    )

    # Total cost across all buses
    total_cost = ext_total_cost + pv_total_cost.sum()
    return total_cost


def dc_then_ac(net_dcopf):
    pp.rundcopp(net_dcopf, delta=1e-6)
    net_dcopf.gen["p_mw"] = net_dcopf.res_gen["p_mw"]
    pp.runpp(net_dcopf, delta=1e-6)


# run collab code over all cases; store results
n, r = 10, 5
cases = [pn.case9, pn.case14, pn.case39, pn.case118]
rows = []
for case_fn in cases:
    # ACOPF
    net_acopf = case_fn()
    ac_time = timeit.repeat(
        lambda: pp.runopp(net_acopf, delta=1e-6), number=n, repeat=r
    )
    ac_time_mean = np.mean(ac_time)
    ac_time_std = np.std(ac_time)
    ac_cost = net_acopf.res_cost

    # DCOPF + AC power flow
    net_dcopf = case_fn()
    dc_time = timeit.repeat(lambda: dc_then_ac(net_dcopf), number=n, repeat=r)
    dc_time_mean = np.mean(dc_time)
    dc_time_std = np.std(dc_time)
    dc_cost = dc_total_cost(net_dcopf)

    # save results
    rows.append(
        {
            "case": case_fn.__name__,
            "cost_ac": ac_cost,
            "cost_dc": dc_cost,
            "cost_pdiff": (dc_cost - ac_cost) / ac_cost,
            "time_ac": f"{ac_time_mean:0.3f} ± {ac_time_std:0.3f}",
            "time_dc": f"{dc_time_mean:0.3f} ± {dc_time_std:0.3f}",
            "time_pdiff": (dc_time_mean - ac_time_mean) / ac_time_mean,
        }
    )

df = pl.DataFrame(rows).with_columns(pl.selectors.numeric().round(3))
print(df)
# shape: (4, 7)
# ┌─────────┬────────────┬────────────┬────────────┬───────────────┬───────────────┬────────────┐
# │ case    ┆ cost_ac    ┆ cost_dc    ┆ cost_pdiff ┆ time_ac       ┆ time_dc       ┆ time_pdiff │
# │ ---     ┆ ---        ┆ ---        ┆ ---        ┆ ---           ┆ ---           ┆ ---        │
# │ str     ┆ f64        ┆ f64        ┆ f64        ┆ str           ┆ str           ┆ f64        │
# ╞═════════╪════════════╪════════════╪════════════╪═══════════════╪═══════════════╪════════════╡
# │ case9   ┆ 5311.912   ┆ 5316.048   ┆ 0.001      ┆ 0.667 ± 0.003 ┆ 0.180 ± 0.002 ┆ -0.73      │
# │ case14  ┆ 8081.526   ┆ 8177.252   ┆ 0.012      ┆ 0.868 ± 0.022 ┆ 0.260 ± 0.008 ┆ -0.701     │
# │ case39  ┆ 41872.301  ┆ 41890.459  ┆ 0.0        ┆ 1.349 ± 0.019 ┆ 0.275 ± 0.017 ┆ -0.796     │
# │ case118 ┆ 129704.738 ┆ 131622.524 ┆ 0.015      ┆ 2.164 ± 0.048 ┆ 0.329 ± 0.001 ┆ -0.848     │
# └─────────┴────────────┴────────────┴────────────┴───────────────┴───────────────┴────────────┘

# ------------------------
# Part 3 - Briefly comment
# ------------------------
# 1. There are clear speedups by initializing ACOPF with DC power
#    set points.
# 2. AC compute time grows with case size; DC compute time
#    also grows, but not as dramatically
# 3. Cost difference also seems to grow with case size, both
#    in absolute and relative values
# %%
