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
# ========================
# Problem 3
# ------------------------
# For each solution, check the following:
# - Unit {A,B,C,D} limits, uptime, downtime
# - Load balance
# - Reserve
#
# Solution 1: Feasible
# - Unit A: uptime, downtime, limits okay
# - Unit B: off the whole time so no violations
# - Unit C: uptime, downtime, limits okay
# - Unid D: uptime, downtime, limits okay
# - Load: balanced every hour
# - Reserve: every hour, from C in 1-3, from A in 4

# Solution 2: Ineasible
# - Unit A: uptime, downtime, limits okay
# - Unit B: off the whole time so no violations
# - Unit C: violation of uptime and downtime constraints
# - Unid D: violation of upper limit in hour 4
# - Load: balanced every hour
# - Reserve: sufficient every hour (ignoring Unit D limit violation)

# Solution 3: Ineasible
# - Unit A: uptime, downtime, limits okay
# - Unit B: violation of downtime and uptime constraints
# - Unit C: violation of lower limit in hour 1
# - Unid D: violation of upper limit in hour 4
# - Load: load imbalance (more generation) in hour 3
# - Reserve: insufficient reserves in hour 4

# %%
# ========================
# Problem 4 -- setup
# ------------------------
# constants
A, B, C = 0, 1, 2
c = 0
load = [80, 110, 120, 80, 70]
pmin = [60, 25, 10]
pmax = [100, 50, 40]
cost_margin = np.array([20, 25, 50])
cost_startup = [0, 0, c]
Tmin_up = [1, 3, 3]
Tmin_down = [3, 1, 1]
u0 = [1, 0, 0]
d0 = [0, 1, 1]
M = 1000  # stands in for an unconstrained pmax

n_p = len(pmin)
T = len(load)

# variables
p = cp.Variable((T, n_p))
u = cp.Variable((T, n_p), boolean=True)


def build_constraints(pmin_units, pmax_units, up_units, down_units):
    # constraints
    const = [cp.sum(p, axis=1) == load]
    const += [p >= 0]
    const += [p[:, g] >= pmin[g] * u[:, g] for g in pmin_units]
    const += [
        p[:, g] <= (pmax[g] if g in pmax_units else M) * u[:, g] for g in [A, B, C]
    ]
    for g in up_units:
        # minimum uptime: a startup at s forces on through s + Tmin_up - 1
        # handle history first
        const += [u[0, g] - u0[g] <= u[tau, g] for tau in range(1, min(Tmin_up[g], T))]
        const += [
            u[s, g] - u[s - 1, g] <= u[tau, g]
            for s in range(1, T)
            for tau in range(s + 1, min(s + Tmin_up[g], T))
        ]
    for g in down_units:
        # minimum downtime: a shutdown at s forces off through s + Tmin_down - 1
        # history: already off, finish the remaining downtime
        if u0[g] == 0:
            const += [u[t, g] == 0 for t in range(min(Tmin_down[g] - d0[g], T))]
        # history: shutdown in hour 1
        const += [
            u0[g] - u[0, g] <= 1 - u[tau, g] for tau in range(1, min(Tmin_down[g], T))
        ]
        const += [
            u[s - 1, g] - u[s, g] <= 1 - u[tau, g]
            for s in range(1, T)
            for tau in range(s + 1, min(s + Tmin_down[g], T))
        ]
    return const


# objective
cost_energy = cp.sum(p @ cost_margin)
cost_start = sum(
    cost_startup[g] * (cp.pos(u[0, g] - u0[g]) + cp.sum(cp.pos(u[1:, g] - u[:-1, g])))
    for g in [A, B, C]
)
obj = cost_energy + cost_start

# %%
# -----------------------
# case (a)
# -----------------------
const_a = build_constraints(
    pmin_units=[A, C],
    pmax_units=[],
    up_units=[],
    down_units=[],
)

# problem
prob = cp.Problem(cp.Minimize(obj), const_a)
prob.solve()
gen_sched = pl.from_numpy(np.maximum(0, p.value), schema=["A", "B", "C"])
print("Total operating cost:", prob.value)
print("Generation schedule:\n", gen_sched)

# Total operating cost: 9200.0
# Generation schedule:
# ┌───────┬─────┬─────┐
# │ A     ┆ B   ┆ C   │
# │ ---   ┆ --- ┆ --- │
# │ f64   ┆ f64 ┆ f64 │
# ╞═══════╪═════╪═════╡
# │ 80.0  ┆ 0.0 ┆ 0.0 │
# │ 110.0 ┆ 0.0 ┆ 0.0 │
# │ 120.0 ┆ 0.0 ┆ 0.0 │
# │ 80.0  ┆ 0.0 ┆ 0.0 │
# │ 70.0  ┆ 0.0 ┆ 0.0 │
# └───────┴─────┴─────┘


# %%
# -----------------------
# case (b)
# -----------------------
const_b = build_constraints(
    pmin_units=[A, C],
    pmax_units=[A],
    up_units=[],
    down_units=[],
)

# problem
prob = cp.Problem(cp.Minimize(obj), const_b)
prob.solve()
gen_sched = pl.from_numpy(np.maximum(0, p.value), schema=["A", "B", "C"])
print("Total operating cost:", prob.value)
print("Generation schedule:\n", gen_sched)

# Total operating cost: 9350.0
# Generation schedule:
# ┌───────┬──────┬─────┐
# │ A     ┆ B    ┆ C   │
# │ ---   ┆ ---  ┆ --- │
# │ f64   ┆ f64  ┆ f64 │
# ╞═══════╪══════╪═════╡
# │ 80.0  ┆ 0.0  ┆ 0.0 │
# │ 100.0 ┆ 10.0 ┆ 0.0 │
# │ 100.0 ┆ 20.0 ┆ 0.0 │
# │ 80.0  ┆ 0.0  ┆ 0.0 │
# │ 70.0  ┆ 0.0  ┆ 0.0 │
# └───────┴──────┴─────┘


# %%
# -----------------------
# case (c)
# -----------------------
const_c = build_constraints(
    pmin_units=[A, B, C],
    pmax_units=[A],
    up_units=[],
    down_units=[],
)

# problem
prob = cp.Problem(cp.Minimize(obj), const_c)
prob.solve()
gen_sched = pl.from_numpy(np.maximum(0, p.value), schema=["A", "B", "C"])
print("Total operating cost:", prob.value)
print("Generation schedule:\n", gen_sched)

# Total operating cost: 9450.0
# Generation schedule:
# ┌──────┬──────┬─────┐
# │ A    ┆ B    ┆ C   │
# │ ---  ┆ ---  ┆ --- │
# │ f64  ┆ f64  ┆ f64 │
# ╞══════╪══════╪═════╡
# │ 80.0 ┆ 0.0  ┆ 0.0 │
# │ 85.0 ┆ 25.0 ┆ 0.0 │
# │ 95.0 ┆ 25.0 ┆ 0.0 │
# │ 80.0 ┆ 0.0  ┆ 0.0 │
# │ 70.0 ┆ 0.0  ┆ 0.0 │
# └──────┴──────┴─────┘


# %%
# -----------------------
# case (d)
# -----------------------
const_d = build_constraints(
    pmin_units=[A, B, C],
    pmax_units=[A],
    up_units=[B],
    down_units=[B],
)

# problem
prob = cp.Problem(cp.Minimize(obj), const_d)
prob.solve()
gen_sched = pl.from_numpy(np.maximum(0, p.value), schema=["A", "B", "C"])
print("Total operating cost:", prob.value)
print("Generation schedule:\n", gen_sched)

# Total operating cost: 9850.0
# Generation schedule:
# ┌──────┬──────┬─────┐
# │ A    ┆ B    ┆ C   │
# │ ---  ┆ ---  ┆ --- │
# │ f64  ┆ f64  ┆ f64 │
# ╞══════╪══════╪═════╡
# │ 80.0 ┆ 0.0  ┆ 0.0 │
# │ 85.0 ┆ 25.0 ┆ 0.0 │
# │ 95.0 ┆ 25.0 ┆ 0.0 │
# │ 0.0  ┆ 80.0 ┆ 0.0 │
# │ 70.0 ┆ 0.0  ┆ 0.0 │
# └──────┴──────┴─────┘

# %%
# -----------------------
# case (e)
# -----------------------
const_e = build_constraints(
    pmin_units=[A, B, C],
    pmax_units=[A],
    up_units=[B],
    down_units=[A, B],
)

# problem
prob = cp.Problem(cp.Minimize(obj), const_e)
prob.solve()
gen_sched = pl.from_numpy(np.maximum(0, p.value), schema=["A", "B", "C"])
print("Total operating cost:", prob.value)
print("Generation schedule:\n", gen_sched)

# Total operating cost: 10100.0
# Generation schedule:
# ┌───────┬─────┬──────┐
# │ A     ┆ B   ┆ C    │
# │ ---   ┆ --- ┆ ---  │
# │ f64   ┆ f64 ┆ f64  │
# ╞═══════╪═════╪══════╡
# │ 80.0  ┆ 0.0 ┆ 0.0  │
# │ 100.0 ┆ 0.0 ┆ 10.0 │
# │ 100.0 ┆ 0.0 ┆ 20.0 │
# │ 80.0  ┆ 0.0 ┆ 0.0  │
# │ 70.0  ┆ 0.0 ┆ 0.0  │
# └───────┴─────┴──────┘

# %%
# -----------------------
# Problem 4.2
# -----------------------
gen_sched_pt1 = gen_sched.clone()
cost_energy = cp.sum(p @ cost_margin)
for c in np.arange(80, 150, 1):
    cost_startup[C] = c
    cost_start = sum(
        cost_startup[g]
        * (cp.pos(u[0, g] - u0[g]) + cp.sum(cp.pos(u[1:, g] - u[:-1, g])))
        for g in [A, B, C]
    )
    obj = cost_energy + cost_start
    prob = cp.Problem(cp.Minimize(obj), const_e)
    prob.solve()
    gen_sched = pl.from_numpy(np.maximum(0, p.value).round(3), schema=["A", "B", "C"])
    if not gen_sched_pt1.equals(gen_sched):
        print("schedule changes at c =", c)
        break
else:
    print("no change up to c =", c)
print(gen_sched)

# schedule changes at c = 102
# ┌──────┬──────┬─────┐
# │ A    ┆ B    ┆ C   │
# │ ---  ┆ ---  ┆ --- │
# │ f64  ┆ f64  ┆ f64 │
# ╞══════╪══════╪═════╡
# │ 80.0 ┆ 0.0  ┆ 0.0 │
# │ 85.0 ┆ 25.0 ┆ 0.0 │
# │ 95.0 ┆ 25.0 ┆ 0.0 │
# │ 0.0  ┆ 80.0 ┆ 0.0 │
# │ 0.0  ┆ 70.0 ┆ 0.0 │
# └──────┴──────┴─────┘

# %%
