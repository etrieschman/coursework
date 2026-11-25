# %%
import timeit
import logging
import cvxpy as cp
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as pn

logging.getLogger("pandapower").setLevel(logging.ERROR)


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


# %%
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

# %%
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


# %%
# ========================
# Problem 6 - Value-based transmission
# ========================
# Part 1 - Max price
# ------------------------
savings_perMW = 25 - 17
print("Max transmission price:", savings_perMW, "[$/MWh]")


# %%
# ========================
# Part 2 - Updated marginal costs
# ------------------------
# value of transmission: price difference
A, B = 0, 1
demand = [2000, 1000]

# ------------------------
# Part 2.a - plot transmission value
# ------------------------
rows = []
for limit in np.arange(0, 1600, 100):
    # step 1 - solve for flow (B->A)
    # 20 + 0.03 * (2000 - f) = 15 + 0.02 * (1000 + f)
    f = 45 / 0.05
    # step 2 - clip to limit
    f_clip = np.clip(f, -limit, limit)
    # step 3 - get marginal costs
    mc_a = 20 + 0.03 * (demand[A] - f_clip)
    mc_b = 15 + 0.02 * (demand[B] + f_clip)
    # step 4 transmission value
    rows.append({"limit": limit, "flow": f_clip, "value": np.abs(mc_a - mc_b)})
df = pl.DataFrame(rows)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(df["limit"], df["value"], marker="o", markersize=4)
ax.set_xlabel("Transmission capacity (MW)")
ax.set_ylabel("Marginal value of transmission ($/MWh)")
ax.set_title("Marginal value of transmission vs. line capacity")
ax.set_xlim(0, 1500)
ax.set_ylim(-10, 50)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %%
# ------------------------
# Part 2.b - transmission demand function
# ------------------------
# define flow f (B->A)
# pi(f) = (20 + 0.03 * (2000 - f)) - (15 + 0.02 * (1000 + f))
#       = 45 - 0.05 * f


# %%
# ------------------------
# Part 2.c - transmission supply function
# ------------------------
length = 500
mc = 210
ann_mc = mc * length
hourly_mc = ann_mc / 8760
print("Annuitized marginal cost of capacity:", np.round(ann_mc, 2), "[$/(MW*km*yr)]")
print("Hourly long run marginal cost:", np.round(hourly_mc, 2), "[$/MWh]")


# %%
# ------------------------
# Part 2.d - Optimal capacity
# ------------------------
# marginal profit = (demand - supply)
# mp(f) = pi(f) - hourly_mc
#       = 45 - 0.05 * f - 12
#       = 33 - 0.05 * f
#     0 = 33 - 0.05 * f (optimal)
f_opt = 33 / 0.05
print("Optimal capacity:", np.round(f_opt, 2), "[MW]")

rows = [
    {
        "capacity": f,
        "marginal_profit": 33 - 0.05 * f,
        "profit": (33 - 0.05 * f) * f,
    }
    for f in np.arange(0, 800, 20)
]
df = pl.DataFrame(rows)
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax2 = ax1.twinx()

l1 = ax1.plot(df["capacity"], df["profit"], color="C0", label="profit")
l2 = ax2.plot(
    df["capacity"], df["marginal_profit"], color="C1", label="marginal profit"
)
ax1.axvline(x=f_opt, color="gray", lw=0.8, linestyle="--")
ax1.set_xlabel("Transmission capacity (MW)")
ax1.set_ylabel("Profit ($/h)", color="C0")
ax2.set_ylabel("Marginal profit ($/MWh)", color="C1")
ax1.tick_params(axis="y", colors="C0")
ax2.tick_params(axis="y", colors="C1")
ax1.set_xlim(0, 800)
ax2.set_ylim(-20, 60)
ax1.set_ylim(-2000, 6000)
ax1.axhline(0, color="gray", lw=0.8)
ax1.grid(True, alpha=0.3)
lines = l1 + l2
ax1.legend(lines, [l.get_label() for l in lines], loc="best")
fig.tight_layout()
plt.show()


# %%
# ========================
# Problem 7 - System frequency response
# ========================
# Part 1 - Load jump
# ------------------------
# Rate of change of system frequency:
# Pm - Pe = -1_000
# domega/dt = -1_000 [MW] / 20_000 [MW/Hz^2] = -0.05 [Hz^2 = Hz/sec]

# Time to 1.5Hz drop if no control:
# -1.5 [Hz] = -0.05 [Hz/sec] * x [sec]
# x = 30 [sec]

# %%
# ========================
# Part 2 - Load jump on isolated system
# ------------------------
# Rate of change of system frequency:
# Pm - Pe = -1_000
# domega/dt = -1_000 [MW] / 3_000 [MW/Hz^2] = -1/3 [Hz^2 = Hz/sec]

# Time to 1.5Hz drop if no control:
# -1.5 [Hz] = -1/3 [Hz/sec] * x [sec]
# x = 4.5 [sec]


# %%
# ========================
# Problem 8 - Frequency control
# ========================
# Part 1 - Operating frequency
# ------------------------
# f = fnom * (1 - rho * p/pmax)
f = 60 * (1 - 0.04 * 30 / 40)
print(f"Operating frequency: {f:0.3f} [Hz]")


# %%
# ========================
# Part 2 - New operating frequency
# ------------------------
# f = fnom * (1 - rho * p/(p1max + p2max))
f = 60 * (1 - 0.04 * 30 / (40 + 30))
print(f"Operating frequency: {f:0.3f} [Hz]")


# %%
# ========================
# Part 3 - Power generation
# ------------------------
# pi = pimax / (p1max + p2max) * p
pmax = np.array([40, 30])
p = 30
pgen = pmax / np.sum(pmax) * p
print(f"Power generation (gen1, gen2):", np.round(pgen, 3), "[kW]")

# %%
