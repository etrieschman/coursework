# %%
import numpy as np
import cvxpy as cp
import polars as pl
import matplotlib.pyplot as plt


# %%
# ==============================
# Problem 5: Market Clearing
print("\n Problem 5: Market Clearing")
# ==============================
# Data
# ------------------------------
offers = pl.DataFrame(
    {
        "genco": ["B", "B", "B", "G", "G", "G", "O", "O", "O", "P", "P", "P"],
        "qty_mw": [300, 100, 50, 150, 200, 50, 250, 200, 50, 250, 200, 50],
        "price": [
            10.0,
            30.0,
            60.0,
            0.0,
            20.0,
            100.0,
            15.0,
            40.0,
            80.0,
            25.0,
            50.0,
            120.0,
        ],  # $/MWh
    }
)

bids = pl.DataFrame(
    {
        "lse": ["M", "M", "M", "O", "O", "O", "F", "F", "F", "L", "L", "L"],
        "qty_mw": [250, 200, 50, 150, 100, 50, 300, 100, 50, 200, 275, 75],
        "price": [
            300.0,
            250.0,
            20.0,
            325.0,
            275.0,
            5.0,
            300.0,
            25.0,
            10.0,
            250.0,
            250.0,
            150.0,
        ],  # $/MWh
    }
)


# %%
# ==============================
# Part 1: Build and plot
print("\n Part 1: Build and plot")


# ------------------------------
def curve(df):
    x = np.concatenate([[0], df["qty_mw"].cum_sum().to_numpy()])
    y = np.concatenate([df["price"].to_numpy()[:1], df["price"].to_numpy()])
    return x, y


# Plot supply and demand curves
plt.figure(figsize=(10, 6))
plt.step(*curve(offers.sort("price")), where="pre", label="Supply")
plt.step(*curve(bids.sort("price", descending=True)), where="pre", label="Demand")
plt.xlabel("Quantity (MW)")
plt.ylabel("Price ($/MWh)")
plt.title("Supply and demand curves")
plt.legend()
plt.show()

# %%
# ==============================
# Part 2: Market price and quantity
print("\n Part 2: Market price and quantity")
# ------------------------------
d = cp.Variable(bids.shape[0])
g = cp.Variable(offers.shape[0])

constraints = [
    d >= 0,
    d <= bids["qty_mw"].to_numpy(),
    g >= 0,
    g <= offers["qty_mw"].to_numpy(),
    cp.sum(d) == cp.sum(g),
]
obj = bids["price"].to_numpy() @ d - offers["price"].to_numpy() @ g
prob = cp.Problem(cp.Maximize(obj), constraints)
prob.solve()
print("Quantity traded:\t", np.round(cp.sum(d).value, 2), "[MWh]")
price = (offers["price"].to_numpy() * (g.value > 1e-4)).max()
print("Market price:\t\t", price, "[$/MWh]")


# %%
# ==============================
# Part 3: Revenue by genco
print("\n Part 3: Revenue by Genco")
# ------------------------------
offers.with_columns(pl.Series("qty_cleared", np.round(g.value))).with_columns(
    (pl.col("qty_cleared") * pl.lit(price)).alias("revenue")
).group_by("genco").agg(pl.sum("revenue").alias("total_revenue")).show()

# %%
# ==============================
# Part 4: Payment by lse
print("\n Part 4: Payment by LSE")
# ------------------------------
bids.with_columns(pl.Series("qty_cleared", np.round(d.value))).with_columns(
    (pl.col("qty_cleared") * pl.lit(price)).alias("payment")
).group_by("lse").agg(pl.sum("payment").alias("total_payment")).show()


# %%
# ==============================
# Problem 6: LMPs
print("\n\n Problem 6: LMPs")
# ==============================
# Part 1: Line AB disconnected
print("\n Part 1: Line AB disconnected")


# ------------------------------
# Generator A supplies Load A
def MC_A(P):
    return 20 + 0.03 * P


print("Marginal cost at Node A:", MC_A(2000), "[$/MWh]")


# Generator B supplies Load B
def MC_B(P):
    return 15 + 0.02 * P


print("Marginal cost at Node B:", MC_B(1000), "[$/MWh]")
# %%
# ==============================
# Part 2: Line AB connected and uncapacitated
print("\n Part 2: Line AB connected and uncapacitated")
# ------------------------------
# Set up as optimization problem and solve
# solution is either i) one gen produces everything or
# ii) both gens produce at equal marginal costs
pA = cp.Variable(nonneg=True)
pB = cp.Variable(nonneg=True)
const = [pA + pB == 3000]
obj = 20 * pA + 0.03 * 1 / 2 * cp.square(pA) + 15 * pB + 0.02 * 1 / 2 * cp.square(pB)
prob = cp.Problem(cp.Minimize(obj), const)
prob.solve()
print("Optimal generation at Node A:", pA.value, "[MW]")
print("Optimal generation at Node B:", pB.value, "[MW]")
print("Marginal cost of Gen A:", np.round(MC_A(pA.value), 2), "[$/MWh]")
print("Marginal cost of Gen B:", np.round(MC_B(pB.value), 2), "[$/MWh]")
print(
    "Check Electricity price:", np.round(-prob.constraints[0].dual_value, 2), "[$/MWh]"
)
