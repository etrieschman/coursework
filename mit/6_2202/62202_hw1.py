# %%
import numpy as np
import matplotlib.pyplot as plt

# constants
In = 9 / 2
Rn = 8 / 3


# equations
def p(R):
    i = In * Rn / (Rn + R)
    return i**2 * R


def dp_dR(R):
    i = In * Rn / (Rn + R)
    dp_dr = i**2
    dp_di = 2 * i * R
    di_dR = -In * Rn / (Rn + R) ** 2
    return dp_dr + dp_di * di_dR


# plot ranges and values
R_range = np.linspace(0, 10, 5000)
p_range = [p(R) for R in R_range]
dp_range = [dp_dR(R) for R in R_range]
R_max = R_range[np.argmin(np.abs(dp_range))]

# plot
plt.plot(R_range, p_range, label="p(R)")
plt.plot(R_range, dp_range, label="dp/dR")
plt.axvline(
    x=R_max,
    color="gray",
    linestyle="--",
    label="dp/dR = 0",
)
plt.xlabel("R")
plt.ylabel("P")
plt.title(f"Power Dissipation\n R at max power: {R_max:.2f}")
plt.legend()
plt.show()
# NOTE: R at max power is 8/3, the same as the thevenin/norton resistance

# %%
