# %%
# =====================
# Question 2.4
# =====================
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
# =====================
# Question 4
# =====================
# Household appliances
# (a) What is the peak power consumed by a vacuum?
# > Dyson V16 Piston Animal Submarine™ wet and dry vacuum has peak wattage of 900W
# > https://www.dyson.com/vacuum-cleaners/cordless/v16-piston/submarine-black-copper

# (b) Assuming an electricity cost of 27 cents/kWh, how much does it cost to vacuum for 20 minutes?
# > 20min * 1hr/60min * 900W * 1kW/1000W * $0.27/kWh = $0.081

# 2. Electric Vehicles
# (a) What is the power supplied by a level 2 electric vehicle charger?
# > 7.2 kW (240V, 30A)
# > https://www.pge.com/en/clean-energy/electric-vehicles/getting-started-with-electric-vehicles/electric-vehicle-charging.html#accordion-1b9c719c3f-item-4a21e424e4

# (b) How much energy per mile is required by electric vehicles?
# > 3.6mi / kW <==> 0.2778 kW/mi
# > https://afdc.energy.gov/vehicles/electric-emissions-sources

# (c) How long do you need to charge to drive 30 miles?
# > 30mi * 1kW/3.6mi * 1hr/7.2kW ~= 1.15hr

# 3. What is the average annual power energy consumption per household in the United States?
# > 10,791kWh
# > https://www.eia.gov/tools/faqs/faq.php?id=97&t=3

# 4. What is the historical peak power demand in the United States?
# > 759kW
# > https://www.eia.gov/todayinenergy/detail.php?id=65864

# %%
