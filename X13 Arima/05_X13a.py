import statsmodels
import pandas as pd

import statsmodels.api as sm
import os
import matplotlib.pyplot as plt


file_path="2. data.csv"
df_population = pd.read_csv(file_path)
XPATH = r"x13as"


df_population['Date']=pd.to_datetime(df_population['Year'], infer_datetime_format=True)
df_population.index=df_population["Year"]
df_population.set_index(df_population["Date"], inplace=True)
# Convert the 'Date' column to datetime format
df_population["Date"] = pd.to_datetime(df_population["Year"])

temp_series=pd.DataFrame(df_population["Population"].values,index=df_population["Date"])
temp_series


import matplotlib.pyplot as plt
import statsmodels.api as sm

# Run X-13ARIMA-SEATS analysis
# x13results = sm.tsa.x13_arima_analysis(endog=temp_series, x12path=XPATH, outlier=True, print_stdout=True)

x13results = sm.tsa.x13_arima_analysis(
    endog=temp_series,
    trading=True,
    maxdiff=(2, 0),         # Increase differencing
    maxorder=(4, 2),        # Increase AR and MA orders
    x12path=XPATH,
    outlier=True,
    print_stdout=True, # Adjust seasonal filter if applicable
)

# Create a larger figure with subplots
fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

# Plot each component separately for clarity
# 1. Original and Seasonally Adjusted Series
axes[0].plot(x13results.observed.index, x13results.observed, label="Original", color='black', linewidth=1.5)
# axes[0].plot(x13results.seasadj.index, x13results.seasadj, label="Seasonally Adjusted", color='blue', linewidth=1.5)
axes[0].set_title("Original", fontsize=16)
axes[0].legend(loc="upper left")
axes[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# 2. Trend Component
axes[1].plot(x13results.trend.index, x13results.trend, label="Trend", color='green', linewidth=1.5)
axes[1].set_title("Trend Component", fontsize=16)
axes[1].legend(loc="upper left")
axes[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# 3. Seasonal Component
axes[2].plot(x13results.seasadj.index, x13results.seasadj, label="Seasonal", color='purple', linewidth=1.5)
axes[2].set_title("Seasonal Component", fontsize=16)
axes[2].legend(loc="upper left")
axes[2].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# 4. Irregular (Residual) Component
axes[3].plot(x13results.irregular.index, x13results.irregular, label="Irregular", color='red', linewidth=1.5)
axes[3].set_title("Irregular Component", fontsize=16)
axes[3].legend(loc="upper left")
axes[3].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# Set the x-axis label on the bottom subplot only
axes[3].set_xlabel("Date", fontsize=14)

# Main title for the entire figure
fig.suptitle("X-13ARIMA-SEATS Decomposition: Korea Population Data", fontsize=16)

# Adjust layout to prevent overlap
fig.tight_layout(rect=[0, 0.03, 1, 0.97])

# Display the plot

plt.savefig("x13_arima.png")
plt.show()