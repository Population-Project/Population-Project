

from statsmodels.tsa.seasonal import STL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.x13 import x13_arima_analysis
# Load the data from CSV file
# Assuming the CSV file has a 'time' column and a 'population' column
df = pd.read_csv('data.csv', parse_dates=['Year'], index_col='Year')
df = df.fillna(0)
# Check if the data was read correctly
print(df.head())

################################################################################################################################



result = seasonal_decompose(df['Population'], model='multiplicative', period=4)

# Set up a larger figure size with 4 subplots
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(18, 8), sharex=True)

# Plotting each component in a separate subplot
result.observed.plot(ax=axes[0], color='blue', linewidth=2)
axes[0].set_title('Observed', fontsize=14)
axes[0].grid()

result.trend.plot(ax=axes[1], color='orange', linewidth=2)
axes[1].set_title('Trend', fontsize=14)
axes[1].grid()

result.seasonal.plot(ax=axes[2], color='green', linewidth=2)
axes[2].set_title('Seasonal', fontsize=14)
axes[2].grid()

result.resid.plot(ax=axes[3], color='red', linewidth=2)
axes[3].set_title('Residual', fontsize=14)
axes[3].grid()

# Adding a common xlabel
plt.xlabel('Year', fontsize=14)

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig('KoreMonthly_mul.PNG')
# Show the plot
plt.show()

