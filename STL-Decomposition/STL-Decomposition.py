import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Load data from CSV
data = pd.read_csv('data.csv', index_col=0, parse_dates=True)

# Perform STL decomposition
stl = STL(data['Population'], seasonal=3)  # Adjust seasonal parameter as needed
result = stl.fit()

# Plot the decomposition with better structure
fig, axes = plt.subplots(4, 1, figsize=(18, 8), sharex=True)

# Observed
axes[0].plot(data['Population'], label='Observed', color='blue')
axes[0].set_title('STL Decomposition of Population Data', fontsize=16)
axes[0].set_ylabel('Population', fontsize=14)
axes[0].legend()
axes[0].grid()

# Trend
axes[1].plot(result.trend, label='Trend', color='orange')
axes[1].set_ylabel('Trend', fontsize=14)
axes[1].legend()
axes[1].grid()

# Seasonal
axes[2].plot(result.seasonal, label='Seasonal', color='green')
axes[2].set_ylabel('Seasonal', fontsize=14)
axes[2].legend()
axes[2].grid()

# Residual
axes[3].plot(result.resid, label='Residual', color='red')
axes[3].set_ylabel('Residual', fontsize=14)
axes[3].legend()
axes[3].grid()

# X-axis label
axes[3].set_xlabel('Date', fontsize=14)

# Adjust layout
plt.tight_layout()

plt.savefig('STLMonthly.PNG')
plt.show()
