
################################## Quarterly 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Load your dataset
df = pd.read_csv('2. Quarterly.csv')  # Update with your actual dataset path

print(df.head())

# Ensure the 'Year' column is in datetime format
df['Year'] = pd.to_datetime(df['Year'], format='%Y-%m-%d')  # Adjust format to match your data

# Set the 'Year' column as the index
df.set_index('Year', inplace=True)

# Resample to monthly frequency if needed
df = df.resample('M').sum()

# Drop any rows with NaN values
df.dropna(inplace=True)

# Perform STL decomposition
stl = STL(df['Population'], seasonal=5)  # Use seasonal=12 for monthly data
result = stl.fit()

# Detecting anomalies in the residual component
threshold = 3  # Define your threshold for anomaly detection
residuals = result.resid

# Identify anomalies in residuals
anomalies = np.abs(residuals) > threshold * np.std(residuals)

# Plotting the components with improved aesthetics
plt.figure(figsize=(18, 12))  # Increased height for better spacing

# Plot the original data
plt.subplot(5, 1, 1)  # 5 rows, 1 column, first subplot
plt.plot(df.index, df['Population'], label='Original Data', color='blue')
plt.title('Population Data', fontsize=16)
plt.ylabel('Population', fontsize=14)
plt.grid()
plt.tight_layout()

# Plot the trend component
plt.subplot(5, 1, 2)  # Second subplot
plt.plot(result.trend, label='Trend', color='orange')
plt.title('Trend Component', fontsize=16)
plt.ylabel('Trend', fontsize=14)
plt.grid()

# Plot the seasonal component
plt.subplot(5, 1, 3)  # Third subplot
plt.plot(result.seasonal, label='Seasonal', color='green')
plt.title('Seasonal Component', fontsize=16)
plt.ylabel('Seasonality', fontsize=14)
plt.grid()

# Plot the residual component
plt.subplot(5, 1, 4)  # Fourth subplot
plt.plot(result.resid, label='Residual', color='red')
plt.title('Residual Component', fontsize=16)
plt.ylabel('Residuals', fontsize=14)
plt.grid()

# Plot anomalies on the residual component
plt.scatter(df.index[anomalies], residuals[anomalies], color='purple', label='Anomalies', s=50)
plt.axhline(y=threshold * np.std(residuals), color='black', linestyle='--', label='Threshold')
plt.axhline(y=-threshold * np.std(residuals), color='black', linestyle='--')

plt.legend()

# Overall title
plt.suptitle('STL Decomposition and Anomalies in Population Data', fontsize=22, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
plt.savefig('STL_Decomposition.jpg') 
plt.show()