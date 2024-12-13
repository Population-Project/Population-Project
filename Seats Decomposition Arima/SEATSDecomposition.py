import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load data from CSV
data = pd.read_csv('data.csv', index_col=0, parse_dates=True)

# Check the data
print(data.head())

# Fit a SARIMA model
# Adjust the order (p, d, q) and seasonal_order (P, D, Q, S) as needed
order = (1, 1, 1)  # Non-seasonal parameters
seasonal_order = (1, 1, 1, 4)  # Seasonal parameters (quarterly)

model = SARIMAX(data['Population'], order=order, seasonal_order=seasonal_order)
results = model.fit(disp=False)

# Get the fitted values and residuals
fitted_values = results.fittedvalues
residuals = results.resid

# Plot the original data, fitted values, and residuals
plt.figure(figsize=(18, 8))

# Original data
plt.subplot(311)
plt.plot(data['Population'], label='Original', color='blue')
plt.plot(fitted_values, label='Fitted', color='orange')
plt.title('Original vs Fitted Values')
plt.legend(loc='upper left')

# Residuals
plt.subplot(312)
plt.plot(residuals, label='Residuals', color='red')
plt.title('Residuals')
plt.axhline(0, color='black', linestyle='--')
plt.legend(loc='upper left')

# ACF and PACF of residuals
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.subplot(313)
plot_acf(residuals, lags=20, ax=plt.gca(), title='ACF of Residuals')
plt.tight_layout()
plt.savefig('KoreaQuarterly_SEATSArima.PNG')
plt.show()

# Summary of the model

print(results.summary())
