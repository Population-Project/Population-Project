

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'data.csv'
data = pd.read_csv(file_path)
# Extract the population data and corresponding years
population_data = data['Population'].values
years = data['Year'].values

# Calculate dispersion measures
mean_population = np.mean(population_data)
range_population = np.ptp(population_data)  # Range: max - min
variance_population = np.var(population_data)  # Variance
std_dev_population = np.std(population_data)  # Standard deviation

# Calculate deviations (difference from the mean for each point)
deviations = population_data - mean_population

# Plot the population data and its dispersion measures
plt.figure(figsize=(18, 8))

# Plot the original population data
plt.plot(years, population_data, label='Population Data', color='blue', lw=2)

# Plot deviations from the mean as points
plt.scatter(years, deviations, color='orange', label='Deviation from Mean', alpha=0.7)

# Add range, variance, and standard deviation as text on the plot
plt.axhline(mean_population, color='red', linestyle='--', label='Mean')
plt.text(1955, mean_population + 5e6, f'Range: {range_population:.0f}', color='purple', fontsize=10)
plt.text(1955, mean_population + 3e6, f'Variance: {variance_population:.0e}', color='green', fontsize=10)
plt.text(1955, mean_population + 1e6, f'Std. Dev.: {std_dev_population:.0f}', color='blue', fontsize=10)

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Population / Deviation')
plt.title('Population Data with Dispersion Measures')
plt.legend()
plt.savefig('Despersion1.png')
# Show the plot
plt.tight_layout()
plt.show()
