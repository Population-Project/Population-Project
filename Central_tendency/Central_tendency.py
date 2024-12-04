import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Re-load the provided CSV file to retrieve the data
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Extract the population data and corresponding years
population_data = data['Population'].values
years = data['Year'].values

# Calculate central tendency measures
mean_population = np.mean(population_data)
median_population = np.median(population_data)
mode_population = stats.mode(population_data)[0][0]  # Select the first mode
trimmed_mean_population = stats.trim_mean(population_data, 0.1)  # 10% trimmed mean

# Plot the population data with central tendency measures
plt.figure(figsize=(18, 8))

# Plot the original population data
plt.plot(years, population_data, label='Population Data', color='blue', lw=2)

# Plot central tendency measures as horizontal lines
plt.axhline(mean_population, color='red', linestyle='--', label=f'Mean: {mean_population:.0f}')
plt.axhline(median_population, color='green', linestyle='--', label=f'Median: {median_population:.0f}')
plt.axhline(mode_population, color='orange', linestyle='--', label=f'Mode: {mode_population:.0f}')
plt.axhline(trimmed_mean_population, color='purple', linestyle='--', label=f'Trimmed Mean: {trimmed_mean_population:.0f}')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Population Data with Central Tendency Measures')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
