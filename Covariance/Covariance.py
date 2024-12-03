import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Drop the 'Year' column for covariance analysis if it's not needed
data_no_year = data.drop('Year', axis=1)

# Clean the data: remove commas and convert to numeric
data_no_year_cleaned = data_no_year.replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce')

# Compute the Covariance matrix
covariance_matrix = data_no_year_cleaned.cov()

# Feature importance: extract covariance of all features with "Population"
population_covariance = covariance_matrix['Population'].drop('Population')

# Sort the covariances by absolute value (importance)
sorted_population_covariance = population_covariance.abs().sort_values(ascending=False)

# Save the sorted covariance values to a CSV file
#output_csv_path = 'results/sorted_population_covariance.csv'
#sorted_population_covariance.to_csv(output_csv_path)

# Create a bar plot for feature importance based on covariance with Population
plt.figure(figsize=(40, 20))  # Increase the figure size
sorted_population_covariance.plot(kind='bar', color='blue')

# Increase font sizes for title, labels, and ticks
#plt.title("Feature Importance Based on Covariance with Population", fontsize=28)
plt.ylabel("Importance", fontsize=35, color='black')
plt.xlabel("Features", fontsize=35, color='black')
plt.xticks(rotation=90, fontsize=34, color='black')
plt.yticks(fontsize=34, color='black')

plt.tight_layout()

# Save the plot as an image file
output_plot_path = 'Covreance.png'
plt.savefig(output_plot_path)  # Save with higher resolution

# Show the plot
plt.show()
