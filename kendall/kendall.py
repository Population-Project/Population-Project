import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
file_path = '1. Korea-new.csv'
data = pd.read_csv(file_path)

# Drop the 'Year' column for correlation analysis if it's not needed
data_no_year = data.drop('Year', axis=1)

# Clean the data: remove commas and convert to numeric
data_no_year_cleaned = data_no_year.replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce')

# Compute Kendall's Tau Correlation matrix
kendall_tau_corr = data_no_year_cleaned.corr(method='kendall')

# Feature importance: extract correlation of all features with "Population"
population_kendall_correlation = kendall_tau_corr['Population'].drop('Population')

# Sort the correlations by absolute value (importance)
sorted_population_kendall_correlation = population_kendall_correlation.abs().sort_values(ascending=False)

# Plotting settings to make the plot more attractive
plt.figure(figsize=(40, 20))  # Set figure size to be tall like the previous horizontal plot

# Use seaborn style for aesthetics
sns.set(style="whitegrid")

# Create a vertical bar plot with a single blue color for bars
sorted_population_kendall_correlation.plot(kind='bar', color='blue', edgecolor='black', width=0.8)

# Add labels and title with enhanced styling
plt.ylabel("Importance", fontsize=35, color='black')
plt.xlabel("Features", fontsize=35, color='black')
#plt.title("Feature Importance Based on Kendall's Tau Correlation with Population", fontsize=20, fontweight='bold', color='darkslategray')

# Increase tick size for readability and rotate x-axis labels for long feature names
plt.xticks(rotation=90, fontsize=30, color='black')
plt.yticks(fontsize=30, color='black')

# Add grid lines for the y-axis
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout to prevent cutting off feature names
plt.tight_layout()

# Save the plot as an image file
output_plot_path = 'Kdl_rotated_blue_tall.png'
plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')  # Save with higher resolution and tight layout

# Display the plot
plt.show()
