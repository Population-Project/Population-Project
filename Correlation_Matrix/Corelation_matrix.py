import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data

file_path = 'data.csv'
data = pd.read_csv(file_path)

# Drop the 'Year' column for correlation analysis if it's not needed
data_no_year_population = data.drop(['Year', 'Population'], axis=1)

# Clean the data: remove commas and convert to numeric
data_no_year_cleaned = data_no_year_population.replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce')

# Compute the Pearson Correlation Matrix
correlation_matrix = data_no_year_cleaned.corr(method='pearson')

# Save the correlation matrix to a CSV file
#output_csv_path = 'D:/Hikmat system/IMLab/LabMembers/Population_project/Methods/pca/Correlation_based_results/correlation_matrix.csv'
#correlation_matrix.to_csv(output_csv_path)

# Create a heatmap of the correlation matrix with increased font size
plt.figure(figsize=(60, 60))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
            xticklabels=data_no_year_cleaned.columns, 
            yticklabels=data_no_year_cleaned.columns)

# Increase font sizes for title, axes, and ticks
plt.title("Correlation Matrix Heatmap", fontsize=30)
plt.xticks(rotation=90, fontsize=30)
plt.yticks(fontsize=30)

plt.tight_layout()

# Save the heatmap as an image file
output_plot_path = 'correlation_matrix_heatmap.png'
plt.savefig(output_plot_path)  # Higher resolution

# Show the heatmap
plt.show()
