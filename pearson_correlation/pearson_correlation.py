import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load your data
file_path = 'Data.csv'
data = pd.read_csv(file_path)

# Drop the 'Year' column for correlation analysis if it's not needed
data_no_year = data.drop('Year', axis=1)

# Clean the data: remove commas and convert to numeric
data_no_year_cleaned = data_no_year.replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce')

# Standardize the cleaned data
scaler = StandardScaler()
data_scaled_cleaned = scaler.fit_transform(data_no_year_cleaned.dropna())

# Compute the Pearson correlation matrix
pearson_corr_cleaned = np.corrcoef(data_scaled_cleaned.T)

# Convert the correlation matrix into a DataFrame for easy handling
correlation_data = pd.DataFrame(pearson_corr_cleaned, columns=data_no_year_cleaned.columns, index=data_no_year_cleaned.columns)

# Feature importance: extract correlation of all features with "Population"
population_correlation = correlation_data['Population'].drop('Population')

# Sort the correlations by absolute value (importance)
sorted_population_correlation = population_correlation.abs().sort_values(ascending=False)

# Save the sorted correlation values to a CSV file
output_csv_path = 'sorted_population_correlation.csv'
sorted_population_correlation.to_csv(output_csv_path)

# Create a bar plot for feature importance
plt.figure(figsize=(18, 12))
sorted_population_correlation.plot(kind='bar', color='blue')

#plt.title("Feature Importance Based on Correlation with Population")
plt.ylabel("Importance", fontsize=20)
plt.xlabel("Features",fontsize=20)
plt.xticks(rotation=90,fontsize=20)

plt.xticks(fontsize=20)  # Increase x-axis tick size
plt.yticks(fontsize=20)
plt.tight_layout()
# Save the plot as an image file
output_plot_path = 'Pearson_feature_importance_plot.png'
plt.savefig(output_plot_path)

# Show the plot
plt.show()
