

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

# Load your data
file_path = 'Data.csv'
data = pd.read_csv(file_path)

# Drop the 'Year' column for analysis if it's not needed
data_no_year = data.drop('Population', axis=1)

# Clean the data: remove commas and convert to numeric
data_no_year_cleaned = data_no_year.replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce')

# Fill any missing values with 0 (or another strategy if preferred)
data_no_year_cleaned = data_no_year_cleaned.fillna(0)

# Apply SVD (TruncatedSVD allows us to choose a number of components)
# Set n_components to be less than the number of features
n_components = min(len(data_no_year_cleaned.columns) - 1, len(data_no_year_cleaned))
svd = TruncatedSVD(n_components=n_components)
X_svd = svd.fit_transform(data_no_year_cleaned)

# Extract the feature importance (loading of the features on the first singular vector)
feature_importance = pd.Series(svd.components_[0], index=data_no_year_cleaned.columns)

# Sort the features by importance (absolute values)
sorted_feature_importance = feature_importance.abs().sort_values(ascending=False)

# Save the sorted feature importances to a CSV file
sorted_feature_importance.to_csv('sorted_population_svd_importance.csv')

# Create a bar plot for feature importance based on SVD
plt.figure(figsize=(14, 8))
sorted_feature_importance.plot(kind='bar', color='blue')

# Customize plot appearance
#plt.title("Feature Importance Based on SVD", fontsize=16)
plt.ylabel("Contribution of Features", fontsize=16)  # Y-axis explanation: Absolute value of loadings
plt.xlabel("Features", fontsize=16)
plt.xticks(rotation=90, fontsize=16)
plt.yticks(fontsize=16)

# Add grid to the plot
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()

# Save the plot
plt.savefig('SVD.png', bbox_inches='tight')

# Show the plot
plt.show()

