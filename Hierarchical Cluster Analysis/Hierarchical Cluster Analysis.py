# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 23:52:51 2024

@author: IAM Lab
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


# Load the data from the uploaded CSV file
path = 'UN_Korea(Oringal).csv'
file_path = path
data = pd.read_csv(file_path)


data = pd.read_csv(file_path, thousands=',')  # Load with thousands separator handled
data.head()
# Select only numeric columns for clustering
data_numeric = data.select_dtypes(include=['number'])

# Check for and handle NaN and infinite values
data_numeric = data_numeric.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
data_numeric = data_numeric.dropna()  # Drop rows with NaN values

# Perform hierarchical clustering on numeric data
linked = linkage(data_numeric, method='ward')

# Extract the 'Year' column for labeling in the dendrogram
years = data_numeric['Year'].astype(str).tolist()

# Plot the dendrogram with years at the leaves
plt.figure(figsize=(18, 8))
dendrogram(linked, labels=years, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical clustering')
plt.xlabel('Year')
plt.ylabel('Euclidean Distance')
plt.tight_layout()
plt.savefig('Hierarvhcal cluste.png')
plt.show()