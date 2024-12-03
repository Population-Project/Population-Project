# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 23:41:48 2024

@author: IAM Lab
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = 'data.csv'  # Replace with the actual path to your CSV file
data = pd.read_csv(file_path)

# Drop the 'Year' column if it's not needed for statistical analysis
data_no_year = data.drop(columns=['Year','Population'], errors='ignore')

# Clean the data: remove commas and convert to numeric
data_cleaned = data_no_year.replace(',', '', regex=True).apply(pd.to_numeric, errors='coerce')

# Calculate descriptive statistics
mean_values = data_cleaned.mean()
median_values = data_cleaned.median()
std_dev_values = data_cleaned.std()

# Plot Mean of Features (Sky Blue)
plt.figure(figsize=(18, 8))
mean_values.plot(kind='bar', color='skyblue', edgecolor='black')
#plt.title("Mean of Features")
plt.ylabel("Mean")
plt.xlabel("Features")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('mean.jpg')
plt.show()

# Plot Median of Features (Green)
plt.figure(figsize=(18, 8))
median_values.plot(kind='bar', color='orange', edgecolor='black')
#plt.title("Median of Features")
plt.ylabel("Median")
plt.yticks(rotation=0)
plt.xlabel("Features")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('median.jpg')
plt.show()

# Plot Standard Deviation of Features (Orange)
plt.figure(figsize=(18, 8))
std_dev_values.plot(kind='bar', color='green', edgecolor='black')
#plt.title("Standard Deviation of Features")
plt.ylabel("Standard Deviation")
plt.xlabel("Features")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('Standard Deviation.jpg')
plt.show()