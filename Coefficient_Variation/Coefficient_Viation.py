# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:41:57 2024

@author: IAM Lab
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset (update the file path to your dataset)
df = pd.read_csv('2. Quarterly.csv')  # Update with your actual dataset path

# Ensure the 'Year' column is in datetime format
df['Year'] = pd.to_datetime(df['Year'], format='%Y-%m-%d')  # Adjust format as necessary

# Set the 'Year' column as the index
df.set_index('Year', inplace=True)

# Calculate the mean and standard deviation of the Population
mean_population = df['Population'].mean()
std_population = df['Population'].std()

# Calculate the Coefficient of Variation (CV)
cv = std_population / mean_population

# Print the CV
print(f"Coefficient of Variation: {cv:.4f}")

# Set a threshold for identifying anomalies based on CV
threshold = 2  # You can adjust this value based on your needs

# Identify anomalies based on CV
df['Anomaly'] = np.where((df['Population'] - mean_population) > (threshold * std_population), 1, 0)

# Count anomalies
num_anomalies = df['Anomaly'].sum()
print(f"Total anomalies detected: {num_anomalies}")

# Plotting the results
plt.figure(figsize=(18, 8))
plt.plot(df.index, df['Population'], label='Population', color='royalblue', linewidth=2)

# Highlight anomalies with larger markers and different color
plt.scatter(df.index[df['Anomaly'] == 1], df['Population'][df['Anomaly'] == 1], color='orange', label='Anomalies', s=150, edgecolor='k', alpha=0.7)

# Adding title and labels with enhanced font sizes
plt.title('Population and Detected Anomalies Based on Coefficient of Variation', fontsize=24, fontweight='bold')
plt.xlabel('Year', fontsize=18)
plt.ylabel('Population', fontsize=18)

# Adding grid and customizations
plt.axhline(y=mean_population, color='green', linestyle='--', label='Mean Population', linewidth=1.5)
plt.grid(which='both', linestyle='--', linewidth=0.7)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('Coefficient_variation.jpg') 
# Show the plot
plt.show()
