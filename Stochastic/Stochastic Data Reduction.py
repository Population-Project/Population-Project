# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:38:55 2024

@author: IAM Lab
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Load the CSV file containing Year and Population columns
file_path = 'data.csv'  # Replace with the actual path to your CSV file
data = pd.read_csv(file_path)

# Apply Gaussian smoothing to the Population column
data['Smoothed Population'] = gaussian_filter1d(data['Population'], sigma=4)

# Plotting the original and smoothed population over time
plt.figure(figsize=(14, 7))
plt.plot(data['Year'], data['Population'], color='red', linestyle='--', linewidth=2, label='Original Population')  # Dashed red line for original data
plt.plot(data['Year'], data['Smoothed Population'], color='blue', linewidth=3, label='Smoothed Population')  # Solid blue line for smoothed data

# Adding a gradient fill under the smoothed line for emphasis
#plt.fill_between(data['Year'], data['Smoothed Population'], color='royalblue', alpha=0.2)

# Adding grid, labels, and title with enhanced fonts
plt.xlabel('Year', fontsize=14)
plt.ylabel('Population', fontsize=14)
#plt.title('Population Over Time (Original vs. Smoothed)', fontsize=16, fontweight='bold', color='darkslategray')
plt.legend(fontsize=12)

# Adding a grid for easier reading of values
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Stochastic Data Reduction.jpg', dpi=300)
# Display the plot
plt.show()

