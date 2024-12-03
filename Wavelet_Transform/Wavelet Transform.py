# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 01:00:59 2024

@author: IAM Lab
"""

# Re-import necessary libraries and perform the transformation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Re-load the provided CSV file to retrieve the data
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Extract the population data and corresponding years
population_data = data['Population'].values
years = data['Year'].values

# Perform Continuous Wavelet Transform (CWT) again
scales = np.arange(1, 128)  # Adjust the scale range as needed
coefficients, frequencies = pywt.cwt(population_data, scales, 'cmor', sampling_period=years[1] - years[0])

# Now let's regenerate the plot as requested
fig, axs = plt.subplots(2, 1, figsize=(18, 8))

# Plot 1: Wavelet transform (time-frequency plot)
axs[0].imshow(np.abs(coefficients), extent=[years.min(), years.max(), scales.min(), scales.max()],
              cmap='Blues', aspect='auto', vmax=abs(coefficients).max(), vmin=0)
axs[0].set_title('Time-Frequency Representation (Wavelet Transform)')
axs[0].set_ylabel('Scale')
axs[0].set_xlabel('Year')

# Plot 2: Original signal (population data)
axs[1].plot(years, population_data, color='blue')
axs[1].set_title('Original Signal (Population Data)')
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Population')

# Layout adjustment
plt.tight_layout()
plt.show()