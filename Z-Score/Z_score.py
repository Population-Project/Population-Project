# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:40:51 2024

@author: IAM Lab
"""

import pandas as pd

# Load the dataset from the file
df = pd.read_csv('data.csv')

# Selecting the relevant columns (Population, Births, Deaths)
columns_of_interest = ['Population', 'Births', 'Deaths']

# Calculate Z-scores for the selected columns
for column in columns_of_interest:
    df[f'Z-score_{column}'] = (df[column] - df[column].mean()) / df[column].std()

# Display the dataframe with Z-scores
print(df)

# Optionally save the output to a new CSV file
df.to_csv('zscore_output.csv', index=False)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the file
df = pd.read_csv('1. Republic of Korea-combine.csv')

# Selecting the relevant columns (Population, Births, Deaths)
columns_of_interest = ['Population', 'Births', 'Deaths']

# Calculate Z-scores for the selected columns
for column in columns_of_interest:
    df[f'Z-score_{column}'] = (df[column] - df[column].mean()) / df[column].std()

# Display the dataframe with Z-scores
print(df)

# Optionally save the output to a new CSV file
df.to_csv('zscore_output.csv', index=False)

# Plotting the Z-scores
plt.figure(figsize=(18, 8))

# Using seaborn to create a line plot for Z-scores
for column in columns_of_interest:
    plt.plot(df['Year'], df[f'Z-score_{column}'], label=f'Z-score of {column}')

# Customizing the plot
plt.title('Z-scores of Population, Births, and Deaths Over Time', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Z-score', fontsize=14)
plt.axhline(0, color='black', linestyle='--', linewidth=0.7)  # Line at Z-score = 0
plt.legend()
plt.grid()
plt.tight_layout()

# Show the plot
plt.show()
