# Smoothing


import pandas as pd


# Load the CSV file again for processing
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Preprocess the data by converting numerical columns to numeric and handling issues like commas
numeric_data_preprocessed = data.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', ''), errors='coerce'))

# Applying the moving average (smoothing) to the entire preprocessed dataset in one go
smoothed_data_processed = numeric_data_preprocessed.rolling(window=20, min_periods=1).mean()

# Display the first few rows of the smoothed data after preprocessing
smoothed_data_processed.head()

import matplotlib.pyplot as plt

# Plot original and smoothed 'Population' data over time
plt.figure(figsize=(18, 8))

# Original data plot
plt.plot(numeric_data_preprocessed['Year'], numeric_data_preprocessed['Population'], label='Original Population', color='red', linestyle='--')

# Smoothed data plot
plt.plot(smoothed_data_processed['Year'], smoothed_data_processed['Population'], label='Smoothed Population', color='blue')

# Add another comparison for feature 'f1'
# plt.plot(numeric_data_preprocessed['Year'], numeric_data_preprocessed['f1'], label='Original f1', color='green', linestyle='--')
# plt.plot(smoothed_data_processed['Year'], smoothed_data_processed['f1'], label='Smoothed f1', color='green')

# Add title and labels with increased font size
plt.title('Comparison of Original and Smoothed Data Over Time', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Values', fontsize=16)
plt.legend(fontsize=16)

# Set font size for tick labels on x and y axes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# Show plot


plt.savefig("tochastic_data_reduction.png")

plt.show()
