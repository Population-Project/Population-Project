import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from minisom import MiniSom
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Set random seeds for reproducibility
seed_value = 2
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Create results directory

# Load the data
data = pd.read_csv('data.csv')

# Convert 'Population' column to numeric, replacing invalid entries with NaN
data['Population'] = pd.to_numeric(data['Population'], errors='coerce')

# Convert all features to numeric (use 'coerce' to handle invalid values) and drop non-numeric rows
data = data.apply(pd.to_numeric, errors='coerce')

# Handle missing values by filling NaNs with the column mean (or other strategies)
data.fillna(data.mean(), inplace=True)

# Drop 'Year' column from features (assuming 'Year' is not a feature used for training)
features = data.drop(columns=['Year'])

# Normalize the features
scaler = StandardScaler()
normalized_data = scaler.fit_transform(features)

# Initialize and train SOM
som_grid_size = (20, 20)
som = MiniSom(x=som_grid_size[0], y=som_grid_size[1], input_len=normalized_data.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(normalized_data)
som.train_random(data=normalized_data, num_iteration=500)

# Visualize and save SOM feature cluster distance map
plt.figure(figsize=(18, 8))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # Distance map as a heatmap
cbar = plt.colorbar(label='Distance')
cbar.ax.tick_params(labelsize=20)  # Increase tick size on color bar
cbar.set_label('Distance', fontsize=20)  # Increase label font size
plt.xlabel("X (Grid Columns)", fontsize=20)
plt.ylabel("Y (Grid Rows)", fontsize=20)
plt.xticks(range(0, 21, 2), fontsize=20)  # Increase x-axis tick size
plt.yticks(range(0, 21, 2), fontsize=20)
plt.grid(True, color='white', linestyle='--', linewidth=0.5)
plt.savefig("SOM.png")
plt.show()
