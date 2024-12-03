import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, kurtosis
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(1)
# Create a directory to save results
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load the CSV file into a pandas DataFrame
file_path_data = r'Data.csv'  # Replace with your actual file path if needed
population_data_user = pd.read_csv(file_path_data)

# Extract and clean the "Year" and "Population" columns
years = population_data_user['Year'].dropna()
population_column = population_data_user['Population'].dropna()

# Ensure both columns have the same length after dropping NaN values
years = years[:len(population_column)]

# Function to generate 3D surface plots in subplots and calculate kurtosis
def plot_3d_distributions(x_data, y_data_list, titles, filename):
    fig = plt.figure(figsize=(18, 8))  # Set figure size to 18x8

    # Loop through and create subplots for each distribution
    for i, (y_data, title) in enumerate(zip(y_data_list, titles)):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')  # Create a 1x3 subplot grid
        
        # Calculate kurtosis (using Fisher kurtosis, so 0 is normal distribution)
        kurt_value = kurtosis(y_data, fisher=True)  # Using fisher=True for excess kurtosis

        # Create meshgrid for 3D plotting
        X, Y = np.meshgrid(x_data, y_data)

        # Use Kernel Density Estimation (KDE) to estimate density
        kde = gaussian_kde(y_data)

        # Calculate Z values (density) for each point in the meshgrid
        Z = np.reshape(kde(Y.ravel()), Y.shape)

        # Plot 3D surface
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

        # Set plot title including excess kurtosis value
        ax.set_title(f"{title}\nExcess Kurtosis: {kurt_value:.2f}", fontsize=14)

        # Keep X and Y axis but remove Z-axis ticks and label
        ax.set_xlabel('Year (X-axis)')
        ax.set_ylabel('Population (Y-axis)')
        ax.set_zlabel('')  # Remove Z-axis label
        ax.set_zticks([])   # Remove Z-axis ticks

    # Save the plot to the results directory
    plt.savefig(os.path.join(results_dir, filename))
    plt.show()

# Create data for all three distributions
platykurtic_data = np.random.uniform(low=min(population_column), high=max(population_column), size=len(years))
mesokurtic_data = np.random.normal(loc=np.mean(population_column), scale=np.std(population_column), size=len(years))
leptokurtic_data = np.random.laplace(loc=np.mean(population_column), scale=np.std(population_column), size=len(years))

# List of data and titles
y_data_list = [platykurtic_data, mesokurtic_data, leptokurtic_data]
titles = [
    "Platykurtic Distribution\n(Low Peakedness, Broad and Flat, Kurtosis < 0)",
    "Mesokurtic Distribution\n(Normal Bell Curve, Kurtosis ≈ 0)",
    "Leptokurtic Distribution\n(High Peakedness, Fat Tails, Kurtosis > 0)"
]

# Generate the plots and save them as one figure
plot_3d_distributions(years, y_data_list, titles, "combined_kurtosis_distributions.png")

print(f"Combined 3D plot saved in the '{results_dir}' folder.")
