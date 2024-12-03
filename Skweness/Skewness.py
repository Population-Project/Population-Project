import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(17)
# Load your dataset (replace 'your_file.csv' with the actual path to your CSV file)
file_path = r'D:\Hikmat system\IMLab\LabMembers\Population_project\Methods\pca\data/Korea(Only_Population).csv'  # Simulated file path for demo
data = pd.read_csv(file_path)

# Assuming the 'Population' column exists in your CSV
log_population = np.log(data['Population'])  # Apply logarithmic scaling

# Generate positive and negative skewed distributions
positive_skew_adjusted = np.random.exponential(scale=np.std(log_population), size=len(data['Population'])) + np.mean(log_population)
negative_skew_adjusted = -np.random.exponential(scale=np.std(log_population), size=len(data['Population'])) + np.mean(log_population)

# Generate a symmetric distribution (smooth, similar to a normal distribution)
mean_value = np.mean(log_population)
symmetric_distribution = np.random.normal(loc=mean_value, scale=np.std(log_population), size=len(data['Population']))

# Set up the plot with the desired figure size
plt.figure(figsize=(34, 20))

# Plot the adjusted Positive Skew distribution
sns.kdeplot(positive_skew_adjusted, color='blue', label='Positive Skew', linewidth=6)

# Plot the adjusted Negative Skew distribution
sns.kdeplot(negative_skew_adjusted, color='red', label='Negative Skew', linewidth=6)

# Plot the smooth symmetric distribution
sns.kdeplot(symmetric_distribution, color='black', label='Symmetric Distribution', linewidth=6)

# Function to calculate the mode (peak) and annotate it
def calculate_mode(data, color, label):
    kde = sns.kdeplot(data, color=color, linewidth=3)
    kde_line = kde.get_lines()[-1]
    x, y = kde_line.get_data()
    mode_x = x[np.argmax(y)]
    mode_y = max(y)
    plt.text(mode_x, mode_y, f'', color=color, fontsize=30, ha='center', va='bottom')

# Calculate and annotate the modes
calculate_mode(positive_skew_adjusted, 'blue', 'Positive Skew')
calculate_mode(negative_skew_adjusted, 'red', 'Negative Skew')
calculate_mode(symmetric_distribution, 'black', 'Symmetric Distribution')

# Add labels and title with increased font size
plt.title('Positive Skew, Negative Skew, and Symmetric Distribution with Modes', fontsize=30)
plt.xlabel('Time', fontsize=30)
plt.ylabel('Frequency', fontsize=30)

# Set the font size for x and y axis tick labels
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# Add a vertical line at the median for reference
median_value = np.median(log_population)
plt.axvline(x=median_value, color='black', linestyle='--', label='Median')

# Add a legend with increased font size
plt.legend(fontsize=30)

# Enable grid
plt.grid(True)

# Save the plot as an image
output_path = 'skew_symmetric_distribution.png'
plt.savefig(output_path)

# Display the image path for the user
output_path
