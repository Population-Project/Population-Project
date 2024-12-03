import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

colors = ['red', 'blue', 'green', 'purple', 'orange']
# Load the data from the uploaded CSV file
path = 'Multi-Variable_data.csv'
file_path = path
data = pd.read_csv(file_path)

# Normalize the numeric columns except 'Year'
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data.drop('Year', axis=1)), columns=data.columns[1:])

# Create the new scatterplot matrix with the requested changes to the diagonal plots
fig, axes = plt.subplots(len(scaled_data.columns), len(scaled_data.columns), figsize=(18, 8))

# Loop through each axis and fill the plots
for i, row_var in enumerate(scaled_data.columns):
    for j, col_var in enumerate(scaled_data.columns):
        ax = axes[i, j]
        if i == j:
            # Diagonal: Bar (histogram) plots with unfilled bins for each variable, overlaying all variables with distinct colors
            for k, var in enumerate(scaled_data.columns):
                ax.hist(scaled_data[var], bins=15, histtype='step', color=colors[k], label=var)
        else:
            # Off-diagonal: Scatter plots with two colors, each representing the respective variable
            ax.scatter(scaled_data[col_var], scaled_data[row_var], color=colors[i], label=row_var, alpha=0.6)
            ax.scatter(scaled_data[col_var], scaled_data[col_var], color=colors[j], label=col_var, alpha=0.6)

        # Only add labels on the edge plots (not all of them)
        if j == 0:
            ax.set_ylabel(row_var)
        if i == len(scaled_data.columns) - 1:
            ax.set_xlabel(col_var)

# Adjust layout to prevent overlaps
plt.tight_layout()

# Save the figure as an image file
output_path = 'scatterplot_matrix.png'  # Change the filename and path as needed
plt.savefig(output_path)

# Show the plot
plt.show()
