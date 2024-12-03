import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = 'data.csv'  # Replace with the path to your CSV file

data = pd.read_csv(file_path)
# Remove commas and convert all values to numeric, if they are strings with commas
data_cleaned = data.drop(columns=['Year']).applymap(lambda x: float(str(x).replace(',', '')) if isinstance(x, str) else x)

# Impute missing values in the dataset with the mean
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data_cleaned)

# Initialize t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)

# Apply t-SNE for the dataset to project to 2D
data_2d = tsne.fit_transform(data_imputed)

# Compute correlation matrix
corr_matrix = np.corrcoef(data_imputed, rowvar=False)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# Plot t-SNE projection with labeled axes
axs[0].scatter(data_2d[:, 0], data_2d[:, 1], color='blue')
axs[0].set_title("t-SNE Projection")
axs[0].set_xlabel("t-SNE Component 1")
axs[0].set_ylabel("t-SNE Component 2")

# Plot correlation matrix with labeled axes
sns.heatmap(corr_matrix, ax=axs[1], annot=False, cmap='Greens', cbar=False, square=True)
axs[1].set_title("Correlation Matrix")
axs[1].set_xlabel("Features")
axs[1].set_ylabel("Features")

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('TSNE1.jpg', dpi=300)
plt.show()
