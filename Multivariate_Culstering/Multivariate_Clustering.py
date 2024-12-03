# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv('Data.csv')

# Clean the dataset by removing commas and converting to numeric values
cleaned_data = data.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', ''), errors='coerce'))

# Filling NaN values with the mean of each column
cleaned_data_filled = cleaned_data.fillna(cleaned_data.mean())

# Standardize the data (important for clustering)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cleaned_data_filled)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Apply PCA to reduce to 2 components for visualization purposes
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Add the PCA components to the dataset for plotting
data['PCA1'] = pca_data[:, 0]
data['PCA2'] = pca_data[:, 1]

# Identify outliers
outlier_indices = []
outlier_threshold = 2  # Define how many standard deviations away to consider an outlier

# Calculate distances to cluster centroids
for cluster in range(3):
    cluster_data = data[data['Cluster'] == cluster]
    centroid = kmeans.cluster_centers_[cluster]
    distances = np.linalg.norm(scaled_data[cluster_data.index] - centroid, axis=1)
    
    # Identify outliers
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    # Find indices of outliers
    outliers = cluster_data[distances > (mean_distance + outlier_threshold * std_distance)]
    outlier_indices.extend(outliers.index)

# Plotting
plt.figure(figsize=(18, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, palette='Set1', s=100, alpha=0.6)

# Highlight outliers
outlier_data = data.loc[outlier_indices]
outlier_plot = plt.scatter(outlier_data['PCA1'], outlier_data['PCA2'], color='red', s=150, label='Outliers', edgecolor='black')

# Calculate the cluster centers in PCA space
cluster_centers = kmeans.cluster_centers_
pca_centers = pca.transform(cluster_centers)

# Plot the cluster centers (means) without labels
mean_plot = plt.scatter(pca_centers[:, 0], pca_centers[:, 1], s=300, c='black', marker='X', label='Cluster Centers')

# Adding elliptical "clouds" around each cluster (2 standard deviations)
for center in pca_centers:
    plt.gca().add_patch(plt.Circle(center, 2, color='black', fill=False, linestyle='--', alpha=0.5))

plt.title('Multivariate Clustering with Outliers Identified')

# Remove x and y axis labels by setting them to empty strings
plt.xlabel('')
plt.ylabel('')
plt.grid(True)
# Show the legend for outliers and mean
plt.legend(handles=[outlier_plot, mean_plot], loc='upper right')
plt.savefig('Multivariate_Clustering_with_Outliers.png')
plt.show()

# Show the mean of each cluster in original space (optional)
cluster_means = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=cleaned_data_filled.columns)
print("Cluster Means:")
print(cluster_means)

# Print outlier information
print("Outlier indices:", outlier_indices)
print("Outlier data:\n", outlier_data)
