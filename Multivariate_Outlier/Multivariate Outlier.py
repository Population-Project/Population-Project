# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 01:19:46 2024

@author: IAM Lab
"""

import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

file_path = 'data.csv'
data = pd.read_csv(file_path)
# Drop 'Year' column since it is not numerical
data_numeric = data.drop(columns=['Year'])
#ata_numeric['f37']= pd.to_numeric(data_numeric['f37'].astype(str).str.replace(',', ''), errors='coerce')
# Clean data by removing rows with NaN values and converting columns to proper numeric types
data_clean = data_numeric.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', ''), errors='coerce')).dropna()

# Try different methods of outlier detection


# 1. Principal Component Analysis (PCA) for outlier detection
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_clean)
pca_outlier_detector = EllipticEnvelope(contamination=0.05)
pca_outlier_predictions = pca_outlier_detector.fit_predict(pca_result)

# 2. Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest_predictions = iso_forest.fit_predict(data_clean)

elliptic_envelope = EllipticEnvelope(contamination=0.05)
elliptic_predictions = elliptic_envelope.fit_predict(data_clean)

# Create a plot comparing the outliers detected by all four methods

plt.figure(figsize=(18, 8))

# Elliptic Envelope-based detection
plt.subplot(4, 1, 1)
plt.scatter(data_clean.index, data_clean.mean(axis=1), c=np.where(elliptic_predictions == 1, 'blue', 'red'), label='Data points')
plt.title('Outliers Detected using Elliptic Envelope (Mahalanobis Distance)')
plt.xlabel('Index')
plt.ylabel('Mean Value')

# PCA-based detection
plt.subplot(4, 1, 2)
plt.scatter(data_clean.index, data_clean.mean(axis=1), c=np.where(pca_outlier_predictions == 1, 'blue', 'red'), label='Data points')
plt.title('Outliers Detected using PCA')
plt.xlabel('Index')
plt.ylabel('Mean Value')

# Isolation Forest-based detection
plt.subplot(4, 1, 3)
plt.scatter(data_clean.index, data_clean.mean(axis=1), c=np.where(iso_forest_predictions == 1, 'blue', 'red'), label='Data points')
plt.title('Outliers Detected using Isolation Forest')
plt.xlabel('Index')
plt.ylabel('Mean Value')

plt.tight_layout()
plt.show()