import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import CCA
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding
from sklearn.impute import SimpleImputer
DSName = "Data"
# Function to read and preprocess the CSV data
def preprocess_csv(input_file, pred_att):
    # Read the CSV file
    dataframe = pd.read_csv(input_file)
    
    # Clean numeric columns with commas
    for col in dataframe.columns:
        if dataframe[col].dtype == object:  # Check if the column contains non-numeric values
            try:
                dataframe[col] = dataframe[col].str.replace(',', '').astype(float)
            except ValueError:
                pass  # If conversion fails (e.g., non-numeric strings), leave the column as is
    
    # Fill NaN values with the mean of each column
    imputer = SimpleImputer(strategy='mean')
    dataframe = pd.DataFrame(imputer.fit_transform(dataframe), columns=dataframe.columns)
    
    # Separate the date column into year and month
    if 'Date' in dataframe.columns:
        dataframe['Year'] = dataframe['Date'].apply(lambda x: int(str(x)[:4]))
        dataframe['Month'] = dataframe['Date'].apply(lambda x: int(str(x)[4:6]))
        dataframe = dataframe.drop('Date', axis=1)
    
    # Separate features and labels
    features = dataframe.drop(pred_att, axis=1)
    features_name = features.columns.tolist()
    labels = dataframe[pred_att]
    labels = np.array(labels)

    return features, labels
def plot_feature_importance(importance, feature_names, title, save_path, DSName):
    plt.figure(figsize=(20, 3))
    plt.bar(feature_names, importance, color='blue')
    plt.title(title + ' - Feature Importance' + ' - '+DSName)
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
# Load and preprocess the data

input_file =  DSName + '.csv'
pred_att = 'Population'

features, labels = preprocess_csv(input_file, pred_att)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Generate synthetic labels for LDA (binarize population into high/low categories)
y = (labels > np.median(labels)).astype(int)


# Apply Factor Analysis
fa = FactorAnalysis(n_components=2)
X_fa = fa.fit_transform(X_scaled)
fa_importance = np.abs(fa.components_).sum(axis=0)
plot_feature_importance(fa_importance, features.columns.tolist(), 'Factor Analysis','FactorAnalysis.png', DSName)
