import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

seed_value =2
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Load and preprocess data
data = pd.read_csv("Data.csv")
data = data.drop(columns=["Year"])  # Drop 'Year' column if present

# Handle non-numeric values and missing values
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(data.mean())

# Separate features and target
X = data.drop(columns=["Population"])
y = data["Population"]

# Normalize features and target
scaler_X = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Define and train the full model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="linear")
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=500, batch_size=10, validation_split=0.2, verbose=1)

# Feature importance analysis based on input layer weights
input_weights = model.layers[0].get_weights()[0]
feature_importance = np.mean(np.abs(input_weights), axis=1)

# Organize and display the top features
features = X.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
top_features = importance_df.head(52)


print(top_features)
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 18))
plt.bar(top_features["Feature"], top_features["Importance"], color='blue')
plt.xticks(rotation=90, fontsize=20)  # Rotate feature names for better readability
plt.xlabel("Feature", fontsize=20)
plt.ylabel("Feature Importance (Mean Absolute Weight)", fontsize=20)
#.title("Top 10 Most Important Features", fontsize=20)
plt.xticks(fontsize=20)  # Increase x-axis tick size
plt.yticks(np.arange(0, 1.45, 0.05), fontsize=20) 
plt.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
plt.tight_layout()  # Adjust layout to fit all feature names
plt.savefig("ANN.png")  # Save the figure
plt.show()
