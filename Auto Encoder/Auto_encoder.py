import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load your dataset (update the file path to your dataset)
df = pd.read_csv('data.csv')  # Update with your actual dataset path

# Select relevant columns for anomaly detection
X = df[['Live births(persons)', 'Deaths(persons)', 'Natural increase(persons)', 
         'Crude birth rate(per 1,000 population)', 'Crude death rate(per 1,000 population)', 
         'Natural increase rate(per 1,000 population)', 'Total fertility rate(persons)', 
         'Masculinity of birth(persons)', 'Marriages(cases)', 
         'Crude marriage rate(per 1,000 population)', 'Divorces(cases)', 
         'Crude divorce rate(per 1,000 population)', 'Life expectancy at birth-total(age)', 
         'Life expectancy at birth-male(age)', 'Life expectancy at birth-female(age)', 
         'Population']].dropna()

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the Autoencoder model
input_dim = X_scaled.shape[1]  # Input dimension
encoding_dim = 8  # Dimension of the encoding, you can adjust this

# Input Layer
input_layer = Input(shape=(input_dim,))
# Encoder Layer
encoder = Dense(encoding_dim, activation='relu')(input_layer)
# Decoder Layer
decoder = Dense(input_dim, activation='sigmoid')(encoder)

# Autoencoder Model
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the Autoencoder
autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=8, shuffle=True, validation_split=0.2)

# Predict using the trained autoencoder
X_pred = autoencoder.predict(X_scaled)

# Calculate reconstruction error
reconstruction_error = np.mean(np.square(X_scaled - X_pred), axis=1)

# Set a threshold for anomalies
threshold = np.percentile(reconstruction_error, 90)  # Using 95th percentile as threshold

# Identify anomalies
df['Anomaly'] = reconstruction_error > threshold

# Display results
print(df[['year', 'Live births(persons)', 'Deaths(persons)', 'Natural increase(persons)', 
           'Anomaly']])

# Visualize reconstruction error
plt.figure(figsize=(18, 8))
plt.plot(df['year'], reconstruction_error, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.xlabel('Year')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error and Anomaly Detection')

# Set x-ticks to show every 2 years
plt.xticks(np.arange(min(df['year']), max(df['year']) + 1, 2))

# Enable grid lines
plt.grid()
plt.legend()
plt.savefig('Autoencoder.png') 
plt.show()
