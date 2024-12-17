# Let's reload the CSV, apply histogramming, and then compare it with the original data (before and after histogramming)
import numpy as   np
import pandas as pd
import matplotlib.pyplot as plt
# Reload the dataset
file_path = 'data.csv'
data = pd.read_csv(file_path)
data=data.iloc[:,1:]
numeric_data_preprocessed = data.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', ''), errors='coerce'))

# # Now we will apply histogramming to reduce the data using binning (5 bins as an example)
def create_histogram_features(column, num_bins=5):
    hist, _ = np.histogram(column.dropna(), bins=num_bins)
    return hist

# # Apply the function to reduce the data by summarizing it into histograms (5 bins for each column)
reduced_data_hist = numeric_data_preprocessed.apply(create_histogram_features)
# # Function to visualize and compare histograms before and after histogramming
def compare_histogram(original_data, histogrammed_data, column_name, num_bins=5):
    plt.figure(figsize=(18, 12))

    # Original data histogram
    plt.subplot(1, 2, 1)
    plt.hist(original_data[column_name].dropna(), bins=num_bins, color='blue', alpha=0.7)
    plt.title(f'Original {column_name} Histogram',fontsize=12)
    plt.xlabel(column_name,  fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Smoothed data histogram
    plt.subplot(1, 2, 2)
    plt.bar(range(num_bins), histogrammed_data[column_name], color='green', alpha=0.7)
    plt.title(f'Histogrammed {column_name}', fontsize=12)
    plt.xlabel('Bins', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Show the comparison
    plt.savefig("histogrammin.png")
    plt.show()
    

# # Apply and compare histograms for 'Population'
compare_histogram(numeric_data_preprocessed, reduced_data_hist, 'Population', num_bins=5)

# # Apply and compare histograms for 'f1'
#compare_histogram(numeric_data_preprocessed, reduced_data_hist, 'f1', num_bins=5)
