import time
import numpy as np
import pandas as pd
from PyEMD import CEEMDAN
from sklearn.cluster import KMeans
from MSE import multiscale_sample_entropy

# -------- CEEMDAN + MSE + KMeans --------
def ceemdan_mse_clustering(df, column_name):

    # Obtain the original time series
    signal = df[column_name].values

    # CEEMDAN decomposition
    ceemdan = CEEMDAN()
    imfs = ceemdan.ceemdan(signal)

    # Calculate the mean multiscale sample entropy for each IMF
    entropy_means = []
    for imf in imfs:
        mse = multiscale_sample_entropy(imf, max_scale=10, m=2, r=0.2)
        mse_mean = np.nanmean(mse)
        entropy_means.append(mse_mean)
    entropy_means = np.array(entropy_means).reshape(-1, 1)

    # Filter NaN
    valid_indices = ~np.isnan(entropy_means.flatten())
    entropy_means_valid = entropy_means[valid_indices]
    imfs_valid = [imfs[i] for i in range(len(imfs)) if valid_indices[i]]

    # KMeans
    kmeans = KMeans(n_clusters=2, random_state=0, algorithm="elkan")
    labels = kmeans.fit_predict(entropy_means)

    # Build results table
    result_df = pd.DataFrame({
        'IMF_Index': np.where(valid_indices)[0],
        'MSE_Mean': entropy_means_valid.flatten(),
        'Cluster_Label': labels
    })

    return imfs_valid, result_df

# Import DataFrame
df = pd.read_excel('dataset.xlsx')

# Execute function
imfs, result_df = ceemdan_mse_clustering(df, 'target')
print(result_df)

