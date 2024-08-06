import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering, SpectralClustering

file_path = f'your/path/data.csv' 
original_df = pd.read_csv(file_path, index_col=0) # Load data
scaler = MinMaxScaler() # Scaler

cluster = AgglomerativeClustering(2) # n_clusters = 2
data_clustering = cluster.fit_predict(original_df)
sil_score = silhouette_score(original_df, data_clustering)
print(f'Silhouette Score: {sil_score}')

data_scaled = scaler.fit_transform(original_df) # Scale with MinMaxScaler
scaled_df = pd.DataFrame(data_scaled, columns=original_df.columns, index=original_df.index) # Get column and index from original dataframe
plt.figure()
clustergrid = sns.clustermap(scaled_df, cmap='magma', figsize=(70, 10), dendrogram_ratio=(0.1, 0.2))
plt.show()

ordered_columns = [scaled_df.columns[i] for i in clustergrid.dendrogram_col.reordered_ind] # Get column order from clustered data

colors = []
for sample in ordered_columns: # Visualization of label
    label = int(sample.split('_')[2])
    severity = int(sample.split('_')[3])
    if label == 1:
        if severity == 1:
            colors.append('lightblue')
        elif severity == 2:
            colors.append('blue')
        else:
            colors.append('darkblue')
    else:
        if severity == 1:
            colors.append('lightcoral')
        elif severity == 2:
            colors.append('red')
        else:
            colors.append('darkred')

# color_labels dictionary
color_labels = {
    'lightblue': 'RP, Severity 1',
    'blue': 'RP, Severity 2',
    'darkblue': 'RP, Severity 3',
    'lightcoral': 'DP, Severity 1',
    'red': 'DP, Severity 2',
    'darkred': 'DP, Severity 3'
}

# plot color bar
plt.figure(figsize=(80, 2))
for i, color in enumerate(colors):
    plt.bar(i, 1, color=color, edgecolor='none', width=1)
plt.axis('off')
plt.show()



