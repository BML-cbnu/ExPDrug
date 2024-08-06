# Preparing the Data
#### Data Format
The input data should be in CSV format.
The data should contain rows as samples and columns as features.
The CSV file should be named as CD4T.csv and placed in the specified directory.

#### File Structure
Ensure the path to your data file is correctly specified in the script. The current placeholder in the script is 'your/path/{celltype}.csv'.
Modify this path to the actual location of your CSV file.

# Running the Script
#### Load Data
The script loads data from the specified CSV file into a pandas DataFrame.

#### Scaling Data
Data is scaled using MinMaxScaler from scikit-learn to normalize the feature values between 0 and 1.

#### Clustering
Hierarchical clustering is performed using AgglomerativeClustering with 2 clusters. You can modify this parameter depends on your data.
The silhouette score is computed to evaluate the quality of the clustering.
The script creates a clustermap using seaborn's clustermap function with the scaled data.
The color of the bars representing samples is determined based on specific label and severity values extracted from sample names.

#### Visualization
The clustermap and a corresponding color bar are displayed using matplotlib.