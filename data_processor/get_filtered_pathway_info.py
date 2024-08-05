import pandas as pd
from tqdm import tqdm

"""
Filtering only the pathways included in Hetionet from the hierarchical information of gene pathways in Reactome.
"""

# Set the file paths
hNet_pathway_file_path = '/path/to/hetionet/data/hetionet-v1.0.hetmat/nodes/Pathway.tsv'
reactome_file_path = '/path/to/reactome/data/NCBI2Reactome_PE_All_Levels.txt'
filtered_reactome_file_path = '/path/to/save/filtered_with_HetioNet_pathway_information.txt'

# Load the Pathway nodes data into a Pandas DataFrame
hNet_pathway_df = pd.read_csv(hNet_pathway_file_path, sep='\t')

# Display the first few rows of the dataframe to verify the contents
print(hNet_pathway_df.head())

# Define column names for the Reactome data
column_names = [
    'id',
    'gene_identifier',
    'gene_description',
    'pathway_identifier',
    'pathway_url',
    'pathway_name',
    'evidence_code',
    'species'
]

# Load the Reactome data into a Pandas DataFrame
reactome_df = pd.read_csv(reactome_file_path, sep='\t', header=None, names=column_names)

# Display the first few rows of the dataframe to verify the contents
print(reactome_df.head())

# Extract pathway names from HetioNet dataframe
hNet_pathway_names = set(hNet_pathway_df['name'])  # Use a set for faster membership testing

# Filter reactome_df to keep only rows where the pathway_name is in hNet_pathway_names
filtered_reactome_df = reactome_df[reactome_df['pathway_name'].isin(hNet_pathway_names)]

# Display the first few rows of the filtered dataframe to verify the contents
print(filtered_reactome_df.head())

# Save the filtered dataframe to a new file
filtered_reactome_df.to_csv(filtered_reactome_file_path, sep='\t', index=False)
