import pandas as pd
from tqdm import tqdm

"""
The process of generating the mapping between gene expression data and pathways through the file 
`filtered_with_HetioNet_pathway_information.txt`, which is the result of executing `get_filtered_pathway_info.py`

The list of genes from the gene expression data and the mapping of pathways are obtained as results. 
The number of samples and the number of genes are confirmed.
"""

# Set the file paths
reactome_file_path = '/path/from/get_filtered_pathway_info/filtered_with_HetioNet_pathway_information.txt'
gene_expression_file_path = '/path/to/gene/expression/data/geDATA.csv'
output_file_path = '/path/to/save/pathway_mapping.csv'

# Read the Reactome data file with low_memory=False
df = pd.read_csv(reactome_file_path, sep='\t', header=0, low_memory=False)

# Display the first few rows to verify the contents
print("Original Reactome DataFrame:")
print(df.head())

# Read the gene expression data file
gene_expression_df = pd.read_csv(gene_expression_file_path)

# Create a list of genes, excluding the last column
gene_list = gene_expression_df.columns[:-1].tolist()

# Initialize lists to store results
not_found_genes = []
result_dfs = []

# Filter the Reactome dataframe based on the gene list
for gene in tqdm(gene_list, desc='Processing'):
    genes = [g.strip() for g in gene.split('or')]
    matched = False
    for g in genes:
        filtered_df = df[df['gene_description'].str.contains(g, regex=False)]
        if not filtered_df.empty:
            result_dfs.append(filtered_df)
            matched = True
    if not matched:
        not_found_genes.append(gene)

# Concatenate all filtered dataframes
if result_dfs:
    result_df = pd.concat(result_dfs).drop_duplicates()
else:
    result_df = pd.DataFrame(columns=df.columns)

# Print genes not found in gene_description column
if not_found_genes:
    print(f"Genes not found in gene_description column: {', '.join(not_found_genes)}")

# Remove genes not found in gene_description column from gene_expression_df
gene_expression_df_filtered = gene_expression_df.drop(columns=not_found_genes)

# Select only the required columns
temp_df = result_df[['gene_identifier', 'gene_description', 'pathway_identifier', 'pathway_name']].copy()

# Remove the text within brackets from the 'gene_description' column
temp_df.loc[:, 'gene_description'] = temp_df['gene_description'].str.replace(r'\[.*?\]', '', regex=True).str.strip()

# Rename the 'gene_description' column to 'gene'
temp_df.rename(columns={'gene_description': 'gene'}, inplace=True)

# Reset the index
temp_df.reset_index(drop=True, inplace=True)

# Display the modified dataframe
print("\nModified DataFrame:")
print(temp_df.head())

# Save the modified dataframe to a CSV file
temp_df.to_csv(output_file_path, index=False)

# Print final sample and gene count
num_samples = gene_expression_df_filtered.shape[0]
num_genes = gene_expression_df_filtered.shape[1] - 1  # Exclude the label column
print(f"\nFinal number of samples: {num_samples}")
print(f"Final number of genes: {num_genes}")
