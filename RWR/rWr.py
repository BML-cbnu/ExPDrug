import pandas as pd
import re
import networkx as nx
import numpy as np
import random
import scipy.sparse
from tqdm import tqdm
import json

"""
The process of performing a random walk using the "pathway_information.csv" file obtained from expNet's main.py

"""

# Load and filter the pathways from CSV
df = pd.read_csv('/ensure/your/result/pathway/info/yourResult.csv')
df = df.sort_values(by='p_value', ascending=True)
filtered_df_005 = df[df['p_value'] <= 0.05]
filtered_df_001 = df[df['p_value'] <= 0.01]

"""
The nodes and edges data are read from Hetionet. You can choose them as you like,
but the following example uses the nodes and edges used in the experiments of this paper.
 
"""

# Load node data
diseases = pd.read_csv('/set/your/hetioNet/data/path/nodes/Disease.tsv', sep='\t')
genes = pd.read_csv('/set/your/hetioNet/data/path/nodes/Gene.tsv', sep='\t')
pathways = pd.read_csv('/set/your/hetioNet/data/path/nodes/Pathway.tsv', sep='\t')
compounds = pd.read_csv('/set/your/hetioNet/data/path/nodes/Compound.tsv', sep='\t')

# Load edge data
CuG = scipy.sparse.load_npz('/set/your/hetioNet/data/path/edges/CuG.sparse.npz')
CdG = scipy.sparse.load_npz('/set/your/hetioNet/data/path/edges/CdG.sparse.npz')
CbG = scipy.sparse.load_npz('/set/your/hetioNet/data/path/edges/CbG.sparse.npz')
GpPW = scipy.sparse.load_npz('/set/your/hetioNet/data/path/edges/GpPW.sparse.npz')
CtD = scipy.sparse.load_npz('/set/your/hetioNet/data/path/edges/CtD.sparse.npz')
DaG = scipy.sparse.load_npz('/set/your/hetioNet/data/path/edges/DaG.sparse.npz')

# Build the graph
G = nx.Graph()
# Add nodes to the graph
for i, disease in diseases.iterrows():
    G.add_node(f'disease_{i}', **disease.to_dict())
for i, gene in genes.iterrows():
    G.add_node(f'gene_{i}', **gene.to_dict())
for i, pathway in pathways.iterrows():
    G.add_node(f'pathway_{i}', **pathway.to_dict())
for i, compound in compounds.iterrows():
    G.add_node(f'compound_{i}', **compound.to_dict())

# Add edges to the graph
for i, j in zip(*CuG.nonzero()):
    G.add_edge(f'compound_{i}', f'gene_{j}')
for i, j in zip(*CdG.nonzero()):
    G.add_edge(f'compound_{i}', f'gene_{j}')
for i, j in zip(*CbG.nonzero()):
    G.add_edge(f'compound_{i}', f'gene_{j}')
for i, j in zip(*GpPW.nonzero()):
    G.add_edge(f'gene_{i}', f'pathway_{j}')
for i, j in zip(*CtD.nonzero()):
    G.add_edge(f'compound_{i}', f'disease_{j}')
for i, j in zip(*DaG.nonzero()):
    G.add_edge(f'disease_{i}', f'gene_{j}')

def create_subgraph(selected_pathways, G):
    subgraph_nodes = set(selected_pathways)
    for pathway in selected_pathways:
        for neighbor in G.neighbors(pathway):
            if neighbor.startswith('gene_') or neighbor.startswith('compound_') or neighbor.startswith('disease_'):
                subgraph_nodes.add(neighbor)
                for second_neighbor in G.neighbors(neighbor):
                    if second_neighbor.startswith('gene_') or second_neighbor.startswith('compound_') or second_neighbor.startswith('disease_'):
                        subgraph_nodes.add(second_neighbor)
    return G.subgraph(subgraph_nodes)

def perform_random_walks_average(H, seed_nodes, seed_values, num_steps=2000, restart_prob=0.7, num_iterations=2000, scale_factor=1000000):
    # Normalize the seed values to the range [0, 1]
    min_seed_value = min(seed_values)
    max_seed_value = max(seed_values)
    if max_seed_value - min_seed_value != 0:
        normalized_seed_values = [(value - min_seed_value) / (max_seed_value - min_seed_value) for value in seed_values]
    else:
        normalized_seed_values = [1.0 for value in seed_values]
    
    # Scale the normalized seed values to ensure they are sufficiently large
    scaled_seed_values = [value * scale_factor for value in normalized_seed_values]
    
    # Initialize a dictionary to keep track of the total visit counts across all iterations
    total_visited_counts = {node: 0 for node in H.nodes}
    
    for _ in tqdm(range(num_iterations), desc="random walking", leave=False):
        values = {node: 0 for node in H.nodes}
        visited_counts = {node: 0 for node in H.nodes}
        
        for seed_node, seed_value in zip(seed_nodes, scaled_seed_values):
            values[seed_node] = seed_value
        
        visited_nodes = set()
        for seed_node in seed_nodes:
            current_node = seed_node
            visited_nodes.add(current_node)
            for i in range(1, num_steps + 1):
                neighbors = list(nx.classes.function.all_neighbors(H, current_node))
                visited_counts[current_node] += 1
                if neighbors:
                    transfer_value = values[current_node] / (len(neighbors) * i)
                    values[current_node] -= transfer_value * len(neighbors)
                    if np.random.rand() > restart_prob:
                        current_node = np.random.choice(neighbors)
                    else:
                        current_node = seed_node
                    values[current_node] += transfer_value
                    visited_nodes.add(current_node)
        
        # Add the visit counts of the current iteration to the total visit counts
        for node in visited_counts:
            total_visited_counts[node] += visited_counts[node]
    
    # Average the visit counts over the number of iterations
    average_visited_counts = {node: count / num_iterations for node, count in total_visited_counts.items()}
    
    return H.subgraph(visited_nodes), average_visited_counts, values

def print_node_composition(H_visited):
    num_genes = sum(1 for node in H_visited.nodes if node.startswith('gene_'))
    num_compounds = sum(1 for node in H_visited.nodes if node.startswith('compound_'))
    num_diseases = sum(1 for node in H_visited.nodes if node.startswith('disease_'))
    num_pathways = sum(1 for node in H_visited.nodes if node.startswith('pathway_'))

    print(f"Number of gene nodes: {num_genes}")
    print(f"Number of compound nodes: {num_compounds}")
    print(f"Number of disease nodes: {num_diseases}")
    print(f"Number of pathway nodes: {num_pathways}")

def print_sorted_compound_visits(visited_counts, values, top_n):
    sorted_compounds_by_visit = sorted([(node, count) for node, count in visited_counts.items() if node.startswith('compound_')], key=lambda x: x[1], reverse=True)
    sorted_compounds_by_spread = sorted([(node, values[node]) for node in visited_counts.keys() if node.startswith('compound_')], key=lambda x: x[1], reverse=True)

    top_n_by_visit = sorted_compounds_by_visit[:top_n]
    top_n_by_spread = sorted_compounds_by_spread[:top_n]

    print(f"Top {top_n} compounds by visit count (descending):")
    for compound, count in top_n_by_visit:
        compound_name = G.nodes[compound].get('name', 'Unknown')
        print(f"{compound} ({compound_name}): {count}, Spread Value: {values[compound]}")

    print(f"\nTop {top_n} compounds by spread value (descending):")
    for compound, spread_value in top_n_by_spread:
        compound_name = G.nodes[compound].get('name', 'Unknown')
        print(f"{compound} ({compound_name}): {visited_counts[compound]}, Spread Value: {spread_value}")

    return top_n_by_visit, top_n_by_spread

def save_cytoscape_json(H_visited, visited_counts, values, filename):
    # Prepare data in Cytoscape JSON format
    cytoscape_data = {
        "data": {},
        "elements": {
            "nodes": [],
            "edges": []
        }
    }

    for node in H_visited.nodes:
        node_data = {
            "data": {
                "id": node,
                "name": G.nodes[node].get('name', node),
                "visited_count": visited_counts.get(node, 0),
                "spread_value": values.get(node, 0.0)
            }
        }
        cytoscape_data["elements"]["nodes"].append(node_data)
    
    for edge in H_visited.edges:
        edge_data = {
            "data": {
                "source": edge[0],
                "target": edge[1]
            }
        }
        cytoscape_data["elements"]["edges"].append(edge_data)

    # Save to JSON file
    with open(filename, 'w') as json_file:
        json.dump(cytoscape_data, json_file, indent=4)
    print(f"Cytoscape data saved to {filename}")

def save_compound_info_to_csv(visited_counts, values, filename):
    compound_data = []
    for node, count in visited_counts.items():
        if node.startswith('compound_'):
            compound_name = G.nodes[node].get('name', 'Unknown')
            spread_value = values[node]
            compound_data.append([compound_name, count, spread_value])
    
    compound_df = pd.DataFrame(compound_data, columns=['Compound', 'Visited', 'Spread Value'])
    compound_df.to_csv(filename, index=False)
    print(f"Compound information saved to {filename}")

top_n = 50

# Create subgraph for p-value <= 0.01 pathways
selected_pathways_001 = [f'pathway_{i}' for i, pathway in pathways.iterrows() if pathway['name'] in filtered_df_001['Pathway_Name'].values]
seed_values_001 = filtered_df_001.sort_values(by='LRP_Score', ascending=False)['LRP_Score'].values
H_001 = create_subgraph(selected_pathways_001, G)

# Perform random walks with averaging and print results for p-value <= 0.01
H_visited_001, average_visited_counts_001, final_values_001 = perform_random_walks_average(H_001, selected_pathways_001, seed_values_001)

# Print node composition
print("Subgraph node composition for p-value <= 0.01 pathways:")
print_node_composition(H_visited_001)

# Print sorted compound visits and get top n by visit and spread
print(f"Random walk results for p-value <= 0.01 pathways (Top {top_n}):")
top_n_by_visit_001, top_n_by_spread_001 = print_sorted_compound_visits(average_visited_counts_001, final_values_001, top_n)

# Save Cytoscape JSON for p-value <= 0.01 pathways
save_cytoscape_json(H_visited_001, average_visited_counts_001, final_values_001, '/set/your/result/output/folder/yourfilename.json')

# Save compound information to CSV for p-value <= 0.01 pathways
save_compound_info_to_csv(average_visited_counts_001, final_values_001, '/set/your/result/output/folder/yourfilename.csv')
