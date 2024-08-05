# Pathway and Random Walk Analysis

This repository contains scripts to analyze biological pathways using random walks on a heterogeneous network. The analysis identifies important compounds related to pathways with significant p-values and visualizes the results.

## Overview

The script performs the following tasks:
1. Loads and filters pathways based on p-value thresholds.
2. Constructs a heterogeneous network using node and edge data.
3. Creates subgraphs for selected pathways.
4. Performs random walks to identify key compounds.
5. Prints and saves the results.

## Data Preparation

Ensure you have the following files in your specified paths:
- Pathway information(LRP score and p-value) result CSV file (`yourResult.csv`).
- Node data files.
- Edge data files.

Update the paths in the script to match your file locations.

## Script Details

### Loading Data

The script begins by loading pathway results from a CSV file and filtering them based on p-value thresholds. It also loads node and edge data from TSV and NPZ files respectively.

### Building the Graph

Nodes and edges are added to a NetworkX graph to construct the heterogeneous network. Nodes represent diseases, genes, pathways, and compounds, while edges represent interactions between these entities.

### Creating Subgraphs

A subgraph is created for pathways with p-values <= 0.01. This subgraph includes the selected pathways and their neighboring nodes.

### Performing Random Walks

Random walks with restarts are performed on the subgraph. Seed nodes are initialized with values from the LRP_Score column. The random walks identify the frequency of visits and spread values for each node.

### Printing Results

The script prints the composition of nodes in the subgraph and the top compounds by visit count and spread value.

### Saving Results

Results are saved in two formats:
- JSON for visualization (`yourfilename.json`).
- CSV file with compound information (`yourfilename.csv`).
