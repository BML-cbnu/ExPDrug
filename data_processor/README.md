
# Gene Pathway Mapping with HetioNet and Reactome

This repository contains scripts to filter pathway data from Reactome based on HetioNet pathways and map gene expression data to these pathways. The repository consists of two main scripts:

1. `get_filtered_pathway_info.py`: Filters Reactome pathways to include only those found in HetioNet.
2. `generate_pathway_mapping.py`: Maps gene expression data to the filtered pathways.

## Files

- `get_filtered_pathway_info.py`: Script to filter Reactome pathway data.
- `generate_pathway_mapping.py`: Script to map gene expression data to filtered pathways.
- `README.md`: This file, providing an overview and instructions.

## Setup

Ensure you have the following dependencies installed:
- pandas
- tqdm

You can install these using pip:
```bash
pip install pandas tqdm
```

## Usage

### Step 1: Filter Reactome Pathway Data

Run `get_filtered_pathway_info.py` to filter the Reactome pathway data based on the pathways included in HetioNet.

**Inputs:**
- Pathway nodes file from HetioNet: `/path/to/hetionet/data/hetionet-v1.0.hetmat/nodes/Pathway.tsv`
- Reactome data file: `/path/to/reactome/data/NCBI2Reactome_PE_All_Levels.txt`

**Output:**
- Filtered Reactome pathway data: `/path/to/save/filtered_with_HetioNet_pathway_information.txt`

**Command:**
```bash
python get_filtered_pathway_info.py
```

### Step 2: Generate Pathway Mapping

Run `generate_pathway_mapping.py` to map gene expression data to the filtered pathways.

**Inputs:**
- Filtered Reactome pathway data: `/path/from/get_filtered_pathway_info/filtered_with_HetioNet_pathway_information.txt`
- Gene expression data: `/path/to/gene/expression/data/geDATA.csv`

**Output:**
- Pathway mapping file: `/path/to/save/pathway_mapping.csv`

**Command:**
```bash
python generate_pathway_mapping.py
```

### Example Workflow

1. **Filter Pathways:**
   - Ensure the paths in `get_filtered_pathway_info.py` are correctly set.
   - Run the script:
     ```bash
     python get_filtered_pathway_info.py
     ```

2. **Map Gene Expression Data:**
   - Ensure the paths in `generate_pathway_mapping.py` are correctly set.
   - Run the script:
     ```bash
     python generate_pathway_mapping.py
     ```

### Code Details

#### `get_filtered_pathway_info.py`
This script performs the following tasks:
- Loads HetioNet pathway nodes.
- Loads Reactome pathway data.
- Filters the Reactome data to include only pathways found in HetioNet.
- Saves the filtered pathways to a file.

#### `generate_pathway_mapping.py`
This script performs the following tasks:
- Loads the filtered Reactome pathway data.
- Loads gene expression data.
- Filters the Reactome data based on gene descriptions found in the gene expression data.
- Generates a mapping between genes and pathways.
- Saves the pathway mapping to a file.

### Notes
- Ensure the file paths are correctly set in the scripts before running them.
- Review the filtered pathway data and gene expression data to ensure correctness.
