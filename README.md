
# ExPDRUG Pipeline

This repository contains the ExPDRUG pipeline, designed for drug discovery using gene expression data. Below you will find instructions on how to set up and run the pipeline.

## File Structure

- `.util/logger.py` : Contains logging functions and result output features.
- `.util/config.py` : Handles file input/output paths and hyperparameter adjustments.
- `.util/data_processor.py` : Manages data handling for model training, including:
  - Handling masking matrices between layers.
  - Shuffling functionality for masking matrices in permutation testing.
- `.util/model.py` : Defines the model for training and relevance score computation, including the loss function.
- `.util/trainer.py` : Handles model training, relevance score computation, and permutation test functionality.
- `main.py` : The main script to run the entire pipeline.

## 1. Data Preparation

### Data Sources

- **Hetionet**: [Hetionet GitHub](https://github.com/hetio/hetionet)
- **Reactome**: [Reactome Data Download](https://reactome.org/download-data)

### Gene Expression Data

- **COVID-19**: [COVID-19 Data](https://coda.nih.go.kr)
- **COVID-19 Severity Score Information**: [Severity Score Information](https://www.kcmo.kr/COVID/)
- **GBM**: [GBM Data](https://github.com/DataX-JieHao/PASNet)
- **Alzheimer's**: [Alzheimer's Data](https://github.com/ChihyunPark/DNN_for_ADprediction)

Ensure that the raw gene expression files for experiments are placed in the `data` folder.

## 2. Data Filtering for ExPNet Training

Run the scripts in the `data_processor` folder to filter the data for training ExPNet. Refer to the pipeline in that folder for specific instructions.

## 3. Model Training, Relevance Score Computation, and Permutation Test

1. Set the file paths and hyperparameters in the `.util/config.py` file.
2. Execute the `main.py` script.

```bash
python main.py
```

## 4. Drug Discovery

Run the scripts in the `RWR` folder to perform drug discovery.

## Installation

Install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
