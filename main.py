from util.config import get_config, set_random_seeds
from util.data_processor import DataProcessor, MatrixHandler, loadEXPdataset
from util.model import EXPNet, LRPNet
from util.trainer import ModelTrainer, p_Test, LRPHandler
from util.logger import Logger, ResultPrinter

import time
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    config = get_config()
    main_logger = Logger("Main Process")
    
    main_logger.info("Setting random seeds...")
    set_random_seeds(42)
    
    main_logger.info("Loading data...")
    df_filtered_gene_expression, df_filtered_gene_pathway = DataProcessor.load_data(config)
    main_logger.info("Data loaded successfully.")
    
    main_logger.info("Processing data...")
    gene_list, gene_count, sorted_pathways, hidden_one_list = DataProcessor.process_data(
        df_filtered_gene_expression, df_filtered_gene_pathway, 
        config["params"]["min_genes_threshold"], config["params"]["max_genes_threshold"],
        config["params"]["top_30_nodes"], config["params"]["middle_30_nodes"], config["params"]["bottom_40_nodes"]
    )
    main_logger.info("Data processed successfully.")

    main_logger.info(f"Number of genes: {gene_count}")
    main_logger.info(f"Number of pathways: {len(sorted_pathways)}")
    main_logger.info(f"Number of hidden nodes: {len(hidden_one_list)}")

    main_logger.info("Loading or creating matrices...")
    gene_pathway_matrix, input_to_h1_masking, h1_to_pathway_masking = MatrixHandler.load_or_create_matrices(
        gene_list, gene_count, sorted_pathways, df_filtered_gene_pathway, hidden_one_list, config
    )
    main_logger.info("Matrices loaded/created successfully.")

    main_logger.info("Loading features and labels...")
    features = df_filtered_gene_expression[gene_list].values
    labels = df_filtered_gene_expression['label'].values

    main_logger.info("Creating dataset...")
    dataset = loadEXPdataset(features, labels)

    main_logger.info("Defining model parameters...")
    input_dim = gene_count
    hidden_one_dim = len(hidden_one_list)
    pathway_dim = len(sorted_pathways)
    hidden_two_dim = 128  # Example value
    output_dim = len(np.unique(labels))

    model_params = {
        'input_dim': input_dim,
        'hidden_one_dim': hidden_one_dim,
        'pathway_dim': pathway_dim,
        'hidden_two_dim': hidden_two_dim,
        'output_dim': output_dim,
        'input_to_h1_masking': input_to_h1_masking,
        'h1_to_pathway_masking': h1_to_pathway_masking
    }

    save_model_path = config["file_paths"]["model_save"]

    main_logger.info("Initializing KFoldTraining for regular training...")
    kfold_training = ModelTrainer(EXPNet, dataset, model_params, save_model_path, k_folds=config["training"]["k_folds"], 
                                    epochs=200, batch_size=config["training"]["batch_size"], learning_rate=0.001, weight_decay=3e-4,
                                    patience=config["training"]["patience"], device=device, use_amp=True, save_model=True)

    main_logger.info("Starting training with model saving enabled...")
    best_model_state_dict, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, best_accuracy, best_roc_auc, best_precision, best_f1 = kfold_training.train(main_logger)

    main_logger.info("Running initial LRP computation...")
    kFold_best_model = LRPNet(**model_params).to(device)
    kFold_best_model.load_state_dict(best_model_state_dict)
    obtained_lrp_scores = LRPHandler.compute_lrp_for_dataset_multiple_times(kFold_best_model, dataset, target_class=1)

    start = time.time()

    main_logger.info("Running permutation test...")
    ptest_trainer = p_Test(LRPNet, dataset, model_params, save_model_path, epochs=1000, batch_size=config["training"]["batch_size"], learning_rate=0.001, weight_decay=3e-4, device=device, use_amp=True)
    K = config["params"]["num_permutation_iterations"]
    all_lrp_scores = ptest_trainer.run_permutation_test(K, gene_pathway_matrix, h1_to_pathway_masking, target_class=1)
    
    main_logger.info("Computing p-values...")
    p_values = ptest_trainer.get_p_values_for_all_pathways(obtained_lrp_scores, all_lrp_scores)

    top_pathways, top_indices = LRPHandler.get_top_lrp_scores(obtained_lrp_scores, sorted_pathways, df_filtered_gene_pathway, top_n=50)
    top_pathways_with_p_values = [(pathway_name, score, gene_count, p_values[idx]) for (pathway_name, score, gene_count), idx in zip(top_pathways, top_indices)]

    ResultPrinter.print_training_results(main_logger, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, best_accuracy, best_roc_auc, best_precision, best_f1)
    ResultPrinter.print_top_lrp_scores(main_logger, top_pathways_with_p_values)
    csv_save_path = config["file_paths"]["result_save"]
    ResultPrinter.print_p_values(main_logger, p_values, sorted_pathways, obtained_lrp_scores, csv_save_path)

    main_logger.info("Permutation test and p-value computation completed.")
    end = time.time()
    print(f"pTEST time... {end - start:.5f} sec")
from util.config import get_config, set_random_seeds
from util.data_processor import DataProcessor, MatrixHandler, loadEXPdataset
from util.model import EXPNet, LRPNet
from util.trainer import ModelTrainer, p_Test, LRPHandler
from util.logger import Logger, ResultPrinter

import time
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    config = get_config()
    main_logger = Logger("Main Process")
    
    # Set random seeds
    main_logger.info("Setting random seeds...")
    set_random_seeds(42)
    
    # Load data
    main_logger.info("Loading data...")
    df_filtered_gene_expression, df_filtered_gene_pathway = DataProcessor.load_data(config)
    main_logger.info("Data loaded successfully.")
    
    # Process data
    main_logger.info("Processing data...")
    gene_list, gene_count, sorted_pathways, hidden_one_list = DataProcessor.process_data(
        df_filtered_gene_expression, df_filtered_gene_pathway, 
        config["params"]["min_genes_threshold"], config["params"]["max_genes_threshold"],
        config["params"]["top_30_nodes"], config["params"]["middle_30_nodes"], config["params"]["bottom_40_nodes"]
    )
    main_logger.info("Data processed successfully.")

    main_logger.info(f"Number of genes: {gene_count}")
    main_logger.info(f"Number of pathways: {len(sorted_pathways)}")
    main_logger.info(f"Number of hidden nodes: {len(hidden_one_list)}")

    # Load or create matrices
    main_logger.info("Loading or creating matrices...")
    gene_pathway_matrix, input_to_h1_masking, h1_to_pathway_masking = MatrixHandler.load_or_create_matrices(
        gene_list, gene_count, sorted_pathways, df_filtered_gene_pathway, hidden_one_list, config
    )
    main_logger.info("Matrices loaded/created successfully.")

    # Load features and labels
    main_logger.info("Loading features and labels...")
    features = df_filtered_gene_expression[gene_list].values
    labels = df_filtered_gene_expression['label'].values

    # Create dataset
    main_logger.info("Creating dataset...")
    dataset = loadEXPdataset(features, labels)

    # Define model parameters
    main_logger.info("Defining model parameters...")
    input_dim = gene_count
    hidden_one_dim = len(hidden_one_list)
    pathway_dim = len(sorted_pathways)
    hidden_two_dim = 128  # Example value
    output_dim = len(np.unique(labels))

    model_params = {
        'input_dim': input_dim,
        'hidden_one_dim': hidden_one_dim,
        'pathway_dim': pathway_dim,
        'hidden_two_dim': hidden_two_dim,
        'output_dim': output_dim,
        'input_to_h1_masking': input_to_h1_masking,
        'h1_to_pathway_masking': h1_to_pathway_masking
    }

    save_model_path = config["file_paths"]["model_save"]

    # Initialize KFoldTraining
    main_logger.info("Initializing KFoldTraining for regular training...")
    kfold_training = ModelTrainer(EXPNet, dataset, model_params, save_model_path, k_folds=config["training"]["k_folds"], 
                                    epochs=200, batch_size=config["training"]["batch_size"], learning_rate=0.001, weight_decay=3e-4,
                                    patience=config["training"]["patience"], device=device, use_amp=True, save_model=True)

    # Start training
    main_logger.info("Starting training with model saving enabled...")
    best_model_state_dict, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, best_accuracy, best_roc_auc, best_precision, best_f1 = kfold_training.train(main_logger)

    # Compute initial LRP scores
    main_logger.info("Running initial LRP computation...")
    kFold_best_model = LRPNet(**model_params).to(device)
    kFold_best_model.load_state_dict(best_model_state_dict)
    obtained_lrp_scores = LRPHandler.compute_lrp_for_dataset_multiple_times(kFold_best_model, dataset, target_class=1)

    start = time.time()

    # Run permutation test
    main_logger.info("Running permutation test...")
    ptest_trainer = p_Test(LRPNet, dataset, model_params, save_model_path, epochs=1000, batch_size=config["training"]["batch_size"], learning_rate=0.001, weight_decay=3e-4, device=device, use_amp=True)
    K = config["params"]["num_permutation_iterations"]
    all_lrp_scores = ptest_trainer.run_permutation_test(K, gene_pathway_matrix, h1_to_pathway_masking, target_class=1)
    
    # Compute p-values
    main_logger.info("Computing p-values...")
    p_values = ptest_trainer.get_p_values_for_all_pathways(obtained_lrp_scores, all_lrp_scores)

    # Print and save results
    top_pathways, top_indices = LRPHandler.get_top_lrp_scores(obtained_lrp_scores, sorted_pathways, df_filtered_gene_pathway, top_n=50)
    top_pathways_with_p_values = [(pathway_name, score, gene_count, p_values[idx]) for (pathway_name, score, gene_count), idx in zip(top_pathways, top_indices)]

    ResultPrinter.print_training_results(main_logger, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, best_accuracy, best_roc_auc, best_precision, best_f1)
    ResultPrinter.print_top_lrp_scores(main_logger, top_pathways_with_p_values)
    csv_save_path = config["file_paths"]["result_save"]
    ResultPrinter.print_p_values(main_logger, p_values, sorted_pathways, obtained_lrp_scores, csv_save_path)

    main_logger.info("Permutation test and p-value computation completed.")
    end = time.time()
    print(f"pTEST time... {end - start:.5f} sec")
