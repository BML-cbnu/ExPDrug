import torch
import numpy as np
import os
from util.utils import get_config, set_random_seeds, Logger, ResultPrinter
from util.data_processor import DataProcessor
from util.dataset import loadEXPdataset
from util.model import EXPNet, LRPNet
from util.trainer import ModelTrainer
from util.handlers import LRPHandler, IGHandler, GSEAHandler

if __name__ == "__main__":
    config = get_config()
    set_random_seeds(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main_logger = Logger("Main Process")

    main_logger.info("Loading data...")
    df_filtered_gene_expression, df_filtered_gene_pathway = DataProcessor.load_data(config)
    main_logger.info("Data loaded successfully.")

    method = config["params"]["interpretation_method"]

    if method in ["LRP", "IG"]:
        main_logger.info("Processing data for model-based interpretation...")
        gene_list, gene_count, sorted_pathways, hidden_one_list = DataProcessor.process_data(
            df_filtered_gene_expression, df_filtered_gene_pathway, 
            config["params"]["min_genes_threshold"], config["params"]["max_genes_threshold"],
            config["params"]["top_30_nodes"], config["params"]["middle_30_nodes"], config["params"]["bottom_40_nodes"]
        )
        main_logger.info("Data processed successfully.")

        main_logger.info("Loading or creating matrices...")
        from util.data_processor import DataProcessor  # if needed
        data_dir = config["data_paths"]["data_dir"]
        # Load matrices
        from util.data_processor import DataProcessor
        from util.dataset import loadEXPdataset
        from util.model import EXPNet
        from util.trainer import ModelTrainer
        from util.handlers import LRPHandler, IGHandler

        gene_pathway_matrix, input_to_h1_masking, h1_to_pathway_masking = DataProcessor.load_or_create_matrices(
            gene_list, gene_count, sorted_pathways, df_filtered_gene_pathway, hidden_one_list, config
        )
        main_logger.info("Matrices loaded/created successfully.")

        main_logger.info("Loading features and labels...")
        features = df_filtered_gene_expression[gene_list].values
        labels = df_filtered_gene_expression['label'].values

        main_logger.info("Creating dataset...")
        dataset = loadEXPdataset(features, labels, device=device)

        main_logger.info("Defining model parameters...")
        input_dim = gene_count
        hidden_one_dim = len(hidden_one_list)
        pathway_dim = len(sorted_pathways)
        hidden_two_dim = 128
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

        main_logger.info("Initializing ModelTrainer for training...")
        model_trainer = ModelTrainer(EXPNet, dataset, model_params, save_model_path, 
                                     k_folds=config["training"]["k_folds"], 
                                     epochs=200, batch_size=config["training"]["batch_size"], 
                                     learning_rate=0.001, weight_decay=3e-4,
                                     patience=config["training"]["patience"], device=device, use_amp=True, save_model=True)

        main_logger.info("Starting training...")
        best_model_state_dict, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, avg_avg_precision, \
        best_accuracy, best_roc_auc, best_precision, best_f1, best_avg_precision = model_trainer.train(main_logger)

        if method == "LRP":
            main_logger.info("Running LRP interpretation...")
            lrp_model = LRPNet(**model_params).to(device)
            lrp_model.load_state_dict(best_model_state_dict)
            obtained_scores = LRPHandler.compute_lrp_for_dataset_multiple_times(
                lrp_model, dataset, target_class=config["params"]["target_class"], num_iterations=config["params"]["num_lrp_iterations"]
            )
            top_pathways, top_indices = LRPHandler.get_top_lrp_scores(obtained_scores, sorted_pathways, df_filtered_gene_pathway, top_n=50)
            ResultPrinter.print_training_results(main_logger, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, avg_avg_precision, 
                                                 best_accuracy, best_roc_auc, best_precision, best_f1, best_avg_precision)
            ResultPrinter.print_top_scores(main_logger, top_pathways, method_name="LRP")
            csv_save_path = config["file_paths"]["result_save"].replace(".csv", "_LRP.csv")
            ResultPrinter.save_scores_to_csv(sorted_pathways, obtained_scores, df_filtered_gene_pathway, csv_save_path, score_name="LRP_Score")

        elif method == "IG":
            main_logger.info("Running IG interpretation...")
            ig_model = EXPNet(**model_params).to(device)
            ig_model.load_state_dict(best_model_state_dict)
            obtained_scores = IGHandler.compute_integrated_gradients_for_dataset_multiple_times(
                ig_model, dataset, gene_pathway_matrix, target_class=config["params"]["target_class"], num_iterations=config["params"]["num_ig_iterations"]
            )
            top_pathways, top_indices = IGHandler.get_top_ig_scores(obtained_scores, sorted_pathways, df_filtered_gene_pathway, top_n=50)
            ResultPrinter.print_training_results(main_logger, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, avg_avg_precision, 
                                                 best_accuracy, best_roc_auc, best_precision, best_f1, best_avg_precision)
            ResultPrinter.print_top_scores(main_logger, top_pathways, method_name="IG")
            csv_save_path = config["file_paths"]["result_save"].replace(".csv", "_IG.csv")
            ResultPrinter.save_scores_to_csv(sorted_pathways, obtained_scores, df_filtered_gene_pathway, csv_save_path, score_name="IG_Score")

    elif method == "GSEA":
        main_logger.info("Running GSEA analysis...")
        gsea_results = GSEAHandler.run_gsea_analysis(
            df_filtered_gene_expression,
            config["file_paths"]["gsea_outdir"],
            config["file_paths"]["gsea_library"],
            config["gsea_params"]
        )
        res_df = gsea_results.res2d
        GSEAHandler.print_gsea_results(res_df)
        res_df.to_csv(config["file_paths"]["gsea_result_save"], index=False)
        main_logger.info(f"GSEA results saved to {config['file_paths']['gsea_result_save']}")

    else:
        main_logger.info("Invalid interpretation method specified.")