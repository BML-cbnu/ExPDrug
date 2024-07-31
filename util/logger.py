import logging
import pandas as pd

class Logger:
    """
    Logger class for handling logging functionality.
    """
    def __init__(self, process_name=""):
        self.logger = self.initialize_logging(process_name)

    @staticmethod
    def initialize_logging(process_name=""):
        """
        Initialize logging with the specified process name.
        """
        logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - {process_name} - %(levelname)s - %(message)s')
        return logging.getLogger(process_name)

    def info(self, message):
        """
        Log an info message.
        """
        self.logger.info(message)

class ResultPrinter:
    """
    ResultPrinter class for printing and saving training results.
    """
    @staticmethod
    def print_training_results(logger, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, best_accuracy, best_roc_auc, best_precision, best_f1):
        """
        Print the average and best training results.
        """
        logger.info('--------------------------------')
        logger.info(f'Average Accuracy: {avg_accuracy}')
        logger.info(f'Average ROC AUC: {avg_roc_auc}')
        logger.info(f'Average Precision: {avg_precision}')
        logger.info(f'Average F1 Score: {avg_f1}')
        logger.info('--------------------------------')
        logger.info(f'Best Accuracy: {best_accuracy}')
        logger.info(f'Best ROC AUC: {best_roc_auc}')
        logger.info(f'Best Precision: {best_precision}')
        logger.info(f'Best F1 Score: {best_f1}')

    @staticmethod
    def print_top_lrp_scores(logger, top_pathways):
        """
        Print the top LRP scores and corresponding pathways.
        """
        logger.info(f"Top {len(top_pathways)} LRP scores and corresponding pathways:")
        logger.info("----------------------------------------")
        for pathway_name, score, gene_count, p_value in top_pathways:
            logger.info(f"Pathway: {pathway_name}, LRP Score: {score}, Number of Genes: {gene_count}, p-value: {p_value}")

    @staticmethod
    def print_p_values(logger, p_values, pathway_list, lrp_scores, save_path):
        """
        Print the p-values for the pathways and save them to a CSV file.
        """
        logger.info(f"p-values for the pathways:")
        logger.info("----------------------------------------")
        pathway_data = []
        for pathway_name, lrp_score, p_value in zip(pathway_list, lrp_scores, p_values):
            logger.info(f"Pathway Name: {pathway_name}, LRP Score: {lrp_score}, p-value: {p_value}")
            pathway_data.append((pathway_name, lrp_score, p_value))
        
        df_pathways = pd.DataFrame(pathway_data, columns=['Pathway_Name', 'LRP_Score', 'p_value'])
        df_pathways.to_csv(save_path, index=False)
        logger.info(f"Pathway data saved to {save_path}")
