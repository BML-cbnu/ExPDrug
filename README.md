# ExPDRUG Pipeline

This repository contains a deep learning training pipeline that includes data loading, preprocessing, model training, and logging. The pipeline is modular and can be configured through environment variables.

## File Structure

- `config.py`: Configuration class that loads settings from environment variables.
- `data_processor.py`: Handles data loading and preprocessing.
- `logger.py`: Sets up logging to a file.
- `model.py`: Defines the machine learning model using TensorFlow/Keras.
- `trainer.py`: Handles the training process of the model.
- `main.py`: Main script to run the entire pipeline.

## Configuration

Configuration is handled through environment variables. The following variables can be set:

- `DATA_PATH`: Path to the data directory (default: `data/`)
- `MODEL_SAVE_PATH`: Path to save the trained model (default: `models/`)
- `LOG_PATH`: Path to save log files (default: `logs/`)
- `EPOCHS`: Number of training epochs (default: 10)
- `BATCH_SIZE`: Batch size for training (default: 32)
- `LEARNING_RATE`: Learning rate for the optimizer (default: 0.001)
