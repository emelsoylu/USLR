# USLR
Underwater Sign Language Recognition

This repository contains the dataset and source code. The dataset was constructed through feature extraction from videos, followed by data augmentation to enhance variability and robustness. The final dataset is provided in CSV format, which can be directly utilized for training purposes.

Since the data is divided into multiple CSV files by class, users are required to merge the CSV files before initiating the training process.

The repository includes the following components:

Feature Extraction and Augmentation Scripts:

Source code for extracting features from raw video data.

Data augmentation procedures applied to increase dataset diversity.

Dataset (CSV format):

Processed and augmented dataset stored as CSV files, one per class.

These files should be merged prior to model training.

Training Notebook (train.ipynb):

Implementation of training pipelines for six different neural network architectures.

Final stage includes 5-fold cross-validation performed on the best-performing model to ensure robust evaluation.
