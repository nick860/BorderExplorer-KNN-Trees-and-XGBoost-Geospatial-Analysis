# KNN and Decision Trees Analysis

This repository contains Python scripts for performing analysis using k-nearest neighbors (KNN) and decision trees algorithms. These algorithms are implemented using various libraries such as scikit-learn and Faiss.

## Overview

The repository consists of the following files:

- `knn.py`: Contains the implementation of the `KNNClassifier` class, which utilizes the Faiss library for efficient k-nearest neighbors classification.
- `decision_trees.py`: Includes functions for training decision tree classifiers and analyzing their performance.
- `data.csv`, `train.csv`, `validation.csv`, `test.csv`, `AD_test.csv`: These CSV files contain data used for training and testing the machine learning models. Each row represents a sample, with features such as longitude, latitude, and country statistics.

## Files Description

### knn.py

The `knn.py` file contains the implementation of the `KNNClassifier` class, which allows for k-nearest neighbors classification. Below are the key components of this file:

- **KNNClassifier Class**: This class implements a k-nearest neighbors classifier using the Faiss library for efficient distance computations. It includes methods for fitting the model to training data (`fit`), predicting class labels for new data (`predict`), and calculating kNN distances (`knn_distance`).
  
### decision_trees.py

The `decision_trees.py` file provides functions for training decision tree classifiers and evaluating their performance. Key functionalities of this file include:

- **Decision Tree Demo**: This function demonstrates the usage of decision tree classifiers using scikit-learn's `DecisionTreeClassifier`. It includes training the classifier, making predictions, and computing accuracy.
- **Random Forest and XGBoost Loading**: This file also includes functions for loading random forest and XGBoost classifiers using scikit-learn's `RandomForestClassifier` and `XGBClassifier`.
- **Plot Decision Boundaries**: A utility function `plot_decision_boundaries` is provided for visualizing decision boundaries of classifiers.
- **Anomaly Detection**: Another function `anomaly_detection_demo` is available for performing anomaly detection using k-nearest neighbors.
- **Train Decision Trees**: This function trains decision trees with different hyperparameters and evaluates their performance on test and validation sets.
- **KNN Analysis**: Lastly, there is a function `KNN` for analyzing the performance of the KNN classifier on different distance metrics and values of k.

## Data Files Description

### data.csv, train.csv, validation.csv, test.csv, AD_test.csv

These CSV files contain data used for training and testing the machine learning models. Each row represents a sample, with features such as longitude, latitude, and country statistics.

## Usage

To utilize the functionalities provided in these files, follow the steps below:

1. Clone the repository to your local machine.
2. Ensure that you have the required dependencies installed (numpy, pandas, matplotlib, scikit-learn, Faiss).
3. Run the desired Python script (e.g., `python decision_trees.py`) to execute the analysis.

## Dependencies

- NumPy
- pandas
- Matplotlib
- scikit-learn
- Faiss
