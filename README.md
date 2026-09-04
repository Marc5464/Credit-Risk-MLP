# Credit-Risk-MLP

# Description
A credit risk python project evaluating borrower default probability on Kaggle's Give Me Some Credit dataset. Built a custom 2-layer MLP from scratch only in NumPy without the use of external machine learning libraries. Opted for LeakyReLU activations, Momentum GD use of weighted class loss to help with heavy class imbalance. Model architectures were benchmarked over a variety hidden layer amounts, H = 8, 12, 16, 20, 32, and compared/evaluated using ROC-AUC and Precision-Recall metrics, combined with LOFO feature pruning, explainability function for rejected data samples and new engineered non linear features.

# Objective
The objective was to understand how an MLP works behind the scenes by fully deriving all the backpropagation gradients and feature contributions', then demonstrating how those results work in to solve an applied problem using only linear algebra.

# My Aproach
1) Cleaned the data: imputed missing values with median (so to exclude outliers), took ln(1+x) of heavily skewed features such as income so when normalised the data points are more distinguishable, engineered new non linear features to help the model more easily find non linear combinations, split the data into a training and validation sets and used training mean and standard deviations to normalise data in both sets to prevent data leakage.
2) Create Model logic:
