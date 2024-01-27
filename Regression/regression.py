import numpy as np
from imblearn.over_sampling import SMOTE, SVMSMOTE

import remove_outliers
from Models.advanced_mean_eval import advanced_mean_eval
from Models.decisionTree import decisionTree
from Models.knn_eval import knn_eval
from Models.linearRegression import linear_regression
from Models.random_forest import random_forest_eval
from Models.sgd_regressor_eval import sgd_regressor
from Models.surprise_cf_cvsklearn_test import surprise_collaborative_filtering_cv
from load_datasets import train_features, train_label, test_features
from sklearn.preprocessing import RobustScaler, StandardScaler
from Models.surprise_cf_eval import surprise_collaborative_filtering
import pandas as pd
import smogn

# Load the dataset
train_features = train_features()
train_label = train_label()
test_features = test_features()

# remove_outliers.remove_outliers_Z(train_features, train_label, threshold=3)
remove_outliers.remove_outliers_iqr(train_features, train_label)

train_df = pd.merge(train_features, train_label, on="Id")
# remove duplicates
train_df = train_df.drop_duplicates(subset=["user", "item"], keep="last")

# Remove the ID and rating column
X_train = train_df.drop("Id", axis=1)
X_test = test_features.drop("Id", axis=1)
X_train = X_train.drop("rating", axis=1)

y_train = train_df["rating"]

# Standardize the data
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reset the index of the DataFrame
#train_df = train_df.reset_index(drop=True)

# Initialize the SMOTE object
smote = SMOTE(sampling_strategy='auto', random_state=31)
#smote = smogn.smoter(data=train_df, y='rating', k=5)
"""
# Sample a subset of the DataFrame
train_df_sample = train_df.sample(n=10000, random_state=1).reset_index(drop=True)

# Check for missing values
if train_df_sample.isnull().values.any():
    print("Warning: Missing values detected. Please handle them before applying smogn.smoter.")

# Check for infinite values
if np.isinf(train_df_sample.values).any():
    print("Warning: Infinite values detected. Please handle them before applying smogn.smoter.")

# Initialize the SMOTE object
smote = smogn.smoter(data=train_df_sample, y='rating', k=5)
"""
# Fit the SMOTE object on your data and create the oversampled data
X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train_scaled, y_train)

# Convert the oversampled data back to a DataFrame
X_train_oversampled_df = pd.DataFrame(X_train_oversampled, columns=X_train.columns)
y_train_oversampled_df = pd.DataFrame(y_train_oversampled, columns=['rating'])

# Merge the oversampled data with the original train_df to maintain the same structure
train_df_oversampled = pd.concat([X_train_oversampled_df, y_train_oversampled_df], axis=1)

print("Generating predictions...")
# Make predictions
# Linear Regression
# predictions = linear_regression(X_train_scaled, y_train, X_test_scaled)
# Decision Tree
# predictions = decisionTree(X_train_scaled, y_train, X_test_scaled)
# SGDRegressor
# predictions = sgd_regressor(X_train_scaled, y_train, X_test_scaled)
# Random Forest
# predictions = random_forest_eval(X_train_scaled, y_train, X_test_scaled)
# KNN
# predictions = knn_eval(X_train_scaled, y_train, X_test_scaled, 5)
# Advanced Mean
# predictions = advanced_mean_eval(train_df, X_test)
# Surprise Collaborative Filtering
# test_predictions, cv_results = surprise_collaborative_filtering(train_df, X_test, 31, 'StratifiedKFold')
# test_predictions = surprise_collaborative_filtering(train_df, X_test, 31, 'TimeSeriesSplit')
test_predictions = surprise_collaborative_filtering(train_df, X_test, 31, 'StratifiedKFold')
# test_predictions = surprise_collaborative_filtering(train_df, X_test, 31, 'KFold')
# test_predictions = surprise_collaborative_filtering(train_df, X_test, 31, 'GroupKFold')
# test_predictions = surprise_collaborative_filtering(train_df, X_test, 31, 'ShuffleSplit')

#test_predictions = surprise_collaborative_filtering_cv(train_df_oversampled, X_test, 31, 'StratifiedKFold')
#test_predictions = surprise_collaborative_filtering_cv(smote, X_test, 31, 'StratifiedKFold')

# Funk SVD
# predictions = funk_svd(train_df, X_test)

# print(test_predictions)
# Create a submission dataframe
predicted_labels_df = pd.DataFrame(
    {"Id": test_features["Id"], "Predicted": test_predictions}
)

predicted_labels_df.to_csv("./regression-data/test_label.csv", index=False)
