import pandas as pd
import xgboost
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

import remove_outliers
from load_datasets import train_features, train_label, test_features

# Load the dataset
train_features = train_features("../")
train_label = train_label("../")
test_features = test_features("../")

#remove_outliers.remove_outliers_Z(train_features, train_label, threshold=3)
remove_outliers.remove_outliers_iqr(train_features, train_label)

train_df = pd.merge(train_features, train_label, on="Id")
# remove duplicates
train_df = train_df.drop_duplicates(subset=["user", "item"], keep="last")
#print(train_df.head())

# Separate features and labels
X = train_df.drop("rating", axis=1)
y = train_df["rating"]

# Initialize SMOTE
smote = SMOTE()

# Fit SMOTE
X_smote, y_smote = smote.fit_resample(X, y)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_smote)


# Define the parameter grid
param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'learning_rate': [0.01, 0.1, 0.5, 0.7],
    'max_depth': [4, 5, 6],
    'alpha': [5, 10, 15],
    'n_estimators': [5, 10, 20, 50, 100]
}

# Initialize the XGBoost regressor
model = xgboost.XGBRegressor(objective ='reg:squarederror')

# Initialize the GridSearchCV object
grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=3, scoring='neg_mean_squared_error')

# Fit the GridSearchCV object
grid_search.fit(X_scaled, y_smote)

# Print the best parameters and the best score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Predict the labels using the best model
y_pred = grid_search.predict(X_scaled)

# Calculate the mean squared error
mse = mean_squared_error(y_smote, y_pred)
print("Mean Squared Error:", mse)

# Apply cross-validation
scores = cross_val_score(grid_search, X_scaled, y_smote, cv=5, scoring="neg_mean_squared_error")
avg_score = scores.mean()

print("Cross-validation scores:", scores)
print("Average cross-validation score:", avg_score)