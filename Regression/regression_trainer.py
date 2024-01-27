from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error
import pandas as pd
from load_datasets import train_features, train_label, test_features
import numpy as np

# Load the dataset
train_features = train_features()
train_label = train_label()

train_df = pd.merge(train_features, train_label, on="Id")

# remove duplicates
train_df = train_df.drop_duplicates(subset=["user", "item"], keep="first")

# Remove the ID and rating column
X_train = train_df.drop("Id", axis=1)
X = X_train.drop("rating", axis=1)
y = train_df["rating"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline for preprocessing and model training
pipeline = Pipeline(
    [("scaler", StandardScaler()), ("model", None)]  # The model will be set dynamically
)

# Define parameter grids for grid search
param_grids = [
    {
        "model": [RandomForestRegressor()],
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5, 10],
    },
    {
        "model": [GradientBoostingRegressor()],
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_depth": [3, 5, 7],
    },
    {
        "model": [SVR()],
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
    },
]

# Define RMSE scorer for GridSearchCV
scorer = make_scorer(
    lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    greater_is_better=False,
)
print("Starting Grid Search...")
# Perform grid search for the best model
grid_search = GridSearchCV(pipeline, param_grids, scoring=scorer, cv=5)
grid_search.fit(X_train, y_train)

# Print the best parameters and corresponding RMSE
print("Best Parameters:", grid_search.best_params_)
print(
    "Best RMSE:", -grid_search.best_score_
)  # Negative because GridSearchCV maximizes the scoring function (minimize RMSE)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE on Test Set:", rmse_test)
