import pandas as pd
from sklearn.preprocessing import StandardScaler
from surprise import Reader, Dataset, SVD
from surprise.model_selection import GridSearchCV

from load_datasets import train_features, train_label, test_features
import remove_outliers

# Load the dataset
train_features = train_features("../")
train_label = train_label("../")
test_features = test_features("../")

remove_outliers.remove_outliers_Z(train_features, train_label, threshold=3)

train_df = pd.merge(train_features, train_label, on="Id")
# remove duplicates
train_df = train_df.drop_duplicates(subset=["user", "item"], keep="last")

# Remove the ID and rating column
X_train = train_df.drop("Id", axis=1)
X_test = test_features.drop("Id", axis=1)
X_train = X_train.drop("rating", axis=1)

y_train = train_df["rating"]

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid
param_grid = {
    "n_epochs": [5, 10, 20],
    "lr_all": [0.002, 0.005, 0.01],
    "reg_all": [0.2, 0.4, 0.6]
}

# Initialize a Reader and a Dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_df[["user", "item", "rating"]], reader)

# Initialize the GridSearchCV object
gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=5)

# Fit the GridSearchCV object
gs.fit(data)

# Get the best parameters and the best score
print(gs.best_params["rmse"])
print(gs.best_score["rmse"])