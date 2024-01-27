import pandas as pd
from sklearn.preprocessing import StandardScaler
from load_datasets import train_features, train_label, test_features
import remove_outliers

# Load the dataset
train_features = train_features()
train_label = train_label()
test_features = test_features()

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
