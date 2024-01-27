from load_datasets import train_features, train_label, test_features
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from LensKit_models.MF import biased_mf, funk_svd, calculate_rmse

# Load the dataset
train_features = train_features()
train_label = train_label()
test_features = test_features()

train_df = pd.merge(train_features, train_label, on="Id")
# remove duplicates
train_df = train_df.drop_duplicates(subset=["user", "item"], keep="last")
train_df.drop("Id", axis=1, inplace=True)
train_df['rating'] = train_df['rating'].astype(float)
# Specify the column order
column_order = ['user', 'item', 'rating', 'timestamp']
train_df = train_df[column_order]

# Remove the ID and rating column
#X_train = train_df.drop("Id", axis=1)
X_test = test_features.drop("Id", axis=1)
#X_train = X_train.drop("rating", axis=1)

y_train = train_df["rating"]

#print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

# Standardize the data
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# predictions = funk_svd(X_train_scaled, y_train, X_test_scaled)
test_predictions = biased_mf(train_df, X_test)
"""
# Assuming test_labels are the actual values
predicted_values = biased_mf(train_df, X_test)
rmse = calculate_rmse(test_labels, test_predictions)
print(f"RMSE: {rmse}")
"""
predicted_labels_df = pd.DataFrame(
    {"Id": test_features["Id"], "Predicted": test_predictions}
)

predicted_labels_df.to_csv("./regression-data/test_label.csv", index=False)
