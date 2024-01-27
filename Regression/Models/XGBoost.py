import pandas as pd
import xgboost
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
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

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Initialize the XGBoost regressor
model = xgboost.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.7,
                max_depth = 5, alpha = 10, n_estimators = 10)

# Fit the model
model.fit(X_scaled, y)

# Predict the labels
y_pred = model.predict(X_scaled)

# Calculate the mean squared error
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)

# Apply cross-validation
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
avg_score = scores.mean()

print("Cross-validation scores:", scores)
print("Average cross-validation score:", avg_score)