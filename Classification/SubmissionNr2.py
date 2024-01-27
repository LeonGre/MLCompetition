from sklearn.ensemble import GradientBoostingClassifier
from load_datasets import train_features, train_label, test_features
from sklearn.preprocessing import RobustScaler
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np

# Load the dataset
train_features = train_features()
train_label = train_label()
test_features = test_features()

# Apply outlier clipping to the training dataset
for col in train_features.columns:
    Q1 = train_features[col].quantile(0.25)
    Q3 = train_features[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_cap = Q1 - 1.5 * IQR
    upper_cap = Q3 + 1.5 * IQR
    train_features[col] = np.clip(train_features[col], lower_cap, upper_cap)

# Remove Id, feature_2 and feature_26 column
X_train = train_features.drop("Id", axis=1).drop(["feature_2", "feature_26"], axis=1)
X_test = test_features.drop("Id", axis=1).drop(["feature_2", "feature_26"], axis=1)
y_train = train_label["label"]

# RobustScaler for better handling of outliers
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
model = GradientBoostingClassifier(
    max_depth=9, n_estimators=200, random_state=42, subsample=1
)

# Fit the model
model.fit(X_train_scaled, y_train)

# Generate predictions for the test set
predicted_labels = model.predict(X_test_scaled)

# Save the predictions to a CSV file
predicted_labels_df = pd.DataFrame(
    {"Id": test_features["Id"], "label": predicted_labels}
)
predicted_labels_df.to_csv(
    "./Classification/classification-data/test_label.csv", index=False
)

# Perform cross-validation and print the average score for the entire training set
scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="f1_macro")
print("Cross-validation F1 macro score:", scores.mean())
print("Done!")
