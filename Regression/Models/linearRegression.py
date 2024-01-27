import pandas as pd
from sklearn.linear_model import LinearRegression

def linear_regression(X_train, y_train, X_test):
    # Fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    return predictions
