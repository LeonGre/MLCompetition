from sklearn.ensemble import RandomForestRegressor


def random_forest_eval(train_features, train_labels, test_features):
        
        # Create and fit the model
        model = RandomForestRegressor()
        model.fit(train_features, train_labels)
    
        # Make predictions
        predictions = model.predict(test_features)
    
        return predictions