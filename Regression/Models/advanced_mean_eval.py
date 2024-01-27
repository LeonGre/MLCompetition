from collections import defaultdict


def advanced_mean_eval(train_df, X_test):
    # Calculate mean ratings for each user using defaultdict
    user_means = defaultdict(lambda: train_df["rating"].mean())
    for user, mean_rating in train_df.groupby("user")["rating"].mean().items():
        user_means[user] = mean_rating

    # Predict mean ratings for test set
    test_predictions = [user_means[row["user"]] for _, row in X_test.iterrows()]

    return test_predictions
