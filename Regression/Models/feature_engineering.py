# feature_engineering.py

import pandas as pd
from Regression.load_datasets import train_df, test_features

def preprocess_data(df):
    # Convert timestamp to datetime and extract time features
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    return df

def compute_user_item_statistics(df):
    if 'rating' not in df.columns:
        return df

    # User statistics
    user_stats = df.groupby('user')['rating'].agg(['mean', 'count', 'var']).reset_index()
    user_stats.columns = ['user', 'user_avg_rating', 'user_rating_count', 'user_rating_var']
    user_stats['user_rating_var'].fillna(0, inplace=True)  # Fill NaNs with 0

    # Item statistics
    item_stats = df.groupby('item')['rating'].agg(['mean', 'count', 'var']).reset_index()
    item_stats.columns = ['item', 'item_avg_rating', 'item_rating_count', 'item_rating_var']
    item_stats['item_rating_var'].fillna(0, inplace=True)  # Fill NaNs with 0

    # Merge statistics back to the main DataFrame
    df = df.merge(user_stats, on='user', how='left')
    df = df.merge(item_stats, on='item', how='left')
    return df

def time_based_split(data:pd.DataFrame, test_size=0.2):
    data = data.sort_values('timestamp')
    # Calculate the index to split the data
    split_index = int((1 - test_size) * len(data))
    train_data = data[:split_index]
    train_data = train_data.sort_values('Id')
    test_data = data[split_index:]
    test_data = test_data.sort_values('Id')
    return train_data, test_data


def prepare_features():
    train_data = train_df("../")
    #test_data = test_features("../")

    train_data, test_data = time_based_split(train_data)

    train_data = preprocess_data(train_data)
    #test_data = preprocess_data(test_data)

    train_data = compute_user_item_statistics(train_data)
    #test_data = compute_user_item_statistics(test_data)

    # Merge user and item statistics from train_data to test_data
    user_stats = train_data[['user', 'user_avg_rating', 'user_rating_count', 'user_rating_var']].drop_duplicates()
    item_stats = train_data[['item', 'item_avg_rating', 'item_rating_count', 'item_rating_var']].drop_duplicates()

    test_data = test_data.merge(user_stats, on='user', how='left')
    test_data = test_data.merge(item_stats, on='item', how='left')

    return train_data, test_data

def prepare_test_features(test_data, train_data):
    test_data = preprocess_data(test_data)

    # Merge user and item statistics from train_data to test_data
    user_stats = train_data[['user', 'user_avg_rating', 'user_rating_count', 'user_rating_var']].drop_duplicates()
    item_stats = train_data[['item', 'item_avg_rating', 'item_rating_count', 'item_rating_var']].drop_duplicates()

    test_data = test_data.merge(user_stats, on='user', how='left')
    test_data = test_data.merge(item_stats, on='item', how='left')

    return test_data

