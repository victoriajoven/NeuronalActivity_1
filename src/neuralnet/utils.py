import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_dataset(path):
    return pd.read_csv(path)

def encode_categoricals(df, columns):
    return pd.get_dummies(df, columns=columns)

def scale_features(X, y):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    return scaler_X.fit_transform(X), scaler_y.fit_transform(y), scaler_X, scaler_y
