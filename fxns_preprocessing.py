import numpy as np
import pandas as pd


# Function for clipping outliers on iqr
def clip_outlier_iqr(df_in, cols, n_iqr = 5):
    df = df_in.copy()
    if isinstance(cols, str):
        cols = [cols]
    q1 = df[cols].quantile(0.25)
    q2 = df[cols].quantile(0.5)
    q3 = df[cols].quantile(0.75)
    mins = q2 - n_iqr * (q3-q1)
    maxs = q2 + n_iqr * (q3-q1)
    df[cols] = df[cols].clip(mins, maxs, axis=1)
    return df

# Function to winsorize training and testing set jointly
def joint_winsorize(train, test, cols, iqr_n = 5):
    train_w = train.copy()
    test_w = test.copy()
    q1 = train_w[cols].quantile(0.25)
    q2 = train_w[cols].quantile(0.25)
    q3 = train_w[cols].quantile(0.75)
    mins = q2 - iqr_n * (q3-q1)
    maxs = q2 + iqr_n * (q3-q1)
    train_w[cols] = train_w[cols].clip(mins, maxs, axis=1)
    test_w[cols] = test_w[cols].clip(mins, maxs, axis=1)
    return [train_w, test_w]

# Define function to standardize training and testing set jointly
def joint_standardize(train, test, cols):
    train_s = train.copy()
    test_s = test.copy()
    mean = train_s[cols].mean()
    sd = train_s[cols].std()
    train_s[cols] = (train_s[cols] - mean) / sd
    test_s[cols] = (test_s[cols] - mean) / sd
    return [train_s, test_s]