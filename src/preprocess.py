import polars as pl
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import gc


def get_x_y(train_df, test_df):
    X = train_df.sort("TransactionDT").drop(
        ["isFraud", "TransactionDT", "TransactionID"]
    )
    y = train_df.sort("TransactionDT")["isFraud"]
    X_test = test_df.drop("TransactionDT", "TransactionID")
    return X, y, X_test


def clean_inf_nan(df):
    float_cols = df.select(pl.col(pl.Float32, pl.Float64)).columns
    return df.with_columns(
        [
            pl.when(pl.col(col).is_infinite())
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
            for col in float_cols
        ]
    )


def encode_cat_cols(X, y, X_test, cat_cols, dataset_dir):
    X_pd = X.to_pandas()
    y_pd = y.to_pandas()
    X_test_pd = X_test.to_pandas()

    gc.collect()

    le = LabelEncoder()
    cat_cols_clean = [c for c in cat_cols if c in X_pd.columns]
    for col in cat_cols_clean:
        X_pd[col] = X_pd[col].astype(str)
        X_test_pd[col] = X_test_pd[col].astype(str)
        le.fit(pd.concat([X_pd[col], X_test_pd[col]]))
        X_pd[col] = le.transform(X_pd[col])
        X_test_pd[col] = le.transform(X_test_pd[col])

    cat_cols_idx = None

    return X_pd, y_pd, X_test_pd
