import os
import pandas as pd
import polars as pl


def load_data(dataset_dir):
    train_identity_path = os.path.join(dataset_dir, "train_identity.csv")
    train_transaction_path = os.path.join(dataset_dir, "train_transaction.csv")
    test_identity_path = os.path.join(dataset_dir, "test_identity.csv")
    test_transaction_path = os.path.join(dataset_dir, "test_transaction.csv")

    train_identity = pl.read_csv(train_identity_path)
    train_transaction = pl.read_csv(train_transaction_path)
    test_identity = pl.read_csv(test_identity_path)
    test_transaction = pl.read_csv(test_transaction_path)

    train_df = train_transaction.join(train_identity, on="TransactionID", how="left")
    test_df = test_transaction.join(test_identity, on="TransactionID", how="left")
    print(f"Dataset sizes | train: {train_df.shape}, test: {test_df.shape}")

    sub = pd.read_csv(os.path.join(dataset_dir, "submission.csv"))
    return train_df, test_df, sub


def rename_test_cols(test_df, train_df):
    test_df = test_df.rename(
        {
            col: col.replace("-", "_")
            for col in test_df.columns
            if "-" in col and col.startswith("id")
        }
    )
    common_id_cols = set([c for c in train_df.columns]) & set(
        [c for c in test_df.columns]
    )
    return test_df
