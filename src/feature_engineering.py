import numpy as np
import polars as pl
from sklearn.model_selection import TimeSeriesSplit
from src.constants import suspicious_null_cols, emails, us_emails


def add_features(df):
    # EDA 1.2
    for col in suspicious_null_cols:
        df = df.with_columns(
            pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_is_null")
        )

    # EDA - Transaction table
    # Categorical
    df = df.with_columns(
        (pl.col("ProductCD") == "C").cast(pl.Int8).alias("is_product_C")
    )
    df = df.with_columns((pl.col("id_07") == 0.0).cast(pl.Int8).alias("id_07_is_0"))
    df = df.with_columns((pl.col("id_07") == 5.0).cast(pl.Int8).alias("id_07_is_5"))
    df = df.with_columns(
        (pl.col("id_08") == -100.0).cast(pl.Int8).alias("id_08_is_-100")
    )
    df = df.with_columns(
        (pl.col("card6") == "debit").cast(pl.Int8).alias("card6_is_debit")
    )
    df = df.with_columns(
        (pl.col("P_emaildomain") == "protonmail.com")
        .cast(pl.Int8)
        .alias("P_emaildomain_is_protonmail.com")
    )
    df = df.with_columns(
        (pl.col("R_emaildomain") == "mail.com")
        .cast(pl.Int8)
        .alias("R_emaildomain_is_mail.com")
    )

    # Continuous
    df = df.with_columns((pl.col("TransactionDT") // 86400).alias("day"))
    df = df.with_columns(((pl.col("TransactionDT") // 3600) % 24).alias("hour"))
    df = df.with_columns(((pl.col("TransactionDT") // 86400) % 7).alias("day_of_week"))
    df = df.with_columns(
        ((pl.col("hour") >= 4) & (pl.col("hour") <= 10))
        .cast(pl.Int8)
        .alias("hour_from_4_to_10")
    )
    df = df.with_columns(
        (pl.col("D1").is_null() | (pl.col("D1") <= 0)).cast(pl.Int8).alias("D1_le_0")
    )

    emails_map = pl.Series(list(emails.keys()), dtype=pl.Utf8), pl.Series(
        list(emails.values()), dtype=pl.String
    )

    for c in ["P_emaildomain", "R_emaildomain"]:
        df = df.with_columns(
            [
                pl.col(c).replace(emails).alias(c + "_bin"),
                pl.col(c).cast(pl.Utf8).str.split(".").list.last().alias(c + "_suffix"),
            ]
        )

        df = df.with_columns(
            pl.when(pl.col(c + "_suffix").is_in(us_emails))
            .then(pl.lit("us"))
            .otherwise(pl.col(c + "_suffix"))
            .alias(c + "_suffix")
        )

    # UID
    df = df.with_columns(
        (
            pl.col("card1").cast(pl.String)
            + "_"
            + pl.col("card2").cast(pl.String)
            + "_"
            + pl.col("addr1").cast(pl.String)
        ).alias("uid1")
    )

    df = df.with_columns(
        (
            pl.col("card1").cast(pl.String)
            + "_"
            + pl.col("addr1").cast(pl.String)
            + "_"
            + pl.col("D1").cast(pl.String)
        ).alias("uid2")
    )

    df = df.with_columns(
        (pl.col("card1").cast(pl.String) + "_" + pl.col("addr1").cast(pl.String)).alias(
            "uid3"
        )
    )

    all_agg_exprs = []
    for uid_col in ["uid1", "uid2", "uid3", "card1", "card2"]:
        all_agg_exprs.extend(
            [
                pl.col("TransactionAmt")
                .mean()
                .over(uid_col)
                .alias(f"TransactionAmt_{uid_col}_mean"),
                pl.col("TransactionAmt")
                .std()
                .over(uid_col)
                .alias(f"TransactionAmt_{uid_col}_std"),
                pl.col("TransactionAmt")
                .max()
                .over(uid_col)
                .alias(f"TransactionAmt_{uid_col}_max"),
                pl.col("TransactionAmt")
                .min()
                .over(uid_col)
                .alias(f"TransactionAmt_{uid_col}_min"),
            ]
        )
    df = df.with_columns(all_agg_exprs)

    df = df.with_columns(
        [
            (
                pl.col("TransactionAmt") / pl.col("TransactionAmt").mean().over("card1")
            ).alias("TransactionAmt_to_mean_card1"),
            (
                pl.col("TransactionAmt") / pl.col("TransactionAmt").std().over("card1")
            ).alias("TransactionAmt_to_std_card1"),
        ]
    )

    count_exprs = []
    for count_col in [
        "card1",
        "uid1",
        "uid2",
        "uid3",
        "P_emaildomain",
        "TransactionAmt",
    ]:
        count_exprs.append(
            pl.col(count_col).count().over(count_col).alias(f"{count_col}_count")
        )
    df = df.with_columns(count_exprs)

    df = df.with_columns(
        [
            pl.col("card1").count().over(["card1", "hour"]).alias("card1_hour_count"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("TransactionAmt") - pl.col("TransactionAmt").cast(pl.Int32)).alias(
                "TransactionAmt_cents"
            ),
            pl.col("TransactionAmt").log1p().alias("TransactionAmt_log"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("TransactionAmt_cents") == 0)
            .cast(pl.Int32)
            .alias("is_round_amount"),
            ((pl.col("TransactionAmt") * 1000 % 10) != 0)
            .cast(pl.Int32)
            .alias("is_foreign_amt"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("DeviceInfo").str.split("/").list.get(0).str.strip_chars()).alias(
                "device_os"
            ),
            (pl.col("id_31").str.split(" ").list.get(0).str.strip_chars()).alias(
                "device_browser"
            ),
            (pl.col("id_31").str.extract(r"(\d+\.\d+)", group_index=1)).alias(
                "browser_version"
            ),
        ]
    )

    df = df.with_columns(
        [
            (
                pl.col("card1").cast(pl.String) + "_" + pl.col("card2").cast(pl.String)
            ).alias("card1_card2"),
            (pl.col("P_emaildomain") == pl.col("R_emaildomain"))
            .cast(pl.Int8)
            .alias("P_R_email_match"),
            (pl.col("addr1") == pl.col("addr2")).cast(pl.Int8).alias("addr_match"),
        ]
    )

    df = df.with_columns(
        pl.col("addr1").n_unique().over("card1").alias("card1_addr1_nunique")
    )
    df = df.with_columns(
        pl.col("card1").n_unique().over("addr1").alias("addr1_card1_nunique")
    )

    df = df.with_columns(
        [
            pl.col("D1").mean().over("card1").alias("D1_card1_mean"),
            pl.col("D1").std().over("card1").alias("D1_card1_std"),
            (pl.col("D1") - pl.col("D1").mean().over("card1")).alias("D1_card1_diff"),
        ]
    )

    return df


def add_target_encoding(train_df, test_df, cat_cols, target_col="isFraud", n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    train_df = train_df.sort("TransactionDT")
    train_pd = train_df.to_pandas()
    test_pd = test_df.to_pandas()

    for col in cat_cols:
        oof_enc = np.zeros(len(train_pd))
        global_mean = train_pd[target_col].mean()

        for train_idx, valid_idx in tscv.split(train_pd):
            # Часть фрода для соответствующего значения объекта из колонки
            fold_map = train_pd.iloc[train_idx].groupby(col)[target_col].mean()
            # Применяем fold map к valid части этого фолда
            oof_enc[valid_idx] = (
                train_pd.iloc[valid_idx][col].map(fold_map).fillna(global_mean)
            )
        # Маппинг по трейну для теста
        full_map = train_pd.groupby(col)[target_col].mean()
        # Train получает OOF энкодинг
        train_pd[f"{col}_target_enc"] = oof_enc
        # тест получает енкодинг по трейну
        test_pd[f"{col}_target_enc"] = test_pd[col].map(full_map).fillna(global_mean)

    return pl.from_pandas(train_pd), pl.from_pandas(test_pd)
