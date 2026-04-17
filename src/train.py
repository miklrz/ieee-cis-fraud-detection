from src.helpers import get_next_run_dir, get_folder_from_num
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from clearml import Task
from tqdm import tqdm
from pathlib import Path
from typing import List
import joblib
import gc
import os


def init_clearml_task(config, params):
    task = Task.init(
        project_name="IEEE Fraud Detection",
        task_name=config.get("clearml_task"),
        tags=["lgbm"],
    )
    task.connect(params)
    return task


def train_lgbm_cv(
    X_pd,
    y_pd,
    X_test_pd,
    params,
    n_fold=5,
    cat_cols_idx=None,
    task: Task = None,
    output_folder_path: Path = None,
):
    tscv = TimeSeriesSplit(n_splits=n_fold)

    y_preds = np.zeros(X_test_pd.shape[0])
    y_oof = np.zeros(X_pd.shape[0])
    scores = []

    feature_importances = pd.DataFrame({"feature": X_pd.columns})
    logger = task.get_logger() if task else None

    for fold_n, (train_index, valid_index) in tqdm(
        enumerate(tscv.split(X_pd, y_pd)), total=n_fold
    ):
        X_train, X_valid = X_pd.iloc[train_index], X_pd.iloc[valid_index]
        y_train, y_valid = y_pd.iloc[train_index], y_pd.iloc[valid_index]

        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=cat_cols_idx,
        )
        valid_data = lgb.Dataset(
            X_valid,
            label=y_valid,
            categorical_feature=cat_cols_idx,
            reference=train_data,
        )
        with tqdm(total=10000, desc=f"Fold {fold_n+1}", leave=False) as pbar:
            clf = lgb.train(
                params,
                train_data,
                num_boost_round=10000,
                valid_sets=[valid_data],
                callbacks=[
                    tqdm_callback(pbar),
                    make_clearml_callback(logger, fold_n),
                    lgb.early_stopping(200, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )

        feature_importances[f"fold_{fold_n + 1}"] = clf.feature_importance()

        y_pred_valid = clf.predict(X_valid)
        y_oof[valid_index] = y_pred_valid
        auc = roc_auc_score(y_valid, y_pred_valid)

        try:
            if logger:
                logger.report_scalar(
                    "Cross-Validation",
                    f"Fold {fold_n+1} - AUC",
                    value=auc,
                    iteration=fold_n,
                )
            else:
                print("Logger is undefined")
        except Exception as e:
            print(f"Не удалось записать метрику: {e}")

        scores.append(auc)
        print(f"Fold {fold_n + 1} | AUC: {auc:.6f}")

        y_preds += clf.predict(X_test_pd) / n_fold

        try:
            model_name = f"fold_{fold_n + 1}_model.txt"
            output_folder_path_for_save = output_folder_path / model_name
            clf.save_model(str(output_folder_path_for_save))
            print(f"Модель {model_name} сохранена")
        except Exception as e:
            print(f"Не удалось сохранить модель на диск: {e}")

        try:
            if logger:
                task.upload_artifact(
                    name=f"model_fold_{fold_n+1}",
                    artifact_object=str(output_folder_path_for_save),
                )
            else:
                print("Logger is undefined")
        except Exception as e:
            print(f"Не удалось записать артефакт в clearml: {e}")

        # Очистка
        del clf, train_data, valid_data, X_train, X_valid, y_train, y_valid
        gc.collect()

    mean_auc = np.mean(scores)
    oof_auc = roc_auc_score(y_pd, y_oof)
    print(f"\nMean AUC = {mean_auc:.6f}")
    print(f"Out-of-Folds AUC = {oof_auc:.6f}")

    try:
        if logger:
            logger.report_scalar("Summary", "Mean AUC", mean_auc, iteration=5)
            logger.report_scalar("Summary", "OOF AUC", oof_auc, iteration=5)
            logger.report_single_value("Mean AUC", mean_auc)
            logger.report_single_value("OOF AUC", oof_auc)
            print("Metrics sent to clearml")
        else:
            print("Logger is undefined")
    except Exception as e:
        print(f"Не удалось записать метрику: {e}")

    if task:
        fi = feature_importances.copy()
        fi["average"] = fi.drop("feature", axis=1).mean(axis=1)
        fi_sorted = (
            fi[["feature", "average"]]
            .sort_values("average", ascending=False)
            .reset_index(drop=True)
        )
        logger.report_table(
            title="Feature Importance",
            series="Average across folds",
            iteration=0,
            table_plot=fi_sorted,
        )

    return y_preds, y_oof, feature_importances, mean_auc


def tqdm_callback(pbar):
    def callback(env):
        pbar.update(1)

    return callback


def make_clearml_callback(logger, fold_n):
    def clearml_callback(env):
        if logger is None:
            return
        iteration = env.iteration
        for data_name, eval_name, value, _ in env.evaluation_result_list:
            logger.report_scalar(
                title=f"Fold {fold_n + 1} / {eval_name}",
                series=data_name,
                value=value,
                iteration=iteration,
            )

    return clearml_callback


def save_importances_and_submission(
    sub, feature_importances, n_fold, output_folder_path
):
    submission_path = output_folder_path / "submission.csv"
    feature_importances_path = output_folder_path / "feature_importances.csv"

    sub.to_csv(submission_path, index=False)
    print(f"submission.csv saved to {submission_path}")

    fold_cols = [f"fold_{i}" for i in range(1, n_fold + 1)]
    feature_importances["average"] = feature_importances[fold_cols].mean(axis=1)
    feature_importances.to_csv(feature_importances_path, index=False)
    print(f"feature_importances.csv saved to {feature_importances_path}")
