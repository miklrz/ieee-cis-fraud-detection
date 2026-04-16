from src.feature_engineering import add_features, add_target_encoding
from src.load import load_data, rename_test_cols
from src.preprocess import get_x_y, clean_inf_nan, encode_cat_cols
from src.constants import cat_cols, te_cols
from src.train import init_clearml_task, train_lgbm_cv, save_importances_and_submission
from src.helpers import get_folder_from_num
from pathlib import Path
import lightgbm as lgb
import os
import gc
import yaml
import argparse


def main(parser):
    CONFIG_PATH = "config.yaml"
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        print("Добавьте yaml config")
        return

    OUTPUT_DIR = Path(config.get("base_dir")) / "output"
    DATASET_DIR = Path(config.get("base_dir")) / "competition_data"

    args = parser.parse_args()
    if (not args.train and not args.eval) or (args.train and args.eval):
        raise AttributeError("Выберите либо train либо eval")

    if args.train:
        print("Начало процесса train")
        lgbm_params = {
            "device": "cuda",
            "max_bin": 255,
            "num_leaves": 64,
            "bagging_freq": 1,
            "min_child_weight": 0.03454472573214212,
            "feature_fraction": 0.3797454081646243,
            "bagging_fraction": 0.4181193142567742,
            "min_data_in_leaf": 200,
            "objective": "binary",
            "max_depth": 8,
            "learning_rate": 0.006883242363721497,
            "boosting_type": "gbdt",
            "bagging_seed": 11,
            "metric": "auc",
            "verbosity": 1,
            "reg_alpha": 0.3899927210061127,
            "reg_lambda": 0.6485237330340494,
            "random_state": 47,
            "n_jobs": -1,
        }

        train_df, test_df, sub = load_data(dataset_dir=DATASET_DIR)
        test_df = rename_test_cols(test_df=test_df, train_df=train_df)

        print("Данные загружены")

        train_df = add_features(train_df)
        test_df = add_features(test_df)

        train_df, test_df = add_target_encoding(train_df, test_df, te_cols)

        print("Feature engineering сделан")

        X, y, X_test = get_x_y(train_df=train_df, test_df=test_df)
        del (train_df, test_df)
        gc.collect()

        X = clean_inf_nan(X)
        X_test = clean_inf_nan(X_test)
        X_pd, y_pd, X_test_pd = encode_cat_cols(X, y, X_test, cat_cols, DATASET_DIR)

        print("Preprocessing сделан")

        del (X, y, X_test)
        gc.collect()

        task = init_clearml_task(config, lgbm_params)
        print("Начало обучения")
        y_preds, y_oof, feature_importances, mean_auc = train_lgbm_cv(
            X_pd,
            y_pd,
            X_test_pd,
            lgbm_params,
            n_fold=config.get("n_fold"),
            cat_cols_idx=None,
            output_dir=OUTPUT_DIR,
        )
        task.close()
        print("Сохранение Feature importances и Submission файлов")
        sub["isFraud"] = y_preds
        save_importances_and_submission(
            sub, feature_importances, n_fold=config.get("n_fold"), output_dir=OUTPUT_DIR
        )
    elif args.eval:
        print("Начало процесса eval")
        eval_run = args.eval_run
        dir = get_folder_from_num(eval_run)
        models = [
            f
            for f in os.listdir(dir)
            if (os.path.isfile(os.path.join(dir, f)) and f.endswith("model.txt"))
        ]
        for model in models:
            clf = lgb.Booster(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-e", "--eval", action="store_true")
    parser.add_argument("--eval_run")
    main(parser)
