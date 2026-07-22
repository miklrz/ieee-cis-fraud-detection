# IEEE-CIS Fraud Detection

Pipeline for the Kaggle IEEE-CIS fraud detection task. The project joins transaction and identity tables, builds tabular features, trains LightGBM with time-based cross-validation, and saves submission and feature-importance artifacts.

## What Is Implemented

- Data loading with Polars from Kaggle CSV files.
- Merge of `transaction` and `identity` tables by `TransactionID`.
- Feature engineering for null indicators, time features, email domains, card/user identifiers, aggregations, counts, device/browser fields, and transaction amount transforms.
- Out-of-fold target encoding with `TimeSeriesSplit`.
- Categorical encoding with `LabelEncoder`.
- LightGBM binary classification with 5-fold time-series CV by default.
- AUC logging, feature importances, model artifacts, and submission generation.
- ClearML integration for experiment tracking.
- Notebooks for EDA, modeling, and SHAP analysis.

## Project Structure

```text
src/
  load.py                 # Read and merge Kaggle data
  feature_engineering.py  # Feature generation and target encoding
  preprocess.py           # X/y split, inf cleanup, categorical encoding
  train.py                # LightGBM CV, ClearML logging, artifacts
  main.py                 # Training entry point
  constants.py            # Feature lists and mappings
notebooks/
  01-eda-ieee-fraud-detection.ipynb
  02-modeling-ieee-fraud-detection.ipynb
  03-shap.ipynb
```

## Data Layout

The code expects Kaggle files in `competition_data/` under `base_dir`:

```text
competition_data/
  train_identity.csv
  train_transaction.csv
  test_identity.csv
  test_transaction.csv
  sample_submission.csv
```

`competition_data/`, `output/`, and `config.yaml` are intentionally ignored by git.

## Configuration

Create `config.yaml` in the project root:

```yaml
base_dir: /path/to/ieee-cis-fraud-detection
n_fold: 5
clearml_task: LightGBM Add features
```

## Run

```bash
poetry install
poetry run python -m src.main --train
```

Training creates a new `output/run_XX/` directory with:

- fold LightGBM model files;
- `submission.csv`;
- `feature_importances.csv`.
