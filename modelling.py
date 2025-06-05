import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
import click
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def none_or_str(value):
    if isinstance(value, str) and value.lower() in ['none', 'null']:
        return None
    return value

def none_or_int(value):
    if isinstance(value, str) and value.lower() in ['none', 'null']:
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Tidak dapat mengkonversi '{value}' ke integer. Menggunakan None.")
        return None

@click.command()
@click.option('--data_file', default='processed_winequality-red.csv', help='Path ke file dataset CSV.')
@click.option('--target_column', default='quality', help='Nama kolom target.')
@click.option('--test_split_size', type=float, default=0.2, help='Proporsi dataset untuk test set.')
@click.option('--random_state_split', type=int, default=42, help='Random state untuk train_test_split.')
@click.option('--n_estimators', type=int, default=100, help='Jumlah trees di Random Forest.')
@click.option('--max_depth_rf', default="None", help='Kedalaman maksimum Random Forest (string "None" atau integer).')
@click.option('--min_samples_leaf_rf', type=int, default=1, help='Jumlah minimum sampel per leaf di Random Forest.')
@click.option('--class_weight_rf', default='balanced', help='Class weight untuk Random Forest ("balanced", "balanced_subsample", atau "None").')
@click.option('--use_smote', type=click.BOOL, default=True, help='Gunakan SMOTE untuk oversampling (True/False).')
@click.option('--random_state_smote', type=int, default=42, help='Random state untuk SMOTE.')
@click.option('--experiment_name', default='WineQuality_CI_Default', help='Nama eksperimen MLflow.')
@click.option('--run_name_prefix', default='CI_RF_Run', help='Prefix untuk nama run MLflow.')
@click.option('--model_artifact_path', default='random_forest_model_ci', help='Path artefak untuk model yang di-log di MLflow.')
@click.option('--tuning', is_flag=True, help='Aktifkan hyperparameter tuning menggunakan GridSearchCV.')
def train_model(data_file, target_column, test_split_size, random_state_split,
                n_estimators, max_depth_rf, min_samples_leaf_rf, class_weight_rf,
                use_smote, random_state_smote,
                experiment_name, run_name_prefix, model_artifact_path,
                tuning):
    logger.info("Memulai proses training model...")

    load_dotenv()

    # Tidak mengatur mlflow.set_tracking_uri, MLflow jalan default lokal

    mlflow.set_experiment(experiment_name)
    logger.info(f"Eksperimen MLflow diatur ke: {experiment_name}")

    try:
        logger.info(f"Memuat data dari: {data_file}")
        data = pd.read_csv(data_file)
        logger.info(f"Data berhasil dimuat. Shape: {data.shape}")
    except FileNotFoundError:
        logger.error(f"ERROR: File dataset '{data_file}' tidak ditemukan.")
        return

    if target_column not in data.columns:
        logger.error(f"ERROR: Kolom target '{target_column}' tidak ditemukan. Kolom tersedia: {data.columns.tolist()}")
        return

    X = data.drop(target_column, axis=1)
    y = data[target_column]
    logger.info(f"Fitur (X) shape: {X.shape}, Target (y) shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split_size, random_state=random_state_split,
        stratify=y if y.nunique() > 1 else None
    )
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    if use_smote:
        logger.info("Menerapkan SMOTE pada data training...")
        smote = SMOTE(random_state=random_state_smote)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"Shape setelah SMOTE: X_train: {X_train.shape}, y_train: {y_train.value_counts().sort_index().to_dict()}")

    run_name = f"{run_name_prefix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id} dimulai untuk run: {run_name}")

        mlflow.log_params({
            "data_file": data_file, "target_column": target_column, "test_split_size": test_split_size,
            "random_state_split": random_state_split, "use_smote": use_smote,
            "random_state_smote": random_state_smote if use_smote else "N/A",
            "tuning": tuning
        })

        if tuning:
            logger.info("Menggunakan GridSearchCV untuk tuning hyperparameter...")

            param_grid = {
                "n_estimators": [100, 150],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2],
                "class_weight": ["balanced", None]
            }

            rf = RandomForestClassifier(random_state=random_state_split, n_jobs=-1)
            grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

            best_params = grid_search.best_params_
            logger.info(f"Best params dari GridSearchCV: {best_params}")
            mlflow.log_params(best_params)
        else:
            actual_max_depth_rf = none_or_int(max_depth_rf)
            actual_class_weight_rf = none_or_str(class_weight_rf)

            mlflow.log_params({
                "n_estimators": n_estimators,
                "max_depth_rf": actual_max_depth_rf,
                "min_samples_leaf_rf": min_samples_leaf_rf,
                "class_weight_rf": actual_class_weight_rf,
            })

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=actual_max_depth_rf,
                min_samples_leaf=min_samples_leaf_rf,
                class_weight=actual_class_weight_rf,
                random_state=random_state_split,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            logger.info("Model berhasil dilatih tanpa tuning.")

        y_pred_test = model.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        report_dict = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)

        precision_test_weighted = report_dict["weighted avg"]["precision"]
        recall_test_weighted = report_dict["weighted avg"]["recall"]
        f1_test_weighted = report_dict["weighted avg"]["f1-score"]

        logger.info(f"Akurasi Test: {accuracy_test:.4f}")
        logger.info(f"Classification Report (Test):\n{classification_report(y_test, y_pred_test, zero_division=0)}")

        mlflow.log_metric("accuracy_test", accuracy_test)
        mlflow.log_metric("precision_test_weighted", precision_test_weighted)
        mlflow.log_metric("recall_test_weighted", recall_test_weighted)
        mlflow.log_metric("f1_test_weighted", f1_test_weighted)
        logger.info("Metrik berhasil di-log ke MLflow.")

        mlflow.sklearn.log_model(model, model_artifact_path)
        logger.info(f"Model berhasil di-log ke MLflow sebagai artefak: {model_artifact_path}")

        print(f"MLFLOW_RUN_ID:{run_id}")
        print(f"MLFLOW_MODEL_PATH:{model_artifact_path}")

        mlflow.set_tag("stage", "ci_training_advanced")
        mlflow.set_tag("data_version", "processed_v1")
        logger.info("Run MLflow selesai.")

if __name__ == '__main__':
    train_model()
