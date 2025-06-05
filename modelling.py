import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv

# Load environment variables dari .env (pastikan sudah buat file .env dengan MLFLOW_TRACKING_USERNAME dan MLFLOW_TRACKING_PASSWORD)
load_dotenv()

username = os.getenv("MLFLOW_TRACKING_USERNAME")
token = os.getenv("MLFLOW_TRACKING_PASSWORD")  # kalau perlu, dipakai juga buat autentikasi jika perlu
dagshub_repo_name = "Membangun_model" 

if username is None or token is None:
    raise ValueError("MLFLOW_TRACKING_USERNAME atau MLFLOW_TRACKING_PASSWORD tidak ditemukan di .env")

# Set tracking URI ke DagsHub
tracking_uri = f"https://dagshub.com/{username}/{dagshub_repo_name}.mlflow"
mlflow.set_tracking_uri(tracking_uri)

experiment_name = "WineQuality_LogisticRegression_Tuning"
mlflow.set_experiment(experiment_name)

def modeling_with_tuning(X_train, X_val, y_train, y_val):
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced']
    }

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

    print("Parameter terbaik:", grid_search.best_params_)
    print("Akurasi:", accuracy)
    print("Classification Report:")
    print(classification_report(y_val, y_pred, zero_division=0))

    return best_model, accuracy, report, grid_search.best_params_

def get_latest_run_id(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' tidak ditemukan!")
    print(f"Experiment ditemukan: {experiment.name} (ID: {experiment.experiment_id})")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if runs.empty:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")
    
    latest_run_id = runs.iloc[0]["run_id"]
    return latest_run_id

if __name__ == "__main__":
    data_path = "processed_winequality-red.csv"
    
    try:
        print(f"Mencoba memuat data dari: {data_path}")
        data = pd.read_csv(data_path)
        print("Data berhasil dimuat.")
        print("Kolom data:", data.columns.tolist())
    except FileNotFoundError:
        print(f"ERROR: File dataset '{data_path}' tidak ditemukan. Pastikan file tersebut ada di lokasi yang benar.")
        exit() 

    target_column = 'quality' 

    if target_column not in data.columns:
        print(f"ERROR: Kolom target '{target_column}' tidak ditemukan dalam dataset.")
        print(f"Kolom yang tersedia adalah: {data.columns.tolist()}")
        exit()

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    print(f"Ukuran X_train: {X_train.shape}")
    print(f"Ukuran X_val: {X_val.shape}")
    print(f"Ukuran y_train: {y_train.shape}")
    print(f"Ukuran y_val: {y_val.shape}")

    with mlflow.start_run(run_name="RF_Tuning_SingleFile"):
        model, accuracy, report_dict, best_params = modeling_with_tuning(
            X_train, X_val, y_train, y_val
        )

        # Log hyperparameter terbaik
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        # Log metrik evaluasi
        mlflow.log_metric("accuracy", accuracy)
        
        if "weighted avg" in report_dict:
            mlflow.log_metric("precision_weighted", report_dict["weighted avg"]["precision"])
            mlflow.log_metric("recall_weighted", report_dict["weighted avg"]["recall"])
            mlflow.log_metric("f1_score_weighted", report_dict["weighted avg"]["f1-score"])
        else:
            print("WARNING: 'weighted avg' tidak ditemukan di classification_report.")

        # Set tag MLflow
        mlflow.set_tag("stage", "tuning")
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("data_source", data_path) 

        # Log model dengan contoh input supaya hilang warning signature
        input_example = X_train.iloc[:5]
        mlflow.sklearn.log_model(
            model, artifact_path="rf_best_model", input_example=input_example,
            conda_env="conda.yaml"
        )

        print(f"Tuning selesai dan model berhasil dicatat ke MLflow DagsHub")

    # Ambil latest run ID di experiment yang sama
    try:
        latest_run_id = get_latest_run_id(experiment_name)
        print(f"Latest run ID dari experiment '{experiment_name}': {latest_run_id}")
    except Exception as e:
        print(f"Error saat ambil latest run ID: {e}")
