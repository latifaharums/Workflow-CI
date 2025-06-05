import argparse
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


parser = argparse.ArgumentParser()
parser.add_argument("--csv_url", type=str, required=True)
parser.add_argument("--target_var", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()


print(f"Mencoba memuat data dari: {args.data_path}")
df = pd.read_csv(args.data_path)
print("Data berhasil dimuat.")
print("Kolom data:", df.columns.tolist())

X = df.drop(columns=[args.target_var])
y = df[args.target_var]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Ukuran X_train:", X_train.shape)
print("Ukuran X_val:", X_val.shape)
print("Ukuran y_train:", y_train.shape)
print("Ukuran y_val:", y_val.shape)


mlflow.set_tracking_uri(uri="https://dagshub.com/latifaharums/Membangun_model.mlflow/")
mlflow.set_experiment("Red_Wine_Tunning")

with mlflow.start_run(run_name="rf_fixed_run") as run:

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced"]
    }

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)

    print("Parameter terbaik:", grid_search.best_params_)
    print("Akurasi:", acc)
    print("Classification Report:\n", report)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", acc)

    signature = infer_signature(X_val, y_pred)
    input_example = pd.DataFrame(X_val[:5], columns=df.drop(columns=[args.target_var]).columns)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="rf_best_model",
        input_example=input_example,
        signature=signature
    )

    print(f"Tuning selesai dan model berhasil dicatat ke MLflow DagsHub (Eksperimen: {run.info.experiment_id}, Run: {run.info.run_id})")
    print(f"View run rf_fixed_run at: https://dagshub.com/latifaharums/Membangun_model.mlflow/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
    print(f"View experiment at: https://dagshub.com/latifaharums/Membangun_model.mlflow/#/experiments/{run.info.experiment_id}")
