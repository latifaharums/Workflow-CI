import pandas as pd
import numpy as np
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


parser = argparse.ArgumentParser()
parser.add_argument("--csv_url", type=str, required=True)
parser.add_argument("--target_var", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

print(f"Mencoba memuat data dari: {args.data_path}")
df = pd.read_csv(args.data_path)
print("Data berhasil dimuat.")
print("Kolom data:", list(df.columns))

X = df.drop(columns=[args.target_var])
y = df[args.target_var]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Ukuran X_train: {X_train.shape}")
print(f"Ukuran X_val: {X_val.shape}")
print(f"Ukuran y_train: {y_train.shape}")
print(f"Ukuran y_val: {y_val.shape}")


params = {
    "n_estimators": [100, 150, 200],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2],
    "class_weight": ["balanced"]
}
clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=params, cv=3)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred)

print("Parameter terbaik:", clf.best_params_)
print("Akurasi:", acc)
print("Classification Report:\n", report)

mlflow.set_experiment("Red_Wine_Tunning")
mlflow.log_params(clf.best_params_)
mlflow.log_metric("accuracy", acc)


mlflow.sklearn.log_model(clf.best_estimator_, "rf_best_model")

print(f"Tuning selesai dan model berhasil dicatat ke MLflow DagsHub (Run: {mlflow.active_run().info.run_id})")
