import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def main(args):
    print(f"Mencoba memuat data dari: {args.data_path}")
    data = pd.read_csv(args.data_path)
    print("Data berhasil dimuat.")
    print("Kolom data:", list(data.columns))

    X = data.drop(columns=[args.target_var])
    y = data[args.target_var]

    # Split data train-test
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Ukuran X_train:", X_train.shape)
    print("Ukuran X_val:", X_val.shape)
    print("Ukuran y_train:", y_train.shape)
    print("Ukuran y_val:", y_val.shape)

    # Tanpa SMOTE, langsung pakai data asli
    X_train_res, y_train_res = X_train, y_train

    param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": [None, "balanced"],
    }

    clf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    with mlflow.start_run(run_name=args.run_name):
        clf.fit(X_train_res, y_train_res)
        y_pred = clf.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        print("Parameter terbaik:", clf.best_params_)
        print("Akurasi:", accuracy)
        print("Classification Report:")
        print(classification_report(y_val, y_pred, zero_division=0))

        # Logging MLflow
        mlflow.log_params(clf.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(clf.best_estimator_, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path ke CSV data")
    parser.add_argument("--target_var", type=str, default="quality", help="Target variabel")
    parser.add_argument("--run_name", type=str, default="rf_fixed_run", help="Nama run MLflow")

    args = parser.parse_args()
    main(args)
