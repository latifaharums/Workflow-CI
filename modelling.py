import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

parser = argparse.ArgumentParser()
parser.add_argument("--csv_url", type=str, required=True)
parser.add_argument("--target_var", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

mlflow.set_tracking_uri("https://dagshub.com/latifaharums/Membangun_model.mlflow/")
mlflow.set_experiment("Red_Wine_Tunning")

print(f"Mencoba memuat data dari: {args.csv_url}")
df = pd.read_csv(args.csv_url)
print("Data berhasil dimuat.")
print("Kolom data:", df.columns.tolist())

X = df.drop(args.target_var, axis=1)
y = df[args.target_var]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("Ukuran X_train:", X_train.shape)
print("Ukuran X_val:", X_val.shape)
print("Ukuran y_train:", y_train.shape)
print("Ukuran y_val:", y_val.shape)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

clf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={
        'n_estimators': [100, 150, 200],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced']
    },
    cv=3,
    n_jobs=-1,
    verbose=1
)
clf.fit(X_train_sm, y_train_sm)

y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred, digits=4)

print("Parameter terbaik:", clf.best_params_)
print("Akurasi:", acc)
print("Classification Report:\n", report)

mlflow.log_params(clf.best_params_)
mlflow.log_metric("accuracy", acc)
mlflow.sklearn.log_model(clf.best_estimator_, "random_forest_model")

print("Tuning selesai dan model berhasil dicatat ke MLflow DagsHub.")
