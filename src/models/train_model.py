import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.data.data_cleaning import load_config, project_path


def load_processed(config):
    return pd.read_csv(project_path(config["data"]["processed_path"]))


def split_data(df, config):
    target = config["data"]["target"]
    X = df.drop(columns=["student_id", target])
    y = df[target]
    return train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )

from sklearn.preprocessing import StandardScaler


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train, config):
    model = LogisticRegression(
        max_iter=config["model"]["max_iter"],
        random_state=config["model"]["random_state"],
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }


def save_model(model, config):
    model_path = project_path(config["output"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def save_metrics(metrics, config):
    metrics_path = project_path(config["output"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def run_training(config):
    df = load_processed(config)
    X_train, X_test, y_train, y_test = split_data(df, config)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)   # tambahkan ini
    model = train_model(X_train_scaled, y_train, config)                      # pakai versi scaled
    metrics = evaluate_model(model, X_test_scaled, y_test)                    # pakai versi scaled
    save_model(model, config)
    save_metrics(metrics, config)
    return model, metrics


if __name__ == "__main__":
    config = load_config()
    model, metrics = run_training(config)
    print("Metrics:", metrics)