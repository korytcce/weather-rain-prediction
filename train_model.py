import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_FEATURES = [
    "temperature_max",
    "temperature_min",
    "wind_speed",
]


def prepare_features(df):
    features_df = df.copy()

    features_df["temperature_avg"] = (
        features_df["temperature_max"] + features_df["temperature_min"]
    ) / 2
    features_df["temperature_range"] = (
        features_df["temperature_max"] - features_df["temperature_min"]
    )
    features_df["wind_temp_ratio"] = (
        features_df["wind_speed"] / (features_df["temperature_avg"].abs() + 1)
    )

    candidate_features = BASE_FEATURES + [
        "temperature_avg",
        "temperature_range",
        "wind_temp_ratio",
    ]

    return features_df[candidate_features]


def preprocess_features(X_train, X_test):
    imputation_values = X_train.median(numeric_only=True)

    X_train_processed = X_train.fillna(imputation_values)
    X_test_processed = X_test.fillna(imputation_values)

    return X_train_processed, X_test_processed, imputation_values


def select_features(X_train, y_train, X_test, k=4):
    k = min(k, X_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    selected_features = X_train.columns[selector.get_support()].tolist()
    feature_scores = pd.DataFrame({
        "feature": X_train.columns,
        "score": selector.scores_,
    }).sort_values("score", ascending=False)

    X_train_selected = pd.DataFrame(
        X_train_selected,
        columns=selected_features,
        index=X_train.index,
    )
    X_test_selected = pd.DataFrame(
        X_test_selected,
        columns=selected_features,
        index=X_test.index,
    )

    return X_train_selected, X_test_selected, selected_features, feature_scores


def train_models():
    df = pd.read_csv("weather_daily.csv")

    df["target"] = (df["precipitation_sum"] > 0).astype(int)
    class_distribution = (
        df["target"]
        .value_counts(normalize=True)
        .rename(index={0: "Без опадів", 1: "Є опади"})
        .mul(100)
        .round(2)
    )

    if df["target"].nunique() < 2:
        raise ValueError(
            "У вибраному періоді є тільки один клас. "
            "Збільш період даних, щоб були дні і з опадами, і без опадів."
        )
    if df["target"].value_counts().min() < 2:
        raise ValueError(
            "Один із класів має замало прикладів для train/test split. "
            "Збільш період даних."
        )

    X = prepare_features(df)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train, X_test, imputation_values = preprocess_features(X_train, X_test)

    (
        X_train_selected,
        X_test_selected,
        selected_features,
        feature_scores,
    ) = select_features(X_train, y_train, X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
        ),
    }

    best_model = None
    best_score = 0
    results = {}

    for name, model in models.items():
        model.fit(X_train_selected, y_train)

        preds = model.predict(X_test_selected)

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        if f1 > best_score:
            best_score = f1
            best_model = model

    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "selected_features": selected_features,
            "imputation_values": imputation_values,
        },
        "models/rain_model.pkl",
    )

    feature_info = {
        "candidate_features": X.columns.tolist(),
        "selected_features": selected_features,
        "feature_scores": feature_scores,
        "class_distribution": class_distribution,
        "imputation_values": imputation_values,
    }

    return results, best_model, feature_info
