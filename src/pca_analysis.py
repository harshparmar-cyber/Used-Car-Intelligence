
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

df = pd.read_csv("data/cleaned_cars.csv")

df = df[df["Kilometers_Driven"] <= 1_000_000].copy()

df["Price_Category"] = pd.qcut(
    df["Price"],
    q=3,
    labels=["Low", "Medium", "High"],
    duplicates="drop"
)

features = [
    "Year",
    "Kilometers_Driven",
    "Mileage",
    "Engine",
    "Power",
    "Seats",
    "Fuel_Type",
    "Transmission",
    "Owner_Type",
    "Location"
]

numeric_features = [
    "Year",
    "Kilometers_Driven",
    "Mileage",
    "Engine",
    "Power",
    "Seats"
]

categorical_features = [
    "Fuel_Type",
    "Transmission",
    "Owner_Type",
    "Location"
]

X = df[features]
y = df["Price_Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

baseline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

pca_model = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", PCA(n_components=0.95, svd_solver="full")),
    ("classifier", LogisticRegression(max_iter=2000))
])

baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

pca_model.fit(X_train, y_train)
pca_predictions = pca_model.predict(X_test)

def evaluate(name, actual, predicted):
    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(actual, predicted), 4))
    print("Precision:", round(precision_score(actual, predicted, average="weighted", zero_division=0), 4))
    print("Recall:", round(recall_score(actual, predicted, average="weighted", zero_division=0), 4))
    print("F1-score:", round(f1_score(actual, predicted, average="weighted", zero_division=0), 4))

evaluate("BASELINE CLASSIFICATION", y_test, baseline_predictions)
evaluate("CLASSIFICATION WITH PCA", y_test, pca_predictions)

pca_step = pca_model.named_steps["pca"]

print("\nPCA RESULTS")
print("Original processed features:", pca_step.n_features_in_)
print("Principal components retained:", pca_step.n_components_)
print(
    "Total explained variance:",
    round(pca_step.explained_variance_ratio_.sum(), 4)
)

Path("outputs").mkdir(exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(pca_step.explained_variance_ratio_) + 1),
    pca_step.explained_variance_ratio_,
    marker="o"
)
plt.title("PCA Explained Variance by Component")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.tight_layout()
plt.savefig("outputs/pca_explained_variance.png")
plt.close()

print("\nPCA chart saved to outputs/pca_explained_variance.png")