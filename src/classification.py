
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("data/cleaned_cars.csv")

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

X = df[features]
y = df["Price_Category"]

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

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("CLASSIFICATION RESULTS")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions, average="weighted", zero_division=0), 4))
print("Recall:", round(recall_score(y_test, predictions, average="weighted", zero_division=0), 4))
print("F1 Score:", round(f1_score(y_test, predictions, average="weighted", zero_division=0), 4))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion Matrix (Low, Medium, High):")
print(confusion_matrix(y_test, predictions, labels=["Low", "Medium", "High"]))

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/classification_model.pkl")
print("\nModel saved to models/classification_model.pkl")
