
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/cleaned_cars.csv")

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
y = df["Price"]

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
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("LINEAR REGRESSION RESULTS")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("MAE:", round(mae, 3), "lakhs")
print("MSE:", round(mse, 3))
print("RMSE:", round(rmse, 3), "lakhs")
print("R2 Score:", round(r2, 3))

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/price_model.pkl")

print("\nModel saved to models/price_model.pkl")

sample = X_test.iloc[[0]]
actual_price = y_test.iloc[0]
predicted_price = model.predict(sample)[0]

print("\nSample Prediction")
print("Car:", df.loc[sample.index[0], "Name"])
print("Actual price:", round(actual_price, 2), "lakhs")
print("Predicted price:", round(predicted_price, 2), "lakhs")
