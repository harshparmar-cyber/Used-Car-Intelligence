
import pandas as pd
import re
from pathlib import Path

DATA_PATH = Path("data/train.csv")
OUTPUT_PATH = Path("data/cleaned_cars.csv")

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df = df.drop_duplicates()

def extract_number(value):
    if pd.isna(value):
        return None
    match = re.search(r"\d+\.?\d*", str(value))
    return float(match.group()) if match else None

for column in ["Mileage", "Engine", "Power"]:
    df[column] = df[column].apply(extract_number)

numeric_columns = [
    "Year",
    "Kilometers_Driven",
    "Mileage",
    "Engine",
    "Power",
    "Seats",
    "Price"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df["Mileage"] = df["Mileage"].replace(0, None)

df = df.dropna(subset=["Price", "Year", "Kilometers_Driven"])
df = df[df["Price"] > 0]

for column in ["Mileage", "Engine", "Power", "Seats"]:
    df[column] = df[column].fillna(df[column].median())

df["Brand"] = df["Name"].str.split().str[0]
df["Car_Age"] = 2026 - df["Year"]

df = df.drop(columns=["New_Price"], errors="ignore")

df.to_csv(OUTPUT_PATH, index=False)

print("\nData cleaning completed!")
print("Cleaned dataset shape:", df.shape)
print("\nRemaining missing values:")
print(df.isnull().sum())
print("\nFirst five cleaned records:")
print(df.head())
print(f"\nCleaned dataset saved to: {OUTPUT_PATH}")
