
import pandas as pd
from scipy.stats import ttest_ind

df = pd.read_csv("data/cleaned_cars.csv")

print("STATISTICAL ANALYSIS OF USED CAR DATASET")

print("\n1. Descriptive Statistics")
print(df[["Price", "Year", "Kilometers_Driven", "Mileage"]].describe())

print("\n2. Mean Car Price")
print("Mean:", round(df["Price"].mean(), 2), "lakhs")

print("\n3. Variance in Car Prices")
print("Variance:", round(df["Price"].var(), 2))

print("\n4. Correlation Between Numerical Features")
print(
    df[["Price", "Year", "Kilometers_Driven", "Mileage", "Engine", "Power"]]
    .corr()
    .round(3)
)

print("\n5. Covariance Between Price and Year")
print(round(df["Price"].cov(df["Year"]), 3))

petrol_prices = df.loc[df["Fuel_Type"] == "Petrol", "Price"]
diesel_prices = df.loc[df["Fuel_Type"] == "Diesel", "Price"]

t_stat, p_value = ttest_ind(
    petrol_prices,
    diesel_prices,
    equal_var=False,
    nan_policy="omit"
)

print("\n6. Welch's Independent T-Test")
print("Petrol sample size:", len(petrol_prices))
print("Diesel sample size:", len(diesel_prices))
print("T-statistic:", round(t_stat, 4))
print("P-value:", p_value)

if p_value < 0.05:
    print("Result: The average prices differ significantly at the 5% level.")
else:
    print("Result: No statistically significant difference was detected at the 5% level.")
