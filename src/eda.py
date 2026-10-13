
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

df = pd.read_csv("data/cleaned_cars.csv")

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(9, 5))
sns.histplot(df["Price"], bins=40, kde=True)
plt.title("Distribution of Used Car Prices")
plt.xlabel("Price (Lakhs)")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.savefig(output_dir / "price_distribution.png")
plt.close()

plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Fuel_Type", hue="Fuel_Type", legend=False)
plt.title("Number of Cars by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.savefig(output_dir / "fuel_types.png")
plt.close()

plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="Transmission", y="Price", hue="Transmission", legend=False)
plt.title("Car Prices by Transmission Type")
plt.xlabel("Transmission")
plt.ylabel("Price (Lakhs)")
plt.tight_layout()
plt.savefig(output_dir / "transmission_prices.png")
plt.close()

plt.figure(figsize=(9, 5))
sns.scatterplot(data=df, x="Year", y="Price", alpha=0.5)
plt.title("Car Price vs Manufacturing Year")
plt.xlabel("Manufacturing Year")
plt.ylabel("Price (Lakhs)")
plt.tight_layout()
plt.savefig(output_dir / "price_vs_year.png")
plt.close()

plt.figure(figsize=(9, 5))
sns.scatterplot(data=df, x="Kilometers_Driven", y="Price", alpha=0.5)
plt.title("Car Price vs Kilometers Driven")
plt.xlabel("Kilometers Driven")
plt.ylabel("Price (Lakhs)")
plt.tight_layout()
plt.savefig(output_dir / "price_vs_kilometers.png")
plt.close()

plt.figure(figsize=(10, 7))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Numerical Features")
plt.tight_layout()
plt.savefig(output_dir / "correlation_heatmap.png")
plt.close()

print("EDA completed successfully!")
print("\nAverage car price by fuel type:")
print(df.groupby("Fuel_Type")["Price"].mean().round(2))

print("\nAverage car price by transmission:")
print(df.groupby("Transmission")["Price"].mean().round(2))

print("\nCharts saved in the outputs folder:")
for file in sorted(output_dir.glob("*.png")):
    print(file)
