
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("data/cleaned_cars.csv")

print("Original records:", len(df))

outliers = df[df["Kilometers_Driven"] > 1_000_000]
print("Records excluded due to extreme mileage:", len(outliers))

if not outliers.empty:
    print(outliers[["Name", "Kilometers_Driven", "Price"]].to_string(index=False))

df = df[df["Kilometers_Driven"] <= 1_000_000].copy()

features = [
    "Car_Age",
    "Kilometers_Driven",
    "Mileage",
    "Engine",
    "Power",
    "Seats"
]

X = df[features]
X_scaled = StandardScaler().fit_transform(X)

k_values = range(2, 9)
inertias = []
scores = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    scores.append(silhouette_score(X_scaled, labels))

best_k = list(k_values)[scores.index(max(scores))]

print("\nK-MEANS RESULTS")
print("Selected clusters:", best_k)
print("Silhouette score:", round(max(scores), 4))

Path("outputs").mkdir(exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig("outputs/kmeans_elbow.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), scores, marker="o")
plt.title("Silhouette Scores")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig("outputs/kmeans_silhouette.png")
plt.close()

final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = final_model.fit_predict(X_scaled)

print("\nCars per cluster:")
print(df["Cluster"].value_counts().sort_index())

print("\nCluster averages:")
print(
    df.groupby("Cluster")[
        ["Price", "Car_Age", "Kilometers_Driven", "Mileage", "Engine", "Power"]
    ].mean().round(2)
)

df.to_csv("data/clustered_cars.csv", index=False)
print("\nSaved filtered clustering results to data/clustered_cars.csv")
