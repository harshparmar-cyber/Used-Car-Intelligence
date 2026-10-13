
import pandas as pd
import heapq

df = pd.read_csv("data/cleaned_cars.csv")


def recommend_cars(budget, min_year, max_km, fuel, transmission, top_n=10):
    candidates = df[
        (df["Price"] <= budget) &
        (df["Year"] >= min_year) &
        (df["Kilometers_Driven"] <= max_km)
    ].copy()

    if fuel != "Any":
        candidates = candidates[candidates["Fuel_Type"] == fuel]

    if transmission != "Any":
        candidates = candidates[candidates["Transmission"] == transmission]

    if candidates.empty:
        return candidates

    candidates["score"] = (
        (budget - candidates["Price"]) / max(budget, 1)
        + (candidates["Year"] - min_year) / max(2026 - min_year, 1)
        + (max_km - candidates["Kilometers_Driven"]) / max(max_km, 1)
    )

    priority_queue = []

    for index, car in candidates.iterrows():
        heapq.heappush(
            priority_queue,
            (-car["score"], index)
        )

    recommendations = []

    while priority_queue and len(recommendations) < top_n:
        _, index = heapq.heappop(priority_queue)
        recommendations.append(candidates.loc[index])

    result = pd.DataFrame(recommendations)
    return result.drop(columns=["score"], errors="ignore")


if __name__ == "__main__":
    budget = float(input("Enter maximum budget in lakhs: "))
    min_year = int(input("Enter minimum manufacturing year: "))
    max_km = int(input("Enter maximum kilometers driven: "))

    fuel = input("Enter fuel type (Diesel/Petrol/Electric/Any): ").strip().title()
    transmission = input("Enter transmission (Manual/Automatic/Any): ").strip().title()

    if fuel not in ["Diesel", "Petrol", "Electric", "Any"]:
        fuel = "Any"

    if transmission not in ["Manual", "Automatic", "Any"]:
        transmission = "Any"

    result = recommend_cars(
        budget, min_year, max_km, fuel, transmission
    )

    if result.empty:
        print("\nNo cars matched your preferences.")
    else:
        print("\nRecommended Cars:")
        print(
            result[
                ["Name", "Year", "Fuel_Type", "Transmission",
                 "Kilometers_Driven", "Price"]
            ].to_string(index=False)
        )
