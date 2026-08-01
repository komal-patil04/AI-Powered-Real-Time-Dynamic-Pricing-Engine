import pandas as pd
import joblib


# Load feature dataset
df = pd.read_csv(
    "data/final/feature_dataset.csv"
)


# Load trained model
model = joblib.load(
    "models/demand_model.pkl"
)


# Predict demand
X = df[
    [
        "current_price",
        "inventory_level",
        "discount_pct",
        "profit_margin",
        "avg_price_elasticity",
        "inventory_risk",
        "weekend",
        "day_of_week",
        "month",
        "inventory_ratio",
        "profit_per_unit"
    ]
]

df["predicted_demand"] = model.predict(X)


# Optimization function
def optimize_price(
    current_price,
    predicted_demand,
    inventory,
    elasticity
):

    new_price = current_price

    if predicted_demand > 150:
        new_price *= 1.15

    elif predicted_demand > 100:
        new_price *= 1.10

    elif predicted_demand < 50:
        new_price *= 0.90

    if inventory < 20:
        new_price *= 1.05

    if elasticity < -2:
        new_price *= 0.95

    return round(new_price, 2)


# Apply optimization
df["recommended_price"] = df.apply(
    lambda x: optimize_price(
        x["current_price"],
        x["predicted_demand"],
        x["inventory_level"],
        x["avg_price_elasticity"]
    ),
    axis=1
)


# Save output
df.to_csv(
    "data/final/optimized_prices.csv",
    index=False
)

print("Price Optimization Completed!")
print(df[
    [
        "product_id",
        "current_price",
        "predicted_demand",
        "recommended_price"
    ]
].head())