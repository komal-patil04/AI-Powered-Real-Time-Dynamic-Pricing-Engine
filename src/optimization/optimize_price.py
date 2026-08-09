
import pandas as pd
import joblib

# --------------------------------------------------
# 1. READ FEATURE DATASET
# --------------------------------------------------

df = pd.read_csv(
    "data/final/feature_dataset.csv"
)

print("Feature dataset loaded successfully!")
print("Rows:", len(df))

# --------------------------------------------------
# 2. LOAD MODEL AND SCALER
# --------------------------------------------------

model = joblib.load(
    "models/demand_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

# --------------------------------------------------
# 3. SELECT FEATURES
# --------------------------------------------------

feature_cols = [
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

X = df[feature_cols]

# --------------------------------------------------
# 4. SCALE FEATURES
# --------------------------------------------------

X_scaled = scaler.transform(X)

# --------------------------------------------------
# 5. PREDICT DEMAND
# --------------------------------------------------

df["predicted_demand"] = model.predict(
    X_scaled
)

# --------------------------------------------------
# 6. DEMAND-BASED PRICE ADJUSTMENT
# --------------------------------------------------

def demand_multiplier(demand):

    if demand > 150:
        return 1.15

    elif demand > 100:
        return 1.10

    elif demand < 50:
        return 0.90

    else:
        return 1.0


df["demand_multiplier"] = (
    df["predicted_demand"]
    .apply(demand_multiplier)
)

# --------------------------------------------------
# 7. INVENTORY ADJUSTMENT
# --------------------------------------------------

df["inventory_multiplier"] = (
    df["inventory_level"]
    .apply(
        lambda x: 1.05 if x < 20 else 1.0
    )
)

# --------------------------------------------------
# 8. ELASTICITY ADJUSTMENT
# --------------------------------------------------

df["elasticity_multiplier"] = (
    df["avg_price_elasticity"]
    .apply(
        lambda x: 0.95 if x < -2 else 1.0
    )
)

# --------------------------------------------------
# 9. FINAL RECOMMENDED PRICE
# --------------------------------------------------

df["recommended_price"] = (
    df["current_price"]
    * df["demand_multiplier"]
    * df["inventory_multiplier"]
    * df["elasticity_multiplier"]
).round(2)

# --------------------------------------------------
# 10. SELECT OUTPUT COLUMNS
# --------------------------------------------------

output_cols = [
    "product_id",
    "current_price",
    "predicted_demand",
    "recommended_price"
]

result = df[output_cols]

# --------------------------------------------------
# 11. SAVE RESULT
# --------------------------------------------------

result.to_csv(
    "data/final/optimized_prices.csv",
    index=False
)

print("Price Optimization Completed!")

print("\nSample optimized prices:")
print(result.head())

print(
    "\nOutput saved to "
    "data/final/optimized_prices.csv"
)

