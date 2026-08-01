
import pandas as pd

df = pd.read_csv(
    "data/processed/master_dataset.csv"
)

#timebased features
df["date"] = pd.to_datetime(
    df["date"]
)

df["day"] = df["date"].dt.day

df["month"] = df["date"].dt.month

df["day_of_week"] = \
    df["date"].dt.dayofweek

df["weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

#price features
df["discount"] = (

    df["base_price"] -

    df["current_price"]

)

df["discount_pct"] = (

    df["discount"] /

    df["base_price"]

) * 100

#inventory features
df["low_stock"] = (

    df["inventory_level"] < 20

).astype(int)

df["inventory_ratio"] = (

    df["units_sold"] /

    (df["inventory_level"] + 1)

)

#profit features
df["profit_per_unit"] = (

    df["current_price"] -

    df["cost_price"]

)

#elasticity features
df["high_elasticity"] = (

    df["avg_price_elasticity"] < -1

).astype(int)

#demand features
df["high_sales"] = (

    df["units_sold"] > 100

).astype(int)

#final features columns
features = [

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



df.to_csv(

    "data/final/feature_dataset.csv",

    index=False
)