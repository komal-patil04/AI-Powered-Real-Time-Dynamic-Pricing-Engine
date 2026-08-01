import pandas as pd
sales = pd.read_csv(
    "data/raw/sales_transactions.csv"
)

catalog = pd.read_csv(
    "data/raw/product_catalog.csv"
)

features = pd.read_csv(
    "data/raw/demand_features.csv"
)

elasticity = pd.read_csv(
    "data/raw/price_elasticity.csv"
)

def clean_sales(df):

    df = df.drop_duplicates()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    numeric_cols = [
        "base_price",
        "current_price",
        "inventory_level",
        "units_sold",
        "revenue",
        "profit"
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna()

    return df

def clean_catalog(df):

    df = df.drop_duplicates()

    df["department"] = df[
        "department"
    ].fillna(
        "Unknown"
    )

    df["aisle"] = df[
        "aisle"
    ].fillna(
        "Unknown"
    )

    return df

def clean_features(df):

    df.fillna(0, inplace=True)

    return df

def clean_elasticity(df):

    df["avg_price_elasticity"] = \
        df["avg_price_elasticity"]\
        .fillna(0)

    return df

if __name__ == "__main__":

    print("Starting preprocessing...")


master_df = sales.merge(

    catalog,

    on=[
        "product_id",
        "product_name",
        "department",
        "store_id"
    ],

    how="left"
)

master_df = master_df.merge(

    elasticity[
        [
            "product_id",
            "avg_price_elasticity"
        ]
    ],

    on="product_id",

    how="left"
)
print(master_df.columns)

master_df["profit_margin"] = (

    master_df["current_price"] -

    master_df["cost_price_y"]

)
master_df.rename(
    columns={
        "base_price_y": "base_price",
        "cost_price_y": "cost_price"
    },
    inplace=True
)

master_df.drop(
    columns=["base_price_x", "cost_price_x"],
    inplace=True
)

master_df["inventory_risk"] = (

    master_df["inventory_level"] < 20

).astype(int)

master_df["high_demand"] = (

    master_df["units_sold"] > 100

).astype(int)

master_df.to_csv(

    "data/processed/master_dataset.csv",

    index=False
)

print("Preprocessing Completed!")

print("Rows:", len(master_df))

print(master_df.head())