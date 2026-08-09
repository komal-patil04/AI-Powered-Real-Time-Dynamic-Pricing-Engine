import glob

from fastapi import FastAPI
import pandas as pd
from datetime import datetime

app = FastAPI()


@app.get("/health")
def health():
    return {
        "status": "running"
    }


@app.get("/live")
def live():

    df = pd.read_csv(
        "data/live_data.csv"
    )

    return df.tail(
        10
    ).to_dict(
        orient="records"
    )


@app.get("/products")
def products():

    df = pd.read_csv(
        "data/raw/product_catalog.csv"
    )

    return df[
        [
            "product_id",
            "product_name"
        ]
    ].to_dict(
        orient="records"
    )


@app.get(
    "/recommended-price/{product_id}"
)
def price(product_id):

    # optimize_price_spark.py writes this as a FOLDER of part-*.csv files,
    # not a single "optimized_prices.csv" file — reading that exact
    # filename directly would raise FileNotFoundError. This finds whichever
    # part-file(s) Spark produced inside the output folder and reads them.
    part_files = glob.glob(
        "data/final/optimized_prices/part-*.csv"
    )

    if not part_files:
        return {
            "error": "optimized_prices output not found — has the "
                     "pricing pipeline run yet?"
        }

    df = pd.concat(
        (pd.read_csv(f) for f in part_files),
        ignore_index=True
    )

    matches = df[df["product_id"] == product_id]

    if matches.empty:
        return {"error": f"product_id {product_id} not found"}

    row = matches.iloc[0]

    return {

        "product":
        row["product_name"],

        "recommended_price":
        row[
            "recommended_price"
        ]
    }