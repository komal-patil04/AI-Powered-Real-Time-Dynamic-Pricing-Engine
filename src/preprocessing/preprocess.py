import sys
import os

# Pin the Spark worker subprocess to this exact Python executable (the
# active venv) rather than letting it resolve "python" via PATH, which can
# find a different system-wide Python version on Windows. Not strictly
# required here (no pandas_udf in this script), but kept consistent with
# optimize_price.py to avoid the same class of bug if one is added later.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.appName("SalesPreprocessing").getOrCreate()

sales = spark.read.csv(
    "data/raw/sales_transactions.csv",
    header=True,
    inferSchema=True
)

catalog = spark.read.csv(
    "data/raw/product_catalog.csv",
    header=True,
    inferSchema=True
)

features = spark.read.csv(
    "data/raw/demand_features.csv",
    header=True,
    inferSchema=True
)

elasticity = spark.read.csv(
    "data/raw/price_elasticity.csv",
    header=True,
    inferSchema=True
)


def clean_sales(df):

    df = df.dropDuplicates()

    df = df.withColumn("date", F.to_date(F.col("date")))

    numeric_cols = [
        "base_price",
        "current_price",
        "inventory_level",
        "units_sold",
        "revenue",
        "profit"
    ]

    for col in numeric_cols:

        df = df.withColumn(
            col,
            F.col(col).cast(DoubleType())
        )

    df = df.dropna()

    return df


def clean_catalog(df):

    df = df.dropDuplicates()

    df = df.withColumn(
        "department",
        F.coalesce(F.col("department"), F.lit("Unknown"))
    )

    df = df.withColumn(
        "aisle",
        F.coalesce(F.col("aisle"), F.lit("Unknown"))
    )

    return df


def clean_features(df):

    df = df.fillna(0)

    return df


def clean_elasticity(df):

    df = df.withColumn(
        "avg_price_elasticity",
        F.coalesce(F.col("avg_price_elasticity"), F.lit(0))
    )

    return df


if __name__ == "__main__":

    print("Starting preprocessing...")


# NOTE: kept identical to the original script — clean_sales/clean_catalog/
# clean_features/clean_elasticity are defined above but never called here,
# so the merge below runs on the raw (uncleaned) DataFrames, exactly as in
# the original pandas version.

join_keys = ["product_id", "product_name", "department", "store_id"]

# pandas' merge() auto-suffixes overlapping non-key columns with _x/_y.
# Spark has no built-in equivalent, so it's replicated manually here.
overlap_cols = (set(sales.columns) & set(catalog.columns)) - set(join_keys)

sales_renamed = sales
catalog_renamed = catalog

for c in overlap_cols:
    sales_renamed = sales_renamed.withColumnRenamed(c, f"{c}_x")
    catalog_renamed = catalog_renamed.withColumnRenamed(c, f"{c}_y")

master_df = sales_renamed.join(
    catalog_renamed,
    on=join_keys,
    how="left"
)

master_df = master_df.join(
    elasticity.select("product_id", "avg_price_elasticity"),
    on="product_id",
    how="left"
)

print(master_df.columns)

master_df = master_df.withColumn(
    "profit_margin",
    F.col("current_price") - F.col("cost_price_y")
)

master_df = master_df.withColumnRenamed(
    "base_price_y", "base_price"
).withColumnRenamed(
    "cost_price_y", "cost_price"
)

master_df = master_df.drop("base_price_x", "cost_price_x")

master_df = master_df.withColumn(
    "inventory_risk",
    (F.col("inventory_level") < 20).cast("int")
)

master_df = master_df.withColumn(
    "high_demand",
    (F.col("units_sold") > 100).cast("int")
)

# coalesce(1) mirrors pandas writing a single CSV file; Spark otherwise
# writes one part-file per partition into the target directory.
master_df.coalesce(1).write.mode("overwrite").option(
    "header", True
).csv("data/processed/master_dataset")

print("Preprocessing Completed!")

print("Rows:", master_df.count())

master_df.show(5)

spark.stop()