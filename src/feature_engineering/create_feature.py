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
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

# NOTE: master_dataset.csv is written by Spark as a directory of part-files
# (see preprocessing_spark.py), so it's read here the same way — Spark
# transparently reads all part-files in the directory as one DataFrame.
df = spark.read.csv(
    "data/processed/master_dataset",
    header=True,
    inferSchema=True
)

# time-based features
df = df.withColumn("date", F.to_date(F.col("date")))

df = df.withColumn("day", F.dayofmonth(F.col("date")))

df = df.withColumn("month", F.month(F.col("date")))

# pandas dayofweek: Monday=0 ... Sunday=6
# Spark's dayofweek(): Sunday=1 ... Saturday=7, so it's remapped to match
df = df.withColumn(
    "day_of_week",
    (F.dayofweek(F.col("date")) + 5) % 7
)

df = df.withColumn(
    "weekend",
    (F.col("day_of_week") >= 5).cast("int")
)

# price features
df = df.withColumn(
    "discount",
    F.col("base_price") - F.col("current_price")
)

df = df.withColumn(
    "discount_pct",
    (F.col("discount") / F.col("base_price")) * 100
)

# inventory features
df = df.withColumn(
    "low_stock",
    (F.col("inventory_level") < 20).cast("int")
)

# LEAKAGE FIX: the original inventory_ratio = units_sold / (inventory_level+1)
# used the SAME ROW's units_sold — the exact value being predicted — so the
# model was trivially "reversing" its own target instead of forecasting it
# (this produced an unrealistically low MAE of ~0.06).
#
# Fixed version uses each product's HISTORICAL average units_sold, computed
# only from rows strictly before the current one (ordered by date). This
# mirrors what's actually knowable at prediction time — you never know
# today's units_sold in advance — and matches the historical-average
# fallback consumer.py already uses for live scoring.
product_window = Window.partitionBy("product_id") \
    .orderBy("date") \
    .rowsBetween(Window.unboundedPreceding, -1)

df = df.withColumn(
    "avg_past_units_sold",
    F.avg("units_sold").over(product_window)
)

df = df.withColumn(
    "inventory_ratio",
    F.coalesce(
        F.col("avg_past_units_sold") / (F.col("inventory_level") + 1),
        F.lit(0.0)
    )
)

df = df.drop("avg_past_units_sold")

# profit features
df = df.withColumn(
    "profit_per_unit",
    F.col("current_price") - F.col("cost_price")
)

# elasticity features
df = df.withColumn(
    "high_elasticity",
    (F.col("avg_price_elasticity") < -1).cast("int")
)

# demand features
df = df.withColumn(
    "high_sales",
    (F.col("units_sold") > 100).cast("int")
)

# final feature columns
# (kept as in the original — defined but not applied as a df.select())
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

df.coalesce(1).write.mode("overwrite").option(
    "header", True
).csv("data/final/feature_dataset")

spark.stop()