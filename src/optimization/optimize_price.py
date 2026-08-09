import sys
import os

# Force the Spark worker subprocess to use THIS exact Python executable
# (the active venv), not whatever "python" it finds first on PATH. Without
# this, on Windows, Spark can spawn a different system-wide Python for
# pandas_udf workers than the one the driver is running under — a version
# mismatch that crashes the worker with an Arrow/socket serialization error
# (this is what caused "Python worker exited unexpectedly" above).
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import joblib
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.appName("PriceOptimization").getOrCreate()

# Load feature dataset
# (written by Spark as a directory of part-files in the feature-engineering
# step, so it's read the same way here)
df = spark.read.csv(
    "data/final/feature_dataset",
    header=True,
    inferSchema=True
)

# Load trained model and scaler
# sklearn/joblib objects don't run natively across a Spark cluster, so
# they're loaded once on the driver and broadcast to every executor. Each
# executor then scores its own partition locally via the pandas_udf below.
# The scaler must be applied before prediction — the model was trained on
# scaled features (see predict.py), so skipping this step would silently
# produce wrong predictions.
model = joblib.load("models/demand_model.pkl")
scaler = joblib.load("models/scaler.pkl")

broadcast_model = spark.sparkContext.broadcast(model)
broadcast_scaler = spark.sparkContext.broadcast(scaler)

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


@F.pandas_udf(DoubleType())
def predict_demand_udf(*cols):
    # Spark hands this function one pandas Series per feature column, per
    # batch/partition — reassembled here into the same column layout the
    # sklearn model was trained on. Scaled with the same scaler used at
    # training time before calling .predict(), exactly as predict.py does
    # for its single hardcoded sample.
    X = pd.concat(cols, axis=1)
    X.columns = feature_cols
    X_scaled = broadcast_scaler.value.transform(X)
    return pd.Series(broadcast_model.value.predict(X_scaled))


# Predict demand
df = df.withColumn(
    "predicted_demand",
    predict_demand_udf(*[F.col(c) for c in feature_cols])
)


# Optimization logic
# Expressed as vectorized when/otherwise multipliers rather than a row-wise
# Python function — same result as the original sequential if/elif
# multiplications (order doesn't matter since it's all multiplicative),
# but runs as a single columnar operation instead of one Python call per row.
demand_multiplier = (
    F.when(F.col("predicted_demand") > 150, 1.15)
    .when(F.col("predicted_demand") > 100, 1.10)
    .when(F.col("predicted_demand") < 50, 0.90)
    .otherwise(1.0)
)

inventory_multiplier = F.when(
    F.col("inventory_level") < 20, 1.05
).otherwise(1.0)

elasticity_multiplier = F.when(
    F.col("avg_price_elasticity") < -2, 0.95
).otherwise(1.0)

df = df.withColumn(
    "recommended_price",
    F.round(
        F.col("current_price")
        * demand_multiplier
        * inventory_multiplier
        * elasticity_multiplier,
        2
    )
)

# Save output
df.coalesce(1).write.mode("overwrite").option(
    "header", True
).csv("data/final/optimized_prices")

print("Price Optimization Completed!")

df.select(
    "product_id",
    "current_price",
    "predicted_demand",
    "recommended_price"
).show(5)

spark.stop()