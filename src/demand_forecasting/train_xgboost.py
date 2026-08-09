import glob

import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.model_selection import \
train_test_split
from sklearn.preprocessing import \
StandardScaler
from sklearn.metrics import \
mean_absolute_error


part_files = glob.glob(
    "data/final/feature_dataset/part-*.csv"
)

if not part_files:
    raise FileNotFoundError(
        "No part files found in data/final/feature_dataset/ — "
        "has create_feature.py been run yet?"
    )

df = pd.concat(
    (pd.read_csv(f) for f in part_files),
    ignore_index=True
)

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

y = df["units_sold"]

X_train, X_test, y_train, y_test = \
train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

model = XGBRegressor(

    n_estimators=200,

    max_depth=8,

    learning_rate=0.05,

    random_state=42

)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    predictions
)

print("MAE:", mae)

joblib.dump(

    model,

    "models/demand_model.pkl"
)

joblib.dump(

    scaler,

    "models/scaler.pkl"
)