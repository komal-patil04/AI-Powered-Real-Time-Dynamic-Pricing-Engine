import joblib

model = joblib.load(
    "models/demand_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

sample = [[

    50,
    100,
    10,
    15,
    -1.2,
    0,
    1,
    6,
    7,
    1.2,
    20

]]

sample = scaler.transform(
    sample
)

prediction = model.predict(
    sample
)

print(
    "Predicted Demand:",
    int(prediction[0])
)