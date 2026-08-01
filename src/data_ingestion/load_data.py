import pandas as pd

class DataLoader:

    def __init__(self):

        self.sales = None
        self.catalog = None
        self.features = None
        self.forecast = None
        self.elasticity = None
        self.optimization = None

    def load_all_data(self):

        print("Loading datasets...")

        self.sales = pd.read_csv(
            "data/raw/sales_transactions.csv"
        )

        self.catalog = pd.read_csv(
            "data/raw/product_catalog.csv"
        )

        self.features = pd.read_csv(
            "data/raw/demand_features.csv"
        )

        self.forecast = pd.read_csv(
            "data/raw/demand_forecast.csv"
        )

        self.elasticity = pd.read_csv(
            "data/raw/price_elasticity.csv"
        )

        self.optimization = pd.read_csv(
            "data/raw/optimal_pricing_output.csv"
        )

        print("Datasets Loaded Successfully!")

        print("\nRows:")
        print("Sales:", len(self.sales))
        print("Catalog:", len(self.catalog))
        print("Features:", len(self.features))
        print("Forecast:", len(self.forecast))
        print("Elasticity:", len(self.elasticity))
        print("Optimization:", len(self.optimization))


if __name__ == "__main__":

    loader = DataLoader()
    loader.load_all_data()