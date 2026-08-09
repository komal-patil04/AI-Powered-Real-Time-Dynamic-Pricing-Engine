import glob

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import os

st.set_page_config(
    page_title="AI Dynamic Pricing",
    layout="wide"
)

# Refresh every 5 seconds
st_autorefresh(
    interval=5000,
    key="refresh"
)

st.title("AI Driven Dynamic Pricing Engine")

# --------------------------
# Load Data
# --------------------------

# optimize_price_spark.py writes this as a FOLDER of part-*.csv files, not
# a single "optimized_prices.csv" — so it's read as a glob of part files
# instead of one fixed filename.
PRICING_DIR = "data/final/optimized_prices"
LIVE_PATH = "data/live_data.csv"

pricing_part_files = glob.glob(
    os.path.join(PRICING_DIR, "part-*.csv")
)

if pricing_part_files:
    pricing_df = pd.concat(
        (pd.read_csv(f) for f in pricing_part_files),
        ignore_index=True
    )
else:
    pricing_df = pd.DataFrame()
    st.error(
        f"Could not find any part-*.csv files in `{PRICING_DIR}` (looked "
        f"in `{os.path.abspath(PRICING_DIR)}`).\n\n"
        "Run these first, from THIS SAME folder, before launching Streamlit:\n"
        "1. `spark-submit preprocess.py`\n"
        "2. `spark-submit create_feature.py`\n"
        "3. `python3 train_xgboost.py`\n"
        "4. `spark-submit optimize_price.py`\n\n"
        "Then confirm with: `ls -la data/final/optimized_prices/`"
    )

if os.path.exists(LIVE_PATH):
    live_df = pd.read_csv(
        LIVE_PATH
    )
else:
    live_df = pricing_df.copy()
    st.info(
        f"`{LIVE_PATH}` not found yet — showing static pricing data instead. "
        "This file only appears once Kafka is running and consumer.py has "
        "received at least one message (`docker compose up`)."
    )


# --------------------------
# KPIs
# --------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Products",
    pricing_df["product_id"].nunique()
    if not pricing_df.empty else 0
)

col2.metric(
    "Messages",
    len(live_df)
)

col3.metric(
    "Avg Demand",
    int(live_df["predicted_demand"].mean())
    if "predicted_demand" in live_df.columns
    else 0
)

col4.metric(
    "Avg Price",
    round(
        live_df["recommended_price"].mean(),
        2
    )
    if "recommended_price" in live_df.columns
    else 0
)

# --------------------------
# Live Feed
# --------------------------

st.subheader("Live Product Feed")

st.dataframe(
    live_df.tail(20)
)

# --------------------------
# Demand Trend
# --------------------------

if "predicted_demand" in live_df.columns:

    fig1 = px.line(
        live_df,
        y="predicted_demand",
        title="Demand Trend"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

# --------------------------
# Price Trend
# --------------------------

if "recommended_price" in live_df.columns:

    fig2 = px.line(
        live_df,
        y="recommended_price",
        title="Price Trend"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# --------------------------
# Elasticity Distribution
# --------------------------

if "avg_price_elasticity" in pricing_df.columns:

    fig3 = px.histogram(
        pricing_df,
        x="avg_price_elasticity",
        title="Elasticity Distribution"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# --------------------------
# Inventory Risk
# --------------------------

if "inventory_risk" in pricing_df.columns:

    risk = pricing_df[
        "inventory_risk"
    ].value_counts()

    fig4 = px.pie(
        values=risk.values,
        names=risk.index,
        title="Inventory Risk"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

# --------------------------
# Last Updated
# --------------------------

st.write(
    "Last Updated:",
    datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )
)