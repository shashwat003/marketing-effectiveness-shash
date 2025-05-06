import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("marketing_data.csv")
    df.columns = df.columns.str.strip()  # Remove whitespace from column names
    df["Income"] = df["Income"].replace("\$", "", regex=True).replace(",", "", regex=True).astype(float)
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%m/%d/%y")
    df["Age"] = 2024 - df["Year_Birth"]
    df["TotalSpend"] = df[[
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]].sum(axis=1)
    return df

# KMeans Clustering
def segment_customers(df, n_clusters=4):
    features = df[["Income", "Age", "Recency", "TotalSpend"]].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)
    features["Segment"] = labels
    return df.join(features["Segment"])

# Streamlit UI Setup
st.set_page_config(page_title="Marketing Effectiveness App", layout="wide")
st.title("📊 Marketing Effectiveness App")

# Sidebar Controls
st.sidebar.header("Segmentation Controls")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)

# Load and process data
data = load_data()
segmented_data = segment_customers(data.copy(), n_clusters=n_clusters)

# Tabs for app sections
tabs = st.tabs(["Segmentation", "Campaign Performance", "A/B Testing", "Causal Inference"])

# --- Tab 1: Segmentation ---
with tabs[0]:
    st.subheader("Customer Segmentation Viewer")
    x_var = st.selectbox("Select X-axis", ["Income", "Age", "Recency", "TotalSpend"])
    y_var = st.selectbox("Select Y-axis", ["Age", "Income", "TotalSpend", "Recency"])

    fig = px.scatter(segmented_data, x=x_var, y=y_var, color=segmented_data["Segment"].astype(str),
                     hover_data=["Marital_Status", "Education", "Country"],
                     title=f"Segments by {x_var} and {y_var}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Overview")
    segment_stats = segmented_data.groupby("Segment")[["Income", "Age", "Recency", "TotalSpend"]].mean().round(2)
    st.dataframe(segment_stats.reset_index())

# --- Tab 2: Campaign Performance ---
with tabs[1]:
    st.subheader("Campaign Response Rates")
    campaign_cols = [col for col in data.columns if col.startswith("AcceptedCmp")]
    campaign_summary = data[campaign_cols + ["Response"]].mean().round(3) * 100
    st.bar_chart(campaign_summary)

    st.subheader("Response by Country")
    response_by_country = data.groupby("Country")["Response"].mean().round(3) * 100
    st.dataframe(response_by_country.reset_index().rename(columns={"Response": "% Response"}))

# --- Tab 3: A/B Testing ---
with tabs[2]:
    st.subheader("A/B Test Simulator")
    size = st.slider("Sample Size per Group", 100, 1000, 300)
    test_data = data.sample(n=size*2, random_state=42).copy()
    test_data["Group"] = ["A"]*size + ["B"]*size

    response_A = test_data[test_data["Group"] == "A"]["Response"]
    response_B = test_data[test_data["Group"] == "B"]["Response"]

    conversion_A = response_A.mean()
    conversion_B = response_B.mean()
    lift = (conversion_B - conversion_A) / conversion_A * 100

    z_stat, p_val = stats.ttest_ind(response_A, response_B)

    st.metric("Conversion A", f"{conversion_A:.2%}")
    st.metric("Conversion B", f"{conversion_B:.2%}")
    st.metric("Lift", f"{lift:.2f}%")
    st.metric("P-Value", f"{p_val:.4f}")

    st.write("""
    - A low p-value (< 0.05) suggests a statistically significant difference between A and B.
    - This is a simulation using the 'Response' field.
    """)

# --- Tab 4: Causal Inference ---
with tabs[3]:
    st.subheader("Causal Inference: Propensity Score Matching")
    st.write("This module estimates the effect of being in a high-income group on campaign response.")

    df = data.copy()
    df = df.dropna(subset=["Income", "Age", "TotalSpend", "Response"])
    df["HighIncome"] = (df["Income"] > df["Income"].median()).astype(int)

    features = df[["Age", "TotalSpend"]]
    treatment = df["HighIncome"]
    outcome = df["Response"]

    X_train, X_test, y_train, y_test = train_test_split(features, treatment, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    propensity_scores = model.predict_proba(features)[:, 1]

    df["Propensity"] = propensity_scores
    df_sorted = df.sort_values(by="Propensity")

    bins = np.linspace(0, 1, 11)
    df_sorted["PropensityBin"] = pd.cut(df_sorted["Propensity"], bins)

    att = df_sorted.groupby("PropensityBin").apply(
        lambda x: x[x["HighIncome"]==1]["Response"].mean() - x[x["HighIncome"]==0]["Response"].mean()
    ).dropna()

    # ✅ FIXED: Flatten the Series to a DataFrame so Streamlit can chart it
    att_df = att.reset_index().rename(columns={0: "ATT"})
    st.line_chart(data=att_df, x="PropensityBin", y="ATT")

    st.write("Average Treatment Effect on the Treated (ATT):", round(att.mean(), 4))
