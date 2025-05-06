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
from prep_utils import preprocess_all, generate_profile_report
from ab_testing import simulate_ab_test
from modeling import train_model


# Load and preprocess data
@st.cache_data
def load_data():
    return preprocess_all("marketing_data.csv")

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
tabs = st.tabs(["🔍 Data Overview", "Segmentation", "Campaign Performance", "A/B Testing", "Causal Inference","📈 Modeling"])

# --- Tab 0: Data Overview ---
with tabs[0]:
    st.subheader("Data Summary and Exploratory Analysis")
    st.dataframe(data.head())

    st.write("### Summary Statistics")
    st.dataframe(data.describe().T)

    st.write("### Distribution: Income")
    st.plotly_chart(px.histogram(data, x="Income", nbins=30), use_container_width=True)

    st.write("### Distribution: Total Spend")
    st.plotly_chart(px.box(data, y="TotalSpend", points="all"), use_container_width=True)

    st.write("### Correlation Heatmap")
    st.plotly_chart(px.imshow(data.corr(), text_auto=True), use_container_width=True)

    # Optional embedded profile report
    try:
        profile_path = generate_profile_report(data)
        st.markdown("### 🧾 Auto Profile Report")
        with open(profile_path, "r", encoding="utf-8") as f:
            html = f.read()
            st.components.v1.html(html, height=900, scrolling=True)
    except:
        st.info("Install `ydata-profiling` to enable full auto report.")

# --- Tab 1: Segmentation ---
with tabs[1]:
    st.subheader("Customer Segmentation Viewer")
    x_var = st.selectbox("Select X-axis", ["Income", "Age", "Recency", "TotalSpend"])
    y_var = st.selectbox("Select Y-axis", ["Age", "Income", "TotalSpend", "Recency"])

    fig = px.scatter(segmented_data, x=x_var, y=y_var, color=segmented_data["Segment"].astype(str),
                     hover_data=["Age", "Income", "TotalSpend"],
                     title=f"Segments by {x_var} and {y_var}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Overview")
    segment_stats = segmented_data.groupby("Segment")[["Income", "Age", "Recency", "TotalSpend"]].mean().round(2)
    st.dataframe(segment_stats.reset_index())

# --- Tab 2: Campaign Performance ---
with tabs[2]:
    st.subheader("Campaign Response Rates")
    campaign_cols = [col for col in data.columns if col.startswith("AcceptedCmp")]
    campaign_summary = data[campaign_cols + ["Response"]].mean().round(3) * 100
    st.bar_chart(campaign_summary)

    


# --- Tab 3: A/B Testing ---
with tabs[3]:
    st.subheader("A/B Test Simulator")
    sample_size = st.slider("Sample Size per Group", 100, 1000, 300)

    ab_result = simulate_ab_test(data, sample_size=sample_size)

    st.metric("Conversion A", f"{ab_result['conversion_A']:.2%}")
    st.metric("Conversion B", f"{ab_result['conversion_B']:.2%}")
    st.metric("Lift", f"{ab_result['lift']:.2f}%")
    st.metric("P-Value", f"{ab_result['p_value']:.4f}")

    st.write("""
    - A low p-value (< 0.05) suggests a statistically significant difference between A and B.
    - This is a simulation using the 'Response' field.
    """)


# --- Tab 4: Causal Inference ---
with tabs[4]:
    st.subheader("Causal Inference: Propensity Score Matching")
    df = data.copy()
    df = df.dropna(subset=["Income", "Age", "TotalSpend", "Response"])
    df["HighIncome"] = (df["Income"] > df["Income"].median()).astype(int)

    features = df[["Age", "TotalSpend"]]
    treatment = df["HighIncome"]

    X_train, X_test, y_train, y_test = train_test_split(features, treatment, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    df["Propensity"] = model.predict_proba(features)[:, 1]

    df_sorted = df.sort_values(by="Propensity")
    bins = np.linspace(0, 1, 11)
    df_sorted["PropensityBin"] = pd.cut(df_sorted["Propensity"], bins)

    att = df_sorted.groupby("PropensityBin").apply(
        lambda x: x[x["HighIncome"]==1]["Response"].mean() - x[x["HighIncome"]==0]["Response"].mean()
    ).dropna()

    att_df = att.reset_index().rename(columns={0: "ATT"})
    att_df["PropensityBin"] = att_df["PropensityBin"].astype(str)
    fig = px.line(att_df, x="PropensityBin", y="ATT", title="ATT by Propensity Score Bin")
    st.plotly_chart(fig, use_container_width=True)

    st.write("Average Treatment Effect on the Treated (ATT):", round(att.mean(), 4))

# --- Tab 5: Modeling ---
with tabs[5]:
    st.subheader("Predictive Modeling")

    st.markdown("Use classification models to predict **Response** from customer attributes.")

    available_features = ["Age", "Income", "Recency", "TotalSpend", "CustomerTenure", "AcceptedTotal"]
    selected_features = st.multiselect("Select Features", available_features, default=available_features[:4])

    model_type = st.selectbox("Choose Model", ["logistic", "random_forest"])

    if st.button("Train Model"):
        model, metrics = train_model(data, selected_features, "Response", model_type=model_type)

        st.success("✅ Model trained successfully!")

        st.subheader("Model Performance")
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        if metrics["roc_auc"] is not None:
            st.metric("ROC AUC", f"{metrics['roc_auc']:.2f}")

        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(metrics["report"]).T.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

        st.subheader("Confusion Matrix")
        st.write(metrics["confusion_matrix"])

