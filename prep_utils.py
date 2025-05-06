# import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Optional: for EDA tab
try:
    import ydata_profiling
    from ydata_profiling import ProfileReport
except ImportError:
    ProfileReport = None


def load_raw_data(path="data/marketing_data.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def clean_and_engineer(df):
    df = df.copy()

    # Convert income
    df["Income"] = (
        df["Income"].replace("\$", "", regex=True).replace(",", "", regex=True).astype(float)
    )

    # Date features
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")
    df["CustomerTenure"] = (pd.to_datetime("today") - df["Dt_Customer"]).dt.days

    # Core engineered features
    df["Age"] = 2024 - df["Year_Birth"]
    df["TotalSpend"] = df[[
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]].sum(axis=1)

    df["AcceptedTotal"] = df[[
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
        "AcceptedCmp4", "AcceptedCmp5"
    ]].sum(axis=1)

    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    df["Income"] = imputer.fit_transform(df[["Income"]])

    return df


def select_features(df):
    features = [
        "Income", "Age", "Recency", "TotalSpend",
        "CustomerTenure", "AcceptedTotal"
    ]
    return df[features + ["Response"]].dropna()


def preprocess_all(path="data/marketing_data.csv"):
    raw = load_raw_data(path)
    cleaned = clean_and_engineer(raw)
    final = select_features(cleaned)
    return final


def generate_profile_report(df):
    if ProfileReport is not None:
        profile = ProfileReport(df, title="Marketing Data EDA", explorative=True)
        profile.to_file("data_profile.html")
        return "data_profile.html"
    else:
        raise ImportError("ydata-profiling is not installed. Run `pip install ydata-profiling`. ")


# Optional for detailed profiling
if __name__ == "__main__":
    df = preprocess_all()
    print(df.describe(include='all'))
    try:
        html_path = generate_profile_report(df)
        print(f"Profile report saved to: {html_path}")
    except ImportError as e:
        print(e)