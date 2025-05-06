import pandas as pd
import numpy as np
from scipy import stats

def simulate_ab_test(df, response_col="Response", sample_size=300, seed=42):
    """
    Simulate an A/B test by splitting the data into two groups and comparing means.
    """
    df = df.copy()
    np.random.seed(seed)
    test_data = df.sample(n=sample_size * 2, random_state=seed).copy()
    test_data["Group"] = ["A"] * sample_size + ["B"] * sample_size

    group_A = test_data[test_data["Group"] == "A"][response_col]
    group_B = test_data[test_data["Group"] == "B"][response_col]

    conversion_A = group_A.mean()
    conversion_B = group_B.mean()
    lift = (conversion_B - conversion_A) / conversion_A * 100

    z_stat, p_val = stats.ttest_ind(group_A, group_B)

    return {
        "conversion_A": conversion_A,
        "conversion_B": conversion_B,
        "lift": lift,
        "p_value": p_val,
    }
