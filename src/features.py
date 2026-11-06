import pandas as pd

TARGET = "Return_Status"
DROP_COLUMNS = ["Order_ID", TARGET]

def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create business features without using the target or the unique order ID."""
    X = frame.drop(columns=[c for c in DROP_COLUMNS if c in frame.columns]).copy()

    # Delay is positive only when delivery was later than expected.
    if {"Actual_Delivery_Days", "Expected_Delivery_Days"}.issubset(X.columns):
        X["Delivery_Delay_Days"] = (
            X["Actual_Delivery_Days"] - X["Expected_Delivery_Days"]
        ).clip(lower=0)

    # Safe denominator; customers with zero previous orders get frequency 0.
    if {"Previous_Returns", "Previous_Orders"}.issubset(X.columns):
        denom = X["Previous_Orders"].replace(0, 1)
        X["Previous_Return_Frequency"] = (
            X["Previous_Returns"] / denom
        ).clip(lower=0, upper=1)

    return X
