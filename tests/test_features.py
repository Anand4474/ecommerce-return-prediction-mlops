import pandas as pd
from src.features import build_features

def test_engineered_features_and_target_removal():
    df = pd.DataFrame([{
        "Order_ID": "X1", "Return_Status": "Returned",
        "Expected_Delivery_Days": 3, "Actual_Delivery_Days": 5,
        "Previous_Returns": 2, "Previous_Orders": 4,
        "Product_Category": "Fashion"
    }])
    result = build_features(df)
    assert "Return_Status" not in result.columns
    assert "Order_ID" not in result.columns
    assert result.loc[0, "Delivery_Delay_Days"] == 2
    assert result.loc[0, "Previous_Return_Frequency"] == 0.5
