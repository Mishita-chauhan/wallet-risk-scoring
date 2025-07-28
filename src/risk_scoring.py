import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

INPUT_CSV = "data/processed_wallet_data.csv"
OUTPUT_CSV = "output/risk_scores.csv"

def calculate_risk_score(df):
    features = ["total_supply", "total_borrow", "supply_balance", "borrow_balance"]
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

    df["raw_risk"] = (
        df_scaled["total_borrow"] + df_scaled["borrow_balance"]
        - df_scaled["total_supply"] - df_scaled["supply_balance"]
    )

    min_risk = df["raw_risk"].min()
    max_risk = df["raw_risk"].max()
    df["score"] = ((df["raw_risk"] - min_risk) / (max_risk - min_risk + 1e-6)) * 1000
    df["score"] = df["score"].round(0).astype(int)

    return df[["wallet_id", "score"]]

def main():
    df = pd.read_csv(INPUT_CSV)
    result_df = calculate_risk_score(df)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved risk scores to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
