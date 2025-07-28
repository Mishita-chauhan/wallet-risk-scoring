import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

INPUT_CSV = "data/compound_wallet_features.csv"
OUTPUT_CSV = "output/risk_scores.csv"

def score_wallets(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    
    df.fillna(0, inplace=True)

    
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[["total_borrow", "borrow_balance", "total_supply", "supply_balance"]] = scaler.fit_transform(
        df[["total_borrow", "borrow_balance", "total_supply", "supply_balance"]]
    )

    
    df_scaled["score_raw"] = (
        (df_scaled["total_borrow"] * 0.4) +
        (df_scaled["borrow_balance"] * 0.3) -
        (df_scaled["total_supply"] * 0.2) -
        (df_scaled["supply_balance"] * 0.1)
    )

    # Normalize scores to 0-1000
    df_scaled["score_raw"] = df_scaled["score_raw"].clip(lower=0)  # prevent negatives
    final_scaler = MinMaxScaler(feature_range=(0, 1000))
    df_scaled["score"] = final_scaler.fit_transform(df_scaled[["score_raw"]])

    
    result = df[["wallet_id"]].copy()
    result["score"] = df_scaled["score"].round().astype(int)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Saved risk scores to {output_csv}")

if __name__ == "__main__":
    score_wallets(INPUT_CSV, OUTPUT_CSV)
