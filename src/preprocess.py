import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_wallet_data(input_csv: str, output_csv: str):
    
    df = pd.read_csv(input_csv)

    
    wallet_ids = df['wallet_id']
    features = df.drop(columns=['wallet_id'])

    # Apply Min-Max scaling
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

   
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_df.insert(0, 'wallet_id', wallet_ids)

   
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    scaled_df.to_csv(output_csv, index=False)
    print(f" Processed data saved to: {output_csv}")


if __name__ == "__main__":
    preprocess_wallet_data(
        input_csv="data/compound_wallet_features.csv",
        output_csv="data/processed_wallet_data.csv"
    )
