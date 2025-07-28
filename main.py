from src.preprocess import preprocess_wallet_data

input_csv = "data/wallets.csv"
output_csv = "data/compound_wallet_features.csv"

preprocess_wallet_data(input_csv, output_csv)
