import pandas as pd
import requests
from tqdm import tqdm

WALLETS_CSV = "data/wallets.csv"
OUTPUT_CSV = "data/compound_wallet_features.csv"
GRAPH_URL = "https://api.thegraph.com/subgraphs/name/graphprotocol/compound-v2"

# GraphQL query template
QUERY_TEMPLATE = """
{{
  account(id: "{wallet}") {{
    id
    tokens {{
      symbol
      lifetimeSupply
      lifetimeBorrow
      supplyBalanceUnderlying
      borrowBalanceUnderlying
    }}
  }}
}}
"""

def query_graph(wallet_address):
    query = QUERY_TEMPLATE.format(wallet=wallet_address.lower())
    response = requests.post(GRAPH_URL, json={'query': query})
    if response.status_code == 200:
        return response.json()
    else:
        return None

def fetch_wallet_features(wallets):
    data = []
    for wallet in tqdm(wallets):
        result = query_graph(wallet)
        if result and result.get("data", {}).get("account"):
            tokens = result["data"]["account"]["tokens"]
            total_supply = sum(float(t.get("lifetimeSupply", 0)) for t in tokens)
            total_borrow = sum(float(t.get("lifetimeBorrow", 0)) for t in tokens)
            supply_balance = sum(float(t.get("supplyBalanceUnderlying", 0)) for t in tokens)
            borrow_balance = sum(float(t.get("borrowBalanceUnderlying", 0)) for t in tokens)
        else:
            total_supply = total_borrow = supply_balance = borrow_balance = 0.0
        data.append({
            "wallet_id": wallet,
            "total_supply": total_supply,
            "total_borrow": total_borrow,
            "supply_balance": supply_balance,
            "borrow_balance": borrow_balance,
        })
    return pd.DataFrame(data)

def main():
    wallets_df = pd.read_csv(WALLETS_CSV)
    wallets = wallets_df["wallet_id"].tolist()
    features_df = fetch_wallet_features(wallets)
    features_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved wallet features to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
