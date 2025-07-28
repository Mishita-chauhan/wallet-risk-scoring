import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("output/risk_scores.csv")
features = pd.read_csv("data/processed_wallet_data.csv")


df = df.merge(features, on="wallet_id")


plt.figure(figsize=(8, 5))
sns.histplot(df["score"], bins=20, kde=True, color='orange')
plt.title("Distribution of Risk Scores")
plt.xlabel("Risk Score")
plt.ylabel("Wallet Count")
plt.tight_layout()
plt.show()


correlations = df.corr(numeric_only=True)["score"].sort_values(ascending=False)
print("\n Top Correlations with Risk Score:\n", correlations)


print("\n Top 5 Riskiest Wallets:")
print(df.sort_values("score", ascending=False)[["wallet_id", "score"]].head())

print("\n Top 5 Safest Wallets:")
print(df.sort_values("score", ascending=True)[["wallet_id", "score"]].head())
