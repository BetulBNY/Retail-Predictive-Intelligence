import pandas as pd

df_retail = pd.read_csv("data/cleaned_retail_data.csv")
df_rfm = pd.read_csv("data/rfm_with_clusters.csv")

print(df_retail.head())
print(df_retail.columns)
print(df_rfm.head())
# docker compose exec analysis_app python src/testc.py