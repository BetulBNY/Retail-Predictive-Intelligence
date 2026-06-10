import warnings
import pandas as pd


df = pd.read_csv("data/cleaned_retail_data.csv")

# -------------------------------- EDA --------------------------------
print("Veri setinin ilk 5 satırı:-----------------------")
print(df.head())
print("\nVeri setinin genel bilgisi:-----------------------")
print(df.info())
print("\nVeri setindeki benzersiz değer sayısı:-----------------------")
print(df.nunique())
print("Base df description:-----------------------")
print(df.describe([0.25, 0.5, 0.75, 0.97]).T)
