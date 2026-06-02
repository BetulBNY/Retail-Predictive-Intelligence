import pandas as pd
from sqlalchemy import create_engine
import os

# 1. Bağlantı Bilgilerim (Docker compose'daki bilgilerle aynı)
user = os.getenv('POSTGRES_USER')
password = os.getenv('POSTGRES_PASSWORD')
db_name = os.getenv('POSTGRES_DB')

host = 'postgres'
port = "5432"

# 2. Connection String ve Engine Oluştur
conn_string = f'postgresql://{user}:{password}@{host}:{port}/{db_name}'
engine = create_engine(conn_string)

# 3. SQL sorgusunu yazarak temilenmiş veriyi pandas DataFrame'ine çekelim.
print("Veriler PostgreSQL'den çekiliyor...")
query = "SELECT * FROM cleaned_retail_data"
df = pd.read_sql(query, engine)

df['invoicedate'] = pd.to_datetime(df['invoicedate']) # PostgreSQL'de normalde çevirmiştim ancak Pandas çekince tekrar object olarak gelmişti.
df['customer_id'] = df['customer_id'].astype('Int64')

print(f"Başarılı! {df.shape[0]} satır ve {df.shape[1]} sütun veri yüklendi.")

# 4. İlk Kontrol
print(df.head())

# 5. Data Types Kontrolü
print(df.dtypes)

# 6. data klasörünün içine temizlenmiş veriyi kaydetme
# df.to_csv("data/cleaned_retail_data.csv", index=False)
df.to_parquet("data/cleaned_retail_data.parquet")
# docker compose exec analysis_app python src/fetch_clean_data.py