import pandas as pd
from sqlalchemy import create_engine
import time
import os

def ingest():
    file_path = "data/online_retail.xlsx" # Dosya yolu
    
    # 1. Veriyi Oku (Hangi sheetler varsa isimlerini kontrol et)
    print("Excel dosyası okunuyor, bu işlem biraz zaman alabilir...")
    # Not: Sayfa isimleri tam olarak 'Year 2009-2010' olmayabilir, kontrol et.
    df1 = pd.read_excel(file_path, sheet_name=0) 
    df2 = pd.read_excel(file_path, sheet_name=1)
    df = pd.concat([df1, df2], ignore_index=True)
    
    print(f"Toplam {len(df)} satır veri okundu.")

    # 2. DB Bağlantısı (Docker içi network adresini kullanıyoruz)
    
    user = os.getenv('POSTGRES_USER')
    password = os.getenv('POSTGRES_PASSWORD')
    db = os.getenv('POSTGRES_DB')

    engine = create_engine(f'postgresql://{user}:{password}@postgres:5432/{db}')
   
    # 3. DB'nin hazır olmasını bekle (Postgres'in ayağa kalkması zaman alabilir)
    retries = 5
    while retries > 0:
        try:
            conn = engine.connect()
            print("Veritabanına bağlanıldı!")
            break
        except Exception as e:
            print(f"Bağlantı bekleniyor... Kalan deneme: {retries}")
            time.sleep(5)
            retries -= 1

    # 4. Veriyi Yaz
    print("Veriler PostgreSQL'e aktarılıyor (raw_retail_data)...")
    df.to_sql('raw_retail_data', engine, if_exists='replace', index=False)
    print("Aktarım tamamlandı!")

if __name__ == "__main__":
    ingest()