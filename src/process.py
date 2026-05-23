import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

df = pd.read_csv("data/cleaned_retail_data.csv")

# -------------------------------- EDA --------------------------------
print(df.head())
print(df.info())
print(df.nunique())
# -------------------------------- Data Preprocessing --------------------------------

# Transforming datatypes
df["invoicedate"] = pd.to_datetime(df["invoicedate"])
df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64") # hatalı customer_id'leri NaN yapar
print(df.info())
# -------------------------------- RFM Analysis --------------------------------

# K-Means algoritmasına çok fazla sütun (özellik) verirsen, algoritmanın kafası karışır (buna Curse of Dimensionality denir). 
# RFM, bir müşterinin değerini %80 oranında özetleyen en güçlü 3 sütundur.

# Analyze date for recency calculation
today = df["invoicedate"].max() + pd.Timedelta(days=1)

# RFM table creation
rfm = df.groupby("customer_id").agg({
    "invoicedate": lambda x: (today - x.max()).days, # recency
    "invoice": "nunique",  # frequency
    "total_revenue": "sum" # monetary
}).reset_index()

rfm.columns = ["customer_id", "recency", "frequency", "monetary"]
print(rfm.head())

# Outlier detection and handling
print("Outlier detection for RFM values:")
print(rfm.describe([0.25, 0.5, 0.75, 0.97]).T)

"""
Outlier detection for RFM values:
              count          mean           std      min       25%      50%       75%         97%        max
customer_id  5878.0  15315.313542   1715.572666  12346.0  13833.25  15314.5  16797.75    18110.69    18287.0
recency      5878.0    201.436883    209.454032      1.0      26.0     96.0     380.0       667.0      739.0
frequency    5878.0      6.279347     12.979594      1.0       1.0      3.0       7.0        28.0      398.0
monetary     5878.0   2869.961142  14002.181211     2.95  330.6825   835.97  2180.865  12880.4929  567769.68

"""
# -------------------------------- Outlier Handling --------------------------------
# IQR yöntemi ile outlierleri tespit edip onlara threshold uygulayarak outlier'ları sınırlandıracağım.

# Winsorization / Cap: Outlier'ları belirli bir eşik değere (threshold) göre sınırlandırma tekniği. 
# Trimming: Outlier'ları tamamen veri setinden çıkarma tekniği. 

def outlier_thresholds (data_frame,column, q1=0.10, q3=0.90):
    quartile1 = data_frame[column].quantile(q1)
    quartile3 = data_frame[column].quantile(q3)
    IQR = quartile3 - quartile1
    min_threshold = quartile1 - 1.5 * IQR
    max_threshold = quartile3 + 1.5 * IQR
    return min_threshold, max_threshold

def replace_with_thresholds(data_frame, column):
    min_threshold, max_threshold = outlier_thresholds (data_frame, column)
    data_frame.loc[(data_frame[column] < min_threshold), column] = min_threshold
    data_frame.loc[(data_frame[column] > max_threshold), column] = max_threshold
    
def check_outlier(data_frame, column):
    min_threshold, max_threshold = outlier_thresholds (data_frame, column)
    if data_frame[(data_frame[column] < min_threshold) | (data_frame[column] > max_threshold)].any(axis=None):
        return True
    else:
        return False
    
for col in ["recency", "frequency", "monetary"]:
    if check_outlier(rfm, col):
        print(f"{col} için outlier var. Threshold'lar uygulanıyor...")
        # outlier olanları bulma:
        min_threshold, max_threshold = outlier_thresholds (rfm, col)
        outlier_indices = rfm[(rfm[col] < min_threshold) | (rfm[col] > max_threshold)].index
        outlier_values = rfm.loc[outlier_indices, col]
        print(f"Outlier değerler:\n{outlier_values}\n")
        # outlier'ları threshold'lara göre sınırlandırma:
        replace_with_thresholds(rfm, col)

print("Outlier'lar sınırlandırıldıktan sonra RFM değerlerinin istatistikleri:")
print(rfm.describe([0.25, 0.5, 0.75, 0.97]).T)

# -------------------------------- Logarithmic Transformation --------------------------------
# RFM değerlerinden F ve M sağa çarpık dağılım gösteryor. Logaritmik dönüşüm, bu tür dağılımları normalize etmek için kullanıldı.
rfm["frequency"] = np.log1p(rfm["frequency"]) # log(0) hatasından kurtulmak için log yerine log1p kullandım (log(1+x))
rfm["monetary"] = np.log1p(rfm["monetary"]) # log(0) hatasından kurtulmak için log yerine log1p kullandım (log(1+x))

print("Logaritmik dönüşüm sonrası RFM değerlerinin istatistikleri:")
print(rfm.describe([0.25, 0.5, 0.75, 0.97]).T)

# -------------------------------- Scaling --------------------------------
# K-Meansx algoritması öklid uzaklık temelli bir algoritma olduğu için, farklı ölçeklerdeki özellikler algoritmanın performansını olumsuz etkileyebilir.
# Bu nedenle, RFM değerlerini aynı ölçeğe getirmek için Standard Scaling uygulayacağım.

rfm_features = rfm[["recency", "frequency", "monetary"]]
standardScaler = StandardScaler()
rfm_scaled = standardScaler.fit_transform(rfm_features)

rfm_scaled_df = pd.DataFrame(rfm_scaled, columns = ["recency", "frequency", "monetary"])

print("Standartlaştırılmış Veri (İlk 5 Satır):")
print(rfm_scaled_df.head())
print("\nİstatistikler (Ortalama 0, Std 1 olmalı):")
print(rfm_scaled_df.describe().round(2))

# -------------------------------- Optimal K Değerinin Belirlenmesi (Elbow Method) --------------------------------

model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(2,10))

visualizer.fit(rfm_scaled) # Senin hazırladığın o scaled veri

# Docker in görselleri gösterebileceği nbir ekranı olmadığı için görseli kaydettim:
visualizer.show(outpath="img/elbow_method.png") 

# Görseli incelediğimde Elbow yöntemiyle optimum K sayısının 5 olduğunu gördüm.

# -------------------------------- Base RFM K-Means  --------------------------------

kmeans = KMeans(n_clusters=5, 
                init='k-means++',      # başlangıç parametresi varsayılan olarak bu aslında, ama ben yine de belirttim
                n_init=10,             # 10 farklı rastgele başlangıç noktasıyla 10 ayrı deneme yapar, en iyisini seçer.
                max_iter=300,          # Her bir denemede merkezleri kaydırma işlemini en fazla 300 adım boyunca sürdürür.
                tol=0.0001,            # Merkezlerin yer değiştirmeyi ne zaman bırakacağını belirleyen durma eşiğidir. max_iter sınırına ulaşılmasa bile, merkezler bu değerden (0.0001) daha az hareket ediyorsa algoritmayı erken durdurarak zaman kazandırır.
                random_state=42)

rfm["cluster"] = kmeans.fit_predict(rfm_scaled)  # Scaled veriyi kullandık ama etiketi orijinal rfm tablosuna ekledik

# rfm_scaled: Adaletli mesafe hesabı için sayıları eşitlediğimiz tablo (Eğitim burada yapılır)
# rfm: Gerçek TL ve gün değerlerinin olduğu, insanların okuyabildiği orijinal tablo (Analiz burada yapılır)

# Her bir kümenin (segmentin) karakterini anlamak için ortalamalarına bakalım
segment_analysis = rfm.groupby('cluster').agg({
    'recency': ['mean', 'median'],
    'frequency': ['mean', 'median'],
    'monetary': ['mean', 'median', 'count']
}).round(1)

print(segment_analysis)
"""
        recency        frequency        monetary             
           mean median      mean median     mean median count
cluster                                                      
0         103.3   72.0       0.9    0.7      5.7    5.8  1263
1          43.5   20.0       2.8    2.7      8.6    8.6  1075
2         402.0  395.0       1.4    1.4      6.9    6.8   852
3         522.0  515.0       0.8    0.7      5.2    5.3  1067
4          66.3   44.0       1.8    1.8      7.3    7.2  1621

Bu sonuçlara baktığımda verilerin medyan ve mean değerleri birbirine çok yakın. Bu da kümelemeyi bozacak outlier
değerlerin olmadığını ve scaling işlemlerimin başarılı çalıştığını gösteriyor. Ayrıca müşteri sayıları da kümelerde 
dengeli dağılmış ve yığılma olmamış.

"""

# docker compose exec analysis_app python src/process.py