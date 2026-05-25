# -------------------------------- LIBRARY IMPORTS --------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging

# Warnings ve logging ayarları
logging.getLogger('matplotlib').setLevel(logging.CRITICAL) # Matplotlib'in sadece kritik hataları basmasını, bilgi uyarısı vermemesini sağlar
warnings.filterwarnings("ignore", message=".*font.*")  

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
# -------------------------------- Data Preprocessing --------------------------------
# Transforming datatypes
df["invoicedate"] = pd.to_datetime(df["invoicedate"])
df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64") # hatalı customer_id'leri NaN yapar

# -------------------------------- RFM Analysis --------------------------------
# K-Means algoritmasına çok fazla feature verirsem, algoritmanın kafası karışır (buna Curse of Dimensionality denir). 
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
print("İlk 5 RFM satırı:-----------------------")
print(rfm.head())

# Outlier detection and handling
print("Outlier detection for RFM values:-----------------------")
print(rfm.describe([0.25, 0.5, 0.75, 0.97]).T)

"""
Outlier detection for RFM values:
              count          mean           std      min       25%      50%       75%         97%        max
customer_id  5878.0  15315.313542   1715.572666  12346.0  13833.25  15314.5  16797.75    18110.69    18287.0
recency      5878.0    201.436883    209.454032      1.0      26.0     96.0     380.0       667.0      739.0
frequency    5878.0      6.279347     12.979594      1.0       1.0      3.0       7.0        28.0      398.0
monetary     5878.0   2869.961142  14002.181211     2.95  330.6825   835.97  2180.865  12880.4929  567769.68

"""
# -------------------------------- Feature Engineering --------------------------------
# Customer segments scatter plot'u inceledikten sonra segmentlerin birbirinden ayrıldığını ancak daha net bir ayrım 
# için bazı ekstra featurelerin eklenmesinin faydalı olabileceğini düşündüm. Bu şekilde kümeleri birbirinden uzaklaştırmış olacağım.

# 1) Average Unit Price (AUP): Müşteri genelde ucuz ürünler mi alıyor yoksa lüks/pahalı ürünler mi? (Monetary'den farklı)
# 2) Product Diversity (Ürün Çeşitliliği): Toplam kaç farklı StockCode satın almış? (Niche bir alıcı mı yoksa her şeyi alan bir genel alıcı mı?).

# Yeni özellikleri ana tablodan (df) çekip RFM tablosuna ekleme:
new_features = df.groupby("customer_id").agg(
    avg_unit_price = ("price", "mean"),
    unique_products = ("stockcode", "nunique")
)

# RFM tablosuyla birleştir
rfm_expanded = rfm.merge(new_features, on="customer_id")

# (Yeni eklediğim özelliklere de Outlier handling, Log ve Scaling yapacağım)

print("RFM tablosuna eklenen yeni özellikler sonrası istatistikler:-----------------------")
print(rfm_expanded.describe([0.25, 0.5, 0.75, 0.97]).T)

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
    
for col in ["recency", "frequency", "monetary", "unique_products", "avg_unit_price"]: 
    if check_outlier(rfm_expanded, col):
        print(f"{col} için outlier var. Threshold'lar uygulanıyor...")
        # outlier olanları bulma:
        min_threshold, max_threshold = outlier_thresholds (rfm_expanded, col)
        outlier_indices = rfm_expanded[(rfm_expanded[col] < min_threshold) | (rfm_expanded[col] > max_threshold)].index
        outlier_values = rfm_expanded.loc[outlier_indices, col]
        print(f"Outlier değerler:\n{outlier_values}\n")
        # outlier'ları threshold'lara göre sınırlandırma:
        replace_with_thresholds(rfm_expanded, col)

print("Outlier'lar sınırlandırıldıktan sonra RFM değerlerinin istatistikleri:-----------------------")
print(rfm_expanded.describe([0.25, 0.5, 0.75, 0.97]).T)

# -------------------------------- Logarithmic Transformation --------------------------------
# RFM değerlerinden F ve M sağa çarpık dağılım gösteryor. Logaritmik dönüşüm, bu tür dağılımları normalize etmek için kullanıldı.
rfm_expanded["frequency"] = np.log1p(rfm_expanded["frequency"]) # log(0) hatasından kurtulmak için log yerine log1p kullandım (log(1+x))
rfm_expanded["monetary"] = np.log1p(rfm_expanded["monetary"]) 
rfm_expanded["unique_products"] = np.log1p(rfm_expanded["unique_products"]) 
rfm_expanded["avg_unit_price"] = np.log1p(rfm_expanded["avg_unit_price"])

print("Logaritmik dönüşüm sonrası RFM değerlerinin ve yeni featurelerin istatistikleri:-----------------------")
print(rfm_expanded.describe([0.25, 0.5, 0.75, 0.97]).T)

# -------------------------------- Scaling --------------------------------
# K-Meansx algoritması öklid uzaklık temelli bir algoritma olduğu için, farklı ölçeklerdeki özellikler algoritmanın performansını olumsuz etkileyebilir.
# Bu nedenle, RFM değerlerini aynı ölçeğe getirmek için Standard Scaling uygulayacağım.

rfm_features = rfm_expanded[["recency", "frequency", "monetary", "unique_products", "avg_unit_price"]]
standardScaler = StandardScaler()
rfm_scaled = standardScaler.fit_transform(rfm_features)

rfm_scaled_df = pd.DataFrame(rfm_scaled, columns = ["recency", "frequency", "monetary", "unique_products", "avg_unit_price"])

print("Standartlaştırılmış Veri (İlk 5 Satır):-----------------------")
print(rfm_scaled_df.head())
print("\nİstatistikler (Ortalama 0, Std 1 olmalı):-----------------------")
print(rfm_scaled_df.describe().round(2))

# -------------------------------- Optimal K Değerinin Belirlenmesi (Elbow Method) --------------------------------
# 1) ELOBOW METHOD: 
model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(2,10))

visualizer.fit(rfm_scaled_df) # Hazırladığım scaled veri

# Docker in görselleri gösterebileceği bir ekranı olmadığı için görseli kaydettim:
visualizer.show(outpath="img/elbow_method.png") 

# Görseli incelediğimde Elbow yöntemiyle optimum K sayısının 5 olduğunu gördüm.

# 2) SILHOUETTE SCORE:
# Sadece Elbow yöntemiyle optimal K değerini belirlemek yeterli olmayabilir, bu yüzden farklı K değerleri için Silhouette skorlarını da hesapladım.
# Silhouette skoru ise "Noktalar kendi kümesine ne kadar yakın, komşu kümeye ne kadar uzak?" sorusuna cevap verir.
# Skor Aralığı: -1 ile +1 arasındadır. +1'e yakın: Kümeleme mükemmel, noktalar birbirinden çok ayrı. 0'a yakın: Noktalar kümelerin sınırında, iç içe geçme çok fazla. -1'e yakın: Kümeleme hatalı, noktalar yanlış kümelere atanmış.

for k in range(2, 8): # Farklı K değerleri için Silhouette skorlarını hesaplayalım
    kms = KMeans(n_clusters=k, random_state=42)
    labels = kms.fit_predict(rfm_scaled_df)
    score = silhouette_score(rfm_scaled_df, labels)
    print(f"K={k} için Silhouette Skoru: {score:.4f}")

"""
K=2 için Silhouette Skoru: 0.4252
K=3 için Silhouette Skoru: 0.4046
K=4 için Silhouette Skoru: 0.3940
K=5 için Silhouette Skoru: 0.3735
K=6 için Silhouette Skoru: 0.3551
K=7 için Silhouette Skoru: 0.3418

Matematiksel Olarak En İyisi K=2 değeri çıktı. Küme sayısı arttıkça skor düzenli olarak düşüyor. 
Matematiksel olarak veri setim en net iki büyük gruba (Örn: Aktifler ve Pasifler) ayrılıyor.
Ancak iş mantığı açısından K = 5 değerini seçmeyi tercih ediyorum. Çünkü K=5 olduğunda segmentlerin karakteristik
özellikleri birbirinden daha net ayrılıyor ve bu da pazarlama stratejileri oluştururken daha anlamlı segmentler oluşturmamı sağlıyor.

"""

# -------------------------------- Base RFM K-Means  --------------------------------
kmeans = KMeans(n_clusters=5, 
                init='k-means++',      # başlangıç parametresi varsayılan olarak bu aslında, ama ben yine de belirttim
                n_init=10,             # 10 farklı rastgele başlangıç noktasıyla 10 ayrı deneme yapar, en iyisini seçer.
                max_iter=300,          # Her bir denemede merkezleri kaydırma işlemini en fazla 300 adım boyunca sürdürür.
                tol=0.0001,            # Merkezlerin yer değiştirmeyi ne zaman bırakacağını belirleyen durma eşiğidir. max_iter sınırına ulaşılmasa bile, merkezler bu değerden (0.0001) daha az hareket ediyorsa algoritmayı erken durdurarak zaman kazandırır.
                random_state=42)

rfm_expanded["cluster"] = kmeans.fit_predict(rfm_scaled_df)  # Scaled veriyi kullandık ama etiketi orijinal rfm tablosuna ekledik




# rfm_scaled: Adaletli mesafe hesabı için sayıları eşitlediğimiz tablo (Eğitim burada yapılır)
# rfm: Gerçek TL ve gün değerlerinin olduğu, insanların okuyabildiği orijinal tablo (Analiz burada yapılır)

# Her bir kümenin (segmentin) karakterini anlamak için ortalamalarına bakalım
segment_analysis = rfm_expanded.groupby('cluster').agg({
    'recency': ['mean', 'median'],
    'frequency': ['mean', 'median'],
    'monetary': ['mean', 'median', 'count'],
    'unique_products': ['mean', 'median'],
    'avg_unit_price': ['mean', 'median']
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

# -------------------------------- SEGMENT NAME MAPPING --------------------------------

# Hangi rakamın hangi isme geleceğini 'segment_analysis' tablosundaki ortalamalara bakarak belirledim.
# Yeni çok boyutlu küme yapısına göre optimize edilmiş segment haritası
# 5 küme yapısına göre optimize edilmiş en doğru isimlendirme
# 5 küme yapısına göre (R-F-M-UP-AP) en tutarlı eşleşme:
seg_map = {
    3: 'Champions',           # R=46 (En taze), F=2.7, M=8.5 (Zirve)
    1: 'Loyal Customers',     # R=122, F=1.6, M=7.1 (Sadık ve düzenli)
    2: 'About to Sleep',      # R=100 (Taze) ama F=1.0, M=5.7 (Harcaması çok düşük)
    0: 'At Risk',             # R=328 (1 yıldır yok) ama Avg_Unit_Price=2.0 (Pahalı ürün almış!)
    4: 'Hibernating'          # R=507 (Kayıp), Tüm metrikler en düşük
}

rfm_expanded['segment'] = rfm_expanded['cluster'].map(seg_map)

# Doğrulamak için örnekleri gör
print(rfm_expanded[['customer_id', 'segment', 'recency', 'monetary']].head())




# Scatter plot
# Grafik oluşturma
plt.figure(figsize=(12, 8))

# Scatter plot: X ekseni Recency, Y ekseni Monetary (Log-scale değerleri daha net ayrım sağlar)
# Eğer rfm_log kullanıyorsan oradan çizmek kümeleri daha yuvarlak ve ayrık gösterir
sns.scatterplot(
    x=rfm_expanded['recency'], 
    y=rfm_expanded['monetary'], 
    hue=rfm_expanded['segment'], 
    palette='bright', 
    s=60,      # Nokta büyüklüğü
    alpha=0.7, # Saydamlık
    edgecolor='w'
)

plt.title('Müşteri Segmentasyonu (K-Means Clustering)', fontsize=15)
plt.xlabel('Recency (Gün)', fontsize=12)
plt.ylabel('Monetary (Toplam Harcama - Log Scale)', fontsize=12)
plt.legend(title='Segmentler', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

# Grafiği kaydet
plt.tight_layout()
plt.savefig('customer_segments_scatter.png')
print("Scatter plot 'customer_segments_scatter.png' olarak kaydedildi!")




from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca_components = pca.fit_transform(rfm_scaled)

pca_df = pd.DataFrame(
    pca_components,
    columns=["PC1", "PC2"]
)

pca_df["segment"] = rfm_expanded["segment"]

plt.figure(figsize=(10,8))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="segment",
    palette="bright",
    alpha=0.7
)

plt.title("PCA Projection of Customer Segments")
plt.show()

# Grafiği kaydet
plt.tight_layout()
plt.savefig('pca_customer_segments2.png')
print("Scatter plot 'pca_customer_segments.png' olarak kaydedildi!")





print(pca.explained_variance_ratio_)


loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PC1", "PC2"],
    index=rfm_features.columns
)

print(loadings)



# docker compose exec analysis_app python src/process.py