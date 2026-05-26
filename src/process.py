# -------------------------------- LIBRARY IMPORTS --------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
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
# today = df["invoicedate"].max() + pd.Timedelta(days=1)  # Veri güncellendiğinde (yeni satır eklendiğinde) tüm müşterilerin Recency değeri değişecek.
# Bu sepeble sabit değer vermeye karar verdim:
print("Last day:", df["invoicedate"].max()) # Last day: 2011-12-09 12:50:00
today = pd.Timestamp("2011-12-10")

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
    min_threshold = max(min_threshold, 0)
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

# Analiz yapmak için log almadığım ancak outlierleri baskılanmış halinin kopyasını alıyorum.
rfm_final_analysis = rfm_expanded.copy() 
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
visualizer.show(outpath="analyze_img/elbow_method.png") 

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
K=2 için Silhouette Skoru: 0.3350
K=3 için Silhouette Skoru: 0.2573
K=4 için Silhouette Skoru: 0.2455
K=5 için Silhouette Skoru: 0.2427
K=6 için Silhouette Skoru: 0.2264
K=7 için Silhouette Skoru: 0.2294

Matematiksel Olarak En İyisi K=2 değeri çıktı. Küme sayısı arttıkça skor düzenli olarak düşüyor. 
Matematiksel olarak veri setim en net iki büyük gruba (Örn: Aktifler ve Pasifler) ayrılıyor.
Ancak iş mantığı açısından K = 5 değerini seçmeyi tercih ediyorum. Çünkü K=5 olduğunda segmentlerin karakteristik
özellikleri birbirinden daha net ayrılıyor ve bu da pazarlama stratejileri oluştururken daha anlamlı segmentler oluşturmamı sağlıyor.

"""
# K=2'de hangi segmentler birleşiyor?
kmeans_2 = KMeans(n_clusters=2, random_state=42)
rfm_final_analysis_test= rfm_final_analysis.copy()
rfm_final_analysis_test['cluster_2'] = kmeans_2.fit_predict(rfm_scaled_df)
print(rfm_final_analysis_test.groupby('cluster_2')[['recency','monetary']].median())
"""
           recency  monetary
cluster_2                   
0             32.0   2247.54
1            332.0    347.61
# Bu şunu ifade ediyor: Veriyi en kaba haliyle ikiye böldüğümüzde 
Cluster 0: Yakın zamanda gelen ve çok harcayanlar
Cluster 1: Çok uzun zamandır gelmeyen ve az harcayanlar
"""

# -------------------------------- Base RFM K-Means  --------------------------------
kmeans = KMeans(n_clusters=5, 
                init='k-means++',      # başlangıç parametresi varsayılan olarak bu aslında, ama ben yine de belirttim
                n_init=10,             # 10 farklı rastgele başlangıç noktasıyla 10 ayrı deneme yapar, en iyisini seçer.
                max_iter=300,          # Her bir denemede merkezleri kaydırma işlemini en fazla 300 adım boyunca sürdürür.
                tol=0.0001,            # Merkezlerin yer değiştirmeyi ne zaman bırakacağını belirleyen durma eşiğidir. max_iter sınırına ulaşılmasa bile, merkezler bu değerden (0.0001) daha az hareket ediyorsa algoritmayı erken durdurarak zaman kazandırır.
                random_state=42)

rfm_final_analysis["cluster"] = kmeans.fit_predict(rfm_scaled_df)  # Scaled veriyi kullandık ama etiketi orijinal rfm tablosuna (rfm_final_analysis) ekledik

# rfm_scaled_df: Adaletli mesafe hesabı için sayıları eşitlediğimiz tablo (Eğitim burada yapılır)
# rfm_final_analysis: Gerçek TL ve gün değerlerinin olduğu, insanların okuyabildiği orijinal tablo (Analiz burada yapılır)

# Her bir kümenin (segmentin) karakterini anlamak için ortalamalarına bakalım
segment_analysis = rfm_final_analysis.groupby('cluster').agg({
    'recency': ['mean', 'median'],
    'frequency': ['mean', 'median'],
    'monetary': ['mean', 'median', 'count'],
    'unique_products': ['mean', 'median'],
    'avg_unit_price': ['mean', 'median'],
    'customer_id': ['count']
}).round(1)

print("Segmentlerin Karakteristiği (Baskılanmış Gerçek Değerler):")
print(segment_analysis)
"""
         recency        frequency        monetary               unique_products        avg_unit_price        customer_id
           mean median      mean median     mean  median count            mean median           mean median       count
cluster                                                                                                                
0         328.4  332.0       1.8    1.0    580.4   321.3   647            14.2   12.0            6.5    5.7         647
1         122.5   66.0       4.5    4.0   1440.1  1198.1  1901            75.8   63.0            3.3    3.1        1901
2         100.7   66.0       1.8    1.0    421.9   314.7   920            20.7   18.0            2.3    2.3         920
3          46.0   22.0      15.5   13.0   6046.1  4771.0  1251           195.2  169.0            3.2    2.9        1251
4         507.9  482.0       1.6    1.0    467.0   305.3  1159            26.8   21.0            2.7    2.7        1159

# Burada da yine orijinal veri üzerinden segmentlere ayırdım veriyi çünkü sonuçları buna göre yorumalamak gerkiyor. 
"""

# -------------------------------- SEGMENT NAME MAPPING --------------------------------

# Hangi rakamın hangi isme geleceğini 'segment_analysis' tablosundaki ortalamalara bakarak belirledim.
# 5 küme yapısına göre (R-F-M-UP-AP) en tutarlı eşleşme:

# 1. Cluster bazlı recency ortalamaları
res = rfm_final_analysis.groupby('cluster')['recency'].mean().sort_values()

# 2. En düşük recency olan cluster numarası:
champions_cluster = res.index[0]
# 3. En yüksek recency olan cluster numarası:
hibernating_cluster = res.index[-1]

print(f"Otomatik Tespit: Champions = Cluster {champions_cluster}, Hibernating = Cluster {hibernating_cluster}")

seg_map = {
    0: 'Champions',           # R=45.5 (En taze), F=15.5, M=6046 (Zirve grup)
    4: 'Loyal Customers',     # R=122.2, F=4.5, M=1439 (Düzenli kitle)
    1: 'About to Sleep',      # R=100.3, F=1.8, M=421 (Taze ama harcaması düşük)
    2: 'At Risk',             # R=326.5 (1 yıldır yok) ama Avg_Unit_Price=6.5 (Değerli ama riskli)
    3: 'Hibernating'          # R=507.5 (En eski), F=1.6, M=466 (Kayıp kitle)
}

rfm_final_analysis['segment'] = rfm_final_analysis['cluster'].map(seg_map)

# Mapping doğrulama: Her segmentin median recency'sine bak
print("Mapping Doğrulama:")
print(rfm_final_analysis.groupby('segment')['recency'].median().sort_values())
# Champions en düşük recency'e sahip olmalı

print("Segmentlere ayrıldıktan sonraki hali:\n")
print(rfm_final_analysis[['customer_id', 'segment', 'recency', 'frequency', 'monetary']].head(10))

# -------------------------------- DATA VISUALIZATION --------------------------------

# 1) SCATTER PLOT
# X ekseninde gerçek dünya recency değerleri, y ekseninde ise verinin çarpıklığını korumak için Log Transform edilmiş monetary değerleri kullanıldı.
# Standard Scaled hali değil Log Transform halini seçmemin sebebi ilki daha çok DS'cilerin kümelerin matematiksel olarak ne kadar iyi
# ayrıldığını kontrol etmek için. Log Transform ise görsel olarak daha anlamlı sayılar sunar. 

# 1. Çizim için özel bir tablo oluşturma
plot_df = rfm_final_analysis.copy()
# 2. X ekseni için gerçek günler (Zaten rfm_final_analysis'te var)
# 3. Y ekseni için Scaled değil, sadece LOG alınmış halini kullandım (Daha anlaşılır olur)
# (rfm_expanded içinde loglanmış halleri vardı.)
plot_df['monetary_log'] = np.log1p(rfm_final_analysis['monetary'])

plt.figure(figsize=(12, 8))

sns.scatterplot( # Scatter plot: X ekseni Recency, Y ekseni Monetary (Log-scale değerleri daha net ayrım sağlar)
    data=plot_df,      # Tek bir kaynak
    x='recency', 
    y='monetary_log',  # Log scale ama -3/+3 değil, 2-10 arası değerler
    hue='segment', 
    palette='bright', 
    s=60,             # Nokta büyüklüğü
    alpha=0.7,        # Saydamlık
    edgecolor='w'
)

plt.title('Müşteri Segmentasyonu (Recency vs Log-Monetary)', fontsize=15)
plt.xlabel('Recency (Gün Sayısı)', fontsize=12)
plt.ylabel('Monetary (Harcama - Logaritmik Ölçek)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Grafiği kaydet
plt.tight_layout()
plt.savefig('analyze_img/customer_segments_final.png')       
print("Scatter plot 'customer_segments_final.png' olarak kaydedildi!")

# 2) PCA CUSTOMER SEGMENTS
# Şu anki verim 5 boyutlu (Recency, Frequency, Monetary, Avg Unit Price, Unique Products).
# Biz insanlar 5 boyutu hayal edemeyiz ve çizemeyiz. PCA ise Bu 5 boyutu, aralarındaki ilişkiyi bozmadan en fazla 
# bilgiyi (varyansı) temsil eden 2 tane yapay eksene (PC1 ve PC2) indirger. 
# Burada ise sadece scaled veriyi kullandım çünkü PCA "Varyans" (Değişkenlik) maksimizasyonu üzerine çalışır.
# Eğer standartlaştırmazsam PCA, 10.000'lik devasa sayıların olduğu Monetary sütununu "en önemli bilgi kaynağı"
# sanar ve diğer tüm sütunları (Frequency, Recency vb.) yok sayar.
# Yani scale etmezsem PCA sadece harcanan parayı çizer, diğer 4 özelliği çöpe atar. Scale ettiğimde ise tüm
# özellikler -2 ile +2 arasına gelir ve PCA her özelliğe "eşit hak" tanıyarak hepsinden birer parça bilgi alır.

pca = PCA(n_components=2)
pca_components = pca.fit_transform(rfm_scaled_df)

pca_df = pd.DataFrame(
    pca_components,
    columns=["PC1", "PC2"]
)

pca_df["segment"] = rfm_final_analysis["segment"]

plt.figure(figsize=(10,8))
sns.scatterplot(
    data=pca_df,
    x="PC1",  # Genellikle verideki en büyük farkı oluşturan bileşendir. Benim verimde muhtmeleen "monetary + frequency"
    y="PC2",  # PC1'in açıklayamadığı ikinci en büyük farkı (belki de sadece Recency veya Avg Unit Price)
    hue="segment",
    palette="bright",
    alpha=0.7
)
plt.title("PCA Projection of Customer Segments")

# Grafiği kaydet
plt.tight_layout()
plt.savefig('analyze_img/pca_customer_segments.png')
print("Scatter plot 'pca_customer_segments.png' olarak kaydedildi!")

# PCA GRAFİĞİ VERİYİ NE KADAR DOĞRU TEMSİL EDİYOR KONTROLÜ:
print("Açıklanan Varyans Oranları:", pca.explained_variance_ratio_)
# OUTPUT : [0.57894881 0.20334714] 
# Yani benim çizdiğim bu 2 boyutlu grafik, orijinal 5 boyutlu verideki toplam bilginin %77'sini temsil ediyor.
# %23'luk bir bilgi kaybım var.

# PCA GRAFİĞİNİN LOADINGS TABLOSU:
# Loadings tablosu, her bileşenin (PC1 ve PC2) içinde hangi malzemeden (R, F, M, UP, AUP) ne kadar olduğunu gösterir.
loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PC1", "PC2"],
    index=rfm_features.columns
)
print("Loadings tablosu sonuçları:\n",loadings)
"""
Loadings tablosu sonuçları:
                       PC1       PC2
recency         -0.403007  0.119693
frequency        0.536497  0.068139
monetary         0.542618  0.105837
unique_products  0.504819 -0.047957
avg_unit_price  -0.021898  0.983631

# PC1 bileşeni; Eğer bir müşterinin PC1 skoru yüksekse; o müşteri çok para bırakmış (Monetary), çok sık gelmiş (Frequency), çok çeşit ürün almış (Unique_products) ve en son gelişi üzerinden çok az zaman geçmiş (Recency - negatif olduğu için ters orantılı).
# PC2 bileşeni; PC2 ise neredeyse tamamen (0.98 loading ile) 'Average Unit Price' üzerinden tanımlanıyor. Bu şunu gösteriyor: Bir müşterinin pahalı ürün tercih etmesi, onun alışveriş sıklığı veya toplam harcamasından bağımsız bir boyut. Bu yüzden modelim,
sadece RFM ile görülemeyen 'Premium' alıcıları bu dikey eksende ayrıştırabildi.
"""
# RFM sonrası verileri kaydetme
rfm_final_analysis.to_csv("data/rfm_with_clusters.csv", index=False)
print("Segmentasyon sonuçları 'data/rfm_with_clusters.csv' olarak kaydedildi.")
# docker compose exec analysis_app python src/process.py