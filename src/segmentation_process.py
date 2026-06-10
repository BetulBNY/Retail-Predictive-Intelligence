# -------------------------------- LIBRARY IMPORTS --------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import logging
# Warnings ve logging ayarları
logging.getLogger('matplotlib').setLevel(logging.CRITICAL) # Matplotlib'in sadece kritik hataları basmasını, bilgi uyarısı vermemesini sağlar
warnings.filterwarnings("ignore", message=".*font.*")  

# -------------------------------- DATA LOADING  --------------------------------
df = pd.read_csv("data/cleaned_retail_data.csv")
# Transforming datatypes
df["invoicedate"] = pd.to_datetime(df["invoicedate"])
df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64") # hatalı customer_id'leri NaN yapar

# -------------------------------- EDA --------------------------------
print("Veri setinin ilk 5 satırı:-----------------------")
print(df.head())
print("\nVeri setinin genel bilgisi:-----------------------")
print(df.info())
print("\nVeri setindeki benzersiz değer sayısı:-----------------------")
print(df.nunique())
print("Base df description:-----------------------")
print(df.describe([0.25, 0.5, 0.75, 0.97]).T)

# -------------------------------- CUT-OFF DATE --------------------------------
# Segmentasyonu da churn modeliyle aynı zaman penceresinde eğitiyorum.
# Böylece KMeans'in öğrendiği segmentler, churn modelinin gördüğü müşteri
# davranışlarıyla tutarlı hale gelir. (Leakage önlemi)
# Veri: 2009-12 → 2011-12. Cut-off: 2011-09-01.
# KMeans sadece df_past (Sep 2011 öncesi) verisiyle eğitiliyor.
cut_off_date = pd.Timestamp('2011-09-01')
df_past = df[df['invoicedate'] < cut_off_date].copy()

print(f"\nToplam kayıt: {len(df):,}")
print(f"df_past (cut-off öncesi) kayıt: {len(df_past):,}")
print(f"df_past benzersiz müşteri: {df_past['customer_id'].nunique():,}")

# -------------------------------- RFM Analysis --------------------------------
# K-Means algoritmasına çok fazla feature verirsem, algoritmanın kafası karışır (buna Curse of Dimensionality denir). 
# RFM, bir müşterinin değerini büyük oranda özetleyen en güçlü 3 sütundur.

# RFM table creation
rfm = df_past.groupby("customer_id").agg({
    "invoicedate": lambda x: (cut_off_date - x.max()).days, # recency
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
              count         mean           std      min      25%      50%      75%        97%        max
customer_id  5249.0  15326.75843   1710.402354  12346.0  13856.0  15317.0  16808.0   18111.56    18287.0
recency      5249.0   204.127262    171.487933      0.0     49.0    161.0    318.0      576.0      638.0
frequency    5249.0       5.6767     11.282189      1.0      1.0      3.0      6.0       25.0      303.0
monetary     5249.0  2538.698632  11676.992068      2.9   307.53   754.49  1994.35  11346.162  440248.41

"""
# -------------------------------- Feature Engineering --------------------------------
# 1) Average Unit Price (AUP): Müşteri genelde ucuz ürünler mi alıyor yoksa lüks/pahalı ürünler mi? (Monetary'den farklı)
# 2) Product Diversity (Ürün Çeşitliliği): Toplam kaç farklı StockCode satın almış? (Niche bir alıcı mı yoksa her şeyi alan bir genel alıcı mı?).

# Yeni özellikleri ana tablodan (df) çekip RFM tablosuna ekleme:
new_features = df_past.groupby("customer_id").agg(
    avg_unit_price = ("price", "mean"),
    unique_products = ("stockcode", "nunique"),
    first_purchase = ("invoicedate", "min"),
    last_purchase = ("invoicedate", "max"),
    is_UK = ("country", lambda x: 1 if x.mode()[0] == "United Kingdom" else 0)
)

# RFM tablosuyla birleştir
rfm_expanded = rfm.merge(new_features, on="customer_id")

# FEATURE 2) AVERAGE SPEND PER ORDER
rfm_expanded["avg_order_value"] =  rfm_expanded["monetary"] / rfm_expanded["frequency"] # it is different then avg_unit_price (we divided total price whole rows per person for avg_unit_price)

# FEATURE 3) ACTIVE LIFESPAN
rfm_expanded['active_lifespan'] = (rfm_expanded['last_purchase'] - rfm_expanded['first_purchase']).dt.days

# (Yeni eklediğim özelliklere de Outlier handling, Log ve Scaling yapacağım)

print("RFM tablosuna eklenen yeni özellikler sonrası istatistikler:-----------------------")
print(rfm_expanded.describe([0.25, 0.5, 0.75, 0.97]).T)

# -------------------------------- Outlier Handling (WINSORIZATION) --------------------------------
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
    data_frame[column] = data_frame[column].astype(float) # float cast → dtype incompatible FutureWarning'i önler (int64 kolonlarda)
    data_frame.loc[(data_frame[column] < min_threshold), column] = min_threshold
    data_frame.loc[(data_frame[column] > max_threshold), column] = max_threshold
    
def check_outlier(data_frame, column):
    min_threshold, max_threshold = outlier_thresholds (data_frame, column)
    if data_frame[(data_frame[column] < min_threshold) | (data_frame[column] > max_threshold)].any(axis=None):
        return True
    else:
        return False
    
OUTLIER_COLS = ["recency", "frequency", "monetary", "unique_products", "avg_unit_price", "active_lifespan"]   

for col in OUTLIER_COLS: 
    if check_outlier(rfm_expanded, col):
        print(f"{col} için outlier var. Threshold'lar uygulanıyor...")
        replace_with_thresholds(rfm_expanded, col)

print("Outlier'lar sınırlandırıldıktan sonra RFM değerlerinin istatistikleri:-----------------------")
print(rfm_expanded.describe([0.25, 0.5, 0.75, 0.97]).T)

# Analiz yapmak ce Churn modelinde kullanmak için log almadığım ancak outlierleri baskılanmış halinin kopyasını alıyorum.
rfm_final_analysis = rfm_expanded.copy() 

# -------------------------------- Logarithmic Transformation --------------------------------
# Sadece KMeans için gerekli; XGBoost bu dönüşümlere ihtiyaç duymaz.
# RFM değerlerinden F ve M sağa çarpık dağılım gösteryor. Logaritmik dönüşüm, bu tür dağılımları normalize etmek için kullanıldı.
rfm_expanded["frequency"] = np.log1p(rfm_expanded["frequency"]) # log(0) hatasından kurtulmak için log yerine log1p kullandım (log(1+x))
rfm_expanded["monetary"] = np.log1p(rfm_expanded["monetary"]) 
rfm_expanded["unique_products"] = np.log1p(rfm_expanded["unique_products"]) 
rfm_expanded["avg_unit_price"] = np.log1p(rfm_expanded["avg_unit_price"])
rfm_expanded["active_lifespan"] = np.log1p(rfm_expanded["active_lifespan"])

print("Logaritmik dönüşüm sonrası RFM değerlerinin ve yeni featurelerin istatistikleri:-----------------------")
print(rfm_expanded.describe([0.25, 0.5, 0.75, 0.97]).T)

# -------------------------------- Scaling --------------------------------
# K-Means algoritması öklid uzaklık temelli bir algoritma olduğu için, farklı ölçeklerdeki özellikler algoritmanın performansını olumsuz etkileyebilir.
# Bu nedenle, RFM değerlerini aynı ölçeğe getirmek için Standard Scaling uygulayacağım.

rfm_features = rfm_expanded[["recency", "frequency", "monetary", "unique_products", "avg_unit_price", "active_lifespan"]]
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)
rfm_scaled_df = pd.DataFrame(rfm_scaled, columns = ["recency", "frequency", "monetary", "unique_products", "avg_unit_price", "active_lifespan"])

print("Standartlaştırılmış Veri (İlk 5 Satır):-----------------------")
print(rfm_scaled_df.head())
print("\nİstatistikler (Ortalama 0, Std 1 olmalı):-----------------------")
print(rfm_scaled_df.describe().round(2))

# -------------------------------- OPTIMAL K: ELBOW + SILHOUETTE--------------------------------
# 1) ELOBOW METHOD: 
print("\n--- Elbow Method ---")
model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(2,10))
visualizer.fit(rfm_scaled_df) # Hazırladığım scaled veri
# Docker in görselleri gösterebileceği bir ekranı olmadığı için görseli kaydettim:
visualizer.show(outpath="analyze_img/elbow_method.png") 

# Görseli incelediğimde Elbow yöntemiyle optimum K sayısının 4 olduğunu gördüm.
# -----------------------------------------
# 2) SILHOUETTE SCORE:
# Sadece Elbow yöntemiyle optimal K değerini belirlemek yeterli olmayabilir, bu yüzden farklı K değerleri için Silhouette skorlarını da hesapladım.
# Silhouette skoru ise "Noktalar kendi kümesine ne kadar yakın, komşu kümeye ne kadar uzak?" sorusuna cevap verir.
# Skor Aralığı: -1 ile +1 arasındadır. +1'e yakın: Kümeleme mükemmel, noktalar birbirinden çok ayrı. 0'a yakın: Noktalar kümelerin sınırında, iç içe geçme çok fazla. -1'e yakın: Kümeleme hatalı, noktalar yanlış kümelere atanmış.
print("\n--- Silhouette Scores ---")
for k in range(2, 8): # Farklı K değerleri için Silhouette skorlarını hesaplayalım
    kms = KMeans(n_clusters=k, random_state=42)
    labels = kms.fit_predict(rfm_scaled_df)
    score = silhouette_score(rfm_scaled_df, labels)
    print(f"K={k} için Silhouette Skoru: {score:.4f}")

"""
--- Silhouette Scores ---
K=2 için Silhouette Skoru: 0.3659
K=3 için Silhouette Skoru: 0.2942
K=4 için Silhouette Skoru: 0.2650
K=5 için Silhouette Skoru: 0.2336
K=6 için Silhouette Skoru: 0.2341
K=7 için Silhouette Skoru: 0.2347

Matematiksel Olarak En İyisi K=2 değeri çıktı. Küme sayısı arttıkça skor düzenli olarak düşüyor. 
Matematiksel olarak veri setim en net iki büyük gruba (Örn: Aktifler ve Pasifler) ayrılıyor.
Ancak iş mantığı açısından K = 4 değerini seçmeyi tercih ediyorum. Çünkü K=4 olduğunda segmentlerin karakteristik
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
0             76.0   1713.89
1            311.0    291.09
# Bu şunu ifade ediyor: Veriyi en kaba haliyle ikiye böldüğümüzde 
Cluster 0: Yakın zamanda gelen ve çok harcayanlar
Cluster 1: Çok uzun zamandır gelmeyen ve az harcayanlar
"""
# -------------------------------- K-MEANS MODELING   --------------------------------
kmeans = KMeans(
                n_clusters=4, 
                init='k-means++',      # başlangıç merkezlerini rastgele seçmek yerine, verinin dağılımına göre daha akıllıca seçen bir yöntem. 
                n_init=10,             # Modeli 10 farklı rastgele başlangıç noktasıyla 10 ayrı şekilde çalıştırır, en iyisini seçer. 
                max_iter=300,          # Her bir denemede merkezleri kaydırma işlemini en fazla 300 adım boyunca sürdürür.
                tol=0.0001,            # Merkezlerin yer değiştirmeyi ne zaman bırakacağını belirleyen durma eşiğidir. max_iter sınırına ulaşılmasa bile, merkezler bu değerden (0.0001) daha az hareket ediyorsa algoritmayı erken durdurarak zaman kazandırır.
                random_state=42
                )

rfm_final_analysis["cluster"] = kmeans.fit_predict(rfm_scaled_df)  # Scaled veriyi kullandık ama etiketi orijinal rfm tablosuna (rfm_final_analysis) ekledik
# rfm_scaled_df: Adaletli mesafe hesabı için sayıları eşitlediğimiz tablo (Eğitim burada yapılır)
# rfm_final_analysis: Gerçek TL ve gün değerlerinin olduğu, insanların okuyabildiği orijinal tablo (Analiz burada yapılır)

# Scaler ve KMeans modelini kaydetme 
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/kmeans_scaler.pkl")
print("\nKMeans ve Scaler modelleri 'models/' klasörüne kaydedildi.")

# -------------------------------- SEGMENT ANALYSIS   --------------------------------
# Her bir kümenin (segmentin) karakterini anlamak için ortalamalarına bakalım
segment_analysis = rfm_final_analysis.groupby('cluster').agg({
    'recency': ['mean', 'median'],
    'frequency': ['mean', 'median'],
    'monetary': ['mean', 'median', 'count'],
    'unique_products': ['mean', 'median'],
    'avg_unit_price': ['mean', 'median'],
    'customer_id': ['count']
}).round(1)

print("Segmentlerin Karakteristiği (Winsorize edilmiş Gerçek Değerler):")
print(segment_analysis)
"""
        recency        frequency        monetary               unique_products        avg_unit_price        customer_id
           mean median      mean median     mean  median count            mean median           mean median       count
cluster                                                                                                                
0         309.3  301.5       1.3    1.0    469.7   250.8   622            13.4   10.5            6.2    5.4         622
1          63.6   37.0      12.7   10.0   4895.2  3685.5  1341           162.8  137.0            3.2    3.0        1341
2         184.6  160.0       3.5    3.0   1036.2   824.5  2026            51.7   43.0            3.3    3.1        2026
3         333.2  326.0       1.2    1.0    343.7   233.2  1260            23.1   18.0            2.4    2.5        1260

# Burada da yine orijinal veri üzerinden segmentlere ayırdım veriyi çünkü sonuçları buna göre yorumalamak gerkiyor. 
"""
# -------------------------------- SEGMENT NAME MAPPING --------------------------------
res = rfm_final_analysis.groupby('cluster')['recency'].mean().sort_values()
freq_by_cluster = rfm_final_analysis.groupby('cluster')['frequency'].mean()

by_recency = res.index.tolist()  # düşük → yüksek recency

# Champions → en düşük recency
champions_cluster = by_recency[0]

# Loyal Customers → ikinci en düşük recency
loyal_cluster = by_recency[1]

# Kalan 2 cluster: At Risk vs Lost
# At Risk → daha yüksek frequency (bir zamanlar aktifti)
# Lost    → daha düşük frequency (zaten hiç aktif olmadı)
remaining = by_recency[2:]

# YENİ — avg_unit_price'a göre ayır: pahalı ürün alıcısı → At Risk (değeri yüksek)
avg_price_by_cluster = rfm_final_analysis.groupby('cluster')['avg_unit_price'].mean()
at_risk_cluster = max(remaining, key=lambda c: avg_price_by_cluster[c])
lost_cluster    = min(remaining, key=lambda c: avg_price_by_cluster[c])

seg_map = {
    int(champions_cluster): 'Champions',
    int(loyal_cluster):     'Loyal Customers',
    int(at_risk_cluster):   'At Risk',
    int(lost_cluster):      'Lost',
}

print(f"\nOtomatik Segment Mapping:")
for cid, name in sorted(seg_map.items()):
    r = rfm_final_analysis[rfm_final_analysis['cluster']==cid]
    print(f"  Cluster {cid} → {name:<16} | recency={r['recency'].median():.0f}g  freq={r['frequency'].median():.1f}  monetary=£{r['monetary'].median():.0f}")

rfm_final_analysis['segment'] = rfm_final_analysis['cluster'].map(seg_map)

print("\nMapping Doğrulama:")
print(rfm_final_analysis.groupby('segment')['recency'].median().sort_values())
print(rfm_final_analysis[['customer_id','segment','recency','frequency','monetary']].head(10))
 
print("\nCluster bazlı detay (tüm metrikler):")
print(rfm_final_analysis.groupby('segment').agg(
    recency_min   = ('recency',        'min'),
    recency_max   = ('recency',        'max'),
    recency_med   = ('recency',        'median'),
    freq_med      = ('frequency',      'median'),
    monetary_med  = ('monetary',       'median'),
    avg_price_med = ('avg_unit_price', 'median'),
    count         = ('customer_id',    'count')
).round(1))
 
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
# Şu anki verim 6 boyutlu (Recency, Frequency, Monetary, Avg Unit Price, Unique Products, active_lifespan).
# Biz insanlar 6 boyutu hayal edemeyiz ve çizemeyiz. PCA ise Bu 6 boyutu, aralarındaki ilişkiyi bozmadan en fazla 
# bilgiyi (varyansı) temsil eden 2 tane yapay eksene (PC1 ve PC2) indirger. 
# Burada ise sadece scaled veriyi kullandım çünkü PCA "Varyans" (Değişkenlik) maksimizasyonu üzerine çalışır.
# Eğer standartlaştırmazsam PCA, 10.000'lik devasa sayıların olduğu Monetary sütununu "en önemli bilgi kaynağı"
# sanar ve diğer tüm sütunları (Frequency, Recency vb.) yok sayar.
# Yani scale etmezsem PCA sadece harcanan parayı çizer, diğer 4 özelliği çöpe atar. Scale ettiğimde ise tüm
# özellikler -2 ile +2 arasına gelir ve PCA her özelliğe "eşit hak" tanıyarak hepsinden birer parça bilgi alır.

pca = PCA(n_components=2)
pca_components = pca.fit_transform(rfm_scaled_df)
pca_df = pd.DataFrame(pca_components, columns=["PC1", "PC2"])
pca_df["segment"] = rfm_final_analysis["segment"].values

plt.figure(figsize=(10,8))
sns.scatterplot(
    data=pca_df,
    x="PC1",  # Genellikle verideki en büyük farkı oluşturan bileşendir. Benim verimde muhtmelen "monetary + frequency"
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
print(f"Toplam Açıklanan Varyans: {pca.explained_variance_ratio_.sum():.2%}")
# OUTPUT : [0.59646389 0.16966971]
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
recency         -0.362658  0.012260
frequency        0.487620  0.035729
monetary         0.480506  0.054105
unique_products  0.434712 -0.097467
avg_unit_price  -0.011929  0.992494
active_lifespan  0.459024  0.033193

# PC1 bileşeni; Eğer bir müşterinin PC1 skoru yüksekse; o müşteri çok para bırakmış (Monetary), çok sık gelmiş (Frequency), çok çeşit ürün almış (Unique_products) ve en son gelişi üzerinden çok az zaman geçmiş (Recency - negatif olduğu için ters orantılı).
# PC2 bileşeni; PC2 ise neredeyse tamamen (0.98 loading ile) 'Average Unit Price' üzerinden tanımlanıyor. Bu şunu gösteriyor: Bir müşterinin pahalı ürün tercih etmesi, onun alışveriş sıklığı veya toplam harcamasından bağımsız bir boyut. Bu yüzden modelim,
sadece RFM ile görülemeyen 'Premium' alıcıları bu dikey eksende ayrıştırabildi.
"""
# -------------------------------- SAVE OUTPUT --------------------------------
save_cols = [
    'customer_id', 'recency', 'frequency', 'monetary',
    'avg_unit_price', 'unique_products', 'is_UK',
    'avg_order_value', 'active_lifespan', 'cluster', 'segment'
]
rfm_final_analysis[save_cols].to_csv("data/rfm_with_clusters.csv", index=False)
print("\nSegmentasyon sonuçları kaydedildi: data/rfm_with_clusters.csv")

# docker compose exec analysis_app python src/segmentation_process.py