

--------------------------------------------------------------------------------------------------
--------------------------------------- DATA QUALITY CHECK --------------------------------------- 
--------------------------------------------------------------------------------------------------
SELECT COUNT(*) FROM raw_retail_data;
SELECT * FROM raw_retail_data LIMIT 100;

---------------------------------------------
-- 1) NULL VALUE ANALYSIS
---------------------------------------------

SELECT
	SUM(CASE WHEN "Invoice" IS NULL THEN 1 ELSE 0 END) AS Invoice_null_count,
	SUM(CASE WHEN "StockCode" IS NULL THEN 1 ELSE 0 END) AS StockCode_null_count,
    SUM(CASE WHEN "Description" IS NULL THEN 1 ELSE 0 END) AS Description_null_count,
    SUM(CASE WHEN "Quantity" IS NULL THEN 1 ELSE 0 END) AS Quantity_null_count,
    SUM(CASE WHEN "InvoiceDate" IS NULL THEN 1 ELSE 0 END) AS InvoiceDate_null_count,
    SUM(CASE WHEN "Price" IS NULL THEN 1 ELSE 0 END) AS Price_null_count,
    SUM(CASE WHEN "Customer ID" IS NULL THEN 1 ELSE 0 END) AS CustomerID_null_count,
    SUM(CASE WHEN "Country" IS NULL THEN 1 ELSE 0 END) AS Country_null_count
FROM raw_retail_data;

-- Description_null_count 4382
-- CustomerID_null_count 243007 
/* - Segmentasyon (RFM) ve kohort analizlerinde bu satırlar "Misafir Kullanıcı (Guest User)" olarak gruplanabilir veya analiz dışı bırakılabilir.
   - Ciro hesaplamalarında ise sisteme toplam para girdiği için bu satırlar korunacak.*/

---------------------------------------------
-- 2) CARDINALITY OF COLUMNS
---------------------------------------------
SELECT 
	COUNT(DISTINCT "StockCode") AS unique_StockCode,   	  -- 5305
	COUNT(DISTINCT "Description") AS unique_Description,  -- 5698
	COUNT(DISTINCT "Quantity") AS unique_Quantity,        -- 1057
	COUNT(DISTINCT "InvoiceDate") AS unique_InvoiceDate,  -- 47635
	COUNT(DISTINCT "Price") AS unique_Price,              -- 2807
	COUNT(DISTINCT "Customer ID") AS unique_Customer_ID,  -- 5942
	COUNT(DISTINCT "Country") AS unique_Country           -- 43
FROM raw_retail_data;
/*
  BULGULAR & NOTLAR:

* StockCode & Description Uyuşmazlığı: Unique ürün kodu (5305) ile ürün açıklaması (5698) sayıları eşit değil. 
  - Bu durum, aynı stok koduna farklı zamanlarda farklı açıklamalar girildiğini (veri kirliliği) gösterir.

* Description (Metin Temizliği İhtiyacı): 
  - "wrongly coded-23343", "check?", "wrongly marked", "?????" gibi manuel girilmiş sistem hata mesajları tespit edilmiştir.
  - Önemli Çıkarım: Gerçek ve geçerli ürün isimleri BÜYÜK HARFLE yazılmışken, sistem hataları/notları genellikle küçük harfle yazılmıştır. 
  - Aksiyon: Model aşamasından önce regex veya metin filtreleme ile bu kalıplar temizlenmelidir.

* Country (Ülke Verisi Anomalileri):
  - "EIRE" (İrlanda), "European Community", "RSA" (Güney Afrika) gibi standart dışı veya birleşik ülke tanımları var.
  - "Unspecified" (756 adet) tanımlanmamış lokasyon mevcut. Coğrafi analizlerde dikkat edilmeli.

* InvoiceDate (Zaman Serisi & Sezonluk Eğilim):
  - Alışveriş yoğunluğu yıl sonlarında (Kasım-Aralık / Black Friday, Yılbaşı etkisi) tepe noktasına ulaşıyor.
  - Ek olarak 2011-06-29 tarihinde ve Temmuz ayında da veri setinde anomalik bir sipariş yoğunluğu göze çarpıyor.
*/

SELECT 
	"Country",
	 COUNT(*)
FROM raw_retail_data
GROUP BY "Country"
ORDER BY COUNT(*);
---------------------------------------------
-- 3) CANCELLED ORDERS
---------------------------------------------
SELECT * 
FROM raw_retail_data 
WHERE "Invoice" LIKE 'C%' 
ORDER BY "Quantity" ASC;  

/* BULGULAR:
* Toplam 19,494 satır iptal/iade faturasıdır (Fatura numarası 'C' ile başlayanlar).
* Veri setindeki negatif "Quantity" (Miktar) değerlerinin ana kaynağı bu iade faturaları olabilir.
* Aksiyon: Net satış cirosu ve saf müşteri davranışını ölçmek için bu 'C'li satırlar ve bunlarla eşleşen orijinal satışlar tespit edilip analiz edilmelidir.
*/

---------------------------------------------
-- 4) FINDING OUTLIERS AND SYSTEM ERRORS
---------------------------------------------

-- A) For Price

SELECT *
FROM raw_retail_data
WHERE "Price" <= 0.1;
-- 6722 value

/*
 BULGULAR (Price):
* Negatif Fiyatlar: Veride 6207 adet negatif fiyat faturası var (Sistemsel düzeltme veya borçlandırma faturası olabilir).
* Sıfır ve Çok Düşük Fiyatlar 6722 satır 0 ile 0.1 arasında (Promosyon, hediye ürün veya test verisi olabilir).
* Uç Değerler (Max Outliers): En yüksek fiyatlar sırasıyla 38970, 25111 ve 18910 olarak gidiyor. 
  - Bu değerler genel ortalamayı (Mean) ciddi şekilde yukarı çekeceğinden, makine öğrenmesi modellerinde dönüşüm yapılmalı veya bu satırlar kırpılmalıdır.
*/

-- B) For Quantity:
-- For Quantity: Burada Iade edilenler zaten negatif. Benim amacım hem iade edilmeyecek hem de negatifleri bulmak:
SELECT * 
FROM raw_retail_data 
WHERE "Invoice" NOT LIKE 'C%' AND "Quantity" <= 0;

/*
 BULGULAR (Quantity):
* Tabloda fatura kodu 'C' ile başlamadığı halde miktar değeri sıfır veya negatif olan 3,457 satır tespit edilmiştir.
* Bu durum sistemsel bir entegrasyon hatasını veya stok düzeltme kayıtlarını gösterir. 
* Aksiyon: Bu 3,457 satır gerçek bir satışı temsil etmediği için veri setinden doğrudan silinmelidir (Drop).
*/


---------------------------------------------
-- 5) COUNTRY DISTRIBUTION
---------------------------------------------

SELECT 
	"Country", 
	 COUNT(*) as order_count 
FROM raw_retail_data 
GROUP BY 1 
ORDER BY 2 DESC 
LIMIT 5;
/*
 BULGULAR:
* Veri setinin ezici bir çoğunluğu (981,330 satır ile) United Kingdom (İngiltere) merkezlidir.
* En az sipariş alan ülke ise sadece 10 kayıt ile Suudi Arabistan'dır.
* Stratejik Çıkarım: Yapılacak müşteri segmentasyonu veya talep tahminleme modelleri "UK" ve "Non-UK (Yurt Dışı)" olarak iki ana kola ayrılırsa model başarısı artacaktır.
*/

