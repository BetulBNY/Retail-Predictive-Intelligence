--------------------------------------------------------------------------------------------------
--------------------------------------- COHORT ANALYSIS ------------------------------------------
--------------------------------------------------------------------------------------------------
SELECT COUNT(*) FROM cleaned_retail_data;

-- 1) FINDING CUSTOMER CHURN RATE:
/*
	cleaned_retail_data tablosunu kullanarak müşterilerin tutundurma oranlarını (retention rate) ve bizi ne kadar 
	sürede terk ettiklerini (churn) analiz ettim. Bu süreçte her müşterinin ilk alışveriş tarihini 
	"doğum ayı" (cohort month) olarak belirledim ve sonraki aylardaki aktivite sürekliliklerini hesapladım.
*/

WITH cohort_birth AS (
SELECT 
	customer_id AS customer_id,
	DATE_TRUNC('month', MIN(invoicedate)) AS first_order_month -- Amacım hepsini "2010 OCak grubu" gibi ayırmak
FROM cleaned_retail_data
GROUP BY customer_id
),
purcashe_months AS (
SELECT DISTINCT
	cb.customer_id,
	first_order_month,
	(DATE_TRUNC('month',invoicedate)) AS purchase_month
FROM cohort_birth AS cb
JOIN cleaned_retail_data AS cr
ON cb.customer_id = cr.customer_id
-- WHERE cb.customer_id = 14513
),
--select * from purcashe_months
month_diffs AS (
SELECT
	*,
	(EXTRACT(YEAR FROM purchase_month)- EXTRACT(YEAR FROM first_order_month)) * 12 + 
	 EXTRACT(MONTH FROM (purchase_month)) - EXTRACT(MONTH FROM (first_order_month)) AS month_number
FROM purcashe_months),

cohort_counts AS (
SELECT 
	first_order_month,
	month_number,
	COUNT(DISTINCT customer_id) as numb_of_custo
FROM month_diffs
GROUP BY first_order_month, month_number)

SELECT 
    first_order_month,
    month_number,
    numb_of_custo,
    -- Bu fonksiyon her grubun ilk satırındaki müşteri sayısını alır
    FIRST_VALUE(numb_of_custo) OVER(PARTITION BY first_order_month ORDER BY month_number) AS base_customers_num,
	ROUND((numb_of_custo*1.0 / FIRST_VALUE(numb_of_custo) OVER(PARTITION BY first_order_month ORDER BY month_number))*100, 2) AS retention_rate
FROM cohort_counts;	

/*
  BULGULAR & NOTLAR:
  Genel Tutundurma ve Churn: Verileri incelediğimde, Aralık 2009 dışındaki tüm başlangıç aylarında (base months) 
  müşteri kaybının ikinci aydan itibaren %80 seviyelerine ulaştığını görüyorum. Bu durum, müşterilerin büyük bir 
  kısmının tek seferlik alışveriş yaptığını ve kalıcı sadakat oluşturmakta zorlandığımızı kanıtlıyor.
  
  Aralık 2009 Kohortunun Farkı: Aralık 2009 grubunda müşteri kaybı %65 seviyesinde kalarak diğer aylara göre daha 
  olumlu bir ayrışma sergiliyor. Bu kitle, tüm kohortlar arasında en yüksek "Retention Rate" değerine sahip. 
  Öyle ki, bu grubun 24. ayındaki tutunma oranı (%19.69), diğer birçok grubun henüz 2. ayında ulaştığı değerle 
  neredeyse aynı. Bu durum, Aralık 2009'daki kampanya veya ürün kalitesinin oldukça etkili olduğunu gösteriyor.

  Mevsimsellik (Seasonality): Kasım ve Aralık aylarında, hem yeni kayıt sayılarında hem de eski müşterilerin geri
  dönüş oranlarında artış gözlemleniyor. Özellikle Aralık 2009 kohortunun 11. ayında (yani bir sonraki yılın
  Aralık ayında) tutunma oranının %49.53'e fırlaması kullanıcıların sezonsal davranışına çok uygun.
  Verileri inceledğimde Aralık 2009 haricindeki base monthlara baktığımda müşterilerin %80'i ikinci aydan
  itibaren azalıyor.
  */ 
