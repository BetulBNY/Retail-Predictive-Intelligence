--------------------------------------------------------------------------------------------------
--------------------------------------- CLEANING DATA -------------------------------------------- 
--------------------------------------------------------------------------------------------------
-- DROP TABLE cleaned_retail_data
CREATE TABLE cleaned_retail_data AS
WITH de_duplicated AS (
    -- Cleaning Duplicated rows
    SELECT *,
           ROW_NUMBER() OVER(PARTITION BY "Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Customer ID", "Country" ORDER BY "InvoiceDate") as rn
    FROM raw_retail_data   
)
SELECT 
	"Invoice" AS invoice,
	"StockCode" AS stockCode,
	UPPER("Description") AS description, 
	"Quantity" AS quantity,
	"InvoiceDate" AS invoiceDate,
	"Price" AS price,
	"Customer ID" AS customer_id,
	"Country" AS country,
	ROUND(("Quantity" * "Price")::numeric,2) AS total_revenue   

FROM de_duplicated
WHERE rn = 1
	AND "Customer ID" IS NOT NULL
	AND "Price" > 0
	AND "Quantity" > 0
	AND "Invoice" NOT LIKE 'C%'
	AND "Description" ~ '^[A-Z]';

-- SELECT * FROM cleaned_retail_data;