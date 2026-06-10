# E-Commerce Customer Intelligence: End-to-End Segmentation & Churn Prediction

This project delivers a production-ready machine learning pipeline designed to analyze customer behavior, identify strategic segments, and predict future churn with a focus on **business impact** and **technical integrity**.

---

## Project Overview

In the competitive e-commerce landscape, understanding *who* your customers are and *when* they might leave is critical. This project implements a dual-stage approach:

1. **Unsupervised Learning:** Multi-dimensional clustering (segmentation) to group customers based on value and behavior.
2. **Supervised Learning:** A leakage-free XGBoost model to predict churn using a temporal cut-off window approach.

### What is the "Cut-off Window" Approach?
To predict customer churn without data leakage, features are built exclusively from historical data up to a defined cut-off date, while the churn label (target variable) is defined in a *separate, future* time window. This ensures the model genuinely learns to predict the future rather than simply describing the past.

---

## Pipeline Steps

### `ingest_data.py` — Data Ingestion
Raw data residing on the local machine is loaded into a PostgreSQL database running inside a Docker container for the first time.

![pgadmin image](img/pgadmin.jpg)

---

### `fetch_clean_data.py` — Data Fetching & Connection Architecture
The `analysis_app` container connects to the `postgres` container over Docker's internal network using `postgres` as the hostname and port `5432`. This is because when two services share the same Docker network, the service name acts as the hostname.

At the same time, pgAdmin running on the host machine connects to the same database via `localhost:5433`, which is mapped to the container's port `5432`. This gives us two separate entry points into the same database.

```
        (HOST MACHINE)
      pgAdmin
          |
          |  localhost:5433
          v
 -------------------------
 |   Docker Engine       |
 |                       |
 |  analysis_app         |
 |        |              |
 |        | postgres:5432 |
 |        v              |
 |     postgres DB       |
 -------------------------
```

---

### `data_quality_check.sql` — Data Quality Assessment

Initial row count: **1,067,371 rows**.

**1) Null Values**
- `Description` null count: **4,382**
- `CustomerID` null count: **243,007**

> **Note:** Rows with null `CustomerID` can either be excluded from segmentation/cohort analyses or grouped under a "Guest User" label. However, since these transactions still represent real revenue flowing into the system, they are retained for financial calculations.

**2) Cardinality Check**

- **StockCode vs. Description Mismatch:** The number of unique product codes (5,305) does not match the number of unique descriptions (5,698). This indicates that the same stock code has been entered with different descriptions at different times — a data quality issue.

- **Description — Text Cleaning Required:** Manual system error messages were detected in the Description field, such as `"wrongly coded-23343"`, `"check?"`, `"wrongly marked"`, and `"?????"`. A key observation: legitimate product names are written in **UPPERCASE**, while system errors/notes are typically written in **lowercase**. These patterns should be filtered out using regex or text-matching logic prior to modeling.

- **Country — Anomalies:** Non-standard or aggregated country entries exist, including `"EIRE"` (Ireland), `"European Community"`, and `"RSA"` (South Africa). There are also **756 rows** labeled `"Unspecified"`, which should be handled carefully in geographic analyses.

- **InvoiceDate — Seasonality:** Order volumes peak toward the end of the year (November–December), consistent with Black Friday and New Year's shopping patterns. An anomalous spike in order density is also visible around **June 29, 2011**, and throughout July 2011.

**3) Cancelled Orders**
- **19,494 rows** are cancellation/return invoices (invoice numbers starting with `'C'`).
- Negative `Quantity` values in the dataset are primarily sourced from these return invoices.
- To accurately measure net sales revenue and true customer behavior, these `'C'`-prefixed rows and their corresponding original transactions should be identified and analyzed separately.

**4) Outlier Detection**

*Price outliers:*
- **Negative Prices:** 6,207 invoices have negative prices. Of these, **3,457** are not cancellation invoices (i.e., they don't start with `'C'`), suggesting a system integration error or stock adjustment record. These 3,457 rows do not represent actual sales and should be dropped.
- **Zero / Very Low Prices:** 6,722 rows have prices between 0 and 0.1 — likely promotional items, gifts, or test records.
- **Maximum Outliers:** The highest prices in the dataset are 38,970, 25,111, and 18,910. These extreme values will significantly inflate the mean and must be capped using defined thresholds before machine learning modeling.

**5) Country Distribution**
- The vast majority of the dataset (**981,330 rows**) is from the **United Kingdom**.
- The least represented country is **Saudi Arabia** with only 10 records.
- **Strategic Insight:** Splitting the dataset into **"UK"** and **"Non-UK (International)"** subsets before modeling can improve both segmentation quality and model performance.

**6) Duplicate Rows**
- **34,335 duplicate rows** were detected and flagged for removal.

---

### `cleaning_data.sql` — Data Cleaning

The following cleaning steps were applied to the raw dataset:

- Removed **34,335 duplicate rows**
- Removed rows with **null `CustomerID`**
- Removed invoices starting with `'C'` (cancellations/returns are deferred for separate analysis)
- Removed rows where **`Price` ≤ 0** or **`Quantity` ≤ 0**

---

### `cohort_analysis.sql` — Cohort Analysis

**What is Cohort Analysis?**
Cohort analysis groups customers with similar characteristics — most commonly, customers who first made a purchase in the same calendar month — and tracks their behavior over subsequent months. By examining retention rates over time, we can gain a high-level understanding of the business's health and customer loyalty trends.

**How it was implemented in SQL:**

1. A CTE is created to find each customer's first purchase date using `DATE_TRUNC` and `MIN` — this becomes the **cohort month**.
2. A second CTE groups all subsequent transactions by customer and converts each purchase date to month format using `DATE_TRUNC`.
3. A third CTE calculates the month difference between each purchase date and the cohort month for each customer.
4. Finally, rows are grouped by cohort month and purchase month offset, and `COUNT(DISTINCT CustomerID)` is used to measure active customers at each step.

**Findings:**
- **High Early Churn:** Except for customers who joined in December 2009, churn across all cohorts reaches approximately **80%** by the second month. This indicates that the majority of customers make a single purchase and do not return, pointing to a significant challenge in building sustained loyalty.
- **Seasonality:** New customer registrations and returning customer activity both increase noticeably in **November and December**, consistent with end-of-year shopping events.

![time based cohort image](img/cohort.jpg)

---

### `fetch_clean_data.py` — Export to DataFrame

The cleaned dataset is retrieved from PostgreSQL into a Pandas DataFrame and saved locally to the `data/` directory for downstream processing.

---

### `segmentation_process.py` — Customer Segmentation

**Feature Engineering**

In addition to standard RFM (Recency, Frequency, Monetary) features, the following behavioral metrics were engineered:

- **Average Unit Price (AUP):** Does this customer typically buy cheap products or premium/luxury items? (Distinct from total Monetary spend)
- **Product Diversity (Unique Products):** How many distinct `StockCode` values has the customer purchased? (Niche buyer vs. broad generalist)
- **`is_UK`:** Boolean flag indicating whether the customer is from the UK
- **`avg_order_value`:** Average value per order
- **`active_lifespan`:** The duration (in days) between a customer's first and last purchase (derived from `max_date - min_date`)

**Outlier Handling**

IQR-based outlier detection was applied to cap extreme values. Before scaling, a copy of the unscaled dataframe was retained — the original values are used for human-readable analysis, while the scaled version is used for model training.

For `Monetary`, the distribution was heavily right-skewed, with a max of 567,769 but a 97th percentile of only 12,880. The treatment decision was context-driven: since the goal was to analyze general customer behavior rather than identify VIP segments, the extreme values were treated as noise. First, IQR-based capping was applied, then log transformation was used to normalize the distribution. The same logic was applied to `Avg Unit Price`.

> **Note:** In a different business context (e.g., fraud detection or VIP tier identification), these outliers could be treated as high-value signals rather than noise. The correct approach is always context-dependent.

**Log Transformation**

Log transformation was applied to features that are positive, heavy-tailed, and ratio-scale — in other words, features that "behave like money" and have a long right tail. Categorical and date-based features were excluded.

**Scaling**

Since K-Means relies on Euclidean distance, features at different scales can distort the clustering. Standard Scaling was applied to bring all features (RFM and engineered metrics) onto the same scale.

**Choosing the Optimal K**

Two methods were used to determine the optimal number of clusters:

*Elbow Method:* Measures inertia (within-cluster sum of squared distances) as K increases. The "elbow" — the point where adding more clusters yields diminishing returns — suggests the optimal K.

![elbow method image](img/embow_m.png)

*Silhouette Score:* Measures how similar each point is to its own cluster relative to neighboring clusters. Scores range from -1 to +1, where values close to +1 indicate well-separated clusters, values near 0 indicate overlapping boundaries, and negative values indicate potential misassignment.

| K | Silhouette Score |
|---|-----------------|
| 2 | 0.3727 |
| 3 | 0.2800 |
| 4 | 0.2601 |
| 5 | 0.2434 |
| 6 | 0.2470 |
| 7 | 0.2574 |

Mathematically, **K=2** is the optimal value — the data most cleanly splits into two broad groups (e.g., Active vs. Inactive customers). However, **K=5 was chosen for business reasons**: with five clusters, the segment profiles become meaningfully distinct from one another, providing more actionable groups for marketing and retention strategy.

**Modeling & Analysis Tables**

K-Means was trained on the scaled feature set, and the resulting cluster labels were appended to the unscaled dataframe as a new `cluster` column.

Two analytical tables were maintained throughout:
- **`rfm_scaled_df`:** Used for model training — all features normalized to the same scale for fair distance calculations.
- **`rfm_final_analysis`:** Used for human interpretation — original monetary values and day counts, readable by stakeholders.

Cluster means were inspected to profile each segment's characteristics, and descriptive business labels were assigned based on these averages.

**Visualizations**

*Customer Segments — Recency vs. Monetary Scatter Plot:*
The X-axis uses original Recency values; the Y-axis uses log-transformed Monetary values. Log scale was chosen over raw values for the Y-axis because it provides clearer visual separation between segments and is more interpretable given the skewed distribution.

![customer segments image](analyze_img/customer_segments_final2.png)

*Customer Segmentation — PCA 2D Projection:*
K-Means was trained on 6 features (Recency, Frequency, Monetary, Avg Unit Price, Unique Products, Active Lifespan). Since humans cannot visualize 6-dimensional space, PCA (Principal Component Analysis) was used to project this data into 2 dimensions while preserving as much variance as possible.

**Why scaled data for PCA?** PCA maximizes variance. Without scaling, the `Monetary` column — with values in the tens of thousands — would completely dominate the variance, causing PCA to ignore all other features. After scaling (all features between -2 and +2), PCA treats each feature equally and extracts information from all of them.

![PCA customer segments image](analyze_img/pca_customer_segments2.png)

*Variance Explained:* `[0.5965, 0.1697]` — the 2D projection captures approximately **77%** of the total information in the original 6-dimensional space, meaning ~23% of information is lost in the projection.

*PCA Loadings Table:*
The loadings table shows how much each original feature contributes to each principal component.

![loadings table image](img/loadings_table.png)

- **PC1** is a composite of overall customer value: a high PC1 score indicates a customer who has spent a lot (Monetary), visited frequently (Frequency), bought a wide variety of products (Unique Products), and purchased recently (Recency — negatively loaded, meaning lower recency values contribute positively).
- **PC2** is almost entirely defined by Average Unit Price (loading: 0.98). This reveals that a customer's preference for expensive products is a dimension entirely independent from their shopping frequency or total spend. This component allowed the model to isolate "Premium" buyers who would not be distinguishable by RFM alone.

---

### `predict_churn.py` — Churn Prediction

**Correction of a Design Mistake**

An important lesson learned: the segmentation (K-Means) step should have been trained only on data *before* the cut-off date (i.e., excluding the final 3 months) from the very beginning. Training K-Means on the full dataset and then using the cluster labels as a feature in the churn model would constitute temporal data leakage — the cluster assignments for the observation window would implicitly encode information from the future prediction window.

To correct this, the segmentation was re-run using only data prior to the cut-off date, producing valid cluster labels that could then be legitimately used as a feature in the churn prediction model.

**Cut-off Window Design**

To prevent **Temporal Data Leakage**, a Fixed Window Approach was implemented:

- **Features (X):** Calculated using data strictly before **September 1, 2011**.
- **Target (y):** Defined by whether a customer made at least one purchase in the 3-month window following the cut-off (Sept 1 – Nov 30, 2011).

This design ensures the model only uses information that would have been available at prediction time, producing a genuine forward-looking churn predictor.

**Model Performance**

- **ROC-AUC: 0.82** — Strong ability to separate churners from active customers.
- **Recall-Focused Threshold Tuning:** The classification threshold was optimized to achieve **91% Recall** for the Churn class, ensuring the business captures almost all at-risk customers (at the cost of some precision — i.e., some false alarms are acceptable to avoid missing true churners).
- **Segment-Level Error Analysis:** Model performance was validated across K-Means clusters. For example, the model performs strongly on the "Hibernating" segment but shows weaker precision on the "Loyal" segment, likely due to fewer distinguishing behavioral signals in that group.

---

## Key Technical Highlights

### 1. Advanced Feature Engineering
Beyond standard RFM, strategic metrics were engineered to capture deeper behavioral patterns:
- **Purchase Momentum:** Ratio of recent (last 90 days) activity to lifetime activity. *(2nd most important feature in the churn model)*
- **Active Lifespan:** Duration between first and last purchase.
- **Avg Unit Price:** Distinguishes "Budget" shoppers from "Premium/Luxury" buyers.

### 2. Leakage-Free Churn Prediction (Cut-off Method)
To prevent **Temporal Data Leakage**, a static snapshot approach was avoided in favor of a **Fixed Window Approach**:
- **Features (X):** Calculated using data strictly before **Sept 1, 2011**.
- **Target (y):** Defined by purchase activity in the following 3 months.
- *Result:* A model that truly predicts the future rather than just describing the past.

### 3. Model Interpretation & Error Analysis
- **ROC-AUC: 0.82** — Strong separability between churners and active users.
- **Recall-Focused Tuning:** Threshold optimized to achieve **91% Recall** for the Churn class, ensuring the business captures almost all at-risk customers.
- **Segment-Level Validation:** Rigorous error analysis across clusters (e.g., the model over-performs on "Hibernating" customers but requires more event-level data for the "Loyal" segment).

---

## Visualizations & Insights

### Customer Segmentation (PCA Projection)
*(Add `pca_customer_segments.png` here)*
> Using PCA, 5D behavioral data was projected into 2D space, demonstrating that the identified segments are geometrically distinct and logically coherent.

### Feature Importance
*(Add `feature_importance_churn.png` here)*
> **Key Insight:** Recency at the cut-off date and Purchase Momentum are the strongest predictors of churn, outweighing total monetary spend.

### Business Dashboard (Looker Studio)
*(Add Looker Studio screenshot here)*
> A live dashboard designed for the marketing team to prioritize high-risk, high-value customers for targeted retention campaigns.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Infrastructure | Docker, Docker Compose (Environment Isolation) |
| Data Engineering | PostgreSQL (Large-scale event processing & Cohort Analysis) |
| Machine Learning | Python — Pandas, Scikit-Learn, XGBoost, PCA |
| Reporting | Looker Studio (Stakeholder Dashboards) |

---

## Project Structure

```bash
├── data/               # Raw and processed datasets
├── src/
│   ├── ingest_data.py       # Data pipeline to PostgreSQL
│   ├── fetch_clean_data.py  # Connection layer & cleaned data export
│   ├── segmentation_process.py  # RFM feature engineering & K-Means clustering
│   └── predict_churn.py     # XGBoost churn modeling & evaluation
├── sql/
│   ├── data_quality_check.sql  # Exploratory data quality queries
│   ├── cleaning_data.sql       # Data cleaning & deduplication
│   └── cohort_analysis.sql     # Time-based cohort retention analysis
├── analyze_img/        # Generated plots and charts
├── img/                # Architecture and reference images
├── docker-compose.yml  # Database & workspace setup
└── README.md
```

---

## How to Run

1. Clone the repository.
2. Ensure Docker is running.
3. Start the containers:
   ```bash
   docker-compose up -d
   ```
4. Ingest the raw data into PostgreSQL:
   ```bash
   docker compose exec analysis_app python src/ingest_data.py
   ```
5. Run the full pipeline (cleaning → segmentation → churn prediction):
   ```bash
   docker compose exec analysis_app python src/segmentation_process.py
   docker compose exec analysis_app python src/predict_churn.py
   ```