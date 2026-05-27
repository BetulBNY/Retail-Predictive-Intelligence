import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from config import CHURN_RECENCY_THRESHOLD

df_model = pd.read_csv("data/rfm_with_clusters.csv")
df_rtail = pd.read_csv("data/cleaned_retail_data.csv")

# -------------------------------- CREATING CHURN LABEL --------------------------------
# If Recency > CHURN_RECENCY_THRESHOLD 1, else 0
df_model['churn'] = (df_model['recency'] > CHURN_RECENCY_THRESHOLD).astype(int)

print("Class Distribution:")
print(df_model['churn'].value_counts(normalize=True))
"""
Class Distribution:
churn
0    0.618408
1    0.381592
# Dataset is not so unbalanced. So, I don't have to apply SMOTE techniques. I can directly model it.
"""
# -------------------------------- CREATING NEW FEATURES --------------------------------
# FEATURE 1) IS UK Feature:
# Because this dataset mosstly includes UK customers.

# Aggregating df_rtail for customer level churn
df_rtail_agg = df_rtail.groupby("customer_id").agg(
    first_purchase=("invoicedate", "min"),
    last_purchase=("invoicedate", "max"),
    country=("country", "first")
).reset_index()

# merge df's
merged_df = df_model.merge(df_rtail_agg, on="customer_id", how="inner")

merged_df["is_UK"] = (merged_df["country"] == "United Kingdom").astype(int)
print(merged_df.head())

# FEATURE 2) AVERAGE SPEND PER ORDER
merged_df["avg_order_value"] =  merged_df["monetary"] / merged_df["frequency"] # it is different then avg_unit_price (we divided total price whole rows per person for avg_unit_price)

# FEATURE 3) ACTIVE LIFESPAN
merged_df['first_purchase'] = pd.to_datetime(merged_df['first_purchase'])
merged_df['last_purchase'] = pd.to_datetime(merged_df['last_purchase'])
merged_df['active_lifespan'] = (merged_df['last_purchase'] - merged_df['first_purchase']).dt.days

# -------------------------------- SELECTING FEATURES --------------------------------
print(merged_df.columns) # ['customer_id', 'recency', 'frequency', 'monetary', 'avg_unit_price','unique_products', 'cluster', 'segment', 'churn', 'first_purchase', 'last_purchase', 'country', 'is_UK', 'avg_order_value']

# I'm not selecting whole features because of data leakage issues.
# For Y column: "churn", because it is target feature
# For X column: "frequency", "monetary", "avg_unit_price", "unique_products"
# For X, I eliminited customer_id, recency because it causes leakage, cluster and segment arised from recency.

# Base Model
X = df_model[['frequency', 'monetary', 'avg_unit_price', 'unique_products']]
y = df_model['churn']

# Improved Model
X2 = merged_df[['frequency', 'monetary', 'avg_unit_price', 'unique_products', 'is_UK', 'avg_order_value', 'active_lifespan']]
y = merged_df['churn']

# Train- test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Base Model
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size=0.2, random_state=42, stratify=y) # Improved Model

# XGBoost
param_grid = {
    'max_depth': [3, 4, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 150, 200, 300, 500],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'scale_pos_weight': [1.25, 1.55, 1.62] # 727/449 = 1.62 (Azınlıkta olan sınıfa ağırlık verme)
}

xgb_search = RandomizedSearchCV(XGBClassifier(eval_metric='logloss'), 
                                 param_distributions=param_grid, 
                                 n_iter=200,
                                 scoring='f1', # F1-skorunu maksimize et 
                                 cv=3, 
                                 random_state=42, 
                                 ) 
"""
xgb_search.fit(X_train, y_train)

best_model = xgb_search.best_estimator_
print("En iyi parametreler:", xgb_search.best_params_)
# En iyi parametreler: {'subsample': 0.7, 'scale_pos_weight': 1.62, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.01, 'colsample_bytree': 0.8}

# Results:
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
"""
"""
              precision    recall  f1-score   support

           0       0.83      0.66      0.74       727
           1       0.59      0.78      0.67       449

    accuracy                           0.71      1176
   macro avg       0.71      0.72      0.70      1176
weighted avg       0.74      0.71      0.71      1176
"""
# When I used the features 'frequency', 'monetary', 'avg_unit_price', and 'unique_products', my F1 score for the "churn" class was 0.67, which is relatively low.
# For this reason, I decided to add new features; however, since the 'recency' feature directly contains information about churn behavior, I could not include it as-is.
# Instead of using "tenure" (customer age), I considered it, but if a customer had only one purchase, it would still essentially correspond to the recency value.
# Therefore, I engineered a new feature representing how long a customer has been active on the platform: "last purchase date - first purchase date".
# I also created the 'is_UK' feature and an "average spending per order" feature.
# For example, customers outside the UK may be more likely to churn due to higher shipping costs.

# X2 VERSION:
xgb_search.fit(X_train2, y_train)

best_model2 = xgb_search.best_estimator_
print("En iyi parametreler 2:", xgb_search.best_params_)
# En iyi parametreler: {'subsample': 0.7, 'scale_pos_weight': 1.62, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.01, 'colsample_bytree': 0.8}

# Results:
y_pred = best_model2.predict(X_test2)
print(classification_report(y_test, y_pred))
"""
En iyi parametreler 2: {'subsample': 0.9, 'scale_pos_weight': 1.55, 'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 0.9}
              precision    recall  f1-score   support

           0       0.90      0.65      0.76       727
           1       0.61      0.88      0.72       449

    accuracy                           0.74      1176
   macro avg       0.75      0.77      0.74      1176
weighted avg       0.79      0.74      0.74      1176
"""
# In churn prediction, I aimed to maximize Recall (0.88) because we are able to capture 88% of customers who are likely to churn in advance.
# Yes, Precision came out as 0.61; meaning that out of every 100 customers we predict as "will churn," 39 actually would not churn (False Positives).
# However, for an e-commerce company, losing a loyal customer completely is much more costly than mistakenly offering a discount coupon to someone who would not churn.

# The feature that contributed most to improving my score was 'active_lifespan'. The other features, 'is_UK' and 'avg_order_value', did not cause any change in my Recall score.












# docker compose exec analysis_app python src/predict_churn.py
