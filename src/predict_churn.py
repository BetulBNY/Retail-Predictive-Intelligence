import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from config import CHURN_RECENCY_THRESHOLD

df_model = pd.read_csv("data/rfm_with_clusters.csv")

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
# -------------------------------- SELECTING FEATURES --------------------------------
print(df_model.columns) # ['customer_id', 'recency', 'frequency', 'monetary', 'avg_unit_price', 'unique_products', 'cluster', 'segment', 'churn']

# I'm not selecting whole features because of data leakage issues.
# For Y column: "churn", because it is target feature
# For X column: "frequency", "monetary", "avg_unit_price", "unique_products"
# For X, I eliminited customer_id, recency because it causes leakage, cluster and segment arised from recency.

X = df_model[['frequency', 'monetary', 'avg_unit_price', 'unique_products']]
y = df_model['churn']

# Train- test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# XGBoost
param_grid = {
    'max_depth': [3, 4, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 150, 200, 300, 500],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'scale_pos_weight': [1.25, 1.55, 1.62] # 727/449 = 1.62 (Azınlıkta olan sınıfa ağırlık verme
}

xgb_search = RandomizedSearchCV(XGBClassifier(eval_metric='logloss'), 
                                 param_distributions=param_grid, 
                                 n_iter=100,
                                 scoring='f1', # F1-skorunu maksimize et 
                                 cv=3, 
                                 random_state=42, 
                                 ) 
xgb_search.fit(X_train, y_train)

best_model = xgb_search.best_estimator_
print("En iyi parametreler:", xgb_search.best_params_)

# Results:
y_pred = xgb_search.predict(X_test)
print(classification_report(y_test, y_pred))









# docker compose exec analysis_app python src/predict_churn.py
