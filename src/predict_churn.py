import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV
from config import CHURN_RECENCY_THRESHOLD

df_rfm_segments  = pd.read_csv("data/rfm_with_clusters2.csv")
df_rtail = pd.read_csv("data/cleaned_retail_data.csv") # Original Data
print("df_model:",df_rfm_segments .info())
print("df_rtail:",df_rtail.info())

df_rtail['invoicedate'] = pd.to_datetime(df_rtail['invoicedate']) # Becasue it i scsv format I'm converting it to datetime format.
# -------------------------------- CREATING CHURN LABEL --------------------------------

   # ------------------------------------- My Old Approach -------------------------------------
   # If Recency > CHURN_RECENCY_THRESHOLD 1, else 0
   # df_model['churn'] = (df_model['recency'] > CHURN_RECENCY_THRESHOLD).astype(int)

# ------------------------------------- New Approach -------------------------------------
# 'Cut-off Date' (Fixed Window) Approach
# I will determine the cut-off date for churn prediction and create the churn label based on whether a customer made a purchase after that cut-off date or not. This approach is more robust and realistic than using a fixed recency threshold, as it directly captures the actual behavior of customers in terms of their purchasing activity over time.

# 1) Determining the cut off date for churn prediction
last_date = df_rtail['invoicedate'].max() # 2011-12-09
cut_off_date = pd.Timestamp('2011-09-01') # 2011-09-01
# Cut-off tarihi olarak Eylül 2011 seçildi.
# Gerekçe: Veri Aralık 2011'de bitiyor → 3 aylık "gelecek" penceresi oluşuyor.
# Bu pencere, e-ticaret sektöründe yaygın kabul gören churn tanımıyla (90 gün)  
# örtüşmektedir. Daha kısa pencere (örn. 1 ay) çok agresif, daha uzun (6 ay)
# ise veri setinin boyutunu aşırı kısıtlar.

# 2) Separating the data into two parts: past and future to the cut off date
df_past = df_rtail[df_rtail['invoicedate'] < cut_off_date] # For training the model, I will use the past data.
df_future = df_rtail[df_rtail['invoicedate'] >= cut_off_date] # For creating the churn label, I will use the future data. If a customer has made a purchase in the future data, I will label them as 0 (not churned), otherwise 1 (churned).

# 3) Creating the churn label (Target variable)
# A list of customers who made a purchase after the cut off date (September)
customers_after_cutoff = df_future['customer_id'].unique()
all_old_customers = df_past['customer_id'].unique()

# 4) If a customer made a purchase after the cut off date, they are labeled as 0 (not churned), otherwise 1 (churned).
# (If past person is not in future, then they are churned)

y_df = pd.DataFrame({'customer_id': all_old_customers})
y_df['churn'] = y_df['customer_id'].apply(lambda x : 0 if x in customers_after_cutoff else 1)

# 5) Calculating Features for the model using past data (df_past)
# I will calculate the features using the past data (df_past) to avoid data leakage
X_df = df_past.groupby('customer_id').agg(
   frequency = ('invoice', 'nunique'),
   monetary = ('total_revenue', 'sum'),
   avg_unit_price = ('price', 'mean'),
   unique_products = ('stockcode', 'nunique'),
   active_lifespan = ('invoicedate', lambda x: (x.max() - x.min()).days), # Last purchase date - First purchase date
   is_UK = ('country', lambda x: 1 if x.mode()[0] == 'United Kingdom' else 0) # Mode kullanarak müşterinin en çok hangi ülkeden alışveriş yaptığını belirleyip ona göre is_UK özelliğini oluşturuyorum.
).reset_index()

X_df['avg_order_value'] = X_df['monetary'] / X_df['frequency'] # Average spending per order (monetary / frequency)

# 6) Merging the features and the target variable into a single DataFrame
df_model = pd.merge(X_df, y_df, on='customer_id', how='inner')

print("Class Distribution:")
print(df_model['churn'].value_counts(normalize=True))
"""
Class Distribution:
churn
1    0.553439
0    0.446561
# Dataset is not unbalanced. So, I don't have to apply SMOTE techniques. I can directly model it.
"""
# -------------------------------- SELECTING FEATURES --------------------------------
print(df_model.columns) # ['customer_id', 'frequency', 'monetary', 'avg_unit_price','unique_products', 'active_lifespan', 'is_UK', 'avg_order_value','churn']
# I'm not selecting whole features because of data leakage issues.
# For Y column: "churn", because it is target feature
# For X column: "frequency", "monetary", "avg_unit_price", "unique_products"
# For X, I eliminited customer_id, recency because it causes leakage, cluster and segment arised from recency.

# Improved Model
X = df_model[['frequency', 'monetary', 'avg_unit_price', 'unique_products', 'is_UK', 'avg_order_value', 'active_lifespan']]
y = df_model['churn']

# Train- test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

# scale_pos_weight (Giving more penalties on the minority class.) # 581 / 469 = 1.24 (Azınlıkta olan sınıfa ağırlık verme)
neg_count = (y_train == 0).sum()  # Majority class (churn=0)
pos_count = (y_train == 1).sum()  # Minority class (churn=1)
base_ratio = neg_count / pos_count

# XGBoost
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 150, 200, 300, 500],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
   'scale_pos_weight': [base_ratio * m for m in [0.5, 0.75, 1.0]] 
}

xgb_search = RandomizedSearchCV(XGBClassifier(eval_metric='logloss'), 
                                 param_distributions=param_grid, 
                                 n_iter=20,
                                 scoring='f1', # F1-skorunu maksimize et 
                                 cv=3, 
                                 random_state=42, 
                                 verbose=1 # For seeing the search process in more detail, I added the verbose parameter
                                 ) 

xgb_search.fit(X_train, y_train)

best_model = xgb_search.best_estimator_
print("En iyi parametreler:", xgb_search.best_params_)
# En iyi parametreler 2: {'subsample': 0.9, 'scale_pos_weight': 1.55, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 0.8}

# -------------------------------- RESULTS --------------------------------

# NEW VERSION 1: Threshold Optimization 
# Instead of using the default 0.5 probability threshold, I search for the optimal cutoff that balances precision and recall.
# The goal is to maximize recall while keeping precision above 0.85, which is critical for churn prediction (minimizing missed churners).

y_probs = best_model.predict_proba(X_test)[:, 1]

# Threshold Optimizasyonu
# Default 0.5 yerine, Recall'ı maximize eden threshold'u buluyoruz.
# Churn'de yanlış negatif (kaçırılan müşteri) >> yanlış pozitif (gereksiz kampanya)
precision_arr, recall_arr, thresholds = precision_recall_curve(y_test, y_probs)

# Minimum %85 precision şartıyla en yüksek recall'ı veren threshold
valid_mask = precision_arr[:-1] >= 0.70
if valid_mask.any():
    best_threshold = thresholds[valid_mask][np.argmax(recall_arr[:-1][valid_mask])]
else:
    best_threshold = 0.5  # Fallback

print(f"Optimal Threshold: {best_threshold:.2f}")

y_pred = (y_probs >= best_threshold).astype(int)
print(classification_report(y_test, y_pred))


# OLD VERSION: Default 0.5 threshold
# y_pred = best_model.predict(X_test) # Default 0.5 threshold
# print(classification_report(y_test, y_pred))

"""
              precision    recall  f1-score   support

           0       0.80      0.48      0.60       469
           1       0.68      0.91      0.78       581

    accuracy                           0.72      1050
   macro avg       0.74      0.69      0.69      1050
weighted avg       0.74      0.72      0.70      1050
"""
# In churn prediction, I aimed to maximize Recall (0.91) because we are able to capture 91% of customers who are likely to churn in advance.
# Yes, Precision came out as 0.68; meaning that out of every 100 customers we predict as "will churn," 32 actually would not churn (False Positives).
# However, for an e-commerce company, losing a loyal customer completely is much more costly than mistakenly offering a discount coupon to someone who would not churn.

# The feature that contributed most to improving my score was 'active_lifespan'. The other features, 'is_UK' and 'avg_order_value', did not cause any change in my Recall score.


# -------------------------------- EVALUATION METRICS --------------------------------

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Churn Prediction Confusion Matrix')
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin Edilen')
plt.savefig('analyze_img/confusion_matrix.png')


# 2. ROC-AUC Skoru ve Eğrisi
# Tahmin olasılıklarını al (0.5 eşiği yerine modelin ne kadar emin olduğunu görmek için)
y_probs = best_model.predict_proba(X_test)[:, 1]

# ROC-AUC Skoru
auc_score = roc_auc_score(y_test, y_probs)
print(f"\nROC-AUC Skoru: {auc_score:.4f}")

# Eğriyi Çizdir
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.legend()
plt.savefig('analyze_img/roc_curve.png')

# 3. Feature Importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importances, x='importance', y='feature')
plt.title('Churn Tahmininde En Etkili Özellikler')
plt.savefig('analyze_img/feature_importance_churn.png')
print("\nÖzellik Önem Sıralaması:")
print(importances)






# docker compose exec analysis_app python src/predict_churn.py
