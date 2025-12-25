# Food Delivery Time Prediction 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv('Food_Delivery_Time_Prediction.csv')

print("First few rows of the dataset:")
print(df.head())

# Data Preprocessing 

# Fill missing numeric values with mean (if any)
numeric_cols = ['Distance', 'Delivery_Time', 'Order_Cost', 'Tip_Amount', 'Restaurant_Rating', 'Customer_Rating']
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

# One-hot encoding for categorical variables
categorical_cols = ['Weather_Conditions', 'Traffic_Conditions', 'Vehicle_Type', 'Order_Priority', 'Order_Time']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Feature Engineering

def extract_lat_long(loc_str):
    loc_str = loc_str.strip('()')
    lat, long = loc_str.split(',')
    return float(lat), float(long)

df[['Customer_Lat', 'Customer_Long']] = df['Customer_Location'].apply(lambda x: pd.Series(extract_lat_long(x)))
df[['Restaurant_Lat', 'Restaurant_Long']] = df['Restaurant_Location'].apply(lambda x: pd.Series(extract_lat_long(x)))

# Add latitude and longitude columns to the encoded dataframe
df_encoded = pd.concat([df_encoded, df[['Customer_Lat', 'Customer_Long', 'Restaurant_Lat', 'Restaurant_Long']]], axis=1)
df_encoded.drop(['Customer_Location', 'Restaurant_Location'], axis=1, inplace=True, errors='ignore')

# Standardize numeric features
features_to_scale = ['Distance', 'Delivery_Person_Experience', 'Order_Cost', 'Tip_Amount',
                     'Restaurant_Rating', 'Customer_Rating', 'Customer_Lat', 'Customer_Long',
                     'Restaurant_Lat', 'Restaurant_Long']
scaler = StandardScaler()
df_encoded[features_to_scale] = scaler.fit_transform(df_encoded[features_to_scale])

#  EDA

print("Dataset Description:")
print(df.describe())

# Correlation heatmap (numeric columns only)
numeric_df = df_encoded.select_dtypes(include=[np.number])
plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Boxplots for outlier detection
for col in ['Distance', 'Delivery_Time', 'Order_Cost', 'Tip_Amount']:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot for {col}')
    plt.show()

# Linear Regression Model (Predict Delivery_Time) 

X = df_encoded.drop(['Order_ID', 'Delivery_Time'], axis=1, errors='ignore')
y = df_encoded['Delivery_Time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

print("Linear Regression Model Evaluation:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("R-squared (R²):", r2_score(y_test, y_pred))

# Logistic Regression Model (Classify Fast or Delayed Delivery)

median_delivery_time = df['Delivery_Time'].median()
df_encoded['Delivery_Status'] = (df['Delivery_Time'] < median_delivery_time).astype(int)  # 1 = Fast, 0 = Delayed

X_cls = df_encoded.drop(['Order_ID', 'Delivery_Time', 'Delivery_Status'], axis=1, errors='ignore')
y_cls = df_encoded['Delivery_Status']

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_cls, y_train_cls)

y_pred_cls = logreg.predict(X_test_cls)
y_pred_proba = logreg.predict_proba(X_test_cls)[:,1]

print("Logistic Regression Model Evaluation:")
print("Accuracy:", accuracy_score(y_test_cls, y_pred_cls))
print("Precision:", precision_score(y_test_cls, y_pred_cls))
print("Recall:", recall_score(y_test_cls, y_pred_cls))
print("F1 Score:", f1_score(y_test_cls, y_pred_cls))
print("Confusion Matrix:\n", confusion_matrix(y_test_cls, y_pred_cls))
print("\nClassification Report:\n", classification_report(y_test_cls, y_pred_cls))

# ROC Curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test_cls, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.show()

print("""
Actionable Insights:
- Optimize delivery routes considering distance and traffic conditions.
- Adjust and scale staffing during busy periods for improved efficiency.
- Target training for delivery staff with less experience.
- Recommend vehicle selection based on weather and traffic patterns.
""")
