# Step 1: Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Reading the Dataset
df = pd.read_csv('Electric Vehicle Sales by State in India.csv')

# Step 3: Basic Overview
print("Dataset shape:", df.shape)
print("Unique Columns:", df.columns.nunique())
print("Column Names:\n", df.columns)
print("First 5 rows:\n", df.head())
print("Last 5 rows:\n", df.tail())

# Step 4: Understanding Yearwise and Statewise Sales
print("Year value counts:\n", df['Year'].value_counts())
print("State value counts:\n", df['State'].value_counts())
print("Vehicle Class value counts:\n", df['Vehicle_Class'].value_counts())
print("Vehicle Category value counts:\n", df['Vehicle_Category'].value_counts())
print("Vehicle Type value counts:\n", df['Vehicle_Type'].value_counts())

# Step 5: Descriptive Stats and Data Quality Check
print("Descriptive Stats (excluding 'Year'):\n", df.drop(columns=['Year']).describe())
print("Duplicate Rows:", df.duplicated().sum())
print("Missing Values:\n", df.isnull().sum())

# Step 6: Data Cleaning & Type Conversion
df['Year'] = df['Year'].astype(int)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
categorical_columns = ['Month_Name', 'State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type']
df[categorical_columns] = df[categorical_columns].astype('category')

print("\nData Types after Cleaning:\n")
print(df.info())

# Step 7: Data Visualization

# Yearly EV Sales
plt.figure(figsize=(6, 4))
plt.title('Yearly Analysis of EV Sales in India')
sns.lineplot(x='Year', y='EV_Sales_Quantity', data=df, marker='o', color='b')
plt.xlabel('Year')
plt.ylabel('EV Sales')
plt.tight_layout()
plt.show()

# Monthly EV Sales
plt.figure(figsize=(6, 4))
plt.title('Monthly Analysis of EV Sales in India')
sns.lineplot(x='Month_Name', y='EV_Sales_Quantity', data=df, marker='o', color='r')
plt.xlabel('Month')
plt.ylabel('EV Sales')
plt.tight_layout()
plt.show()

# State-wise EV Sales
plt.figure(figsize=(6, 7))
plt.title('State-Wise Analysis of EV Sales')
sns.barplot(y='State', x='EV_Sales_Quantity', data=df, hue='State', palette='bright')
plt.xlabel('States')
plt.ylabel('EV Sales')
plt.legend().remove()
plt.tight_layout()
plt.show()

# Vehicle Class
plt.figure(figsize=(15, 4))
sns.barplot(x='Vehicle_Class', y='EV_Sales_Quantity', data=df, hue='Vehicle_Class', palette='bright')
plt.title('Analysis by Vehicle Class')
plt.xlabel('Vehicle Class')
plt.ylabel('EV Sales')
plt.xticks(rotation=90)
plt.legend().remove()
plt.tight_layout()
plt.show()

# Vehicle Category
plt.figure(figsize=(6, 4))
sns.barplot(x='Vehicle_Category', y='EV_Sales_Quantity', data=df, hue='Vehicle_Category', palette='bright')
plt.title('Analysis by Vehicle Category')
plt.xlabel('Vehicle Category')
plt.ylabel('EV Sales')
plt.xticks(rotation=0)
plt.legend().remove()
plt.tight_layout()
plt.show()

# Vehicle Type
plt.figure(figsize=(6, 4))
sns.barplot(x='Vehicle_Type', y='EV_Sales_Quantity', data=df, hue='Vehicle_Type', palette='bright')
plt.title('Analysis by Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('EV Sales')
plt.xticks(rotation=90)
plt.legend().remove()
plt.tight_layout()
plt.show()
# Step 8: Feature Engineering
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=[
    'State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type'
], drop_first=True)

# Drop unneeded columns
df_encoded.drop(['Date', 'Month_Name'], axis=1, inplace=True)

# Step 9: Preparing Features & Target
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = df_encoded.drop('EV_Sales_Quantity', axis=1)
y = df_encoded['EV_Sales_Quantity']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 11: Model Prediction
y_pred = model.predict(X_test)

# Step 12: Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse:.2f}')

# Step 13: Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs Predicted EV Sales')
plt.xlabel('Actual EV Sales')
plt.ylabel('Predicted EV Sales')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 14: Feature Importance
importance = model.feature_importances_
feature_importance = pd.Series(importance, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importance[:20].plot(kind='bar')  # top 20 features
plt.title('Top 20 Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
