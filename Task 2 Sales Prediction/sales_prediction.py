
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib


df = pd.read_csv("advertising.csv")  

print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())
print("Missing values per column:\n", df.isnull().sum())

df.dropna(inplace=True)  
print("After cleaning, dataset shape:", df.shape)


plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

sns.pairplot(df)
plt.show()


X = df.drop('Sales', axis=1)
y = df['Sales']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nFeature Importance (Coefficients):")
print(coeff_df)


y_pred = model.predict(X_test)
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, line_kws={"color":"red"})
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Regression Line: Actual vs Predicted Sales')
plt.show()


