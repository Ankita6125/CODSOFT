from preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  
import os

#  Step 1: Data Load
df = preprocess_data("data/IMDB-Movie-Data.csv")

X = df.drop('Rating', axis=1)
y = df['Rating']

#  Step 2: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Step 3: Model Train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#  Step 4: Predictions & Evaluation
preds = model.predict(X_test)

#  Fixed: Manual RMSE Calculation
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

print(" RMSE:", round(rmse, 2))
print(" R²:", round(r2, 2))

#  Step 5: Save model
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/movie_rating_model.pkl", "wb"))

#  Step 6: Show some predictions
df_results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': preds
})
print("\n Sample Predictions:")
print(df_results.head(10))

#  Step 7: Feature Importance Graph
importances = model.feature_importances_
feature_names = X.columns

feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.barh(feat_df['Feature'], feat_df['Importance'], color='skyblue')
plt.xlabel("Importance")
plt.title("Top 10 Feature Importance")
plt.gca().invert_yaxis()

os.makedirs("output", exist_ok=True)
plt.savefig("output/feature_importance.png")
print("\n Feature importance graph saved to output/feature_importance.png")
