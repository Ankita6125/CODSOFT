import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 1: CSV file se data load karo
df = pd.read_csv('iris.csv')

# Step 2: Data ka basic info
print("First 5 rows of dataset:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nSpecies count:")
print(df['species'].value_counts())

# Step 3: Data Visualization

# Pairplot
sns.pairplot(df, hue='species')
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()

# Sepal length vs Sepal width
plt.figure(figsize=(8,5))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=df, s=100)
plt.title('Sepal Length vs Sepal Width')
plt.show()

# Petal length vs Petal width
plt.figure(figsize=(8,5))
sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=df, s=100)
plt.title('Petal Length vs Petal Width')
plt.show()

# Step 4: Label Encoding
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['species'])

# Step 5: Features and Target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species_encoded']

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Accuracy Score:", accuracy_score(y_test, rf_pred))
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred, target_names=le.classes_))

# Confusion Matrix - Random Forest
cm_rf = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm_rf, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Step 8: Feature Importance
importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(6,4))
sns.barplot(x=importances, y=features)
plt.title('Feature Importance from Random Forest')
plt.show()

# Step 9: Support Vector Machine (SVM) Model
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

print("\nSVM Accuracy Score:", accuracy_score(y_test, svm_pred))
print("\nSVM Classification Report:")
print(classification_report(y_test, svm_pred, target_names=le.classes_))

# Confusion Matrix - SVM
cm_svm = confusion_matrix(y_test, svm_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm_svm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')
plt.show()


