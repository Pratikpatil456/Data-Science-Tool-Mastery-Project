# ==========================================
# DATA SCIENCE TOOL MASTERY PROJECT
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

# 1. LOADING DATA (Pandas Mastery)
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

print("--- DATA OVERVIEW ---")
print(df.head())
print("\n--- DATA INFO ---")
print(df.info())
print("\n--- SUMMARY STATISTICS ---")
print(df.describe().round(2))

# 2. NUMPY MASTERY (Feature Engineering)
# Create a new feature using NumPy: Petal Area
df['petal_area'] = df['petal length (cm)'] * df['petal width (cm)']
print("\nNew feature 'petal_area' created using NumPy.")

# 3. VISUALIZATION (Matplotlib & Seaborn Mastery)
sns.set_theme(style="whitegrid")

# Chart 1: Histogram
plt.figure(figsize=(6, 4))
sns.histplot(df['petal length (cm)'], bins=20, kde=True, color='#E8650A')
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.savefig('chart1_hist.png', dpi=150)
plt.show()

# Chart 2: Correlation Heatmap
plt.figure(figsize=(6, 4))
corr = df[iris.feature_names + ['petal_area']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Iris Features')
plt.savefig('chart2_heatmap.png', dpi=150)
plt.show()

# Chart 3: Boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(x='species_name', y='petal length (cm)', data=df, palette='Set2')
plt.title('Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.savefig('chart3_box.png', dpi=150)
plt.show()

# 4. MACHINE LEARNING (Scikit-learn Mastery)
X = df[iris.feature_names + ['petal_area']]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n--- MODEL EVALUATION ---")
print(f"Model Accuracy: {acc * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
