# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# ==============================
# 2. LOAD DATASET
# ==============================
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name='class')
df = pd.concat([X, y], axis=1)
print(df.head())

# ==============================
# 3. GRAPH 1: PAIRPLOT (First 5 features)
# ==============================
sns.pairplot(df.iloc[:, :6].join(df['class']), hue='class')
plt.suptitle("Pairplot of Wine Features (First 5) by Class", y=1.02)
plt.show()

# ==============================
# 4. GRAPH 2: HEATMAP (Correlation)
# ==============================
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Wine Features")
plt.show()

# ==============================
# 5. GRAPH 3: BOXPLOTS (First 5 features)
# ==============================
plt.figure(figsize=(15,5))
for i, col in enumerate(wine.feature_names[:5]):
    plt.subplot(1,5,i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# ==============================
# 6. PREPROCESSING
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==============================
# 7. MODEL TRAINING
# ==============================
# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ==============================
# 8. ACCURACY BAR CHART
# ==============================
acc_lr = accuracy_score(y_test, lr_pred)
acc_dt = accuracy_score(y_test, dt_pred)
acc_rf = accuracy_score(y_test, rf_pred)

models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracy = [acc_lr, acc_dt, acc_rf]

plt.figure()
sns.barplot(x=models, y=accuracy)
plt.ylim(0,1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

print("Accuracies:")
print("Logistic Regression:", acc_lr)
print("Decision Tree:", acc_dt)
print("Random Forest:", acc_rf)

# ==============================
# 9. CONFUSION MATRIX (Random Forest)
# ==============================
cm = confusion_matrix(y_test, rf_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# 10. CLASSIFICATION REPORT
# ==============================
print("Classification Report (Random Forest):")
print(classification_report(y_test, rf_pred))
