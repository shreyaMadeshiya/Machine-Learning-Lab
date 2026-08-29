import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = load_breast_cancer()

# Convert dataset into Pandas DataFrame
df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

# Add target column
df['target'] = data.target

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nTarget Values:")
print(df['target'].value_counts())


# ============================================================
# 2. SELECT FEATURES AND TARGET
# ============================================================

# Selecting two features for visualization
X = df[['mean radius', 'mean texture']]

# Binary target variable
y = df['target']

print("\nSelected Features:")
print(X.head())

print("\nTarget:")
print(y.head())


# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ============================================================
# 4. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nFirst 5 Scaled Training Samples:")
print(X_train_scaled[:5])


# ============================================================
# 5. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression()

model.fit(X_train_scaled, y_train)

print("\nLogistic Regression Model trained successfully!")


# ============================================================
# 6. PREDICTIONS
# ============================================================

# Predicted binary labels
y_pred = model.predict(X_test_scaled)

# Probability of class 1
y_prob = model.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# 7. ACTUAL VS PREDICTED TABLE
# ============================================================

comparison = pd.DataFrame({
    'Actual Label': y_test.values,
    'Predicted Label': y_pred,
    'P(Y = 1)': y_prob
})

print("\nActual vs Predicted:")
print(comparison.to_string(index=False))


# ============================================================
# 8. DECISION BOUNDARY
# ============================================================

# Create a grid of points
x_min = X_train_scaled[:, 0].min() - 1
x_max = X_train_scaled[:, 0].max() + 1

y_min = X_train_scaled[:, 1].min() - 1
y_max = X_train_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

# Predict class for each point in the grid
Z = model.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3
)

plt.scatter(
    X_train_scaled[:, 0],
    X_train_scaled[:, 1],
    c=y_train,
    edgecolor='k'
)

plt.xlabel('Mean Radius (Scaled)')
plt.ylabel('Mean Texture (Scaled)')
plt.title('Logistic Regression Decision Boundary')

plt.show()


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Confusion Matrix")

plt.show()


# ============================================================
# 10. PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)


print("\nPerformance Metrics:")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-Score :", f1)
print("ROC-AUC  :", roc_auc)


# ============================================================
# 11. METRICS TABLE
# ============================================================

metrics = pd.DataFrame({
    'Metric': [
        'Accuracy',
        'Precision',
        'Recall',
        'F1-Score',
        'ROC-AUC'
    ],
    'Score': [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

print("\nMetrics Table:")
print(metrics.to_string(index=False))


# ============================================================
# 12. ROC-AUC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f'ROC Curve (AUC = {roc_auc:.2f})'
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    label='Random Classifier'
)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC-AUC Curve')

plt.legend()

plt.show()