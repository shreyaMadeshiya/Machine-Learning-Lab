import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("Assignment1/titanic.csv")

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# ============================================================
# 2. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================

# Select numeric columns
numeric_df = df.select_dtypes(include=np.number)

print("\nDescriptive Statistics:")
print(numeric_df.describe())

# Additional statistics
print("\nMean:")
print(numeric_df.mean())

print("\nMedian:")
print(numeric_df.median())

print("\nStandard Deviation:")
print(numeric_df.std())

print("\nMinimum:")
print(numeric_df.min())

print("\nMaximum:")
print(numeric_df.max())

print("\nQuantiles:")
print(numeric_df.quantile([0.25, 0.50, 0.75]))


# ============================================================
# 4. HISTOGRAMS + KDE PLOTS
# ============================================================

for column in numeric_df.columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=numeric_df,
        x=column,
        kde=True,
        bins=20
    )

    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


# ============================================================
# 5. BOX PLOTS
# ============================================================

for column in numeric_df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        y=numeric_df[column]
    )

    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. PAIRWISE SCATTER PLOTS
# ============================================================

sns.pairplot(
    numeric_df
)

plt.suptitle(
    'Pairwise Scatter Plots',
    y=1.02
)

plt.show()


# ============================================================
# 7. CORRELATION MATRIX
# ============================================================

correlation = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation)


# ============================================================
# 8. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)

plt.title('Correlation Heatmap')

plt.tight_layout()
plt.show()


# ============================================================
# 9. OUTLIER DETECTION USING IQR
# ============================================================

print("\nPotential Outliers:")

for column in numeric_df.columns:

    Q1 = numeric_df[column].quantile(0.25)
    Q3 = numeric_df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = numeric_df[
        (numeric_df[column] < lower_bound) |
        (numeric_df[column] > upper_bound)
    ]

    print(f"{column}: {len(outliers)} potential outliers")


# ============================================================
# 10. SUMMARY OF FINDINGS
# ============================================================

print("\nMajor Analytical Findings:")

print("1. The dataset contains both numerical and categorical variables.")

print("2. Histograms and KDE plots show the distribution and spread of numerical features.")

print("3. Box plots help identify potential outliers in numerical variables.")

print("4. Pairwise scatter plots show relationships and patterns between numerical features.")

print("5. The correlation heatmap identifies positively and negatively correlated variables.")

print("6. Variables with correlation values close to +1 or -1 have strong linear relationships.")