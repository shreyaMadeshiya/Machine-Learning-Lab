import pandas as pd

# Read the Titanic dataset
df = pd.read_csv("titanic.csv")

# ==================================================
# Q2(a) Descriptive statistics for numerical columns
# ==================================================

# Select numerical columns
numerical_columns = df.select_dtypes(include="number").columns

print("Numerical columns:")
print(numerical_columns.tolist())

# Calculate required statistics
summary = pd.DataFrame({
    "Mean": df[numerical_columns].mean(),
    "Median": df[numerical_columns].median(),
    "Standard Deviation": df[numerical_columns].std(),
    "Minimum": df[numerical_columns].min(),
    "Maximum": df[numerical_columns].max()
})

# Display the summary table
print("\n========== NUMERICAL SUMMARY STATISTICS ==========")
print(summary.round(2))



# ==================================================
# Q2(b) Interpretation of numerical statistics
# ==================================================

print("\n========== INTERPRETATIONS ==========")

print("""
1. Age:
   The mean age is about 28.11 years and the median is 27 years.
   The mean is slightly higher than the median, suggesting a small
   right-skew caused by some older passengers. Ages range from 2 to 54.

2. Fare:
   The mean fare is 27.02 while the median is 16.10.
   The higher mean compared with the median suggests that some
   passengers paid considerably higher fares. The maximum fare
   of 71.28 may indicate a possible high-value observation.

3. Pclass:
   The mean passenger class is 2.30 and the median is 3.
   This indicates that third-class passengers form a substantial
   portion of this sample.

4. SibSp:
   The mean number of siblings/spouses aboard is 0.70 and the
   median is 0.5. Most passengers had few or no siblings/spouses
   travelling with them.

5. Parch:
   The mean number of parents/children aboard is 0.30 and the
   median is 0. Most passengers had no parents or children aboard.

6. Survived:
   The mean is 0.50, meaning 50% of the passengers in this small
   sample survived and 50% did not.

Note:
   PassengerId is an identifier rather than a meaningful measurement,
   so its mean and standard deviation are not useful for interpretation.
""")



# ==================================================
# Q2(c) Frequency counts for categorical columns
# ==================================================

print("\n========== CATEGORICAL FREQUENCY COUNTS ==========")

# Frequency count for Sex
print("\nSex:")
print(df["Sex"].value_counts())

# Frequency count for Embarked
print("\nEmbarked:")
print(df["Embarked"].value_counts())

# Frequency count for Pclass
print("\nPclass:")
print(df["Pclass"].value_counts())


# ==================================================
# Comments on categorical distributions
# ==================================================

print("\n========== CATEGORICAL INTERPRETATIONS ==========")

print("""
1. Sex:
   The dataset contains more male passengers than female passengers.
   Therefore, the gender distribution is not perfectly balanced.

2. Embarked:
   Most passengers embarked from Southampton (S), while fewer
   passengers embarked from Cherbourg (C) and Queenstown (Q).

3. Pclass:
   Third-class passengers form the largest group in this sample,
   followed by first-class and second-class passengers.
""")


# ==================================================
# Final 5 Insights
# ==================================================

print("\n========== FINAL INSIGHTS ==========")

print("""
• The average passenger age is 28.11 years, with a median of 27 years.
  The small difference between mean and median suggests a slight
  right-skew in the age distribution.

• Fare has a mean of 27.02 and a median of 16.10. Since the mean is
  considerably higher than the median, some higher fares may be
  influencing the average and may represent possible high-value
  observations.

• Third-class passengers are the majority, with 6 out of 10 passengers
  belonging to Pclass 3. First class has 3 passengers and second class
  has only 1 passenger.

• The Sex distribution is perfectly balanced in this sample, with
  5 male and 5 female passengers.

• Southampton (S) is the most common embarkation point, accounting
  for 7 of the 10 passengers, followed by Cherbourg (C) with 2 and
  Queenstown (Q) with 1 passenger.
""")