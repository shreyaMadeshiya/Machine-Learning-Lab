import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the Titanic dataset
df = pd.read_csv("titanic.csv")


# ==================================================
# Q3(a) BAR GRAPH - Passenger Class
# ==================================================

class_counts = df["Pclass"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.bar(
    class_counts.index.astype(str),
    class_counts.values
)

plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.title("Number of Passengers by Passenger Class")

plt.show()


# ==================================================
# Q3(b) HISTOGRAM - Age Distribution
# ==================================================

plt.figure(figsize=(7, 5))

plt.hist(
    df["Age"].dropna(),
    bins=5,
    edgecolor="black"
)

plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.title("Distribution of Passenger Ages")

plt.show()


# ==================================================
# Q3(c) PIE CHART - Passenger Sex
# ==================================================

sex_counts = df["Sex"].value_counts()

plt.figure(figsize=(6, 6))

plt.pie(
    sex_counts.values,
    labels=sex_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Passenger Composition by Sex")

plt.show()


# ==================================================
# Q3(d) BOX PLOT - Fare
# ==================================================

plt.figure(figsize=(7, 5))

plt.boxplot(
    df["Fare"].dropna(),
    vert=True
)

plt.ylabel("Fare")
plt.title("Box Plot of Passenger Fares")

plt.show()


# ==================================================
# Q3(e) CORRELATION HEATMAP
# ==================================================

# Select numerical columns
numerical_data = df.select_dtypes(include="number")

# Calculate correlation matrix
correlation_matrix = numerical_data.corr()

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Variables")

plt.show()


# ==================================================
# INTERPRETATIONS
# ==================================================

print("\n========== Q3 INTERPRETATIONS ==========")

print("""
1. Bar Graph:
   Third-class passengers are the most frequent group, with 6 passengers.
   First-class has 3 passengers, while second-class has only 1 passenger.

2. Histogram:
   Passenger ages are mainly distributed between the teenage years and
   early 40s, with the highest frequency occurring around the 33–44 range.

3. Pie Chart:
   Male and female passengers are equally represented, with each group
   accounting for 50% of the sample.

4. Box Plot:
   Fare values show considerable variability, ranging from about 7.25
   to 71.28. No individual points appear beyond the whiskers, so there
   are no clear outliers according to the box-plot criterion.

5. Correlation Heatmap:
   Pclass and Fare show a very strong negative correlation (-0.96),
   while Age and SibSp show a notable negative correlation (-0.70).
   These relationships should be interpreted cautiously because the
   dataset contains only 10 passengers.
""")