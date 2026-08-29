# Q-1: Implementation of Linear Regression

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Read dataset and perform initial data inspection


df = pd.read_csv("Salary_Data.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())



# Split dataset into training and testing sets

X = df[["YearsExperience"]]   # Independent variable
y = df["Salary"]              # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("\nTraining set size:", len(X_train))
print("Testing set size:", len(X_test))



#Train Linear Regression model


model = LinearRegression()

model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\nModel trained successfully!")


# Display regression equation
print("\nRegression Equation:")
print(
    f"Salary = {model.intercept_:.2f} + "
    f"{model.coef_[0]:.2f} × YearsExperience"
)


# --------------------------------------------------
# d) Actual vs Predicted values for test set
# --------------------------------------------------

result = pd.DataFrame({
    "YearsExperience": X_test["YearsExperience"].values,
    "Actual Salary": y_test.values,
    "Predicted Salary": y_test_pred
})

result = result.sort_values("YearsExperience")

print("\nActual vs Predicted Salary:")
print(result.to_string(index=False))


# --------------------------------------------------
# e) Plot regression line for training dataset
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    X_train,
    y_train,
    label="Training Data"
)

plt.plot(
    X_train,
    y_train_pred,
    label="Regression Line"
)

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression - Training Dataset")
plt.legend()
plt.grid(True)

plt.show()


# --------------------------------------------------
# Plot regression line for testing dataset
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    X_test,
    y_test,
    label="Testing Data"
)

plt.plot(
    X_test,
    y_test_pred,
    label="Regression Line"
)

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression - Testing Dataset")
plt.legend()
plt.grid(True)

plt.show()


# --------------------------------------------------
# f) Calculate evaluation metrics
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)


# --------------------------------------------------
# Evaluation Metrics Summary
# --------------------------------------------------

metrics = pd.DataFrame({
    "Metric": [
        "Mean Absolute Error (MAE)",
        "Mean Squared Error (MSE)",
        "R2 Score"
    ],
    "Value": [
        mae,
        mse,
        r2
    ]
})

print("\nEvaluation Metrics Summary:")
print(metrics.to_string(index=False))