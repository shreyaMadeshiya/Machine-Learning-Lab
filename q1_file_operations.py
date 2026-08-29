import pandas as pd


# ==================================================
# Q1(a) READ DATASET FROM CSV
# ==================================================

print("\n========== Q1(a) READ CSV ==========")

# Read the CSV dataset into a DataFrame
df = pd.read_csv("titanic.csv")

print("First 5 rows:")
print(df.head())

print("\nShape of DataFrame:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# ==================================================
# Q1(b) WRITE DATA TO EXCEL AND JSON
# ==================================================

print("\n========== Q1(b) WRITE EXCEL AND JSON ==========")

# Write DataFrame to Excel
df.to_excel("titanic_output.xlsx", index=False)

# Write DataFrame to JSON
df.to_json(
    "titanic_output.json",
    orient="records",
    indent=4
)

print("Excel file created successfully!")
print("JSON file created successfully!")


# ==================================================
# Q1(c) READ DATA BACK AND VERIFY CONSISTENCY
# ==================================================

print("\n========== Q1(c) READ ALL FORMATS ==========")

# Read CSV
df_csv = pd.read_csv("titanic.csv")

# Read Excel
df_excel = pd.read_excel("titanic_output.xlsx")

# Read JSON
df_json = pd.read_json("titanic_output.json")


# --------------------------------------------------
# Compare shapes
# --------------------------------------------------

print("\nCSV shape   :", df_csv.shape)
print("Excel shape :", df_excel.shape)
print("JSON shape  :", df_json.shape)


# --------------------------------------------------
# Compare column names
# --------------------------------------------------

print("\nCSV columns:")
print(df_csv.columns.tolist())

print("\nExcel columns:")
print(df_excel.columns.tolist())

print("\nJSON columns:")
print(df_json.columns.tolist())


# Check whether column names are same
columns_same = (
    df_csv.columns.tolist()
    == df_excel.columns.tolist()
    == df_json.columns.tolist()
)

print("\nAre column names same?")
print(columns_same)


# Check whether shapes are same
shapes_same = (
    df_csv.shape
    == df_excel.shape
    == df_json.shape
)

print("\nAre shapes same?")
print(shapes_same)


# --------------------------------------------------
# Basic value check
# --------------------------------------------------

print("\nFirst passenger name:")

print("CSV   :", df_csv.iloc[0]["Name"])
print("Excel :", df_excel.iloc[0]["Name"])
print("JSON  :", df_json.iloc[0]["Name"])


# ==================================================
# Q1(d) APPEND NEW PASSENGER ROWS
# ==================================================

print("\n========== Q1(d) APPEND NEW ROWS ==========")


# Read original CSV
df_csv = pd.read_csv("titanic.csv")

# Create fresh copies for Excel and JSON
# This prevents duplicate rows when the program
# is run multiple times.
df_excel = df_csv.copy()
df_json = df_csv.copy()


# --------------------------------------------------
# Create 3 new passenger rows
# --------------------------------------------------

new_rows = pd.DataFrame([
    {
        "PassengerId": 1001,
        "Survived": 1,
        "Pclass": 1,
        "Name": "Test Passenger One",
        "Sex": "female",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "TEST001",
        "Fare": 100.0,
        "Cabin": "T1",
        "Embarked": "S"
    },
    {
        "PassengerId": 1002,
        "Survived": 0,
        "Pclass": 2,
        "Name": "Test Passenger Two",
        "Sex": "male",
        "Age": 30,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "TEST002",
        "Fare": 50.0,
        "Cabin": "T2",
        "Embarked": "C"
    },
    {
        "PassengerId": 1003,
        "Survived": 1,
        "Pclass": 3,
        "Name": "Test Passenger Three",
        "Sex": "female",
        "Age": 22,
        "SibSp": 0,
        "Parch": 1,
        "Ticket": "TEST003",
        "Fare": 30.0,
        "Cabin": "T3",
        "Embarked": "Q"
    }
])

print("New passenger rows:")
print(new_rows)


# --------------------------------------------------
# Append rows to CSV
# --------------------------------------------------

df_csv = pd.concat(
    [df_csv, new_rows],
    ignore_index=True
)

df_csv.to_csv(
    "titanic_output.csv",
    index=False
)

print("\nNew rows added to CSV successfully!")


# --------------------------------------------------
# Append rows to Excel
# --------------------------------------------------

df_excel = pd.concat(
    [df_excel, new_rows],
    ignore_index=True
)

df_excel.to_excel(
    "titanic_output.xlsx",
    index=False
)

print("New rows added to Excel successfully!")


# --------------------------------------------------
# Append rows to JSON
# --------------------------------------------------

df_json = pd.concat(
    [df_json, new_rows],
    ignore_index=True
)

df_json.to_json(
    "titanic_output.json",
    orient="records",
    indent=4
)

print("New rows added to JSON successfully!")


# --------------------------------------------------
# Reload files and verify appended rows
# --------------------------------------------------

print("\n========== VERIFYING NEW ROWS ==========")

check_csv = pd.read_csv("titanic_output.csv")
check_excel = pd.read_excel("titanic_output.xlsx")
check_json = pd.read_json("titanic_output.json")


# Display updated shapes
print("\nUpdated CSV shape   :", check_csv.shape)
print("Updated Excel shape :", check_excel.shape)
print("Updated JSON shape  :", check_json.shape)


# --------------------------------------------------
# Check whether new PassengerIds exist
# --------------------------------------------------

new_ids = [1001, 1002, 1003]

for passenger_id in new_ids:

    csv_present = passenger_id in check_csv["PassengerId"].values
    excel_present = passenger_id in check_excel["PassengerId"].values
    json_present = passenger_id in check_json["PassengerId"].values

    print(f"\nPassengerId {passenger_id}:")

    print("CSV   :", csv_present)
    print("Excel :", excel_present)
    print("JSON  :", json_present)


# --------------------------------------------------
# Display newly appended rows
# --------------------------------------------------

print("\nLast 3 rows of CSV:")
print(check_csv.tail(3))

print("\nLast 3 rows of Excel:")
print(check_excel.tail(3))

print("\nLast 3 rows of JSON:")
print(check_json.tail(3))


# ==================================================
# Q1(e) COMPARISON OF CSV, EXCEL AND JSON
# ==================================================

print("\n========== Q1(e) FORMAT COMPARISON ==========")

print("""
CSV:
- Simple tabular format.
- Lightweight and easy to read.
- Widely supported for data exchange.
- Does not preserve formatting, formulas, multiple sheets
  or much metadata.
- Commonly used for datasets and data exchange.

Excel:
- Uses the .xlsx format.
- Supports multiple worksheets.
- Supports formatting, formulas, charts and other
  spreadsheet features.
- Useful for reports and spreadsheet-based analysis.
- Preserves more spreadsheet structure than CSV.

JSON:
- Stores data using key-value pairs and records.
- Commonly used for APIs and web applications.
- Supports structured and nested data better than CSV.
- Human-readable but can be more verbose.
- Useful for data exchange between applications.

Overall:
- CSV is best for simple tabular data exchange.
- Excel is best for spreadsheets, reports and formatted data.
- JSON is best for APIs, web applications and structured data.
""")