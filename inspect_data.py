# Import the Pandas library and give it the short name 'pd'
# A library is a collection of ready-made Python functions
import pandas as pd


# Read the CSV file and store its data in a variable called relations
# read_csv() is a Pandas function used to read CSV files
# A DataFrame is a table containing rows and columns
relations = pd.read_csv("data/sectivia-relations.csv")


# Display the first 5 rows of the dataset
# head() displays the first 5 records by default
print("\nFIRST 5 RECORDS")
print(relations.head())


# Display the number of rows and columns
# shape returns (rows, columns)
print("\nDATASET SHAPE")
print(relations.shape)


# Display all column names present in the dataset
# columns is used to access the column names
print("\nCOLUMN NAMES")
print(relations.columns)


# Display information about the dataset
# info() shows column names, data types and non-missing values
print("\nDATASET INFORMATION")
relations.info()


# Check for missing values in the dataset
# isnull() identifies empty or missing values
# sum() counts how many missing values are present in each column
print("\nMISSING VALUES")
print(relations.isnull().sum())


# Check for duplicate rows in the dataset
# duplicated() identifies repeated rows
# sum() counts the number of duplicate rows
print("\nDUPLICATE RECORDS")
print(relations.duplicated().sum())