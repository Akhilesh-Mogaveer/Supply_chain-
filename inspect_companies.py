
import pandas as pd

# Load the company dataset into a table called companies
companies = pd.read_csv("data/sectivia-companies.csv")

# Display the first 5 company records
print("\nFIRST 5 COMPANIES")
print(companies.head())

# Display the number of rows and columns
print("\nDATASET SHAPE")
print(companies.shape)

# Display all column names
print("\nCOLUMN NAMES")
print(companies.columns)

# Display information about each column
# info() shows data types and how many values are available
print("\nDATASET INFORMATION")
companies.info()

# Count missing values in each column
print("\nMISSING VALUES")
print(companies.isnull().sum())

# Count completely repeated rows
print("\nDUPLICATE RECORDS")
print(companies.duplicated().sum())
