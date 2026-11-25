
import pandas as pd

# Load the original company dataset
companies = pd.read_csv("data/sectivia-companies.csv")

# Select the company details needed for our project
clean_companies = companies[
    ["id", "name", "country", "sector", "segment"]
].copy()

# Remove extra spaces from text values
for column in clean_companies.columns:
    clean_companies[column] = clean_companies[column].str.strip()

# Remove rows where important company details are missing
clean_companies = clean_companies.dropna(
    subset=["id", "name"]
)

# Remove duplicate company IDs
clean_companies = clean_companies.drop_duplicates(
    subset=["id"]
)

# Save the cleaned company dataset
clean_companies.to_csv(
    "data/processed/clean_companies.csv",
    index=False
)

# Display the prepared company data
print("\nCLEAN COMPANY DATA")
print(clean_companies.head())

print("\nFINAL DATASET SHAPE")
print(clean_companies.shape)

print("\nMISSING VALUES")
print(clean_companies.isnull().sum())

print("\nDUPLICATE COMPANY IDs")
print(clean_companies["id"].duplicated().sum())

print("\nCLEAN COMPANY DATA SAVED SUCCESSFULLY!")
