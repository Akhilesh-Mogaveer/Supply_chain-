
import pandas as pd

# Load the original supplier-customer dataset
relations = pd.read_csv("data/sectivia-relations.csv")

# Select only the columns needed to build our supply-chain network
# supplier_id and customer_id identify the connected companies
# supplier_name and customer_name store their company names
clean_relations = relations[
    ["supplier_id", "supplier_name", "customer_id", "customer_name"]
].copy()

# Remove extra spaces from company IDs and names
# strip() removes spaces from the beginning and end of text
for column in clean_relations.columns:
    clean_relations[column] = clean_relations[column].str.strip()

# Remove rows where important company information is missing
# dropna() removes rows containing missing values in selected columns
clean_relations = clean_relations.dropna(
    subset=["supplier_id", "supplier_name", "customer_id", "customer_name"]
)

# Remove repeated supplier-customer relationships if any exist
clean_relations = clean_relations.drop_duplicates(
    subset=["supplier_id", "customer_id"]
)

# Save the prepared relationship dataset
# to_csv() writes the table into a CSV file
# index=False prevents Pandas from adding an extra index column
clean_relations.to_csv("data/processed/clean_relations.csv", index=False)

# Display the final prepared dataset information
print("\nCLEAN RELATIONSHIP DATA")
print(clean_relations.head())

print("\nFINAL DATASET SHAPE")
print(clean_relations.shape)

print("\nMISSING VALUES")
print(clean_relations.isnull().sum())

print("\nDUPLICATE RELATIONSHIPS")
print(clean_relations.duplicated(
    subset=["supplier_id", "customer_id"]
).sum())

print("\nCLEAN DATASET SAVED SUCCESSFULLY!")
