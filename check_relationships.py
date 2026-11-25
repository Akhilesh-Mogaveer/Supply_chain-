
import pandas as pd

# Load the supplier-customer CSV file into a table called relations
relations = pd.read_csv("data/sectivia-relations.csv")

# Count how many different suppliers are present
# nunique() counts different, non-repeated IDs
print("\nUNIQUE SUPPLIERS")
print(relations["supplier_id"].nunique())

# Count how many different customers are present
print("\nUNIQUE CUSTOMERS")
print(relations["customer_id"].nunique())

# Combine supplier IDs and customer IDs
# set() keeps only different IDs
all_companies = set(relations["supplier_id"]) | set(relations["customer_id"])

# Count all different companies in the network
print("\nTOTAL UNIQUE COMPANIES")
print(len(all_companies))

# Check whether a company supplies itself
# == compares the supplier ID and customer ID in each row
print("\nSELF-RELATIONSHIPS")
print((relations["supplier_id"] == relations["customer_id"]).sum())

# Check repeated supplier-customer pairs
# subset checks only these two columns, not all 14 columns
print("\nREPEATED SUPPLIER-CUSTOMER PAIRS")
print(relations.duplicated(subset=["supplier_id", "customer_id"]).sum())
