
import pandas as pd

# Load both CSV files
relations = pd.read_csv("data/sectivia-relations.csv")
companies = pd.read_csv("data/sectivia-companies.csv")

# Get all company IDs from the company dataset
# set() keeps only unique IDs
company_ids = set(companies["id"])

# Get all supplier IDs from the relationship dataset
supplier_ids = set(relations["supplier_id"])

# Get all customer IDs from the relationship dataset
customer_ids = set(relations["customer_id"])

# Combine supplier and customer IDs into one collection
relationship_ids = supplier_ids | customer_ids

# Find IDs used in relationships but missing from company details
# difference() finds items that exist in the first set but not the second
missing_ids = relationship_ids.difference(company_ids)

# Display the total number of unique companies in each dataset
print("\nCOMPANIES IN COMPANY DATASET")
print(len(company_ids))

print("\nCOMPANIES IN RELATIONSHIP DATASET")
print(len(relationship_ids))

# Display IDs that could not be matched
print("\nUNMATCHED COMPANY IDs")
print(missing_ids)

# Display how many IDs could not be matched
print("\nNUMBER OF UNMATCHED IDs")
print(len(missing_ids))
