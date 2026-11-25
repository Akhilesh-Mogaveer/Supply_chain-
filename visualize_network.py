
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --------------------------------------------------
# STEP 1: LOAD THE CLEANED DATASETS
# --------------------------------------------------

# Read the cleaned company information
companies = pd.read_csv("data/processed/clean_companies.csv")

# Read the cleaned supplier-customer relationships
relations = pd.read_csv("data/processed/clean_relations.csv")


# --------------------------------------------------
# STEP 2: CREATE A DIRECTED GRAPH
# --------------------------------------------------

# DiGraph means Directed Graph.
# It represents connections that have a specific direction.
# In our project: Supplier -> Customer

G = nx.DiGraph()


# --------------------------------------------------
# STEP 3: ADD COMPANIES AS NODES
# --------------------------------------------------

# iterrows() reads the DataFrame one row at a time.
# Each company will become one node in the graph.

for index, company in companies.iterrows():

    # Add company ID as the node ID.
    # Store the company name as an attribute.
    G.add_node(
        company["id"],
        name=company["name"]
    )


# --------------------------------------------------
# STEP 4: ADD SUPPLY RELATIONSHIPS AS EDGES
# --------------------------------------------------

# Each relationship connects a supplier to a customer.
# add_edge() creates a directed connection.

for index, relation in relations.iterrows():

    G.add_edge(
        relation["supplier_id"],
        relation["customer_id"]
    )


# --------------------------------------------------
# STEP 5: CALCULATE DEGREE INFORMATION
# --------------------------------------------------

# In-degree = Number of incoming connections.
# It tells us how many suppliers supply a company.

in_degree = dict(G.in_degree())


# Out-degree = Number of outgoing connections.
# It tells us how many customers a company supplies.

out_degree = dict(G.out_degree())


# Out-degree centrality = Normalized out-degree.
# It compares outgoing connections relative to the
# maximum possible outgoing connections in the graph.

out_centrality = nx.out_degree_centrality(G)


# --------------------------------------------------
# STEP 6: GET COMPANY NAMES
# --------------------------------------------------

# Retrieve company names stored in the graph nodes.

company_names = nx.get_node_attributes(G, "name")


# --------------------------------------------------
# STEP 7: CREATE A DEGREE ANALYSIS TABLE
# --------------------------------------------------

degree_data = []

# Visit every company in the graph.

for company_id in G.nodes():

    # Store the details of each company in a dictionary.

    degree_data.append({

        "Company ID": company_id,

        "Company Name": company_names.get(
            company_id, "Unknown"
        ),

        "In-Degree": in_degree[company_id],

        "Out-Degree": out_degree[company_id],

        "Out-Degree Centrality": out_centrality[company_id]

    })


# Convert the list of dictionaries into a Pandas DataFrame.

degree_df = pd.DataFrame(degree_data)


# --------------------------------------------------
# STEP 8: FIND TOP 10 COMPANIES
# --------------------------------------------------

# Sort companies based on Out-Degree.
# ascending=False means highest value comes first.

degree_df = degree_df.sort_values(
    by="Out-Degree",
    ascending=False
)


# Display the first 10 companies.

print("\n========== TOP 10 COMPANIES BY OUT-DEGREE ==========")

print(
    degree_df.head(10).to_string(index=False)
)


# --------------------------------------------------
# STEP 9: DISPLAY NETWORK INFORMATION
# --------------------------------------------------

print("\n========== NETWORK INFORMATION ==========")

print("Total Companies (Nodes):", G.number_of_nodes())

print("Total Relationships (Edges):", G.number_of_edges())


# --------------------------------------------------
# STEP 10: CREATE THE NETWORK VISUALIZATION
# --------------------------------------------------

# spring_layout() calculates positions for the nodes.
# seed=42 keeps the layout consistent between runs.

positions = nx.spring_layout(G, seed=42)


# Draw companies as blue nodes.

nx.draw_networkx_nodes(
    G,
    positions,
    node_size=35,
    node_color="skyblue"
)


# Draw supplier-customer relationships as arrows.

nx.draw_networkx_edges(
    G,
    positions,
    edge_color="gray",
    arrows=True,
    arrowsize=5,
    width=0.5
)


# Add graph title.

plt.title("Supply Chain Network")


# Hide unnecessary axis lines.

plt.axis("off")


# Display the graph AFTER printing the results.

plt.show()
