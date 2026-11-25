
import os
from pathlib import Path
import pandas as pd
import networkx as nx

def resolve_data_paths():
    """Locate cleaned company and relationship CSV files reliably from any working directory."""
    base_dir = Path(__file__).resolve().parent
    comp_candidates = [
        base_dir / "processed" / "clean_companies.csv",
        base_dir.parent / "data" / "processed" / "clean_companies.csv",
        Path("data/processed/clean_companies.csv"),
        Path("processed/clean_companies.csv")
    ]
    rel_candidates = [
        base_dir / "processed" / "clean_relations.csv",
        base_dir.parent / "data" / "processed" / "clean_relations.csv",
        Path("data/processed/clean_relations.csv"),
        Path("processed/clean_relations.csv")
    ]
    comp_path = next((p for p in comp_candidates if p.exists()), None)
    rel_path = next((p for p in rel_candidates if p.exists()), None)
    if not comp_path or not rel_path:
        raise FileNotFoundError(f"Could not locate cleaned datasets. Checked: {comp_candidates}")
    return str(comp_path), str(rel_path)

def build_network(comp_csv=None, rel_csv=None):
    """
    Construct the directed supply chain graph from cleaned company and relationship datasets.
    Nodes: Company IDs with attributes (name, country, sector, segment)
    Edges: Directed connections (supplier_id -> customer_id)
    """
    if comp_csv is None or rel_csv is None:
        c_path, r_path = resolve_data_paths()
        comp_csv = comp_csv or c_path
        rel_csv = rel_csv or r_path

    # Load our two cleaned datasets
    companies = pd.read_csv(comp_csv)
    relations = pd.read_csv(rel_csv)

    # Create an empty directed graph
    # DiGraph means Directed Graph: connections have arrows
    graph = nx.DiGraph()

    # Add all companies as nodes
    # iterrows() reads the company table one row at a time
    for index, company in companies.iterrows():
        # add_node() adds one company to our graph
        # The extra information is stored as attributes of that node
        graph.add_node(
            company["id"],
            name=company["name"],
            country=company["country"],
            sector=company["sector"],
            segment=company["segment"]
        )

    # Add supplier-customer relationships as directed edges
    for index, relation in relations.iterrows():
        # add_edge() connects two nodes with an arrow
        # The arrow starts at the supplier and points to the customer
        graph.add_edge(
            relation["supplier_id"],
            relation["customer_id"]
        )

    return graph

# Build the global graph variable G
G = build_network()

if __name__ == "__main__":
    # Display the total number of companies (nodes)
    print("\nTOTAL COMPANIES (NODES)")
    print(G.number_of_nodes())

    # Display the total number of supply relationships (edges)
    print("\nTOTAL RELATIONSHIPS (EDGES)")
    print(G.number_of_edges())

    # Display the first 10 companies in the graph
    print("\nFIRST 10 COMPANIES")
    print(list(G.nodes())[:10])

    # Display the first 10 supply relationships
    print("\nFIRST 10 RELATIONSHIPS")
    print(list(G.edges())[:10])

    # Display a company's details stored in the graph
    print("\nDETAILS OF AGILENT")
    print(G.nodes["a"])
