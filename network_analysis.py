"""
Supply Chain Cascade Simulator - Advanced Network Analysis
============================================================
This module performs advanced network science analysis on the directed
supply chain network graph built from Sectivia cleaned datasets.

Analyses performed:
1. Degree Analysis (In-degree, Out-degree, In-degree centrality, Out-degree centrality)
2. Betweenness Centrality (nx.betweenness_centrality with normalized=True)
3. Connected Components Analysis (Weakly and Strongly connected components, component sizes, isolates)
4. Global Network Statistics (Density, averages, DAG verification, sources, sinks)
5. Critical Company Analysis (Multi-perspective evaluation of chokepoints, sources, and sinks)
6. Visualization & Reporting (PNG plots, CSV datasets, JSON summary, and formatted terminal output)
"""

import os
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

# Import reusable graph builder from build_network
from build_network import build_network, resolve_data_paths


def get_results_dir() -> Path:
    """Ensure and return the results/analysis output directory."""
    # Place network analysis results in results/analysis subfolder
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results" / "analysis"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def calculate_degree_metrics(G: nx.DiGraph) -> dict:
    """
    Calculate directed degree metrics:
    - In-Degree: number of incoming edges (suppliers supplying this firm)
    - Out-Degree: number of outgoing edges (customers this firm supplies)
    - In-Degree Centrality: in_degree / (N - 1)
    - Out-Degree Centrality: out_degree / (N - 1)
    """
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    in_centrality = nx.in_degree_centrality(G)
    out_centrality = nx.out_degree_centrality(G)

    return {
        "in_degrees": in_degrees,
        "out_degrees": out_degrees,
        "in_centrality": in_centrality,
        "out_centrality": out_centrality,
    }


def calculate_betweenness_metrics(G: nx.DiGraph) -> dict:
    """
    Calculate exact normalized betweenness centrality for all nodes in the directed graph.
    Formula:
        g(v) = sum_{s != v != t} (sigma_st(v) / sigma_st)
    normalized=True scales values by 1 / ((N - 1) * (N - 2)) for directed graphs.
    """
    betweenness = nx.betweenness_centrality(G, normalized=True)
    return betweenness


def calculate_components_metrics(G: nx.DiGraph) -> dict:
    """
    Calculate connected components for directed graphs:
    - Weakly connected components (WCC): reachability ignoring edge directions
    - Strongly connected components (SCC): mutual directed reachability
    - Largest WCC size
    - Isolated nodes (degree == 0)
    """
    wccs = list(nx.weakly_connected_components(G))
    wccs.sort(key=len, reverse=True)
    
    sccs = list(nx.strongly_connected_components(G))
    
    isolates = list(nx.isolates(G))
    wcc_sizes = [len(c) for c in wccs]

    # Map each node to its WCC ID (1-indexed, ordered by size) and size
    node_wcc_map = {}
    for idx, comp in enumerate(wccs, start=1):
        for node in comp:
            node_wcc_map[node] = {
                "component_id": f"WCC-{idx}",
                "component_size": len(comp)
            }

    return {
        "num_wcc": len(wccs),
        "num_scc": len(sccs),
        "wccs": wccs,
        "wcc_sizes": wcc_sizes,
        "largest_wcc_size": wcc_sizes[0] if wcc_sizes else 0,
        "isolates": isolates,
        "num_isolates": len(isolates),
        "node_wcc_map": node_wcc_map,
    }


def calculate_network_statistics(G: nx.DiGraph, degree_metrics: dict, component_metrics: dict) -> dict:
    """
    Calculate global network structural properties:
    - Total nodes and edges
    - Network density
    - Average in-degree and average out-degree
    - DAG (Directed Acyclic Graph) verification
    - Source nodes (in-degree == 0) and Sink nodes (out-degree == 0)
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    density = nx.density(G)
    
    in_deg_values = list(degree_metrics["in_degrees"].values())
    out_deg_values = list(degree_metrics["out_degrees"].values())
    
    avg_in_degree = sum(in_deg_values) / num_nodes if num_nodes > 0 else 0.0
    avg_out_degree = sum(out_deg_values) / num_nodes if num_nodes > 0 else 0.0
    
    is_dag = nx.is_directed_acyclic_graph(G)
    
    sources = [n for n, deg in degree_metrics["in_degrees"].items() if deg == 0]
    sinks = [n for n, deg in degree_metrics["out_degrees"].items() if deg == 0]
    
    from collections import Counter
    size_distribution = dict(Counter(component_metrics["wcc_sizes"]))

    return {
        "total_nodes": num_nodes,
        "total_edges": num_edges,
        "network_density": float(density),
        "average_in_degree": float(avg_in_degree),
        "average_out_degree": float(avg_out_degree),
        "is_dag": bool(is_dag),
        "number_of_weakly_connected_components": component_metrics["num_wcc"],
        "number_of_strongly_connected_components": component_metrics["num_scc"],
        "size_of_largest_weakly_connected_component": component_metrics["largest_wcc_size"],
        "number_of_isolated_companies": component_metrics["num_isolates"],
        "number_of_source_nodes": len(sources),
        "number_of_sink_nodes": len(sinks),
        "weakly_connected_component_size_distribution": size_distribution,
    }


def build_full_metrics_dataframe(G: nx.DiGraph, degree_metrics: dict, betweenness: dict, component_metrics: dict) -> pd.DataFrame:
    """
    Assemble complete node-level metrics table for all nodes in the graph.
    """
    node_names = nx.get_node_attributes(G, "name")
    node_countries = nx.get_node_attributes(G, "country")
    node_sectors = nx.get_node_attributes(G, "sector")
    node_segments = nx.get_node_attributes(G, "segment")
    node_wcc_map = component_metrics["node_wcc_map"]

    rows = []
    for node in G.nodes():
        in_d = degree_metrics["in_degrees"][node]
        out_d = degree_metrics["out_degrees"][node]
        wcc_info = node_wcc_map.get(node, {"component_id": "Unknown", "component_size": 1})
        
        rows.append({
            "company_id": node,
            "company_name": node_names.get(node, node),
            "country": node_countries.get(node, "Unknown"),
            "sector": node_sectors.get(node, "Unknown"),
            "segment": node_segments.get(node, "Unknown"),
            "in_degree": in_d,
            "out_degree": out_d,
            "total_degree": in_d + out_d,
            "in_degree_centrality": degree_metrics["in_centrality"][node],
            "out_degree_centrality": degree_metrics["out_centrality"][node],
            "betweenness_centrality": betweenness[node],
            "is_source": (in_d == 0),
            "is_sink": (out_d == 0),
            "is_isolated": (in_d == 0 and out_d == 0),
            "component_id": wcc_info["component_id"],
            "component_size": wcc_info["component_size"],
        })

    df = pd.DataFrame(rows)
    # Sort primarily by betweenness descending, then out_degree descending
    df = df.sort_values(by=["betweenness_centrality", "out_degree", "in_degree"], ascending=[False, False, False]).reset_index(drop=True)
    return df


def classify_structural_role(row: pd.Series, max_in: int, max_out: int, max_betw: float) -> str:
    """
    Classify company into transparent, network-science-backed structural roles:
    1. Dual Hub & Intermediary: High In-degree, High Out-degree, and High Betweenness
    2. Major Upstream Supplier / Source Hub: High Out-degree, In-degree == 0
    3. Major Downstream Consumer / Sink Hub: High In-degree, Out-degree == 0
    4. Critical Chokepoint / Bridge: Disproportionately high betweenness relative to degree
    5. Upstream Vendor: Out-degree > In-degree
    6. Downstream Buyer: In-degree > Out-degree
    7. Isolated Firm: Degree == 0
    8. Peripheral Node: Lower connectivity
    """
    in_d = row["in_degree"]
    out_d = row["out_degree"]
    betw = row["betweenness_centrality"]

    if in_d == 0 and out_d == 0:
        return "Isolated Firm"
    if in_d >= 10 and out_d >= 10 and betw > 0.005:
        return "Dual Hub & Intermediary"
    if in_d == 0 and out_d >= 5:
        return "Major Upstream Source Hub"
    if out_d == 0 and in_d >= 8:
        return "Major Downstream Sink Hub"
    if betw >= 0.0015 and (in_d < 8 or out_d < 8):
        return "Critical Intermediary Chokepoint"
    if out_d >= 5:
        return "Major Supplier / Distributor"
    if in_d >= 5:
        return "Major Consumer / Integrator"
    if out_d > in_d:
        return "Upstream Vendor"
    if in_d > out_d:
        return "Downstream Customer"
    return "Balanced Node"


def perform_critical_company_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-perspective comparison of companies based on:
    - Out-Degree (Direct Customer Reach)
    - In-Degree (Supplier Dependency)
    - Betweenness Centrality (Network Intermediation & Chokepoint Potential)

    NOTE ON MULTI-DIMENSIONAL CRITICALITY:
    Rather than imposing an arbitrary opaque risk score, this analysis provides:
    1. Ranks for each individual dimension (In-Degree Rank, Out-Degree Rank, Betweenness Rank)
    2. Documented Multi-Perspective Criticality Index (MPCI):
       Formula: MPCI = (1/3) * (In_Degree / max(In_Degree) + Out_Degree / max(Out_Degree) + Betweenness / max(Betweenness))
       Assumptions:
       - Gives equal (33.3%) weight to upstream exposure, downstream impact reach, and path intermediation.
       - Each component is min-max normalized by the empirical network maximum.
       - Disclosed transparently alongside raw underlying metrics so users can evaluate each dimension independently.
    """
    crit_df = df.copy()

    max_in = crit_df["in_degree"].max()
    max_out = crit_df["out_degree"].max()
    max_betw = crit_df["betweenness_centrality"].max()

    # Ranks (1 = highest)
    crit_df["in_degree_rank"] = crit_df["in_degree"].rank(ascending=False, method="min").astype(int)
    crit_df["out_degree_rank"] = crit_df["out_degree"].rank(ascending=False, method="min").astype(int)
    crit_df["betweenness_rank"] = crit_df["betweenness_centrality"].rank(ascending=False, method="min").astype(int)

    # Normalized component values for transparent comparison
    crit_df["norm_in_degree"] = crit_df["in_degree"] / max_in if max_in > 0 else 0.0
    crit_df["norm_out_degree"] = crit_df["out_degree"] / max_out if max_out > 0 else 0.0
    crit_df["norm_betweenness"] = crit_df["betweenness_centrality"] / max_betw if max_betw > 0 else 0.0

    # Documented Multi-Perspective Criticality Index (MPCI)
    crit_df["documented_mpci_score"] = (
        (1.0 / 3.0) * (crit_df["norm_in_degree"] + crit_df["norm_out_degree"] + crit_df["norm_betweenness"])
    ).round(6)

    # Classify structural roles
    crit_df["structural_role"] = crit_df.apply(
        lambda r: classify_structural_role(r, max_in, max_out, max_betw), axis=1
    )

    # Order columns logically
    ordered_cols = [
        "company_id", "company_name", "sector", "country", "structural_role",
        "betweenness_centrality", "betweenness_rank",
        "out_degree", "out_degree_rank",
        "in_degree", "in_degree_rank",
        "total_degree",
        "norm_betweenness", "norm_out_degree", "norm_in_degree",
        "documented_mpci_score", "is_source", "is_sink", "is_isolated"
    ]
    crit_df = crit_df[ordered_cols].sort_values(
        by=["documented_mpci_score", "betweenness_centrality", "out_degree"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return crit_df


def save_visualizations(df: pd.DataFrame, comp_metrics: dict, output_dir: Path):
    """
    Generate and save required network visualization plots as high-resolution PNGs:
    1. Top 10 Companies by Betweenness Centrality
    2. Top 10 Companies by Out-Degree
    3. Top 10 Companies by In-Degree
    4. Network Component Size Distribution
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # -------------------------------------------------------------
    # 1. Top 10 Companies by Betweenness Centrality
    # -------------------------------------------------------------
    top_betw = df.sort_values(by="betweenness_centrality", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.barh(
        [f"{name} ({cid})" for name, cid in zip(top_betw["company_name"], top_betw["company_id"])],
        top_betw["betweenness_centrality"],
        color="#2b5c8f",
        edgecolor="#1a365d",
        alpha=0.9
    )
    ax.set_title("Top 10 Companies by Betweenness Centrality (Chokepoints / Bridges)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Normalized Betweenness Centrality", fontsize=11, labelpad=8)
    ax.set_ylabel("Company", fontsize=11)
    
    # Annotate bar values
    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{w:.6f}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
            color="#1a202c"
        )
    ax.set_xlim(0, max(top_betw["betweenness_centrality"]) * 1.22)
    plt.tight_layout()
    plot_path1 = output_dir / "top_10_betweenness_centrality.png"
    plt.savefig(plot_path1)
    plt.close()

    # -------------------------------------------------------------
    # 2. Top 10 Companies by Out-Degree
    # -------------------------------------------------------------
    top_out = df.sort_values(by=["out_degree", "out_degree_centrality"], ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.barh(
        [f"{name} ({cid})" for name, cid in zip(top_out["company_name"], top_out["company_id"])],
        top_out["out_degree"],
        color="#20744a",
        edgecolor="#144d31",
        alpha=0.9
    )
    ax.set_title("Top 10 Companies by Out-Degree (Direct Customer Reach)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Out-Degree (Number of Customers Supplied)", fontsize=11, labelpad=8)
    ax.set_ylabel("Company", fontsize=11)
    
    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{int(w)}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#1a202c"
        )
    ax.set_xlim(0, max(top_out["out_degree"]) * 1.15)
    plt.tight_layout()
    plot_path2 = output_dir / "top_10_out_degree.png"
    plt.savefig(plot_path2)
    plt.close()

    # -------------------------------------------------------------
    # 3. Top 10 Companies by In-Degree
    # -------------------------------------------------------------
    top_in = df.sort_values(by=["in_degree", "in_degree_centrality"], ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.barh(
        [f"{name} ({cid})" for name, cid in zip(top_in["company_name"], top_in["company_id"])],
        top_in["in_degree"],
        color="#a83232",
        edgecolor="#701a1a",
        alpha=0.9
    )
    ax.set_title("Top 10 Companies by In-Degree (Supplier Dependency)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("In-Degree (Number of Direct Suppliers)", fontsize=11, labelpad=8)
    ax.set_ylabel("Company", fontsize=11)
    
    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{int(w)}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#1a202c"
        )
    ax.set_xlim(0, max(top_in["in_degree"]) * 1.15)
    plt.tight_layout()
    plot_path3 = output_dir / "top_10_in_degree.png"
    plt.savefig(plot_path3)
    plt.close()

    # -------------------------------------------------------------
    # 4. Network Component Size Distribution
    # -------------------------------------------------------------
    from collections import Counter
    size_counts = Counter(comp_metrics["wcc_sizes"])
    sorted_sizes = sorted(size_counts.keys())
    counts = [size_counts[s] for s in sorted_sizes]
    labels = [f"Size {s}" for s in sorted_sizes]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.bar(
        range(len(sorted_sizes)),
        counts,
        tick_label=labels,
        color="#6b46c1",
        edgecolor="#44337a",
        alpha=0.85,
        width=0.55
    )
    ax.set_title("Weakly Connected Component Size Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Component Size (Number of Nodes)", fontsize=11, labelpad=8)
    ax.set_ylabel("Frequency (Number of Components)", fontsize=11, labelpad=8)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{int(h)}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#1a202c"
        )
    ax.set_ylim(0, max(counts) * 1.2)
    plt.tight_layout()
    plot_path4 = output_dir / "component_size_distribution.png"
    plt.savefig(plot_path4)
    plt.close()

    return [plot_path1, plot_path2, plot_path3, plot_path4]


def print_terminal_summary(stats: dict, df: pd.DataFrame, crit_df: pd.DataFrame):
    """
    Display a comprehensive, readable terminal summary of all network statistics and Top 10 rankings.
    """
    divider = "=" * 80
    subdivider = "-" * 80

    print("\n" + divider)
    print("      SUPPLY CHAIN CASCADE SIMULATOR - ADVANCED NETWORK ANALYSIS")
    print(divider)

    print("\n[1] GLOBAL NETWORK TOPOLOGY & STATISTICS")
    print(subdivider)
    print(f"  Total Companies (Nodes):               {stats['total_nodes']}")
    print(f"  Total Supply Relationships (Edges):   {stats['total_edges']}")
    print(f"  Network Density:                      {stats['network_density']:.6f}")
    print(f"  Average In-Degree:                    {stats['average_in_degree']:.4f}")
    print(f"  Average Out-Degree:                   {stats['average_out_degree']:.4f}")
    print(f"  Is Directed Acyclic Graph (DAG):      {stats['is_dag']}")
    print(f"  Weakly Connected Components (WCC):    {stats['number_of_weakly_connected_components']}")
    print(f"  Strongly Connected Components (SCC):  {stats['number_of_strongly_connected_components']}")
    print(f"  Largest Weakly Connected Component:   {stats['size_of_largest_weakly_connected_component']} nodes ({(stats['size_of_largest_weakly_connected_component'] / stats['total_nodes'] * 100):.1f}% of network)")
    print(f"  Isolated Companies (Degree = 0):      {stats['number_of_isolated_companies']}")
    print(f"  Source Nodes (In-Degree = 0):         {stats['number_of_source_nodes']} (pure suppliers / initial tier)")
    print(f"  Sink Nodes (Out-Degree = 0):          {stats['number_of_sink_nodes']} (pure consumers / end-assemblers)")
    print(f"  WCC Size Distribution:                {stats['weakly_connected_component_size_distribution']}")

    print("\n[2] TOP 10 COMPANIES BY BETWEENNESS CENTRALITY (CRITICAL CHOKEPOINTS)")
    print(subdivider)
    top_betw = df.sort_values(by="betweenness_centrality", ascending=False).head(10)
    print(f"  {'Rank':<5} {'ID':<8} {'Company Name':<30} {'Betweenness':<14} {'In-Deg':<8} {'Out-Deg':<8} {'Sector'}")
    print(f"  {'-'*4} {'-'*6} {'-'*28} {'-'*12} {'-'*6} {'-'*6} {'-'*15}")
    for idx, (_, row) in enumerate(top_betw.iterrows(), start=1):
        print(f"  {idx:<5} {row['company_id']:<8} {row['company_name'][:28]:<30} {row['betweenness_centrality']:<14.6f} {row['in_degree']:<8} {row['out_degree']:<8} {row['sector']}")

    print("\n[3] TOP 10 COMPANIES BY OUT-DEGREE (DIRECT CUSTOMER REACH)")
    print(subdivider)
    top_out = df.sort_values(by=["out_degree", "out_degree_centrality"], ascending=False).head(10)
    print(f"  {'Rank':<5} {'ID':<8} {'Company Name':<30} {'Out-Degree':<12} {'Out-Cent':<12} {'Betweenness':<14} {'Sector'}")
    print(f"  {'-'*4} {'-'*6} {'-'*28} {'-'*10} {'-'*10} {'-'*12} {'-'*15}")
    for idx, (_, row) in enumerate(top_out.iterrows(), start=1):
        print(f"  {idx:<5} {row['company_id']:<8} {row['company_name'][:28]:<30} {row['out_degree']:<12} {row['out_degree_centrality']:<12.6f} {row['betweenness_centrality']:<14.6f} {row['sector']}")

    print("\n[4] TOP 10 COMPANIES BY IN-DEGREE (SUPPLIER DEPENDENCY)")
    print(subdivider)
    top_in = df.sort_values(by=["in_degree", "in_degree_centrality"], ascending=False).head(10)
    print(f"  {'Rank':<5} {'ID':<8} {'Company Name':<30} {'In-Degree':<12} {'In-Cent':<12} {'Betweenness':<14} {'Sector'}")
    print(f"  {'-'*4} {'-'*6} {'-'*28} {'-'*10} {'-'*10} {'-'*12} {'-'*15}")
    for idx, (_, row) in enumerate(top_in.iterrows(), start=1):
        print(f"  {idx:<5} {row['company_id']:<8} {row['company_name'][:28]:<30} {row['in_degree']:<12} {row['in_degree_centrality']:<12.6f} {row['betweenness_centrality']:<14.6f} {row['sector']}")

    print("\n[5] CRITICAL COMPANY COMPARISON & STRUCTURAL ARCHETYPES")
    print(subdivider)
    print("  Key Archetypes Identified:")
    print("  * Dual Hubs & Bridges (e.g. Amazon, Microsoft): Both critical suppliers and major aggregators.")
    print("  * Critical Chokepoints (e.g. Nvidia, Equinix, Intel, Micron, AMD): Vital bridges connecting tiers.")
    print("  * Pure Out-Degree Hubs (e.g. Visa, S&P Global, Teledyne): Many customers but 0 betweenness (sources).")
    print("  * Pure In-Degree Sinks (e.g. Lockheed Martin, Boeing): Highly dependent on suppliers, 0 betweenness.")
    print("\n  Top 10 Multi-Perspective Critical Companies:")
    print(f"  {'Rank':<5} {'ID':<8} {'Company Name':<28} {'Role':<32} {'MPCI':<8} {'In-D':<6} {'Out-D':<6} {'Betw'}")
    print(f"  {'-'*4} {'-'*6} {'-'*26} {'-'*30} {'-'*6} {'-'*4} {'-'*4} {'-'*10}")
    for idx, (_, row) in enumerate(crit_df.head(10).iterrows(), start=1):
        print(f"  {idx:<5} {row['company_id']:<8} {row['company_name'][:26]:<28} {row['structural_role']:<32} {row['documented_mpci_score']:<8.4f} {row['in_degree']:<6} {row['out_degree']:<6} {row['betweenness_centrality']:.6f}")

    print("\n" + divider + "\n")


def run_advanced_network_analysis():
    """
    Main orchestration function:
    - Builds network using build_network
    - Computes all degree, betweenness, component, and global statistics
    - Generates and writes all CSV, JSON, and PNG deliverables
    - Prints full terminal summary
    """
    results_dir = get_results_dir()

    # 1. Reuse existing graph builder
    G = build_network()

    # Verify input preservation
    assert G.number_of_nodes() == 250, f"Expected 250 nodes, got {G.number_of_nodes()}"
    assert G.number_of_edges() == 445, f"Expected 445 edges, got {G.number_of_edges()}"

    # 2. Compute metrics
    degree_metrics = calculate_degree_metrics(G)
    betweenness = calculate_betweenness_metrics(G)
    component_metrics = calculate_components_metrics(G)
    stats = calculate_network_statistics(G, degree_metrics, component_metrics)

    # 3. Build dataframes
    full_df = build_full_metrics_dataframe(G, degree_metrics, betweenness, component_metrics)
    crit_df = perform_critical_company_analysis(full_df)

    # Prepare specific betweenness dataframe sorted descending
    betweenness_df = full_df[["company_id", "company_name", "sector", "country", "betweenness_centrality", "in_degree", "out_degree", "total_degree"]].copy()
    betweenness_df.insert(0, "rank", range(1, len(betweenness_df) + 1))

    # Add Top 10 lists to JSON summary
    top_10_betw_records = betweenness_df.head(10).to_dict(orient="records")
    top_10_out_records = full_df.sort_values(by=["out_degree", "out_degree_centrality"], ascending=False).head(10)[
        ["company_id", "company_name", "sector", "out_degree", "out_degree_centrality", "betweenness_centrality"]
    ].reset_index(drop=True)
    top_10_out_records.insert(0, "rank", range(1, 11))

    top_10_in_records = full_df.sort_values(by=["in_degree", "in_degree_centrality"], ascending=False).head(10)[
        ["company_id", "company_name", "sector", "in_degree", "in_degree_centrality", "betweenness_centrality"]
    ].reset_index(drop=True)
    top_10_in_records.insert(0, "rank", range(1, 11))

    stats["top_10_betweenness_centrality"] = top_10_betw_records
    stats["top_10_out_degree"] = top_10_out_records.to_dict(orient="records")
    stats["top_10_in_degree"] = top_10_in_records.to_dict(orient="records")

    # 4. Save CSV and JSON deliverables
    advanced_metrics_path = results_dir / "advanced_network_metrics.csv"
    betweenness_path = results_dir / "betweenness_centrality.csv"
    crit_path = results_dir / "critical_company_analysis.csv"
    summary_path = results_dir / "network_summary.json"

    full_df.to_csv(advanced_metrics_path, index=False)
    betweenness_df.to_csv(betweenness_path, index=False)
    crit_df.to_csv(crit_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    # 5. Generate and save PNG visualizations
    plots = save_visualizations(full_df, component_metrics, results_dir)

    # 6. Terminal Summary
    print_terminal_summary(stats, full_df, crit_df)

    print(f"Results successfully saved to: {results_dir}")
    print(f" - CSV: {advanced_metrics_path.name}")
    print(f" - CSV: {betweenness_path.name}")
    print(f" - CSV: {crit_path.name}")
    print(f" - JSON: {summary_path.name}")
    for p in plots:
        print(f" - Chart: {p.name}")

    return {
        "stats": stats,
        "full_df": full_df,
        "crit_df": crit_df,
        "betweenness_df": betweenness_df,
        "results_dir": str(results_dir)
    }


if __name__ == "__main__":
    run_advanced_network_analysis()

