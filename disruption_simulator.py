"""
Supply Chain Cascade Simulator - Supplier Disruption Simulation Engine
=======================================================================
This module simulates supply chain disruptions by evaluating the downstream
propagation of a supplier firm's unavailability.

Key Features & Methodological Principles:
1. Graph Isolation: Clones the master graph (G.copy()) for every simulation.
   The original master graph is strictly read-only and never mutated.
2. Case-Insensitive Entity Resolution: Supports company ID or company name.
3. Breadth-First Search (BFS) Propagation:
   Traces topological downstream cascade paths level-by-level (Tier 1, Tier 2, etc.).
4. Exposure Classification:
   - Direct Exposure (Tier 1): Immediate customers of the disrupted supplier.
   - Indirect Exposure (Tier 2+): Customers of customers reached transitively.
5. Multi-Sourcing & Structural Redundancy Check:
   - Evaluates whether exposed firms have alternative recorded supplier relationships.
   - Does NOT assume all downstream reachable firms experience operational failure.
   - Distinguishes 'Sole Dependency' from 'Multi-Sourced / Buffered' firms.
6. Publication-Quality Visualization:
   Generates dual-panel visualizations showing cascade topology and tier resilience.
7. Explicit Limitations & Scientific Assumptions:
   - Dataset captures structural topology only (no inventory buffers, lead times,
     production volumes, contract values, or dynamic rerouting capacities).
   - Metrics reflect structural vulnerability and reach, not financial losses.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import deque, Counter
from typing import Dict, List, Tuple, Any, Optional

import matplotlib
matplotlib.use("Agg")  # Headless non-interactive rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import networkx as nx

# Reuse existing graph builder and data resolver
from build_network import build_network, resolve_data_paths


# Documented Scientific Assumptions & Limitations
SIMULATION_ASSUMPTIONS = {
    "structural_reach_only": (
        "Cascade reach is determined strictly by topological directed paths. "
        "A reachable downstream firm has structural exposure, but not necessarily "
        "operational failure."
    ),
    "absence_of_flow_data": (
        "The Sectivia dataset lacks transaction volumes, inventory buffer stock (days of inventory), "
        "order frequency, component criticality, and contract terms. All relationships are "
        "treated as unweighted binary dependencies."
    ),
    "no_instantaneous_rerouting": (
        "The model reports existing recorded alternative suppliers. It does not model "
        "dynamic spot-market contracting, supplier switching costs, or qualified replacement suppliers."
    ),
    "non_deterministic_impact": (
        "Firms with alternative recorded suppliers may absorb or mitigate the disruption. "
        "Firms with sole dependency on the disrupted supplier represent the highest structural risk."
    ),
    "no_financial_claims": (
        "This simulation does not estimate dollar financial losses, revenue impacts, or stock price changes."
    )
}


def get_results_dir() -> Path:
    """Ensure and return the output results/simulation directory."""
    # Place disruption simulation results in results/simulation subfolder
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results" / "simulation"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def resolve_company_node(G: nx.DiGraph, query: str) -> str:
    """
    Resolve a user-provided company query (ID or company name) to a canonical node ID.
    Supports case-insensitive matching and partial matching.
    """
    clean_q = query.strip().lower()

    # 1. Exact node ID match (case-insensitive)
    id_map = {str(n).lower(): n for n in G.nodes()}
    if clean_q in id_map:
        return id_map[clean_q]

    # 2. Exact company name match (case-insensitive)
    name_map = {}
    for n, d in G.nodes(data=True):
        nm = d.get("name", "")
        if nm:
            name_map[nm.lower()] = n
    if clean_q in name_map:
        return name_map[clean_q]

    # 3. Substring match in company name
    partial_matches = [
        (n, d.get("name", n)) for n, d in G.nodes(data=True)
        if clean_q in d.get("name", "").lower() or clean_q in str(n).lower()
    ]
    if len(partial_matches) == 1:
        return partial_matches[0][0]
    elif len(partial_matches) > 1:
        options = ", ".join([f"{name} ({nid})" for nid, name in partial_matches[:5]])
        raise ValueError(
            f"Ambiguous query '{query}'. Multiple matches found: {options}"
        )

    raise KeyError(f"Company '{query}' was not found in the supply chain network.")


class DisruptionSimulator:
    """
    Engine for simulating single-supplier unavailability and evaluating downstream cascade reach.
    """

    def __init__(self, master_graph: Optional[nx.DiGraph] = None):
        if master_graph is None:
            self.master_graph = build_network()
        else:
            self.master_graph = master_graph

        # Store immutable node attributes
        self.node_names = nx.get_node_attributes(self.master_graph, "name")
        self.node_sectors = nx.get_node_attributes(self.master_graph, "sector")
        self.node_countries = nx.get_node_attributes(self.master_graph, "country")
        self.node_segments = nx.get_node_attributes(self.master_graph, "segment")

    def run_simulation(self, target_query: str) -> Dict[str, Any]:
        """
        Execute a disruption simulation on a target company:
        1. Resolve target company.
        2. Create an isolated copy of the graph.
        3. Identify direct customers (Tier 1).
        4. Run BFS downstream to record cascade depth / tiers.
        5. Evaluate alternative supplier paths for every exposed firm.
        6. Compute summary metrics and build data tables.
        """
        target_id = resolve_company_node(self.master_graph, target_query)
        target_name = self.node_names.get(target_id, target_id)
        target_sector = self.node_sectors.get(target_id, "Unknown")
        target_country = self.node_countries.get(target_id, "Unknown")

        # Snapshot baseline graph counts to verify non-mutation
        orig_node_count = self.master_graph.number_of_nodes()
        orig_edge_count = self.master_graph.number_of_edges()

        # Step 4: Create isolated copy of graph
        sim_graph = self.master_graph.copy()

        # Direct customers in original graph
        direct_customers = list(self.master_graph.successors(target_id))

        # Step 7: BFS downstream traversal to trace reachable companies and cascade levels
        # Queue stores: (current_node, current_level, path_from_target)
        visited_depth: Dict[str, int] = {target_id: 0}
        parent_map: Dict[str, str] = {}
        sample_path: Dict[str, List[str]] = {target_id: [target_id]}

        queue = deque([target_id])
        while queue:
            curr = queue.popleft()
            curr_lvl = visited_depth[curr]
            for succ in sim_graph.successors(curr):
                if succ not in visited_depth:
                    visited_depth[succ] = curr_lvl + 1
                    parent_map[succ] = curr
                    sample_path[succ] = sample_path[curr] + [succ]
                    queue.append(succ)

        # Downstream exposed companies (excluding the target company itself)
        downstream_nodes = [n for n in visited_depth if n != target_id]
        downstream_set = set(downstream_nodes)

        # Classify exposure levels and supplier alternatives
        exposed_records = []
        tier_counts: Dict[int, int] = Counter()
        tier_sole_dep_counts: Dict[int, int] = Counter()
        tier_with_alt_counts: Dict[int, int] = Counter()

        num_sole_dependent = 0
        num_with_alt_suppliers = 0
        num_with_unaffected_suppliers = 0

        for comp_id in downstream_nodes:
            depth = visited_depth[comp_id]
            tier_counts[depth] += 1
            is_direct = (depth == 1)

            # All incoming suppliers in master graph
            all_suppliers = set(self.master_graph.predecessors(comp_id))
            total_suppliers = len(all_suppliers)

            # Direct alternative suppliers excluding the disrupted target node
            alt_suppliers_direct = all_suppliers - {target_id}
            num_alt_direct = len(alt_suppliers_direct)

            # Unaffected suppliers outside the entire downstream cascade
            unaffected_suppliers = all_suppliers - downstream_set - {target_id}
            num_unaffected = len(unaffected_suppliers)

            # Redundancy classification
            if num_alt_direct == 0:
                redundancy_status = "Sole Dependency"
                num_sole_dependent += 1
                tier_sole_dep_counts[depth] += 1
            else:
                num_with_alt_suppliers += 1
                tier_with_alt_counts[depth] += 1
                if num_unaffected > 0:
                    redundancy_status = "Multi-Sourced (Buffered)"
                else:
                    redundancy_status = "Multi-Sourced (Cascade Constrained)"

            if num_unaffected > 0:
                num_with_unaffected_suppliers += 1

            # Format supplier names for reporting
            alt_names = [self.node_names.get(s, s) for s in sorted(alt_suppliers_direct)]
            unaffected_names = [self.node_names.get(s, s) for s in sorted(unaffected_suppliers)]

            exposed_records.append({
                "company_id": comp_id,
                "company_name": self.node_names.get(comp_id, comp_id),
                "sector": self.node_sectors.get(comp_id, "Unknown"),
                "country": self.node_countries.get(comp_id, "Unknown"),
                "cascade_depth": depth,
                "exposure_type": "Direct Customer (Tier 1)" if is_direct else f"Indirect Exposure (Tier {depth})",
                "is_direct_customer": is_direct,
                "total_recorded_suppliers": total_suppliers,
                "alternative_suppliers_count": num_alt_direct,
                "unaffected_suppliers_count": num_unaffected,
                "has_alternative_suppliers": (num_alt_direct > 0),
                "is_sole_dependent": (num_alt_direct == 0),
                "redundancy_status": redundancy_status,
                "immediate_supplier_in_cascade": self.node_names.get(parent_map.get(comp_id, ""), parent_map.get(comp_id, "")),
                "propagation_path": " -> ".join([self.node_names.get(p, p) for p in sample_path[comp_id]]),
                "alternative_suppliers_list": "; ".join(alt_names) if alt_names else "None",
                "unaffected_suppliers_list": "; ".join(unaffected_names) if unaffected_names else "None"
            })

        # Build DataFrame
        exposed_df = pd.DataFrame(exposed_records)
        if not exposed_df.empty:
            exposed_df = exposed_df.sort_values(
                by=["cascade_depth", "is_sole_dependent", "total_recorded_suppliers"],
                ascending=[True, False, True]
            ).reset_index(drop=True)

        # Compute summary metrics
        total_direct = len(direct_customers)
        total_downstream = len(downstream_nodes)
        total_indirect = total_downstream - total_direct
        max_depth = max(visited_depth.values()) if downstream_nodes else 0

        # Percentage of network affected
        pct_network_exposed = round((total_downstream / orig_node_count) * 100, 2)

        # Verify master graph was not modified
        assert self.master_graph.number_of_nodes() == orig_node_count, "Master graph node count altered!"
        assert self.master_graph.number_of_edges() == orig_edge_count, "Master graph edge count altered!"

        simulation_result = {
            "target_company_id": target_id,
            "target_company_name": target_name,
            "target_sector": target_sector,
            "target_country": target_country,
            "total_direct_customers": total_direct,
            "total_downstream_reachable": total_downstream,
            "total_indirectly_exposed": total_indirect,
            "maximum_cascade_depth": max_depth,
            "companies_with_alternative_suppliers": num_with_alt_suppliers,
            "companies_with_unaffected_suppliers": num_with_unaffected_suppliers,
            "solely_dependent_companies": num_sole_dependent,
            "percentage_of_network_exposed": pct_network_exposed,
            "tier_distribution": dict(tier_counts),
            "tier_sole_dependency_distribution": dict(tier_sole_dep_counts),
            "tier_with_alternative_distribution": dict(tier_with_alt_counts),
            "direct_customers_list": [self.node_names.get(c, c) for c in direct_customers],
            "assumptions": SIMULATION_ASSUMPTIONS,
            "exposed_df": exposed_df
        }

        return simulation_result


def generate_disruption_visualization(sim_result: Dict[str, Any], G: nx.DiGraph, output_dir: Path) -> Path:
    """
    Generate a publication-quality dual-panel figure for the disruption scenario:
    - Left Panel: Hierarchical Cascade Subgraph (Tiers arranged left-to-right,
                  color-coded by exposure level and dependency risk).
    - Right Panel: Tier Resilience & Exposure Breakdown Bar Chart.
    """
    target_id = sim_result["target_company_id"]
    target_name = sim_result["target_company_name"]
    max_depth = sim_result["maximum_cascade_depth"]
    exposed_df = sim_result["exposed_df"]

    fig = plt.figure(figsize=(15, 7.5), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.28)

    # -------------------------------------------------------------
    # Panel 1: Cascade Subgraph
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Extract subgraph of target and all downstream nodes
    nodes_in_subgraph = [target_id] + list(exposed_df["company_id"])
    subG = G.subgraph(nodes_in_subgraph).copy()

    # Assign coordinates based on cascade depth
    depth_map = {target_id: 0}
    for _, row in exposed_df.iterrows():
        depth_map[row["company_id"]] = row["cascade_depth"]

    # Calculate hierarchical layout positions: x = depth, y = spaced within depth
    tier_nodes: Dict[int, List[str]] = {}
    for n in nodes_in_subgraph:
        d = depth_map[n]
        tier_nodes.setdefault(d, []).append(n)

    pos = {}
    for d, n_list in tier_nodes.items():
        # Sort nodes within tier: sole-dependent first, then by name
        if d == 0:
            pos[n_list[0]] = (0.0, 0.0)
        else:
            k = len(n_list)
            y_coords = [-(i - (k - 1) / 2.0) * (2.2 / max(k, 1)) for i in range(k)]
            for i, nid in enumerate(n_list):
                pos[nid] = (float(d) * 2.5, y_coords[i])

    # Node coloring and styling
    node_colors = []
    edgecolors = []
    node_sizes = []

    # Map node to sole-dependency status
    sole_dep_map = dict(zip(exposed_df["company_id"], exposed_df["is_sole_dependent"]))

    for n in subG.nodes():
        if n == target_id:
            node_colors.append("#dc2626")  # Vibrant Red
            edgecolors.append("#7f1d1d")
            node_sizes.append(900)
        elif sole_dep_map.get(n, False):
            # Sole dependent node: Coral/Salmon with dark red border
            node_colors.append("#f87171")
            edgecolors.append("#991b1b")
            node_sizes.append(500)
        else:
            # Multi-sourced node: Soft Teal/Sky Blue with slate border
            node_colors.append("#38bdf8")
            edgecolors.append("#0369a1")
            node_sizes.append(420)

    # Draw edges with soft gray arrows
    nx.draw_networkx_edges(
        subG, pos, ax=ax1,
        edge_color="#94a3b8",
        arrows=True,
        arrowsize=12,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        width=1.0,
        alpha=0.65
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        subG, pos, ax=ax1,
        node_color=node_colors,
        edgecolors=edgecolors,
        linewidths=1.8,
        node_size=node_sizes,
        alpha=0.92
    )

    # Label nodes (display name for root and key nodes; company ID for compact clarity)
    labels = {}
    for n in subG.nodes():
        nm = G.nodes[n].get("name", n)
        if n == target_id:
            labels[n] = f"{nm}\n[ORIGIN]"
        elif len(subG.nodes()) <= 25:
            labels[n] = f"{nm[:14]}"
        else:
            labels[n] = n

    font_size = 9 if len(subG.nodes()) <= 25 else 7.5
    nx.draw_networkx_labels(
        subG, pos, labels=labels, ax=ax1,
        font_size=font_size,
        font_weight="bold",
        font_color="#0f172a"
    )

    ax1.set_title(
        f"Propagation Topology: {target_name} ({sim_result['total_downstream_reachable']} Downstream Reachable)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax1.axis("off")

    # Legend for Panel 1
    legend_handles = [
        mpatches.Patch(color="#dc2626", label=f"Disrupted Supplier: {target_name}"),
        mpatches.Patch(color="#f87171", label="Sole Dependent (No Other Suppliers)"),
        mpatches.Patch(color="#38bdf8", label="Multi-Sourced (Alternative Suppliers Exist)")
    ]
    ax1.legend(handles=legend_handles, loc="lower left", fontsize=8.5, frameon=True, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel 2: Tier Breakdown & Resilience Chart
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    
    tier_dist = sim_result["tier_distribution"]
    tier_sole = sim_result["tier_sole_dependency_distribution"]
    tier_alt = sim_result["tier_with_alternative_distribution"]

    if tier_dist:
        tiers = sorted(tier_dist.keys())
        sole_vals = [tier_sole.get(t, 0) for t in tiers]
        alt_vals = [tier_alt.get(t, 0) for t in tiers]
        labels_tier = [f"Tier {t}" + (" (Direct)" if t == 1 else "") for t in tiers]

        x_indices = range(len(tiers))
        width = 0.55

        ax2.bar(x_indices, sole_vals, width, label="Sole Dependency (High Risk)", color="#ef4444", edgecolor="#991b1b", alpha=0.9)
        ax2.bar(x_indices, alt_vals, width, bottom=sole_vals, label="Has Alternative Suppliers", color="#0ea5e9", edgecolor="#0369a1", alpha=0.9)

        # Annotate totals on top of bars
        for idx in x_indices:
            total_h = sole_vals[idx] + alt_vals[idx]
            sole_h = sole_vals[idx]
            alt_h = alt_vals[idx]
            ax2.text(idx, total_h + 0.3, f"Total: {total_h}\n({sole_h} sole / {alt_h} alt)", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax2.set_xticks(x_indices)
        ax2.set_xticklabels(labels_tier, fontsize=9.5)
        ax2.set_ylabel("Number of Exposed Companies", fontsize=10, fontweight="bold")
        ax2.set_ylim(0, max([s + a for s, a in zip(sole_vals, alt_vals)] or [1]) * 1.35)
    else:
        ax2.text(0.5, 0.5, "No Downstream Reachable Customers\n(Company is a Pure Sink / Final Consumer)", ha="center", va="center", fontsize=11, color="#64748b")
        ax2.set_axis_off()

    ax2.set_title(
        f"Downstream Exposure by Cascade Tier\n(Total Exposed: {sim_result['total_downstream_reachable']})",
        fontsize=11.5, fontweight="bold", pad=12
    )
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Global Footnote Documenting Assumptions
    plt.figtext(
        0.5, 0.02,
        "Methodology Note: Downstream reach indicates topological exposure. Firms with alternative recorded suppliers may mitigate impacts. "
        "Inventory buffers and contract quantities are unrecorded in this dataset.",
        ha="center", fontsize=7.5, color="#475569", style="italic"
    )

    chart_filename = f"disruption_cascade_{target_id}.png"
    chart_path = output_dir / chart_filename
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()

    return chart_path


def save_simulation_artifacts(sim_result: Dict[str, Any], output_dir: Path) -> Tuple[Path, Path]:
    """
    Save simulation deliverables:
    1. Detailed cascade nodes table as CSV.
    2. Comprehensive simulation summary as JSON.
    """
    target_id = sim_result["target_company_id"]
    csv_path = output_dir / f"disruption_cascade_{target_id}.csv"
    json_path = output_dir / f"disruption_summary_{target_id}.json"

    # Save CSV
    df: pd.DataFrame = sim_result["exposed_df"]
    df.to_csv(csv_path, index=False)

    # Prepare JSON serializable dictionary
    json_payload = {k: v for k, v in sim_result.items() if k != "exposed_df"}
    # Add top exposed records to JSON
    json_payload["exposed_companies_sample"] = df.head(15).to_dict(orient="records") if not df.empty else []

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=4)

    return csv_path, json_path


def print_simulation_terminal_summary(sim_result: Dict[str, Any]):
    """
    Display a clear, structured terminal report for the simulated disruption scenario.
    """
    divider = "=" * 80
    subdivider = "-" * 80

    print("\n" + divider)
    print(f"   SUPPLIER DISRUPTION SIMULATION: {sim_result['target_company_name'].upper()} ({sim_result['target_company_id']})")
    print(divider)

    print("\n[1] SCENARIO TARGET INFORMATION")
    print(subdivider)
    print(f"  Target Company Name:                   {sim_result['target_company_name']}")
    print(f"  Target Company ID:                     {sim_result['target_company_id']}")
    print(f"  Sector:                                {sim_result['target_sector']}")
    print(f"  Country:                               {sim_result['target_country']}")

    print("\n[2] CASCADE PROPAGATION METRICS")
    print(subdivider)
    print(f"  Direct Customers (Tier 1):             {sim_result['total_direct_customers']}")
    print(f"  Total Downstream Reachable:            {sim_result['total_downstream_reachable']} companies ({sim_result['percentage_of_network_exposed']}% of entire network)")
    print(f"  Indirectly Exposed (Tier 2+):          {sim_result['total_indirectly_exposed']} companies")
    print(f"  Maximum Cascade Depth:                 Tier {sim_result['maximum_cascade_depth']}")
    print(f"  Tier Breakdown (Level Distribution):   {sim_result['tier_distribution']}")

    print("\n[3] RESILIENCE & MULTI-SOURCING ANALYSIS")
    print(subdivider)
    print(f"  Companies with Alternative Suppliers: {sim_result['companies_with_alternative_suppliers']} / {sim_result['total_downstream_reachable']}")
    print(f"  Companies with Unaffected Suppliers:   {sim_result['companies_with_unaffected_suppliers']} / {sim_result['total_downstream_reachable']} (suppliers outside cascade)")
    print(f"  Solely Dependent on Disrupted Firm:    {sim_result['solely_dependent_companies']} (highest structural vulnerability)")
    print(f"  Sole Dependency by Tier:               {sim_result['tier_sole_dependency_distribution']}")

    exposed_df = sim_result["exposed_df"]
    if not exposed_df.empty:
        print("\n[4] SAMPLE DOWNSTREAM EXPOSED COMPANIES (FIRST 10)")
        print(subdivider)
        print(f"  {'Tier':<6} {'ID':<8} {'Company Name':<28} {'Redundancy Status':<30} {'Total-Suppliers'}")
        print(f"  {'-'*4} {'-'*6} {'-'*26} {'-'*28} {'-'*15}")
        for _, row in exposed_df.head(10).iterrows():
            print(f"  {row['cascade_depth']:<6} {row['company_id']:<8} {row['company_name'][:26]:<28} {row['redundancy_status']:<30} {row['total_recorded_suppliers']}")

    print("\n[5] SCIENTIFIC ASSUMPTIONS & LIMITATIONS")
    print(subdivider)
    for k, v in sim_result["assumptions"].items():
        print(f"  * {k}: {v}")

    print("\n" + divider + "\n")


def run_comparative_suite(companies: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run simulations across multiple benchmark companies and produce comparative cross-scenario tables.
    """
    simulator = DisruptionSimulator()
    results_dir = get_results_dir()

    summary_rows = []
    full_scenarios_dict = {}

    print("\n" + "=" * 80)
    print("      EXECUTING SUPPLIER DISRUPTION SIMULATION TEST SUITE")
    print("=" * 80)

    for comp in companies:
        sim_res = simulator.run_simulation(comp)
        print_simulation_terminal_summary(sim_res)
        
        # Save artifacts
        csv_p, json_p = save_simulation_artifacts(sim_res, results_dir)
        chart_p = generate_disruption_visualization(sim_res, simulator.master_graph, results_dir)
        print(f"  Saved artifacts for {sim_res['target_company_name']}:")
        print(f"   - CSV:   {csv_p.name}")
        print(f"   - JSON:  {json_p.name}")
        print(f"   - Chart: {chart_p.name}\n")

        summary_rows.append({
            "target_id": sim_res["target_company_id"],
            "target_name": sim_res["target_company_name"],
            "sector": sim_res["target_sector"],
            "direct_customers": sim_res["total_direct_customers"],
            "downstream_reachable": sim_res["total_downstream_reachable"],
            "indirectly_exposed": sim_res["total_indirectly_exposed"],
            "max_cascade_depth": sim_res["maximum_cascade_depth"],
            "with_alternative_suppliers": sim_res["companies_with_alternative_suppliers"],
            "with_unaffected_suppliers": sim_res["companies_with_unaffected_suppliers"],
            "solely_dependent": sim_res["solely_dependent_companies"],
            "pct_network_exposed": sim_res["percentage_of_network_exposed"]
        })

        # Keep JSON version
        clean_json = {k: v for k, v in sim_res.items() if k != "exposed_df"}
        full_scenarios_dict[sim_res["target_company_id"]] = clean_json

    comp_df = pd.DataFrame(summary_rows)
    comp_csv_path = results_dir / "disruption_comparative_summary.csv"
    comp_df.to_csv(comp_csv_path, index=False)

    all_json_path = results_dir / "disruption_all_scenarios.json"
    with open(all_json_path, "w", encoding="utf-8") as f:
        json.dump(full_scenarios_dict, f, indent=4)

    print("=" * 80)
    print("COMPARATIVE SCENARIOS SUMMARY TABLE:")
    print(comp_df.to_string(index=False))
    print("=" * 80)
    print(f"Comparative summary saved to: {comp_csv_path.name}")
    print(f"All scenarios JSON saved to: {all_json_path.name}\n")

    return comp_df, full_scenarios_dict


def main():
    parser = argparse.ArgumentParser(description="Supply Chain Cascade Simulator - Supplier Disruption Simulation")
    parser.add_argument("--company", type=str, default=None, help="Company ID or name to simulate (e.g. 'Amazon', 'nvda', 'Microsoft')")
    parser.add_argument("--suite", action="store_true", help="Run comparative benchmark suite on Amazon, Microsoft, Nvidia")
    args = parser.parse_args()

    results_dir = get_results_dir()

    if args.company:
        simulator = DisruptionSimulator()
        sim_res = simulator.run_simulation(args.company)
        print_simulation_terminal_summary(sim_res)
        csv_p, json_p = save_simulation_artifacts(sim_res, results_dir)
        chart_p = generate_disruption_visualization(sim_res, simulator.master_graph, results_dir)
        print(f"Deliverables saved to {results_dir}:")
        print(f" - CSV:   {csv_p.name}")
        print(f" - JSON:  {json_p.name}")
        print(f" - Chart: {chart_p.name}")
    else:
        # Default behavior: run benchmark suite on Amazon, Microsoft, Nvidia and additional test companies
        test_firms = ["Amazon", "Microsoft", "Nvidia"]
        run_comparative_suite(test_firms)


if __name__ == "__main__":
    main()

