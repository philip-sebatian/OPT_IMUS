"""
Streamlit-powered dashboard for generating random Optimus scenarios
and visualising the results on an animated map.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import streamlit as st
from streamlit import components

from workspace_optimus import CuOptEnhancedPlanner
from workspace_optimus.visualization.web_map import create_interactive_route_map
from workspace_optimus.visualization.graph_animation import create_graph_animation_html


def make_complete_graph(coords: np.ndarray):
    node_count = coords.shape[0]
    offsets = [0]
    edges = []
    weights = []
    for origin in range(node_count):
        for destination in range(node_count):
            if origin == destination:
                continue
            edges.append(destination)
            distance = float(np.linalg.norm(coords[origin] - coords[destination]))
            weights.append(distance)
        offsets.append(len(edges))
    return (
        np.array(offsets, dtype=int),
        np.array(edges, dtype=int),
        np.array(weights, dtype=float),
    )


def build_random_problem(
    num_depots: int,
    num_stores: int,
    num_vehicles: int,
    seed: int,
):
    rng = np.random.default_rng(seed)

    base_lat, base_lon = 37.7749, -122.4194  # San Francisco anchor
    depot_angles = np.linspace(0, 2 * np.pi, num_depots, endpoint=False)
    depot_coords = np.column_stack(
        (
            base_lat + 0.3 * np.sin(depot_angles),
            base_lon + 0.3 * np.cos(depot_angles),
        )
    )

    store_coords = rng.uniform(
        low=[base_lat - 0.25, base_lon - 0.35],
        high=[base_lat + 0.25, base_lon + 0.35],
        size=(num_stores, 2),
    )

    coords = np.vstack((depot_coords, store_coords))
    offsets, edges, weights = make_complete_graph(coords)

    target_locations = np.arange(num_depots, num_depots + num_stores)
    depot_locations = list(range(num_depots))
    vehicle_capacities = rng.integers(45, 80, size=num_vehicles).tolist()
    initial_stock = [
        int(max(5, cap - rng.integers(0, int(cap * 0.3) + 1))) for cap in vehicle_capacities
    ]
    delivery_demands = rng.integers(5, 25, size=num_stores)

    vehicle_start_locations = rng.choice(
        depot_locations,
        size=num_vehicles,
        replace=True,
    ).tolist()

    location_coordinates: Dict[int, Tuple[float, float]] = {
        node: (coords[node, 0], coords[node, 1]) for node in range(len(coords))
    }

    planner = CuOptEnhancedPlanner(
        offsets=offsets,
        edges=edges,
        weights=weights,
        time_to_travel=weights.copy(),
        target_locations=target_locations,
        depot_locations=depot_locations,
        vehicle_capacities=vehicle_capacities,
        initial_stock=initial_stock,
        delivery_demands=delivery_demands,
        solver_time_limit=30.0,
        vehicle_start_locations=vehicle_start_locations,
    )

    planner_result = planner.solve()
    summary = planner_result.as_dict()

    graph_context = {
        "offsets": offsets,
        "edges": edges,
        "weights": weights,
        "depot_locations": depot_locations,
        "store_locations": target_locations.tolist(),
    }

    return planner_result, summary, location_coordinates, graph_context


def run_dashboard() -> None:
    st.set_page_config(page_title="Optimus Route Dashboard", layout="wide")
    st.title("Optimus Route Dashboard")

    with st.sidebar:
        st.header("Scenario builder")
        num_depots = st.number_input("Depots", min_value=1, max_value=6, value=3, step=1)
        num_stores = st.number_input("Stores", min_value=3, max_value=60, value=15, step=1)
        num_vehicles = st.number_input(
            "Vehicles", min_value=1, max_value=50, value=8, step=1
        )
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=7, step=1)
        step_minutes = st.slider("Timeline step (minutes)", 1, 30, 5)
        run_button = st.button("Generate scenario", use_container_width=True)

    if not run_button:
        st.info("Pick your parameters in the sidebar and click **Generate scenario**.")
        return

    planner_result, summary, coords, graph_data = build_random_problem(
        num_depots=num_depots,
        num_stores=num_stores,
        num_vehicles=num_vehicles,
        seed=int(seed),
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total cost", f"{summary.get('total_cost', 0.0):.1f}")
    with col2:
        st.metric("Vehicles used", summary.get("vehicles_used", 0))
    with col3:
        st.metric(
            "Strategies",
            ", ".join(f"{k}: {v}" for k, v in summary.get("strategy_counts", {}).items()),
        )

    st.subheader("Animated route timeline")

    map_tab, graph_tab = st.tabs(["Map view", "Graph view"])

    with map_tab:
        with tempfile.TemporaryDirectory() as tmpdir:
            map_path = Path(tmpdir) / "workspace_optimus_routes_map.html"
            create_interactive_route_map(
                planner_result,
                location_coordinates=coords,
                filename=str(map_path),
                step_minutes=int(step_minutes),
                graph_offsets=graph_data.get("offsets"),
                graph_edges=graph_data.get("edges"),
                depot_locations=graph_data.get("depot_locations"),
                store_locations=graph_data.get("store_locations"),
            )
            html = map_path.read_text()
            components.v1.html(html, height=650, scrolling=False)

    with graph_tab:
        graph_html = create_graph_animation_html(
            planner_result=planner_result,
            location_coordinates=coords,
            step_minutes=int(step_minutes),
            graph_offsets=graph_data.get("offsets"),
            graph_edges=graph_data.get("edges"),
            graph_weights=graph_data.get("weights"),
            depot_locations=graph_data.get("depot_locations"),
            store_locations=graph_data.get("store_locations"),
        )
        components.v1.html(graph_html, height=650, scrolling=False)

    st.subheader("Assignments")
    st.dataframe(summary.get("assignments", []))

    st.subheader("Vehicle plans")
    st.json(summary.get("vehicle_plans", []))


if __name__ == "__main__":
    run_dashboard()
