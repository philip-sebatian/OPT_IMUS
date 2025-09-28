"""
Interactive web map visualizations using Folium.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import folium
from folium import plugins
from branca.element import Element

LocationCoordMap = Dict[int, Tuple[float, float]]


def _compute_map_center(coords: Iterable[Tuple[float, float]]) -> Tuple[float, float]:
    xs, ys = zip(*coords)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _ensure_latlon(lat: float, lon: float) -> Tuple[float, float]:
    return float(lat), float(lon)


def create_interactive_route_map(
    planner_result,
    location_coordinates: LocationCoordMap,
    filename: str = "workspace_optimus_routes_map.html",
    base_time: Union[datetime, None] = None,
    step_minutes: int = 5,
    graph_offsets: Optional[Sequence[int]] = None,
    graph_edges: Optional[Sequence[int]] = None,
    depot_locations: Optional[Iterable[int]] = None,
    store_locations: Optional[Iterable[int]] = None,
) -> str:
    """Render an interactive Leaflet timeline map describing vehicle movements."""
    summary = (
        planner_result.as_dict()
        if hasattr(planner_result, "as_dict")
        else dict(planner_result)
    )

    vehicle_plans = summary.get("vehicle_plans", [])
    if not vehicle_plans:
        raise ValueError("Planner result must include `vehicle_plans` for visualization")

    coords_required = {node for plan in vehicle_plans for node in plan.get("route", [])}
    missing = coords_required - location_coordinates.keys()
    if missing:
        raise KeyError(
            "Missing coordinates for locations: " + ", ".join(map(str, sorted(missing)))
        )

    latlon_values = [_ensure_latlon(*location_coordinates[loc]) for loc in coords_required]
    center_lat, center_lon = _compute_map_center(latlon_values)
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=8, control_scale=True)

    palette = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999",
    ]

    base_time = base_time or datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    delta = timedelta(minutes=max(1, step_minutes))
    features = []

    if depot_locations is None and getattr(planner_result, "depot_manager", None):
        depot_locations = getattr(planner_result.depot_manager, "depot_locations", None)
        if callable(getattr(planner_result.depot_manager, "get_depot_locations", None)):
            depot_locations = planner_result.depot_manager.get_depot_locations()

    if store_locations is None and getattr(planner_result, "delivery_tasks", None):
        store_locations = [task.location for task in planner_result.delivery_tasks]

    depot_locations = set(int(loc) for loc in depot_locations or [])
    store_locations = set(int(loc) for loc in store_locations or [])

    network_group = None
    if graph_offsets is not None and graph_edges is not None:
        offsets: List[int] = [int(val) for val in graph_offsets]
        edges_list: List[int] = [int(val) for val in graph_edges]
        unique_edges = set()
        for origin in range(len(offsets) - 1):
            start = offsets[origin]
            end = offsets[origin + 1] if origin + 1 < len(offsets) else len(edges_list)
            for edge_idx in range(int(start), int(end)):
                destination = edges_list[edge_idx]
                if origin == destination:
                    continue
                key = tuple(sorted((int(origin), int(destination))))
                unique_edges.add(key)

        if unique_edges:
            network_group = folium.FeatureGroup(name="Network Graph", show=False)
            for origin, destination in sorted(unique_edges):
                if origin not in location_coordinates or destination not in location_coordinates:
                    continue
                start_point = _ensure_latlon(*location_coordinates[origin])
                end_point = _ensure_latlon(*location_coordinates[destination])
                folium.PolyLine(
                    [start_point, end_point],
                    color="#b0b0b0",
                    weight=1,
                    opacity=0.35,
                    dash_array="3,6",
                ).add_to(network_group)

            network_group.add_to(fmap)

    depot_group = folium.FeatureGroup(name="Depots", show=True)
    store_group = folium.FeatureGroup(name="Stores", show=True)
    waypoint_group = folium.FeatureGroup(name="Waypoints", show=False)

    for node, coord in sorted(location_coordinates.items()):
        lat, lon = _ensure_latlon(*coord)
        tooltip = f"Node {node} • ({lat:.4f}, {lon:.4f})"
        popup = f"<b>Node {node}</b><br>Latitude: {lat:.4f}<br>Longitude: {lon:.4f}"

        if node in depot_locations:
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color="#1f78b4",
                fill=True,
                fill_color="#1f78b4",
                fill_opacity=0.9,
                tooltip=tooltip,
                popup=popup,
            ).add_to(depot_group)
        elif node in store_locations:
            folium.CircleMarker(
                location=[lat, lon],
                radius=7,
                color="#e31a1c",
                fill=True,
                fill_color="#fb6a4a",
                fill_opacity=0.9,
                tooltip=tooltip,
                popup=popup,
            ).add_to(store_group)
        else:
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color="#636363",
                fill=True,
                fill_color="#969696",
                fill_opacity=0.8,
                tooltip=tooltip,
                popup=popup,
            ).add_to(waypoint_group)

    depot_group.add_to(fmap)
    store_group.add_to(fmap)
    if waypoint_group._children:
        waypoint_group.add_to(fmap)

    route_summaries: List[str] = []

    for idx, plan in enumerate(vehicle_plans):
        route = plan.get("route", [])
        if not route:
            continue

        color = palette[idx % len(palette)]
        polyline_points = [location_coordinates[node] for node in route if node in location_coordinates]
        if not polyline_points:
            continue

        vehicle_group = folium.FeatureGroup(name=f"Vehicle {plan.get('vehicle_id', idx)} route", show=True)
        folium.PolyLine(polyline_points, color=color, weight=4, opacity=0.85).add_to(vehicle_group)
        vehicle_group.add_to(fmap)

        route_summary = " → ".join(str(node) for node in route)
        route_summaries.append(
            f"<strong>Vehicle {plan.get('vehicle_id', idx)}</strong>: {route_summary}"
        )

        operations = plan.get("operations", [])
        operations_by_node: Dict[int, List[Dict[str, Union[str, int, float]]]] = {}
        for op in operations:
            location = op.get("location")
            if location is None:
                continue
            operations_by_node.setdefault(int(location), []).append(op)

        current_time = base_time

        for node in route:
            if node not in location_coordinates:
                current_time += delta
                continue
            lat, lon = _ensure_latlon(*location_coordinates[node])
            node_ops = operations_by_node.get(int(node), [])

            popup_lines = [f"<strong>Vehicle {plan.get('vehicle_id', idx)}</strong>"]
            popup_lines.append(f"Node {node}")
            popup_lines.append(f"Lat/Lon: {lat:.4f}, {lon:.4f}")

            for op in node_ops:
                if op.get("type") == "delivery":
                    popup_lines.append(
                        f"Delivery • Qty {op.get('quantity', 'n/a')} • Remaining {op.get('remaining_stock', 'n/a')}"
                    )
                elif op.get("type") == "refill":
                    popup_lines.append(
                        f"Refill • Stock {op.get('remaining_stock', 'n/a')}"
                    )
                elif op.get("type") == "return_to_depot":
                    popup_lines.append("Return to depot")
                else:
                    popup_lines.append(op.get("type", "Event").title())

            popup_html = "<br>".join(popup_lines)

            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {
                        "time": current_time.isoformat(),
                        "popup": popup_html,
                        "icon": "circle",
                        "iconstyle": {
                            "color": color,
                            "fillColor": color,
                            "fillOpacity": 0.95,
                            "radius": 8,
                        },
                    },
                }
            )

            current_time += delta

    if features:
        plugins.TimestampedGeoJson(
            {"type": "FeatureCollection", "features": features},
            period=f"PT{max(1, step_minutes)}M",
            add_last_point=True,
            auto_play=False,
            loop=False,
            max_speed=10,
            loop_button=True,
        ).add_to(fmap)

    if route_summaries:
        route_html = (
            "<div style='position: fixed; top: 15px; right: 20px; width: 280px; z-index: 9999; "
            "background-color: rgba(255, 255, 255, 0.95); padding: 12px 14px; "
            "border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.2); font-size: 13px;'>"
            "<h4 style='margin: 0 0 8px 0; font-size: 14px;'>Vehicle Routes</h4>"
            "<div style='max-height: 180px; overflow-y: auto;'>"
        )
        route_html += "<br>".join(route_summaries)
        route_html += "</div></div>"
        fmap.get_root().html.add_child(Element(route_html))

    if location_coordinates:
        rows = []
        for node, coord in sorted(location_coordinates.items()):
            lat, lon = _ensure_latlon(*coord)
            rows.append(
                f"<tr><td>{node}</td><td>{lat:.4f}</td><td>{lon:.4f}</td></tr>"
            )

        table_html = (
            "<div style='position: fixed; bottom: 20px; left: 20px; width: 260px; z-index: 9999; "
            "background-color: rgba(255, 255, 255, 0.95); padding: 12px 14px; "
            "border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.2); font-size: 12px;'>"
            "<h4 style='margin: 0 0 8px 0; font-size: 14px;'>Node Coordinates</h4>"
            "<div style='max-height: 240px; overflow-y: auto;'>"
            "<table style='width: 100%; border-collapse: collapse;'>"
            "<thead><tr><th style='text-align:left;'>Node</th>"
            "<th style='text-align:right;'>Lat</th>"
            "<th style='text-align:right;'>Lon</th></tr></thead>"
            "<tbody>"
        )
        table_html += "".join(rows)
        table_html += "</tbody></table></div></div>"
        fmap.get_root().html.add_child(Element(table_html))

    folium.LayerControl(collapsed=True).add_to(fmap)

    fmap.save(filename)
    return filename
