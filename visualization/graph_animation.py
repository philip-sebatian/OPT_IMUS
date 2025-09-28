"""Interactive graph animation for Optimus vehicle routes.

The helper generates a self-contained HTML bundle that renders the routing
network and animates vehicle movements using a lightweight canvas-based engine.
Edge lengths remain proportional to the projected geographic distances so users
can compare utilised versus available connections at a glance.
"""

from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from html import escape

LocationCoordMap = Dict[int, Tuple[float, float]]


def _project_coordinates(location_coordinates: LocationCoordMap) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    """Project latitude/longitude pairs onto a 2D plane for rendering."""

    if not location_coordinates:
        return {}, {"minX": 0.0, "maxX": 0.0, "minY": 0.0, "maxY": 0.0}

    lats = [coord[0] for coord in location_coordinates.values()]
    lons = [coord[1] for coord in location_coordinates.values()]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    scale = 111_000.0  # metres per degree (approx) for small regions
    cos_lat = math.cos(math.radians(center_lat)) or 1.0

    projected: Dict[int, Dict[str, float]] = {}
    xs: List[float] = []
    ys: List[float] = []

    for node_id, (lat, lon) in location_coordinates.items():
        x = (lon - center_lon) * scale * cos_lat
        y = (lat - center_lat) * scale
        projected[int(node_id)] = {"lat": float(lat), "lon": float(lon), "x": x, "y": y}
        xs.append(x)
        ys.append(y)

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    offset_x = (min_x + max_x) / 2.0
    offset_y = (min_y + max_y) / 2.0

    for node in projected.values():
        node["x"] -= offset_x
        node["y"] -= offset_y

    bounds = {
        "minX": min_x - offset_x,
        "maxX": max_x - offset_x,
        "minY": min_y - offset_y,
        "maxY": max_y - offset_y,
    }
    return projected, bounds


def _build_edge_list(
    offsets: Optional[Sequence[int]],
    edges: Optional[Sequence[int]],
    projected: Dict[int, Dict[str, float]],
) -> List[Dict[str, float]]:
    """Construct unique undirected edges with geometry."""

    if offsets is None or edges is None:
        return []

    offsets = [int(val) for val in offsets]
    edges = [int(val) for val in edges]

    unique = set()
    results: List[Dict[str, float]] = []

    for origin in range(len(offsets) - 1):
        start = offsets[origin]
        end = offsets[origin + 1] if origin + 1 < len(offsets) else len(edges)
        for idx in range(int(start), int(end)):
            destination = edges[idx]
            if destination == origin:
                continue
            key = tuple(sorted((origin, destination)))
            if key in unique:
                continue
            unique.add(key)
            if origin not in projected or destination not in projected:
                continue
            start_pt = projected[origin]
            end_pt = projected[destination]
            length = math.dist((start_pt["x"], start_pt["y"]), (end_pt["x"], end_pt["y"]))
            results.append(
                {
                    "from": int(origin),
                    "to": int(destination),
                    "length": float(length),
                    "sx": start_pt["x"],
                    "sy": start_pt["y"],
                    "ex": end_pt["x"],
                    "ey": end_pt["y"],
                }
            )

    return results


def _build_vehicle_paths(
    planner_result,
    projected: Dict[int, Dict[str, float]],
    palette: Sequence[str],
) -> Tuple[List[Dict[str, object]], float]:
    """Derive per-vehicle movement segments for animation."""

    summary = planner_result.as_dict() if hasattr(planner_result, "as_dict") else planner_result
    vehicle_plans = summary.get("vehicle_plans", [])

    vehicles: List[Dict[str, object]] = []
    max_time = 0.0

    for idx, plan in enumerate(vehicle_plans):
        route = plan.get("route", [])
        color = palette[idx % len(palette)]
        segments: List[Dict[str, float]] = []
        if len(route) >= 2:
            current_time = 0.0
            for start, end in zip(route[:-1], route[1:]):
                if start not in projected or end not in projected:
                    continue
                start_point = projected[int(start)]
                end_point = projected[int(end)]
                segment = {
                    "tStart": current_time,
                    "tEnd": current_time + 1.0,
                    "start": {
                        "x": start_point["x"],
                        "y": start_point["y"],
                    },
                    "end": {
                        "x": end_point["x"],
                        "y": end_point["y"],
                    },
                    "startNode": int(start),
                    "endNode": int(end),
                }
                segments.append(segment)
                current_time += 1.0
            max_time = max(max_time, current_time)

        vehicles.append(
            {
                "id": int(plan.get("vehicle_id", idx)),
                "color": color,
                "segments": segments,
                "label": f"Vehicle {plan.get('vehicle_id', idx)}",
            }
        )

    return vehicles, max_time


def create_graph_animation_html(
    planner_result,
    location_coordinates: LocationCoordMap,
    step_minutes: int,
    graph_offsets: Optional[Sequence[int]] = None,
    graph_edges: Optional[Sequence[int]] = None,
    graph_weights: Optional[Sequence[float]] = None,  # kept for API symmetry (unused)
    depot_locations: Optional[Iterable[int]] = None,
    store_locations: Optional[Iterable[int]] = None,
) -> str:
    """Generate a Three.js animation HTML fragment for the routing graph."""

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

    projected, bounds = _project_coordinates(location_coordinates)

    depot_set = {int(loc) for loc in (depot_locations or [])}
    store_set = {int(loc) for loc in (store_locations or [])}

    nodes = []
    for node_id, values in projected.items():
        node_type = "other"
        if node_id in depot_set:
            node_type = "depot"
        elif node_id in store_set:
            node_type = "store"
        nodes.append(
            {
                "id": node_id,
                "lat": round(values["lat"], 6),
                "lon": round(values["lon"], 6),
                "x": values["x"],
                "y": values["y"],
                "type": node_type,
            }
        )

    edges = _build_edge_list(graph_offsets, graph_edges, projected)
    vehicles, max_time = _build_vehicle_paths(planner_result, projected, palette)

    summary = planner_result.as_dict() if hasattr(planner_result, "as_dict") else planner_result
    vehicle_plans = summary.get("vehicle_plans", [])

    time_labels: List[Dict[str, object]] = []
    for idx, plan in enumerate(vehicle_plans):
        for step, node in enumerate(plan.get("route", [])):
            if node not in projected:
                continue
            time_labels.append(
                {
                    "vehicle": int(plan.get("vehicle_id", idx)),
                    "time": step,
                    "node": int(node),
                }
            )

    route_summaries: List[str] = []
    for idx, plan in enumerate(vehicle_plans):
        route = plan.get("route", []) or []
        if not route:
            continue
        vehicle_label = escape(str(plan.get("vehicle_id", idx)))
        route_text = escape(" -> ".join(str(int(node)) for node in route))
        route_summaries.append(f"<strong>Vehicle {vehicle_label}</strong>: {route_text}")

    if route_summaries:
        routes_panel_html = "<br>".join(route_summaries)
    else:
        routes_panel_html = "No routes planned yet."

    node_rows: List[str] = []
    for entry in sorted(nodes, key=lambda item: item["id"]):
        node_rows.append(
            f"<tr><td>{entry['id']}</td><td>{entry['lat']:.4f}</td><td>{entry['lon']:.4f}</td></tr>"
        )
    node_table_html = "".join(node_rows)

    data = {
        "nodes": nodes,
        "edges": edges,
        "vehicles": vehicles,
        "steps": max(1.0, max_time),
        "stepDuration": max(1, int(step_minutes)),
        "bounds": bounds,
        "timeline": time_labels,
    }

    data_json = json.dumps(data, separators=(",", ":"))
    timeline_max = max(1, int(math.ceil(max_time)))

    html = f"""

<div id="workspace-optimus-graph-container" style="width: 100%; height: 640px; position: relative; background: #0f141d; border-radius: 8px; overflow: hidden;">
  <canvas id="workspace-optimus-graph-canvas" style="position:absolute; inset:0; width:100%; height:100%; display:block;"></canvas>
  <div id="graph-controls" style="position:absolute; top:12px; left:12px; z-index:10; background: rgba(15, 18, 28, 0.72); color:#f5f7fa; padding:10px 12px; border-radius:6px; font-family: 'Segoe UI', sans-serif; font-size:13px; box-shadow:0 4px 12px rgba(0,0,0,0.35);">
    <div style="display:flex; align-items:center; gap:8px;">
      <button id="graph-play-toggle" style="border:none; background:#4dabf7; color:#08192c; padding:6px 12px; border-radius:4px; cursor:pointer; font-weight:600;">Pause</button>
      <span id="graph-time-label">t = 0</span>
    </div>
    <input id="graph-timeline" type="range" min="0" max="{timeline_max}" value="0" step="0.01" style="width:220px; margin-top:8px;">
  </div>
  <div id="graph-legend" style="position:absolute; bottom:12px; left:12px; z-index:10; background: rgba(15, 18, 28, 0.72); color:#f5f7fa; padding:10px 12px; border-radius:6px; font-family:'Segoe UI', sans-serif; font-size:12px; line-height:1.6; box-shadow:0 4px 12px rgba(0,0,0,0.35);">
    <div style="font-weight:600; margin-bottom:6px;">Legend</div>
    <div><span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:#1f78b4; margin-right:6px;"></span>Depot</div>
    <div><span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:#fb6a4a; margin-right:6px;"></span>Store</div>
    <div><span style="display:inline-block; width:14px; height:3px; background:#b5b8c3; margin-right:6px;"></span>Available edge</div>
  </div>
  <div id="graph-routes-panel" style="position:absolute; top:12px; right:12px; z-index:10; background: rgba(15, 18, 28, 0.72); color:#f5f7fa; padding:10px 12px; border-radius:6px; font-family:'Segoe UI', sans-serif; font-size:12px; line-height:1.6; width:280px; box-shadow:0 4px 12px rgba(0,0,0,0.35);">
    <div style="font-weight:600; margin-bottom:6px;">Vehicle Routes</div>
    <div style="max-height:220px; overflow-y:auto;">{routes_panel_html}</div>
  </div>
  <div id="graph-node-panel" style="position:absolute; bottom:12px; right:12px; z-index:10; background: rgba(15, 18, 28, 0.72); color:#f5f7fa; padding:10px 12px; border-radius:6px; font-family:'Segoe UI', sans-serif; font-size:12px; width:260px; box-shadow:0 4px 12px rgba(0,0,0,0.35);">
    <div style="font-weight:600; margin-bottom:6px;">Node Coordinates</div>
    <div style="max-height:220px; overflow-y:auto;">
      <table style="width:100%; border-collapse:collapse;">
        <thead><tr><th style="text-align:left; padding-bottom:4px;">Node</th><th style="text-align:right; padding-bottom:4px;">Lat</th><th style="text-align:right; padding-bottom:4px;">Lon</th></tr></thead>
        <tbody>{node_table_html}</tbody>
      </table>
    </div>
  </div>
  <div id="graph-tooltip" style="display:none; position:absolute; pointer-events:none; background: rgba(17,24,39,0.93); color:#f9fafb; padding:6px 8px; border-radius:4px; font-size:12px; font-family:'Segoe UI', sans-serif; box-shadow:0 2px 10px rgba(0,0,0,0.4);"></div>
</div>
<script>
(function() {{
  const data = {data_json};
  const container = document.getElementById('workspace-optimus-graph-container');
  if (!container) return;

  const canvas = document.getElementById('workspace-optimus-graph-canvas');
  const ctx = canvas.getContext('2d');
  const playButton = document.getElementById('graph-play-toggle');
  const timeline = document.getElementById('graph-timeline');
  const timeLabel = document.getElementById('graph-time-label');
  const tooltip = document.getElementById('graph-tooltip');

  const totalSteps = Math.max(1, data.steps);
  timeline.max = totalSteps;

  let animationTime = 0;
  let playing = true;
  let lastFrame = null;
  let width = 0;
  let height = 0;
  let scale = 1;
  let offsetX = 0;
  let offsetY = 0;
  const padding = 48;

  const depotColor = '#1f78b4';
  const storeColor = '#fb6a4a';
  const otherColor = '#8f96a3';
  const edgeColor = 'rgba(181, 184, 195, 0.35)';

  const projectPoint = (point) => {{
    return {{
      x: offsetX + point.x * scale,
      y: offsetY - point.y * scale,
    }};
  }};

  const updateViewport = () => {{
    const rect = container.getBoundingClientRect();
    width = rect.width;
    height = rect.height;
    const pixelRatio = window.devicePixelRatio || 1;
    canvas.width = width * pixelRatio;
    canvas.height = height * pixelRatio;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

    const spanX = (data.bounds.maxX - data.bounds.minX) || 1;
    const spanY = (data.bounds.maxY - data.bounds.minY) || 1;
    const usableWidth = Math.max(1, width - padding * 2);
    const usableHeight = Math.max(1, height - padding * 2);
    scale = Math.min(usableWidth / spanX, usableHeight / spanY);
    offsetX = (width - spanX * scale) / 2 - data.bounds.minX * scale;
    offsetY = (height - spanY * scale) / 2 + data.bounds.maxY * scale;
  }};

  const drawStaticGraph = () => {{
    ctx.clearRect(0, 0, width, height);

    ctx.lineWidth = 1.4;
    ctx.strokeStyle = edgeColor;
    ctx.beginPath();
    data.edges.forEach(edge => {{
      const start = projectPoint({{x: edge.sx, y: edge.sy}});
      const end = projectPoint({{x: edge.ex, y: edge.ey}});
      ctx.moveTo(start.x, start.y);
      ctx.lineTo(end.x, end.y);
    }});
    ctx.stroke();

    data.nodes.forEach(node => {{
      const point = projectPoint(node);
      let radius = 5;
      let color = otherColor;
      if (node.type === 'depot') {{
        radius = 7;
        color = depotColor;
      }} else if (node.type === 'store') {{
        radius = 6;
        color = storeColor;
      }}
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.strokeStyle = '#0f141d';
      ctx.lineWidth = 1.2;
      ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }});
  }};

  const getVehiclePosition = (vehicle, time) => {{
    if (!vehicle.segments.length) return null;
    const segments = vehicle.segments;
    if (time <= segments[0].tStart) {{
      return projectPoint(segments[0].start);
    }}
    const lastSegment = segments[segments.length - 1];
    if (time >= lastSegment.tEnd) {{
      return projectPoint(lastSegment.end);
    }}
    for (let i = 0; i < segments.length; i += 1) {{
      const seg = segments[i];
      if (time >= seg.tStart && time <= seg.tEnd) {{
        const alpha = (time - seg.tStart) / (seg.tEnd - seg.tStart || 1);
        const px = seg.start.x + (seg.end.x - seg.start.x) * alpha;
        const py = seg.start.y + (seg.end.y - seg.start.y) * alpha;
        return projectPoint({{x: px, y: py}});
      }}
    }}
    return null;
  }};

  const drawVehicles = () => {{
    data.vehicles.forEach(vehicle => {{
      const pos = getVehiclePosition(vehicle, animationTime);
      if (!pos) return;
      ctx.beginPath();
      ctx.fillStyle = vehicle.color;
      ctx.strokeStyle = '#0b1320';
      ctx.lineWidth = 1.8;
      ctx.arc(pos.x, pos.y, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }});
  }};

  const drawVehicleTrails = () => {{
    data.vehicles.forEach(vehicle => {{
      if (!vehicle.segments.length) return;
      ctx.lineWidth = 3.2;
      ctx.strokeStyle = vehicle.color;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      vehicle.segments.forEach(seg => {{
        if (animationTime <= seg.tStart) {{
          return;
        }}

        const startProjected = projectPoint(seg.start);
        const endProjected = projectPoint(seg.end);

        ctx.beginPath();
        ctx.moveTo(startProjected.x, startProjected.y);

        if (animationTime >= seg.tEnd) {{
          ctx.lineTo(endProjected.x, endProjected.y);
          ctx.stroke();
          return;
        }}

        const alpha = (animationTime - seg.tStart) / (seg.tEnd - seg.tStart || 1);
        if (alpha <= 0) {{
          return;
        }}

        const partialProjected = projectPoint({{
          x: seg.start.x + (seg.end.x - seg.start.x) * alpha,
          y: seg.start.y + (seg.end.y - seg.start.y) * alpha,
        }});
        ctx.lineTo(partialProjected.x, partialProjected.y);
        ctx.stroke();
      }});
    }});
  }};

  const renderScene = () => {{
    drawStaticGraph();
    drawVehicleTrails();
    drawVehicles();
    timeline.value = animationTime;
    timeLabel.textContent = 't = ' + animationTime.toFixed(2);
  }};

  const animate = (timestamp) => {{
    if (lastFrame === null) {{
      lastFrame = timestamp;
    }}
    const delta = (timestamp - lastFrame) / 1000;
    lastFrame = timestamp;

    if (playing) {{
      const animationSpeed = Math.max(0.15, 0.35);
      animationTime += delta * animationSpeed;
      if (animationTime > totalSteps) {{
        animationTime = 0;
      }}
    }}

    renderScene();
    requestAnimationFrame(animate);
  }};

  playButton.addEventListener('click', () => {{
    playing = !playing;
    playButton.textContent = playing ? 'Pause' : 'Play';
  }});

  timeline.addEventListener('input', (event) => {{
    const value = parseFloat(event.target.value);
    if (!Number.isFinite(value)) return;
    animationTime = Math.max(0, Math.min(totalSteps, value));
    playing = false;
    playButton.textContent = 'Play';
    renderScene();
  }});

  const hitRadius = 9;
  container.addEventListener('mousemove', (event) => {{
    const rect = container.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    let found = null;
    for (let i = 0; i < data.nodes.length; i += 1) {{
      const node = data.nodes[i];
      const point = projectPoint(node);
      const dx = point.x - mx;
      const dy = point.y - my;
      if (Math.abs(dx) <= hitRadius && Math.abs(dy) <= hitRadius) {{
        if (dx * dx + dy * dy <= hitRadius * hitRadius) {{
          found = {{ node, point }};
          break;
        }}
      }}
    }}

    if (found) {{
      tooltip.style.display = 'block';
      const left = Math.min(width - 140, Math.max(12, found.point.x + 12));
      const top = Math.min(height - 80, Math.max(12, found.point.y - 12));
      tooltip.style.left = left + 'px';
      tooltip.style.top = top + 'px';
      tooltip.innerHTML = '<strong>Node ' + found.node.id + '</strong><br>Lat: ' + found.node.lat.toFixed(4) + '<br>Lon: ' + found.node.lon.toFixed(4);
    }} else {{
      tooltip.style.display = 'none';
    }}
  }});

  container.addEventListener('mouseleave', () => {{
    tooltip.style.display = 'none';
  }});

  const handleResize = () => {{
    updateViewport();
    renderScene();
  }};

  window.addEventListener('resize', handleResize);
  updateViewport();
  renderScene();
  requestAnimationFrame(animate);
}})();
</script>

"""

    return html
