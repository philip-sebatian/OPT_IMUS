# workspace_optimus: Advanced Vehicle Routing with Live Visualisations

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![cuOpt](https://img.shields.io/badge/cuOpt-optimized-green.svg)](https://developer.nvidia.com/cuopt)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**workspace_optimus** is a vehicle-routing optimisation toolkit that extends the Pickup and Delivery Problem (PDP) with intelligent refills, split deliveries, and GPU-accelerated solving through NVIDIA cuOpt. The project now includes a dual-visualisation Streamlit dashboard featuring a Folium map and a fully interactive canvas-based network animation that highlight every available edge, vehicle route, and depot/store type in real time.

---

## ✨ Highlights

- **cuOptEnhancedPlanner** – builds full cuOpt data models, runs the GPU solver when available, and gracefully falls back to the heuristic engine on CPU-only hosts.
- **Automatic Strategies** – evaluates direct, refill, reassignment, and split strategies, selecting the most cost-effective plan.
- **Dynamic Refills** – always returns the vehicle to the nearest depot when stock is depleted.
- **Live Streamlit Dashboard**
  - **Map View**: interactive Folium map with animated timeline, depot/store highlighting, and configurable playback speed.
  - **Graph View**: WebGL-free canvas animation that renders the full graph, colours travelled edges per vehicle, and exposes tooltips plus route/coordinate panels.
- **Richer Visual Outputs** – all routes, assignments, and per-vehicle plans are surfaced in the dashboard metrics and exported HTML.

---

## 📁 Project Layout

```
workspace_optimus/
├── __init__.py                  # Exposes high-level APIs
├── dashboard/                   # Streamlit dashboard
│   └── app.py
├── visualization/               # Folium + canvas visualisations
│   ├── graph_animation.py
│   └── web_map.py
├── src/                         # Core optimisation library
│   ├── algorithms/
│   ├── core/
│   └── utils/
├── examples/                    # CLI and scriptable demos
├── tests/                       # Unit / integration tests
├── docs/                        # Extended documentation
├── requirements.txt             # Python dependencies
└── setup.py                     # Packaging metadata
```

---

## 🚀 Getting Started

### 1. Install
```bash
git clone <repository-url>
cd workspace_optimus
pip install -r requirements.txt
pip install -e .
```

### 2. Launch the Dashboard
```bash
# Inside the repo root
export PYTHONPATH=$(pwd)
streamlit run workspace_optimus/dashboard/app.py --server.port=8501 --server.address=0.0.0.0
```

Open `http://0.0.0.0:8501` to explore:
- **Map View** for geo-based playback.
- **Graph View** for network-level animation with coloured trails, tooltips, and route summaries.

### 3. Basic API Usage
```python
import numpy as np
from workspace_optimus import CuOptEnhancedPlanner

offsets = np.array([0, 3, 5, 7])
edges = np.array([1, 2, 0, 2, 0, 1, 0])
weights = np.array([5.4, 3.2, 5.4, 2.9, 3.2, 2.9, 5.4])

planner = CuOptEnhancedPlanner(
    offsets=offsets,
    edges=edges,
    weights=weights,
    time_to_travel=weights,
    target_locations=np.array([1, 2]),
    depot_locations=[0],
    vehicle_capacities=[15, 12],
    initial_stock=[10, 10],
    delivery_demands=[8, 5],
)
result = planner.solve().as_dict()
print(result["total_cost"])
```

> **Tip:** If you run on a laptop without CUDA/cuOpt, the same API works; the planner transparently uses the deterministic heuristic fallback.

---

## 🧭 Dashboard Feature Guide

| Area | Highlights |
| --- | --- |
| **Map View** | Animated Folium timeline, depot/store colour coding, and an assignments grid for quick verification. |
| **Graph View** | Canvas-driven animation (no external CDN), coloured per-vehicle trails, hover tooltips, depot/store legend, route summary drawer, and node coordinate table. |
| **Sidebar Controls** | Random scenario builder with sliders for depots, stores, vehicles, solver timestep, and random seed. |

The dashboard writes temporary HTML outputs to `/tmp` only while rendering, keeping the workspace clean.

---

## 🧪 Tests

```bash
cd workspace_optimus
pytest
```

---

## 📚 Further Reading

- `docs/api_reference.md` – detailed class and function reference
- `docs/cuopt_enhanced_planner.md` – cuOpt integration notes
- `examples/` – runnable scripts for planners and refills

---

## 🤝 Contributing

1. Fork and clone the repository.
2. Create a branch for your feature/fix.
3. Add tests for new behaviour and run `pytest`.
4. Submit a pull request once everything passes.

Please file issues or feature requests via your chosen tracker and reference the observed behaviour plus reproduction steps.

---

## 📄 License

Released under the MIT License. See [LICENSE](LICENSE) for details.
