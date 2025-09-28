# CuOptEnhancedPlanner Overview

This document explains the end-to-end flow of the GPU-accelerated planner introduced in this release. The class lives in `workspace_optimus/src/algorithms/cuopt_enhanced.py` and is exposed as `workspace_optimus.CuOptEnhancedPlanner`.

## High-Level Workflow

1. **Input Normalisation**
   - Accepts the graph in CSR format (`offsets`, `edges`, `weights`), per-edge travel times, and the usual PDP inputs (stores, depots, vehicle capacities, initial stock, demand).
   - Validates lengths and prepares reusable vehicle/task objects for visualisation and pretty-printing.

2. **Distance Engine Preparation**
   - When the CUDA/cuOpt runtime is available, the planner instantiates `cuopt.distance_engine.WaypointMatrix` to compute GPU-accelerated cost and time matrices for the union of depot and store nodes.
   - If cuOpt is not importable (e.g., laptop without an NVIDIA GPU) the planner transparently falls back to a CPU Dijkstra implementation to build the same matrices. This guarantees that the downstream DepotManager and CostCalculator remain usable in both modes.

3. **Job Expansion**
   - Customer demand that exceeds the largest vehicle capacity is automatically split into multiple jobs. Each job records the originating task ID so that split deliveries can be reconstructed in the result metadata.

4. **cuOpt Data Model Construction**
   - A `routing.DataModel` is created with the dense cost/time matrices, depot index, vehicle capacities, and the job list generated above.
   - Vehicles are anchored to the primary depot by default; downstream applications can customise this by manipulating the data model before solving.

5. **Solver Invocation**
   - A `routing.Solver` is configured with the requested time limit (default 30 seconds) and executed. When the solver reports a feasible solution, routes, loads, and job timings are extracted.

6. **Result Normalisation**
   - Vehicle routes, distances, and per-stop operations (refills, deliveries, pass-throughs) are captured in `VehiclePlan` objects.
   - All job assignments are converted back into the legacy dictionary format so that existing utilities (`PrettyPrinter`, visualisers) continue to operate without modification. The method `PlannerResult.as_dict()` exposes this compatibility layer.

7. **Automatic Fallback**
   - Any exception raised during solver setup or execution results in a graceful fallback to an internal deterministic heuristic. The heuristic still honours capacities, refills at the primary depot, and produces the same `PlannerResult` structure so downstream tooling behaves consistently.
   - The `meta` dictionary included in every result advertises whether the GPU solver or the fallback handled the request (`meta['solver'] in {'cuOpt', 'heuristic'}`).

## Integration Points

- **Visualisation**: pass `planner_result.vehicles`, `planner_result.delivery_tasks`, `planner_result.depot_manager`, and `planner_result.as_dict()` to `visualization.static_visualizer.create_static_visualization()` to render a proportional static plot.
- **Pretty Printer**: the existing `PrettyPrinter` now consumes the richer `vehicle_plans` list when present, printing per-vehicle operations directly.
- **Legacy Systems**: the heuristic `OptimizedRefillSystem` and `EnhancedRefillSystem` remain available for backwards compatibility or comparative testing.

## When to Use Which Solver

| Scenario | Recommended Solver |
|----------|-------------------|
| Production environment with NVIDIA GPU | `CuOptEnhancedPlanner` (full GPU path) |
| Development laptop / CI without GPU | `CuOptEnhancedPlanner` (fallback path) |
| Need to inspect individual strategy costs | `OptimizedRefillSystem` or `EnhancedRefillSystem` |

The planner is designed to be the default choice: it maximises cuOpt usage when the runtime is present, and still returns a structurally identical plan when it is not.
