# Optimus Mathematical Overview

This document summarises the mathematical model that the Optimus planner solves and highlights practical levers for producing shorter routes.

## 1. Problem Formulation

The optimisation extends the classical Capacitated Vehicle Routing Problem (CVRP) with depots, refill operations, and optional split deliveries. We work on a directed complete graph \(G=(V, E)\) built from:

- **Depots** \(D\subset V\)
- **Stores** \(S = V\setminus D\)
- **Vehicles** \(K\)

### Decision Variables (cuOpt model)
- \(x_{ijk} \in \{0,1\}\): vehicle \(k\) travels from node \(i\) to node \(j\)
- \(y_{ik}\): load of vehicle \(k\) after visiting node \(i\)
- \(t_{ik}\): arrival (or service completion) time at node \(i\) by vehicle \(k\)

### Objective
Minimise total travel cost
\[
\min \sum_{k \in K} \sum_{(i,j) \in E} c_{ij} x_{ijk}
\]
where \(c_{ij}\) is the distance or generic cost between nodes \(i\) and \(j\).

### Constraints
1. **Flow conservation**: each store is entered and exited exactly once across all vehicles.
2. **Depot balance**: vehicles start at their assigned start nodes and finish at the nearest depot (enforced post-solve when required).
3. **Capacity**: delivered quantity never exceeds vehicle load; refills reset load to capacity at depots.
4. **Demand fulfilment**: every store’s demand is met. If a demand exceeds a single vehicle capacity the planner partitions it into multiple jobs (split deliveries).
5. **Optional time limits**: carried by cuOpt through transit-time matrices and solver settings.

## 2. Distance & Time Matrices

The graph is supplied in compressed-sparse-row format. `WaypointMatrix` (if available) computes:

- **Cost matrix** \(C\): shortest-path distances between all relevant depots, stores, and per-vehicle start nodes
- **Time matrix** \(T\): analogous shortest travel times, allowing travel-time-aware optimisation

On CPU-only environments we reproduce the same matrices through Dijkstra’s algorithm, preserving identical downstream behaviour.

## 3. Job Expansion & Vehicle Starts

For each store with demand \(d_i\), the planner creates jobs of size \(\leq \max_k \text{capacity}_k\). These jobs keep a reference to their source store, which enables post-solve reconstruction of split deliveries.

Each vehicle may start at a different depot or even a store. Start indices feed directly into the cuOpt data model (`set_vehicle_locations`). After the solver returns, routes are extended to the nearest depot using the pre-computed cost matrix.

## 4. Solver Pipeline
1. Construct matrices and mappings (location id \(\leftrightarrow\) CSR indices).
2. Build `routing.DataModel` with cost/time matrices, vehicle capacities, start nodes, and job demands.
3. Call `routing.Solver` with a configurable time limit.
4. Extract vehicle tours, loads, and job assignments.
5. Post-process to:
   - Convert cuOpt job assignments back to store-level deliveries (with split detection).
   - Append nearest depots to vehicle itineraries.
   - Produce per-vehicle operation logs (refill/delivery/pass-through) for the pretty printer and static visualiser.
6. If the GPU runtime is unavailable, run the deterministic heuristic fallback that mirrors the same result structure.

## 5. Heuristic Fallback

The CPU fallback follows a cyclic assignment:
1. Iterates over tasks, allocating them to vehicles in round-robin order.
2. Refills vehicles at their nearest depot whenever remaining stock is insufficient.
3. After each delivery, routes the vehicle back to its closest depot and refills to full capacity to maintain consistency with the solver interface.

## 6. Producing Shorter Routes – Practical Suggestions

1. **Tighten cost matrices**: ensure distances reflect realistic travel costs (e.g., road network travel time rather than Euclidean distance). Lower costs lead to proportionally shorter routes.
2. **Vehicle start optimisation**: place vehicles at depots or stores close to their assigned clusters. The planner now accepts `vehicle_start_locations`, so you can seed starting positions to reduce deadhead travel.
3. **Demand clustering**: pre-cluster stores (e.g., k-means on coordinates) and assign vehicles to clusters via their start nodes before calling cuOpt. This often shortens tours.
4. **Solver parameter tuning**: adjust cuOpt solver heuristics (initial solution strategy, time limit). Longer time budgets or alternative heuristics (e.g., guided local search) can yield lower-cost tours.
5. **Introduce penalties**: if certain legs are undesirable, inflate their costs in the weight matrix to discourage them and redirect the solver to shorter alternatives.
6. **Post-optimisation local search**: apply 2-opt or swap-based heuristic refinement on the retrieved routes for marginal gains, especially if the fallback path was used.
7. **Capacity slack**: leave 5–10% headroom in vehicle capacities to reduce refill arcs. Excessive refills lengthen tours.

## 7. Further Enhancements

- **Time-window support**: incorporate store availability windows; cuOpt natively supports time constraints and will produce feasible, often shorter, tours.
- **Energy or speed models**: replace scalar cost with energy usage or travel time to capture the true optimisation objective.
- **Multi-day planning**: expand `vehicle_start_locations` and end-of-day depot logic to encompass multi-shift operations.

This mathematical and procedural foundation should help you interpret results, justify solutions, and identify tuning levers for achieving shorter, more efficient routes.
