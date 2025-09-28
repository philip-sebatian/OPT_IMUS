"""GPU-accelerated planner built on cuOpt for multi-vehicle delivery.

This module provides a high-level orchestration layer that leverages NVIDIA's
cuOpt solver to compute capacity-aware routes, automatically splits oversized
orders, and enriches the solution with delivery/refill operations that the rest
of the Optimus stack can consume (pretty printer, visualizers, etc.).

When a compatible GPU/cuOpt environment is not available the planner gracefully
falls back to the heuristic EnhancedRefillSystem to maintain functionality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.vehicle import VehicleState
from ..core.delivery_task import DeliveryTask
from ..core.depot_manager import DepotManager
from ..core.cost_calculator import CostCalculator


try:  # pragma: no cover - imported at runtime if cuOpt is available
    from cuopt.distance_engine import WaypointMatrix
    from cuopt import routing
except Exception:  # pragma: no cover - fallback path exercised in CPU tests
    WaypointMatrix = None  # type: ignore
    routing = None  # type: ignore


@dataclass
class DeliveryJob:
    """Represents an individual delivery demand fed to cuOpt."""

    job_id: int
    location: int
    demand: int
    source_task: int


@dataclass
class VehiclePlan:
    """Detailed plan for a single vehicle returned by the planner."""

    vehicle_id: int
    route: List[int] = field(default_factory=list)
    operations: List[Dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0
    total_distance: float = 0.0
    total_time: float = 0.0


@dataclass
class PlannerResult:
    """Normalized planner response consumed by pretty printer & visualizers."""

    vehicles: List[VehicleState]
    delivery_tasks: List[DeliveryTask]
    depot_manager: DepotManager
    vehicle_plans: List[VehiclePlan]
    job_assignments: List[Dict[str, Any]]
    meta: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        """Return a dictionary version for compatibility with legacy callers."""

        strategy_counts, strategy_lookup = self._compute_strategy_data()

        assignments: List[Dict[str, Any]] = []
        for assignment in self.job_assignments:
            task_id = assignment.get("task_id")
            strategy = strategy_lookup.get(task_id, "direct")
            enriched = dict(assignment)
            enriched["strategy"] = strategy
            if strategy == "split":
                enriched["vehicles"] = sorted(
                    {
                        entry["vehicle_id"]
                        for entry in self.job_assignments
                        if entry.get("task_id") == task_id
                    }
                )
            else:
                enriched["vehicles"] = [enriched.get("vehicle_id")]
            assignments.append(enriched)

        return {
            "total_cost": float(sum(plan.total_cost for plan in self.vehicle_plans)),
            "total_distance": float(sum(plan.total_distance for plan in self.vehicle_plans)),
            "total_time": float(sum(plan.total_time for plan in self.vehicle_plans)),
            "vehicles_used": sum(1 for plan in self.vehicle_plans if plan.route),
            "strategy_counts": strategy_counts,
            "assignments": assignments,
            "vehicle_routes": {
                plan.vehicle_id: plan.route for plan in self.vehicle_plans if plan.route
            },
            "vehicle_plans": [
                {
                    "vehicle_id": plan.vehicle_id,
                    "route": plan.route,
                    "operations": plan.operations,
                    "total_cost": float(plan.total_cost),
                    "total_distance": float(plan.total_distance),
                    "total_time": float(plan.total_time),
                }
                for plan in self.vehicle_plans
            ],
            "meta": self.meta,
        }

    def _compute_strategy_data(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        counts: Dict[str, int] = {"direct": 0, "split": 0}
        lookup: Dict[int, str] = {}
        per_task: Dict[int, List[int]] = {}

        for assignment in self.job_assignments:
            per_task.setdefault(assignment["task_id"], []).append(assignment["vehicle_id"])

        for task_id, vehicles in per_task.items():
            if len(set(vehicles)) > 1:
                counts["split"] += 1
                lookup[task_id] = "split"
            else:
                counts["direct"] += 1
                lookup[task_id] = "direct"

        return counts, lookup


class CuOptEnhancedPlanner:
    """High-level planner that maximises cuOpt usage while preserving ergonomics."""

    def __init__(
        self,
        offsets: np.ndarray,
        edges: np.ndarray,
        weights: np.ndarray,
        time_to_travel: np.ndarray,
        target_locations: np.ndarray,
        depot_locations: List[int],
        vehicle_capacities: List[int],
        initial_stock: List[int],
        delivery_demands: List[int],
        solver_time_limit: float = 30.0,
        vehicle_start_locations: Optional[List[int]] = None,
    ) -> None:
        self.offsets = offsets
        self.edges = edges
        self.weights = weights
        self.time_to_travel = time_to_travel
        self.target_locations = target_locations
        self.depot_locations = depot_locations
        self.vehicle_capacities = vehicle_capacities
        self.initial_stock = initial_stock
        self.delivery_demands = delivery_demands
        self.solver_time_limit = solver_time_limit

        if vehicle_start_locations is None:
            vehicle_start_locations = [depot_locations[0]] * len(vehicle_capacities)
        if len(vehicle_start_locations) != len(vehicle_capacities):
            raise ValueError("vehicle_start_locations length must match vehicle count")
        self.vehicle_start_locations = [int(loc) for loc in vehicle_start_locations]

        self._ensure_inputs()

        # Lazy initialisation; set during planning
        self.w_matrix: Optional[WaypointMatrix] = None
        self.cost_matrix: Optional[np.ndarray] = None
        self.time_matrix: Optional[np.ndarray] = None
        self.all_locations: Optional[np.ndarray] = None
        self.location_to_index: Dict[int, int] = {}
        self.index_to_location: Dict[int, int] = {}

        self.vehicles = []
        for i, (cap, stock, start_loc) in enumerate(
            zip(vehicle_capacities, initial_stock, self.vehicle_start_locations)
        ):
            self.vehicles.append(
                VehicleState(
                    id=i,
                    capacity=cap,
                    current_stock=stock,
                    position=start_loc,
                )
            )

        self.delivery_tasks = [
            DeliveryTask(location=int(loc), demand=int(demand))
            for loc, demand in zip(target_locations, delivery_demands)
        ]

        self.depot_manager: Optional[DepotManager] = None
        self.cost_calculator: Optional[CostCalculator] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> PlannerResult:
        """Compute an optimised plan, falling back to heuristics as needed."""

        self._prepare_distance_engine()

        if routing is None or self.w_matrix is None:
            return self._fallback("cuOpt runtime unavailable")

        try:
            data_model, jobs = self._build_data_model()
            solver = routing.Solver(data_model)
            solver.set_time_limit(self.solver_time_limit)
            solution = solver.solve()

            if not solution or not solution.is_feasible():
                return self._fallback("cuOpt solver did not find a feasible solution")

            vehicle_plans, job_assignments = self._extract_solution(solution, jobs)
            meta = {
                "solver": "cuOpt",
                "time_limit": self.solver_time_limit,
                "fallback": False,
                "status": "ok",
            }

            return PlannerResult(
                vehicles=self.vehicles,
                delivery_tasks=self.delivery_tasks,
                depot_manager=self.depot_manager,  # type: ignore[arg-type]
                vehicle_plans=vehicle_plans,
                job_assignments=job_assignments,
                meta=meta,
            )

        except Exception as exc:  # pragma: no cover - GPU path not executed in CI
            return self._fallback(f"cuOpt exception: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_inputs(self) -> None:
        if len(self.depot_locations) == 0:
            raise ValueError("At least one depot location is required")
        if len(self.vehicle_capacities) != len(self.initial_stock):
            raise ValueError("vehicle_capacities and initial_stock length mismatch")
        if len(self.target_locations) != len(self.delivery_demands):
            raise ValueError("target_locations and delivery_demands length mismatch")

    def _prepare_distance_engine(self) -> None:
        if self.cost_matrix is not None and self.time_matrix is not None:
            return

        start_array = np.array(self.vehicle_start_locations, dtype=int)
        self.all_locations = np.unique(
            np.concatenate(
                [
                    self.target_locations,
                    np.array(self.depot_locations, dtype=int),
                    start_array,
                ]
            )
        )

        if WaypointMatrix is not None:
            self.w_matrix = WaypointMatrix(self.offsets, self.edges, self.weights)
            self.cost_matrix = self.w_matrix.compute_cost_matrix(self.all_locations)
            self.time_matrix = self.w_matrix.compute_shortest_path_costs(
                self.all_locations, self.time_to_travel
            )
        else:
            self.cost_matrix = self._compute_cost_matrix_cpu(self.all_locations)
            self.time_matrix = np.array(self.cost_matrix)
        self.location_to_index = {
            int(loc): int(idx) for idx, loc in enumerate(self.all_locations.tolist())
        }
        self.index_to_location = {
            int(idx): int(loc) for idx, loc in enumerate(self.all_locations.tolist())
        }

        self.depot_manager = DepotManager(
            depot_locations=self.depot_locations,
            cost_matrix=self.cost_matrix,
            location_to_index=self.location_to_index,
        )
        self.cost_calculator = CostCalculator(self.depot_manager)

    # ------------------------------------------------------------------
    # cuOpt data model construction & solution extraction
    # ------------------------------------------------------------------

    def _build_data_model(self) -> Tuple[routing.DataModel, List[DeliveryJob]]:
        assert self.cost_matrix is not None
        assert self.time_matrix is not None
        assert self.all_locations is not None

        jobs = self._create_jobs()

        data_model = routing.DataModel(
            n_locations=len(self.all_locations),
            n_fleet=len(self.vehicle_capacities),
            n_orders=len(jobs),
        )

        primary_depot = self.depot_locations[0]
        data_model.set_depot(self.location_to_index[primary_depot])
        data_model.add_cost_matrix(self.cost_matrix)
        data_model.add_transit_time_matrix(self.time_matrix)

        # Vehicle configuration
        start_locations = [
            self.location_to_index[int(loc)] for loc in self.vehicle_start_locations
        ]
        data_model.set_vehicle_locations(start_locations)

        for vehicle_id, capacity in enumerate(self.vehicle_capacities):
            data_model.set_vehicle_capacity(vehicle_id, capacity)

        # Order configuration
        order_locations = [self.location_to_index[job.location] for job in jobs]
        data_model.set_order_locations(order_locations)
        data_model.demand = [job.demand for job in jobs]
        data_model.set_order_service_times([0] * len(jobs))

        return data_model, jobs

    def _create_jobs(self) -> List[DeliveryJob]:
        jobs: List[DeliveryJob] = []
        max_vehicle_capacity = max(self.vehicle_capacities)
        job_id = 0

        for task_id, (location, demand) in enumerate(
            zip(self.target_locations.astype(int), self.delivery_demands.astype(int))
        ):
            remaining = int(demand)

            while remaining > 0:
                chunk = min(remaining, max_vehicle_capacity)
                jobs.append(
                    DeliveryJob(
                        job_id=job_id,
                        location=location,
                        demand=chunk,
                        source_task=task_id,
                    )
                )
                job_id += 1
                remaining -= chunk

        return jobs

    def _extract_solution(
        self, solution: routing.Solution, jobs: List[DeliveryJob]
    ) -> Tuple[List[VehiclePlan], List[Dict[str, Any]]]:
        assert self.depot_manager is not None

        vehicle_plans: List[VehiclePlan] = []
        job_assignments: List[Dict[str, Any]] = []

        # Pre-compute job ownership for split detection & operations
        job_lookup: Dict[Tuple[int, int], List[Tuple[float, DeliveryJob]]] = {}

        for job in jobs:
            vehicle_id = solution.get_job_vehicle(job.job_id)
            if vehicle_id < 0:
                continue

            delivery_time = solution.get_job_delivery_time(job.job_id)
            job_lookup.setdefault((vehicle_id, job.location), []).append((delivery_time, job))

            job_assignments.append(
                {
                    "job_id": job.job_id,
                    "task_id": job.source_task,
                    "vehicle_id": vehicle_id,
                    "location": job.location,
                    "demand": job.demand,
                    "delivery_time": delivery_time,
                }
            )

        # Sort deliveries per location in chronological order
        for deliveries in job_lookup.values():
            deliveries.sort(key=lambda item: item[0])

        for vehicle in self.vehicles:
            route_indices = solution.get_vehicle_route(vehicle.id)
            plan = VehiclePlan(vehicle_id=vehicle.id)

            if len(route_indices) == 0:
                vehicle_plans.append(plan)
                continue

            route_locations = [self.index_to_location[int(idx)] for idx in route_indices]
            plan.route = route_locations
            vehicle.route = route_locations
            vehicle.position = route_locations[-1]

            try:
                plan.total_distance = float(solution.get_vehicle_distance(vehicle.id))
            except Exception:  # pragma: no cover
                plan.total_distance = self._estimate_distance(route_locations)

            try:  # pragma: no cover
                plan.total_time = float(solution.get_vehicle_time(vehicle.id))
            except Exception:
                plan.total_time = 0.0

            current_stock = min(vehicle.current_stock, vehicle.capacity)
            operations: List[Dict[str, Any]] = []

            for idx, location in enumerate(route_locations):
                if location in self.depot_locations:
                    current_stock = vehicle.capacity
                    operations.append(
                        {
                            "type": "refill",
                            "location": location,
                            "remaining_stock": current_stock,
                            "note": "Vehicle refilled at depot",
                        }
                    )
                    continue

                deliveries = job_lookup.get((vehicle.id, location), [])
                if not deliveries:
                    operations.append(
                        {
                            "type": "pass_through",
                            "location": location,
                            "remaining_stock": current_stock,
                        }
                    )
                    continue

                _, job = deliveries.pop(0)
                current_stock -= job.demand
                current_stock = max(current_stock, 0)

                operations.append(
                    {
                        "type": "delivery",
                        "location": location,
                        "quantity": job.demand,
                        "remaining_stock": current_stock,
                        "task_id": job.source_task,
                    }
                )

            plan.operations = operations
            plan.total_cost = plan.total_distance or self._estimate_distance(route_locations)
            vehicle.total_cost = plan.total_cost
            vehicle.current_stock = current_stock

            final_location = plan.route[-1]
            nearest_depot, depot_distance = self.depot_manager.find_nearest_depot(final_location)
            if nearest_depot != final_location:
                plan.route.append(nearest_depot)
                plan.operations.append(
                    {
                        "type": "return_to_depot",
                        "location": nearest_depot,
                        "remaining_stock": vehicle.capacity,
                    }
                )
                plan.total_distance += depot_distance
                plan.total_cost += depot_distance
                vehicle.total_cost = plan.total_cost
                vehicle.current_stock = vehicle.capacity
                final_location = nearest_depot

            vehicle.position = final_location
            vehicle_plans.append(plan)

        return vehicle_plans, job_assignments

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _estimate_distance(self, route: List[int]) -> float:
        if not route or len(route) < 2 or self.depot_manager is None:
            return 0.0

        distance = 0.0
        for origin, destination in zip(route[:-1], route[1:]):
            distance += self.depot_manager.get_distance(origin, destination)
        return float(distance)

    # ------------------------------------------------------------------
    # Fallback handling
    # ------------------------------------------------------------------

    def _fallback(self, reason: str) -> PlannerResult:
        if self.depot_manager is None:
            self._prepare_distance_engine()

        plans: Dict[int, VehiclePlan] = {}
        current_stock: Dict[int, int] = {}

        for vehicle in self.vehicles:
            start_location = self.vehicle_start_locations[vehicle.id]
            plans[vehicle.id] = VehiclePlan(
                vehicle_id=vehicle.id,
                route=[start_location],
                operations=[],
            )
            current_stock[vehicle.id] = vehicle.current_stock

        job_assignments: List[Dict[str, Any]] = []
        vehicle_index = 0
        vehicles_count = len(self.vehicles)

        for task_id, task in enumerate(self.delivery_tasks):
            remaining = task.demand
            while remaining > 0 and vehicles_count:
                vehicle = self.vehicles[vehicle_index % vehicles_count]
                plan = plans[vehicle.id]
                vehicle_index += 1

                current_location = plan.route[-1]
                capacity = vehicle.capacity

                needed = min(remaining, capacity)
                if current_stock[vehicle.id] < needed:
                    refill_depot, distance_to_refill = self.depot_manager.find_nearest_depot(current_location)
                    if refill_depot != current_location:
                        plan.total_cost += distance_to_refill
                        plan.route.append(refill_depot)
                        plan.operations.append(
                            {
                                "type": "refill",
                                "location": refill_depot,
                                "remaining_stock": capacity,
                            }
                        )
                        current_location = refill_depot
                    current_stock[vehicle.id] = capacity

                delivered_amount = min(remaining, current_stock[vehicle.id])
                remaining -= delivered_amount
                current_stock[vehicle.id] -= delivered_amount

                if current_location != task.location:
                    plan.total_cost += self.depot_manager.get_distance(current_location, task.location)
                    plan.route.append(task.location)
                    current_location = task.location

                plan.operations.append(
                    {
                        "type": "delivery",
                        "location": task.location,
                        "quantity": delivered_amount,
                        "remaining_stock": current_stock[vehicle.id],
                        "task_id": task_id,
                    }
                )

                job_assignments.append(
                    {
                        "job_id": len(job_assignments),
                        "task_id": task_id,
                        "vehicle_id": vehicle.id,
                        "location": task.location,
                        "demand": delivered_amount,
                        "delivery_time": float(len(job_assignments)),
                    }
                )

                nearest_depot, depot_distance = self.depot_manager.find_nearest_depot(current_location)
                if nearest_depot != current_location:
                    plan.total_cost += depot_distance
                    plan.route.append(nearest_depot)
                    current_location = nearest_depot
                plan.operations.append(
                    {
                        "type": "refill",
                        "location": current_location,
                        "remaining_stock": capacity,
                    }
                )
                current_stock[vehicle.id] = capacity

        for vehicle in self.vehicles:
            plan = plans[vehicle.id]
            plan.total_distance = plan.total_cost
            plan.total_time = plan.total_distance
            vehicle.route = plan.route
            vehicle.total_cost = plan.total_cost
            vehicle.current_stock = current_stock[vehicle.id]
            vehicle.position = plan.route[-1]

        meta = {
            "solver": "heuristic",
            "fallback": True,
            "reason": reason,
        }

        return PlannerResult(
            vehicles=self.vehicles,
            delivery_tasks=self.delivery_tasks,
            depot_manager=self.depot_manager,  # type: ignore[arg-type]
            vehicle_plans=list(plans.values()),
            job_assignments=job_assignments,
            meta=meta,
        )

    def _compute_cost_matrix_cpu(self, locations: np.ndarray) -> np.ndarray:
        adjacency = self._build_adjacency()
        matrix = np.full((len(locations), len(locations)), np.inf)

        for i, source in enumerate(locations):
            distances = self._dijkstra(int(source), adjacency)
            for j, target in enumerate(locations):
                matrix[i, j] = distances.get(int(target), np.inf)

        np.fill_diagonal(matrix, 0.0)
        return matrix

    def _build_adjacency(self) -> Dict[int, List[Tuple[int, float]]]:
        adjacency: Dict[int, List[Tuple[int, float]]] = {}
        node_count = max(len(self.offsets) - 1, 0)
        for origin in range(node_count):
            start = int(self.offsets[origin])
            end = int(self.offsets[origin + 1]) if origin + 1 < len(self.offsets) else len(self.edges)
            adjacency.setdefault(origin, [])
            for edge_idx in range(start, end):
                destination = int(self.edges[edge_idx])
                weight = float(self.weights[edge_idx])
                adjacency[origin].append((destination, weight))
        return adjacency

    def _dijkstra(
        self, source: int, adjacency: Dict[int, List[Tuple[int, float]]]
    ) -> Dict[int, float]:
        import heapq

        distances: Dict[int, float] = {source: 0.0}
        queue: List[Tuple[float, int]] = [(0.0, source)]

        while queue:
            cost, node = heapq.heappop(queue)
            if cost > distances.get(node, float("inf")):
                continue

            for neighbour, weight in adjacency.get(node, []):
                new_cost = cost + weight
                if new_cost < distances.get(neighbour, float("inf")):
                    distances[neighbour] = new_cost
                    heapq.heappush(queue, (new_cost, neighbour))

        return distances
