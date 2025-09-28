"""
Static visualizer module for Optimus routing system.

This module provides non-interactive graph visualization capabilities for
environments without display capabilities (e.g., headless servers).
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import networkx as nx
import numpy as np
import textwrap
from typing import Dict, Any, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.vehicle import VehicleState
    from src.core.delivery_task import DeliveryTask
    from src.core.depot_manager import DepotManager
else:  # pragma: no cover - runtime fallback for environments without src imports
    VehicleState = Any  # type: ignore
    DeliveryTask = Any  # type: ignore
    DepotManager = Any  # type: ignore


class StaticOptimusVisualizer:
    """
    Static visualizer for the Optimus routing system.
    
    This version saves visualizations to files instead of displaying them interactively.
    """
    
    def __init__(self, vehicles: List[VehicleState], delivery_tasks: List[DeliveryTask], 
                 depot_manager: DepotManager, results: Dict[str, Any]):
        """
        Initialize the static visualizer.
        
        Args:
            vehicles: List of vehicles in the system
            delivery_tasks: List of delivery tasks
            depot_manager: Depot manager for location information
            results: Optimization results containing assignments and routes
        """
        self.vehicles = {v.id: v for v in vehicles}
        self.delivery_tasks = {i: task for i, task in enumerate(delivery_tasks)}
        self.depot_manager = depot_manager
        self.results = results
        
        # Create the graph
        self.G = self._create_graph()
        
        # Generate colors for vehicles
        self.vehicle_colors = self._generate_vehicle_colors()
        
    def _create_graph(self) -> nx.Graph:
        """Create the network graph from depot and delivery locations."""
        G = nx.Graph()
        
        # Add depot nodes
        depot_positions = {}
        for depot_id in self.depot_manager.depot_locations:
            pos = self._get_location_position(depot_id)
            depot_positions[depot_id] = pos
            G.add_node(depot_id, node_type='depot', position=pos)
        
        # Add delivery location nodes (stores)
        store_positions = {}
        for task_id, task in self.delivery_tasks.items():
            pos = self._get_location_position(task.location)
            store_positions[task.location] = pos
            G.add_node(task.location, node_type='store', position=pos, 
                      demand=task.demand, task_id=task_id)
        
        # Add edges between all nodes
        all_nodes = list(depot_positions.keys()) + list(store_positions.keys())
        for i, node1 in enumerate(all_nodes):
            for node2 in all_nodes[i+1:]:
                distance = self.depot_manager.get_distance(node1, node2)
                if distance != float('inf'):
                    G.add_edge(node1, node2, weight=distance)
        
        return G
    
    def _create_distance_based_layout(self) -> Dict[int, Tuple[float, float]]:
        """Create a layout where edge lengths are proportional to actual distances."""
        node_count = max(len(self.G.nodes), 1)
        optimal_k = 3.0 / np.sqrt(node_count)
        pos = nx.spring_layout(self.G, weight='weight', k=optimal_k, iterations=200, seed=42)
        
        # Scale the positions to a reasonable range
        if pos:
            # Get the current range
            x_coords = [pos[node][0] for node in pos]
            y_coords = [pos[node][1] for node in pos]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Scale to 0-100 range
            scale_factor = 80  # Leave some margin
            if x_max - x_min > 0:
                x_scale = scale_factor / (x_max - x_min)
            else:
                x_scale = 1
            if y_max - y_min > 0:
                y_scale = scale_factor / (y_max - y_min)
            else:
                y_scale = 1
            
            # Apply scaling and centering
            for node in pos:
                pos[node] = (
                    10 + (pos[node][0] - x_min) * x_scale,  # 10-90 range
                    10 + (pos[node][1] - y_min) * y_scale   # 10-90 range
                )
        
        # Apply overlap prevention
        pos = self._prevent_overlap(pos, min_distance=10.0)
        
        return pos
    
    def _prevent_overlap(self, pos: Dict[int, Tuple[float, float]], min_distance: float = 8.0) -> Dict[int, Tuple[float, float]]:
        """Prevent nodes from overlapping by adjusting positions."""
        nodes = list(pos.keys())
        max_iterations = 50
        
        for iteration in range(max_iterations):
            moved = False
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes[i+1:], i+1):
                    x1, y1 = pos[node1]
                    x2, y2 = pos[node2]
                    
                    # Calculate distance between nodes
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    
                    if distance < min_distance and distance > 0:
                        # Nodes are too close, move them apart
                        # Calculate direction vector
                        dx = x2 - x1
                        dy = y2 - y1
                        
                        # Normalize direction vector
                        length = np.sqrt(dx**2 + dy**2)
                        if length > 0:
                            dx = dx / length
                            dy = dy / length
                            
                            # Move nodes apart
                            move_distance = (min_distance - distance) / 2
                            pos[node1] = (x1 - dx * move_distance, y1 - dy * move_distance)
                            pos[node2] = (x2 + dx * move_distance, y2 + dy * move_distance)
                            moved = True
            
            if not moved:
                break  # No more overlaps to fix
        
        # Ensure all positions are within bounds
        for node in pos:
            x, y = pos[node]
            x = max(5, min(95, x))  # Keep within 5-95 range
            y = max(5, min(95, y))
            pos[node] = (x, y)
        
        return pos
    
    def _get_location_position(self, location_id: int) -> Tuple[float, float]:
        """Get the position of a location (depot or store) based on actual distances."""
        # This will be overridden by the distance-based layout algorithm
        # Return a default position that will be updated
        return (0, 0)
    
    def _generate_vehicle_colors(self) -> Dict[int, str]:
        """Generate distinct colors for each vehicle."""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 
                 'olive', 'cyan', 'magenta', 'yellow', 'navy', 'lime', 'maroon']
        vehicle_colors = {}
        for i, vehicle_id in enumerate(self.vehicles.keys()):
            vehicle_colors[vehicle_id] = colors[i % len(colors)]
        return vehicle_colors
    
    def create_visualization(self, filename: str = "workspace_optimus_routing.png", 
                           show_legend: bool = True, figsize: Tuple[int, int] = (24, 18)) -> str:
        """
        Create and save a static visualization with route and summary panels.

        Args:
            filename: Output filename for the visualization
            show_legend: Whether to show the legend on the graph panel
            figsize: Figure size as (width, height)
        """
        fig = plt.figure(figsize=figsize)
        fig.suptitle('Optimus Routing System - Vehicle Paths and Deliveries', fontsize=16, fontweight='bold')

        grid = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[3, 2], height_ratios=[3, 1], hspace=0.4, wspace=0.35)
        ax_graph = fig.add_subplot(grid[:, 0])
        ax_routes = fig.add_subplot(grid[0, 1])
        ax_summary = fig.add_subplot(grid[1, 1])

        # Layout with overlap prevention
        pos = self._create_distance_based_layout()
        for node, position in pos.items():
            self.G.nodes[node]['position'] = position

        self._draw_graph(ax_graph, pos, show_legend)
        self._populate_route_table(ax_routes)
        self._populate_summary_panel(ax_summary)

        fig.subplots_adjust(top=0.9)
        fig.savefig(filename, dpi=300)
        plt.close(fig)

        print(f"Visualization saved to {filename}")
        return filename

    def _draw_graph(self, ax, pos: Dict[int, Tuple[float, float]], show_legend: bool) -> None:
        """Render the depot/store graph and vehicle paths on the primary axis."""
        ax.set_facecolor('#f9f9f9')

        nx.draw_networkx_edges(self.G, pos, ax=ax, alpha=0.2, edge_color='#999999', width=1.2)

        depot_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['node_type'] == 'depot']
        if depot_nodes:
            nx.draw_networkx_nodes(
                self.G,
                pos,
                nodelist=depot_nodes,
                node_color='#d95f02',
                node_size=900,
                node_shape='s',
                ax=ax,
                alpha=0.9,
                label='Depots'
            )
            depot_labels = {n: f'Depot-{n}' for n in depot_nodes}
            depot_label_pos = {n: (pos[n][0], pos[n][1] + 3.5) for n in depot_nodes}
            nx.draw_networkx_labels(self.G, depot_label_pos, depot_labels, ax=ax, font_size=12, font_weight='bold')

        store_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['node_type'] == 'store']
        if store_nodes:
            nx.draw_networkx_nodes(
                self.G,
                pos,
                nodelist=store_nodes,
                node_color='#1b9e77',
                node_size=750,
                node_shape='o',
                ax=ax,
                alpha=0.85,
                label='Stores'
            )
            store_labels = {n: f'Store-{n}\n({self.G.nodes[n]["demand"]})' for n in store_nodes}
            store_label_pos = {n: (pos[n][0], pos[n][1] - 3.5) for n in store_nodes}
            nx.draw_networkx_labels(self.G, store_label_pos, store_labels, ax=ax, font_size=11)

        for vehicle_id, vehicle in self.vehicles.items():
            if not vehicle.route or len(vehicle.route) < 2:
                continue

            color = self.vehicle_colors[vehicle_id]
            path_edges = [(vehicle.route[i], vehicle.route[i + 1]) for i in range(len(vehicle.route) - 1)]

            for index, edge in enumerate(path_edges):
                if edge[0] not in pos or edge[1] not in pos:
                    continue

                x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
                y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
                ax.plot(
                    x_coords,
                    y_coords,
                    color=color,
                    linewidth=2.5,
                    alpha=0.85,
                    label=f'Vehicle {vehicle_id}' if index == 0 else ""
                )

        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            legend_mapping = {label: handle for handle, label in zip(handles, labels) if label}
            if legend_mapping:
                ax.legend(
                    legend_mapping.values(),
                    legend_mapping.keys(),
                    loc='upper left',
                    bbox_to_anchor=(0.0, 1.02),
                    ncol=2,
                    frameon=False,
                    fontsize=10
                )

        ax.set_title('Vehicle Paths and Delivery Information', fontsize=12, loc='left')
        ax.grid(True, which='major', alpha=0.2, linestyle='--')
        ax.margins(0.08)
        ax.set_aspect('equal', adjustable='datalim')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    def _populate_route_table(self, ax) -> None:
        """Populate the vehicle route table on the secondary axis."""
        ax.set_title('Vehicle Routes', fontsize=12, fontweight='bold', loc='left')
        ax.axis('off')

        if not self.vehicles:
            ax.text(0.0, 0.5, 'No vehicle data available', fontsize=10, va='center', ha='left')
            return

        column_labels = ['Vehicle', 'Route', 'Cost', 'Stock']
        cell_data: List[List[str]] = []
        row_colors: List[Tuple[float, float, float]] = []

        for vehicle_id in sorted(self.vehicles.keys()):
            vehicle = self.vehicles[vehicle_id]
            route_text = self._format_route(vehicle)
            cell_data.append([
                f'#{vehicle_id}',
                route_text,
                f'{vehicle.total_cost:.1f}',
                f'{vehicle.current_stock}/{vehicle.capacity}'
            ])
            row_colors.append(self._lighten_color(self.vehicle_colors[vehicle_id], amount=0.75))

        table = ax.table(cellText=cell_data, colLabels=column_labels, loc='upper left', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        table.auto_set_column_width(list(range(len(column_labels))))

        header_color = '#3f3f3f'
        for column_index in range(len(column_labels)):
            header_cell = table[0, column_index]
            header_cell.set_facecolor(header_color)
            header_cell.set_edgecolor('#dddddd')
            header_cell.set_text_props(color='white', weight='bold')

        for row_index, row_color in enumerate(row_colors, start=1):
            for column_index in range(len(column_labels)):
                cell = table[row_index, column_index]
                cell.set_facecolor(row_color)
                cell.set_edgecolor('#f0f0f0')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def _populate_summary_panel(self, ax) -> None:
        """Populate the summary panel with aggregate metrics and assignments."""
        ax.set_title('Summary', fontsize=12, fontweight='bold', loc='left')
        ax.axis('off')

        total_cost = float(self.results.get('total_cost', 0.0))
        vehicles_used = self.results.get('vehicles_used', len(self.vehicles))
        strategy_counts = self.results.get('strategy_counts', {})
        assignments = self.results.get('assignments', [])

        summary_lines = [
            f'Total Cost: {total_cost:.1f}',
            f'Vehicles Used: {vehicles_used}'
        ]
        ax.text(0.0, 0.95, "\n".join(summary_lines), fontsize=10, va='top', ha='left', transform=ax.transAxes)

        if strategy_counts:
            ax.text(0.0, 0.6, 'Strategy Counts:', fontsize=10, fontweight='bold', ha='left', transform=ax.transAxes)
            formatted_counts = [f"- {key.replace('_', ' ').title()}: {value}" for key, value in sorted(strategy_counts.items())]
            ax.text(0.02, 0.6 - 0.05, "\n".join(formatted_counts), fontsize=9, ha='left', va='top', transform=ax.transAxes)

        if assignments:
            preview_header_y = 0.3
            ax.text(0.0, preview_header_y, 'Assignments Preview:', fontsize=10, fontweight='bold', ha='left', transform=ax.transAxes)
            preview_lines = self._format_assignment_preview(assignments)
            ax.text(0.02, preview_header_y - 0.05, "\n".join(preview_lines), fontsize=9, ha='left', va='top', transform=ax.transAxes)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def _format_route(self, vehicle: VehicleState) -> str:
        """Format a vehicle's route for tabular display with wrapping."""
        if not vehicle.route:
            return 'No recorded route'

        formatted_nodes = []
        for node in vehicle.route:
            label = f'Depot-{node}' if self.depot_manager.is_depot(node) else f'Store-{node}'
            formatted_nodes.append(label)

        route_string = " → ".join(formatted_nodes)
        return textwrap.fill(route_string, width=40)

    def _format_assignment_preview(self, assignments: List[Dict[str, Any]]) -> List[str]:
        """Build a short textual preview of assignments to avoid crowding the plot."""
        preview_lines: List[str] = []
        max_preview = 5

        for assignment in assignments[:max_preview]:
            task_id = assignment.get('task_id', '-')
            location = assignment.get('location', '-')
            vehicles = assignment.get('vehicles', [])
            vehicle_str = ', '.join(f'#{vid}' for vid in vehicles) if vehicles else 'n/a'
            strategy = assignment.get('strategy', 'unknown')
            preview_lines.append(f'Task {task_id}: {vehicle_str} → {location} ({strategy})')

        if len(assignments) > max_preview:
            preview_lines.append('...')

        return preview_lines

    def _lighten_color(self, color: str, amount: float = 0.7) -> Tuple[float, float, float]:
        """Lighten a matplotlib color by mixing it with white."""
        try:
            base_color = np.array(mcolors.to_rgb(color))
        except ValueError:
            base_color = np.array([0.6, 0.6, 0.6])

        white = np.array([1.0, 1.0, 1.0])
        mixed = base_color + (white - base_color) * amount
        return tuple(np.clip(mixed, 0.0, 1.0))


def create_static_visualization(vehicles: List[VehicleState], delivery_tasks: List[DeliveryTask], 
                               depot_manager: DepotManager, results: Dict[str, Any],
                               filename: str = "workspace_optimus_routing.png") -> str:
    """
    Create a static visualization of the routing system.
    
    Args:
        vehicles: List of vehicles
        delivery_tasks: List of delivery tasks
        depot_manager: Depot manager
        results: Optimization results
        filename: Output filename
        
    Returns:
        Filename of the saved visualization
    """
    visualizer = StaticOptimusVisualizer(vehicles, delivery_tasks, depot_manager, results)
    return visualizer.create_visualization(filename)
