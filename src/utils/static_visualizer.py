"""
Static visualizer module for Optimus routing system.

This module provides non-interactive graph visualization capabilities for
environments without display capabilities (e.g., headless servers).
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from ..core.vehicle import VehicleState
from ..core.delivery_task import DeliveryTask
from ..core.depot_manager import DepotManager


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
    
    def _get_location_position(self, location_id: int) -> Tuple[float, float]:
        """Get the position of a location (depot or store)."""
        # Generate positions based on location ID for consistency
        np.random.seed(location_id)
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)
        return (x, y)
    
    def _generate_vehicle_colors(self) -> Dict[int, str]:
        """Generate distinct colors for each vehicle."""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 
                 'olive', 'cyan', 'magenta', 'yellow', 'navy', 'lime', 'maroon']
        vehicle_colors = {}
        for i, vehicle_id in enumerate(self.vehicles.keys()):
            vehicle_colors[vehicle_id] = colors[i % len(colors)]
        return vehicle_colors
    
    def create_visualization(self, filename: str = "workspace_optimus_routing.png", 
                           show_legend: bool = True, figsize: Tuple[int, int] = (18, 10)):
        """
        Create and save a static visualization with side table for routes.
        
        Args:
            filename: Output filename for the visualization
            show_legend: Whether to show the legend
            figsize: Figure size as (width, height)
        """
        # Create figure with subplots - main graph and side table
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], hspace=0.3)
        
        # Main graph subplot
        ax = fig.add_subplot(gs[0, 0])
        fig.suptitle('Optimus Routing System - Vehicle Paths and Deliveries', 
                    fontsize=16, fontweight='bold')
        
        pos = nx.get_node_attributes(self.G, 'position')
        
        # Draw edges
        nx.draw_networkx_edges(self.G, pos, ax=ax, alpha=0.3, edge_color='gray')
        
        # Draw depot nodes
        depot_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['node_type'] == 'depot']
        if depot_nodes:
            nx.draw_networkx_nodes(self.G, pos, nodelist=depot_nodes, 
                                 node_color='red', node_size=600, 
                                 node_shape='s', ax=ax, alpha=0.8, label='Depots')
            # Add depot labels with better formatting
            depot_labels = {n: f'Depot-{n}' for n in depot_nodes}
            nx.draw_networkx_labels(self.G, pos, depot_labels, ax=ax, font_size=10, font_weight='bold')
        
        # Draw store nodes
        store_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['node_type'] == 'store']
        if store_nodes:
            nx.draw_networkx_nodes(self.G, pos, nodelist=store_nodes, 
                                 node_color='blue', node_size=400, 
                                 node_shape='o', ax=ax, alpha=0.8, label='Stores')
            # Add store labels with demand
            store_labels = {}
            for n in store_nodes:
                demand = self.G.nodes[n]['demand']
                store_labels[n] = f'Store-{n}\n({demand})'
            nx.draw_networkx_labels(self.G, pos, store_labels, ax=ax, font_size=9, font_weight='bold')
        
        # Draw vehicle paths with route information
        for vehicle_id, vehicle in self.vehicles.items():
            if not vehicle.route or len(vehicle.route) < 2:
                continue
                
            color = self.vehicle_colors[vehicle_id]
            
            # Draw the path with arrows to show direction
            path_edges = [(vehicle.route[i], vehicle.route[i+1]) 
                         for i in range(len(vehicle.route)-1)]
            
            for i, edge in enumerate(path_edges):
                if self.G.has_edge(edge[0], edge[1]):
                    x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
                    y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
                    
                    # Draw the line
                    ax.plot(x_coords, y_coords, color=color, linewidth=4, 
                           alpha=0.8, label=f'Vehicle {vehicle_id}' if i == 0 else "")
                    
                    # Add arrow to show direction
                    mid_x = (x_coords[0] + x_coords[1]) / 2
                    mid_y = (y_coords[0] + y_coords[1]) / 2
                    dx = x_coords[1] - x_coords[0]
                    dy = y_coords[1] - y_coords[0]
                    
                    # Normalize the direction vector
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        dx = dx / length
                        dy = dy / length
                        
                        # Draw arrow
                        ax.arrow(mid_x, mid_y, dx*3, dy*3, head_width=2, head_length=2, 
                               fc=color, ec=color, alpha=0.8)
            
            # Route information will be shown in side table instead
        
        # Draw delivery information
        store_deliveries = {}
        for assignment in self.results.get('assignments', []):
            store_id = assignment['location']
            if store_id not in store_deliveries:
                store_deliveries[store_id] = []
            
            strategy = assignment['strategy']
            if strategy == 'split':
                for v_id, amount in assignment['details']['vehicle_amounts']:
                    store_deliveries[store_id].append(f'Vehicle-{v_id}: {amount} units')
            else:
                for v_id in assignment['vehicles']:
                    store_deliveries[store_id].append(f'Vehicle-{v_id}: {assignment["demand"]} units')
        
        # Draw delivery information boxes
        for store_id, deliveries in store_deliveries.items():
            if store_id in pos:
                x, y = pos[store_id]
                info_text = f'Deliveries to Store-{store_id}:\n' + '\n'.join(deliveries)
                ax.text(x, y + 12, info_text, fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8),
                       ha='center', va='bottom')
        
        # Add legend
        if show_legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add summary information
        total_cost = self.results.get('total_cost', 0)
        vehicles_used = self.results.get('vehicles_used', 0)
        strategy_counts = self.results.get('strategy_counts', {})
        
        summary_text = f"""Summary:
Total Cost: {total_cost:.1f}
Vehicles Used: {vehicles_used}
Strategies: {strategy_counts}"""
        
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Vehicle Paths and Delivery Information', fontsize=12)
        
        # Create side table for vehicle routes
        self._create_side_table(fig, gs)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {filename}")
        return filename
    
    def _create_side_table(self, fig, gs):
        """Create a side table showing vehicle routes and information."""
        # Create table subplot
        ax_table = fig.add_subplot(gs[0, 1])
        ax_table.axis('off')
        ax_table.set_title('Vehicle Routes & Information', fontsize=14, fontweight='bold', pad=20)
        
        # Prepare table data
        table_data = []
        headers = ['Vehicle', 'Route', 'Cost', 'Stock']
        
        depot_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['node_type'] == 'depot']
        
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle.route:
                # Format route with proper names
                route_str = " → ".join([f"Depot-{r}" if r in depot_nodes else f"Store-{r}" for r in vehicle.route])
            else:
                route_str = f"Depot-{vehicle.position}"
            
            table_data.append([
                f"Vehicle-{vehicle_id}",
                route_str,
                f"{vehicle.total_cost:.1f}",
                f"{vehicle.current_stock}/{vehicle.capacity}"
            ])
        
        # Create table
        if table_data:
            table = ax_table.table(cellText=table_data, colLabels=headers,
                                 cellLoc='left', loc='center',
                                 colWidths=[0.15, 0.5, 0.15, 0.2])
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Color the header row
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color vehicle rows with their respective colors
            for i, (vehicle_id, _) in enumerate(self.vehicles.items()):
                color = self.vehicle_colors[vehicle_id]
                for j in range(len(headers)):
                    table[(i+1, j)].set_facecolor(color)
                    table[(i+1, j)].set_alpha(0.3)
        
        # Add delivery summary
        ax_table.text(0.05, 0.85, 'Delivery Summary:', fontsize=12, fontweight='bold', 
                     transform=ax_table.transAxes)
        
        # Show delivery information
        y_pos = 0.8
        for assignment in self.results.get('assignments', []):
            store_id = assignment['location']
            strategy = assignment['strategy']
            demand = assignment['demand']
            
            if strategy == 'split':
                vehicles_involved = [f"V{v_id}" for v_id in assignment['vehicles']]
                delivery_text = f"Store-{store_id}: {demand} units (split: {', '.join(vehicles_involved)})"
            else:
                vehicle_id = assignment['vehicles'][0]
                delivery_text = f"Store-{store_id}: {demand} units (Vehicle-{vehicle_id})"
            
            ax_table.text(0.05, y_pos, f"• {delivery_text}", fontsize=9, 
                         transform=ax_table.transAxes, wrap=True)
            y_pos -= 0.05


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
