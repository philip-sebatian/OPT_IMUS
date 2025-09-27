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

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.vehicle import VehicleState
from src.core.delivery_task import DeliveryTask
from src.core.depot_manager import DepotManager


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
        # Use NetworkX's spring layout with distance weights and better parameters
        pos = nx.spring_layout(self.G, weight='weight', k=2, iterations=100, seed=42)
        
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
        pos = self._prevent_overlap(pos)
        
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
    
    def create_visualization(self, filename: str = "optimus_routing.png", 
                           show_legend: bool = True, figsize: Tuple[int, int] = (24, 18)):
        """
        Create and save a static visualization with route information displayed cleanly.
        
        Args:
            filename: Output filename for the visualization
            show_legend: Whether to show the legend
            figsize: Figure size as (width, height)
        """
        # Create single figure for main graph
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle('Optimus Routing System - Vehicle Paths and Deliveries', 
                    fontsize=16, fontweight='bold')
        
        # Create distance-based layout where edge lengths are proportional to actual distances
        pos = self._create_distance_based_layout()
        
        # Update the graph with the new positions
        for node, position in pos.items():
            self.G.nodes[node]['position'] = position
        
        # Draw edges
        nx.draw_networkx_edges(self.G, pos, ax=ax, alpha=0.3, edge_color='gray')
        
        # Draw depot nodes
        depot_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['node_type'] == 'depot']
        if depot_nodes:
            nx.draw_networkx_nodes(self.G, pos, nodelist=depot_nodes, 
                                 node_color='red', node_size=1000, 
                                 node_shape='s', ax=ax, alpha=0.8, label='Depots')
            # Add depot labels with better formatting
            depot_labels = {n: f'Depot-{n}' for n in depot_nodes}
            nx.draw_networkx_labels(self.G, pos, depot_labels, ax=ax, font_size=14, font_weight='bold')
        
        # Draw store nodes
        store_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['node_type'] == 'store']
        if store_nodes:
            nx.draw_networkx_nodes(self.G, pos, nodelist=store_nodes, 
                                 node_color='blue', node_size=800, 
                                 node_shape='o', ax=ax, alpha=0.8, label='Stores')
            # Add store labels with demand
            store_labels = {}
            for n in store_nodes:
                demand = self.G.nodes[n]['demand']
                store_labels[n] = f'Store-{n}\n({demand})'
            nx.draw_networkx_labels(self.G, pos, store_labels, ax=ax, font_size=12, font_weight='bold')
        
        # Draw vehicle paths with arrows to show direction
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
                    
                    # Draw the line (thinner, no arrows)
                    ax.plot(x_coords, y_coords, color=color, linewidth=2, 
                           alpha=0.8, label=f'Vehicle {vehicle_id}' if i == 0 else "")
        
        # Delivery information removed to reduce clutter
        
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
        
        # Add route information as text boxes positioned outside the graph area
        self._add_route_information(ax, pos)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {filename}")
        return filename
    
    def _add_route_information(self, ax, pos):
        """Add route information as text boxes positioned to avoid overlap."""
        depot_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['node_type'] == 'depot']
        
        # Get the bounds of the graph
        x_coords = [pos[node][0] for node in pos]
        y_coords = [pos[node][1] for node in pos]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Position route information outside the graph area
        route_x = x_max + (x_max - x_min) * 0.1  # Right side of graph
        route_y_start = y_max - (y_max - y_min) * 0.1  # Top of graph
        
        # Add route information for each vehicle
        for i, (vehicle_id, vehicle) in enumerate(self.vehicles.items()):
            if vehicle.route:
                # Format route with proper names
                route_str = " → ".join([f"Depot-{r}" if r in depot_nodes else f"Store-{r}" for r in vehicle.route])
            else:
                route_str = f"Depot-{vehicle.position}"
            
            # Create route text box
            route_text = f"Vehicle-{vehicle_id}:\n{route_str}\nCost: {vehicle.total_cost:.1f}\nStock: {vehicle.current_stock}/{vehicle.capacity}"
            
            # Position the text box
            current_y = route_y_start - (i * 0.15 * (y_max - y_min))
            
            # Color the text box with vehicle color
            color = self.vehicle_colors[vehicle_id]
            
            ax.text(route_x, current_y, route_text, 
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.3, edgecolor=color),
                   ha='left', va='top')
        
        # Delivery summary removed to reduce clutter


def create_static_visualization(vehicles: List[VehicleState], delivery_tasks: List[DeliveryTask], 
                               depot_manager: DepotManager, results: Dict[str, Any],
                               filename: str = "optimus_routing.png") -> str:
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
