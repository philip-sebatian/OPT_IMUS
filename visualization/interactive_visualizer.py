"""
Interactive visualizer module for Optimus routing system.

This module provides interactive graph visualization capabilities to display
the routing network, vehicle paths, and delivery information in a user-friendly UI.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, CheckButtons
import networkx as nx
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import random

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.vehicle import VehicleState
from src.core.delivery_task import DeliveryTask
from src.core.depot_manager import DepotManager


class OptimusVisualizer:
    """
    Interactive visualizer for the Optimus routing system.
    
    Features:
    - Graph display with depot and store nodes
    - Colored vehicle paths
    - Stock delivery information display
    - Interactive controls for toggling vehicle paths
    """
    
    def __init__(self, vehicles: List[VehicleState], delivery_tasks: List[DeliveryTask], 
                 depot_manager: DepotManager, results: Dict[str, Any]):
        """
        Initialize the visualizer.
        
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
        
        # Track which vehicle paths are visible
        self.visible_vehicles = set(self.vehicles.keys())
        
        # Create the figure and setup
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        self.fig.suptitle('Optimus Routing System - Vehicle Paths and Deliveries', fontsize=16, fontweight='bold')
        
        # Setup interactive controls
        self._setup_controls()
        
    def _create_graph(self) -> nx.Graph:
        """Create the network graph from depot and delivery locations."""
        G = nx.Graph()
        
        # Add depot nodes
        depot_positions = {}
        for depot_id in self.depot_manager.depot_locations:
            # Get position from depot manager or use default
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
        
        # Add edges between all nodes (simplified - in reality would be based on actual connections)
        all_nodes = list(depot_positions.keys()) + list(store_positions.keys())
        for i, node1 in enumerate(all_nodes):
            for node2 in all_nodes[i+1:]:
                distance = self.depot_manager.get_distance(node1, node2)
                if distance != float('inf'):
                    G.add_edge(node1, node2, weight=distance)
        
        return G
    
    def _get_location_position(self, location_id: int) -> Tuple[float, float]:
        """Get the position of a location (depot or store)."""
        # For now, generate positions based on location ID
        # In a real system, this would come from actual coordinates
        np.random.seed(location_id)  # Ensure consistent positioning
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
    
    def _setup_controls(self):
        """Setup interactive controls for the visualization."""
        # Create checkboxes for vehicle visibility
        vehicle_labels = [f'Vehicle {vid}' for vid in sorted(self.vehicles.keys())]
        vehicle_status = [True] * len(vehicle_labels)
        
        # Position the checkboxes
        ax_check = plt.axes([0.02, 0.02, 0.15, 0.3])
        self.check_buttons = CheckButtons(ax_check, vehicle_labels, vehicle_status)
        self.check_buttons.on_clicked(self._toggle_vehicle)
        
        # Add refresh button
        ax_refresh = plt.axes([0.02, 0.35, 0.15, 0.05])
        self.refresh_button = Button(ax_refresh, 'Refresh')
        self.refresh_button.on_clicked(self._refresh_plot)
        
        # Add legend
        self._add_legend()
    
    def _add_legend(self):
        """Add legend to the plot."""
        legend_elements = [
            patches.Patch(color='red', label='Depots'),
            patches.Patch(color='blue', label='Stores'),
            patches.Patch(color='gray', label='Vehicle Paths')
        ]
        
        # Add vehicle colors to legend
        for vid, color in self.vehicle_colors.items():
            legend_elements.append(patches.Patch(color=color, label=f'Vehicle {vid}'))
        
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    def _toggle_vehicle(self, label):
        """Toggle visibility of a vehicle's path."""
        vehicle_id = int(label.split()[1])
        if vehicle_id in self.visible_vehicles:
            self.visible_vehicles.remove(vehicle_id)
        else:
            self.visible_vehicles.add(vehicle_id)
        self._refresh_plot(None)
    
    def _refresh_plot(self, event):
        """Refresh the plot with current settings."""
        self.ax.clear()
        self._draw_graph()
        self._draw_vehicle_paths()
        self._draw_delivery_info()
        self._add_legend()
        self.fig.canvas.draw()
    
    def _draw_graph(self):
        """Draw the base graph with nodes and edges."""
        pos = nx.get_node_attributes(self.G, 'position')
        
        # Draw edges
        nx.draw_networkx_edges(self.G, pos, ax=self.ax, alpha=0.3, edge_color='gray')
        
        # Draw depot nodes
        depot_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['node_type'] == 'depot']
        if depot_nodes:
            nx.draw_networkx_nodes(self.G, pos, nodelist=depot_nodes, 
                                 node_color='red', node_size=500, 
                                 node_shape='s', ax=self.ax, alpha=0.8, label='Depots')
            # Add depot labels
            depot_labels = {n: f'Depot-{n}' for n in depot_nodes}
            nx.draw_networkx_labels(self.G, pos, depot_labels, ax=self.ax, font_size=9, font_weight='bold')
        
        # Draw store nodes
        store_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['node_type'] == 'store']
        if store_nodes:
            nx.draw_networkx_nodes(self.G, pos, nodelist=store_nodes, 
                                 node_color='blue', node_size=350, 
                                 node_shape='o', ax=self.ax, alpha=0.8, label='Stores')
            # Add store labels with demand
            store_labels = {}
            for n in store_nodes:
                demand = self.G.nodes[n]['demand']
                store_labels[n] = f'Store-{n}\n({demand})'
            nx.draw_networkx_labels(self.G, pos, store_labels, ax=self.ax, font_size=8, font_weight='bold')
    
    def _draw_vehicle_paths(self):
        """Draw vehicle paths with different colors."""
        pos = nx.get_node_attributes(self.G, 'position')
        
        for vehicle_id in self.visible_vehicles:
            vehicle = self.vehicles[vehicle_id]
            if not vehicle.route or len(vehicle.route) < 2:
                continue
                
            color = self.vehicle_colors[vehicle_id]
            
            # Draw the path
            path_edges = [(vehicle.route[i], vehicle.route[i+1]) 
                         for i in range(len(vehicle.route)-1)]
            
            for edge in path_edges:
                if self.G.has_edge(edge[0], edge[1]):
                    x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
                    y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
                    self.ax.plot(x_coords, y_coords, color=color, linewidth=2, 
                               alpha=0.7, label=f'Vehicle {vehicle_id}')
    
    def _draw_delivery_info(self):
        """Draw delivery information for each store."""
        pos = nx.get_node_attributes(self.G, 'position')
        
        # Get delivery information from assignments
        store_deliveries = {}
        for assignment in self.results.get('assignments', []):
            store_id = assignment['location']
            if store_id not in store_deliveries:
                store_deliveries[store_id] = []
            
            strategy = assignment['strategy']
            if strategy == 'split':
                # For split deliveries, show individual vehicle contributions
                for v_id, amount in assignment['details']['vehicle_amounts']:
                    if v_id in self.visible_vehicles:
                        store_deliveries[store_id].append(f'V{v_id}: {amount}')
            else:
                # For single vehicle deliveries
                for v_id in assignment['vehicles']:
                    if v_id in self.visible_vehicles:
                        store_deliveries[store_id].append(f'V{v_id}: {assignment["demand"]}')
        
        # Draw delivery information boxes
        for store_id, deliveries in store_deliveries.items():
            if store_id in pos:
                x, y = pos[store_id]
                info_text = '\n'.join(deliveries)
                self.ax.text(x, y + 5, info_text, fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                           ha='center', va='bottom')
    
    def show(self):
        """Display the visualization."""
        self._draw_graph()
        self._draw_vehicle_paths()
        self._draw_delivery_info()
        self.ax.set_title('Optimus Routing System - Vehicle Paths and Deliveries')
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def save(self, filename: str):
        """Save the visualization to a file."""
        self._draw_graph()
        self._draw_vehicle_paths()
        self._draw_delivery_info()
        self.ax.set_title('Optimus Routing System - Vehicle Paths and Deliveries')
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {filename}")


def create_visualization(vehicles: List[VehicleState], delivery_tasks: List[DeliveryTask], 
                        depot_manager: DepotManager, results: Dict[str, Any]) -> OptimusVisualizer:
    """
    Create a visualization of the routing system.
    
    Args:
        vehicles: List of vehicles
        delivery_tasks: List of delivery tasks
        depot_manager: Depot manager
        results: Optimization results
        
    Returns:
        OptimusVisualizer instance
    """
    return OptimusVisualizer(vehicles, delivery_tasks, depot_manager, results)
