"""
Graph manipulation utilities for the Optimus routing system.

This module provides utilities for working with graphs, including
graph creation, validation, and manipulation functions.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import networkx as nx


class GraphUtils:
    """
    Utility class for graph manipulation and analysis.
    """
    
    @staticmethod
    def create_graph_from_waypoint_matrix(offsets: np.ndarray, edges: np.ndarray, 
                                        weights: np.ndarray) -> nx.Graph:
        """
        Create a NetworkX graph from cuOpt waypoint matrix format.
        
        Args:
            offsets: Graph offsets
            edges: Graph edges
            weights: Edge weights
            
        Returns:
            NetworkX graph object
        """
        G = nx.Graph()
        
        # Add nodes
        num_nodes = len(offsets) - 1
        for i in range(num_nodes):
            G.add_node(i)
        
        # Add edges
        for i in range(num_nodes):
            start_idx = offsets[i]
            end_idx = offsets[i + 1]
            
            for j in range(start_idx, end_idx):
                neighbor = edges[j]
                weight = weights[j]
                G.add_edge(i, neighbor, weight=weight)
        
        return G
    
    @staticmethod
    def validate_graph_connectivity(offsets: np.ndarray, edges: np.ndarray, 
                                  weights: np.ndarray) -> bool:
        """
        Validate that the graph is connected.
        
        Args:
            offsets: Graph offsets
            edges: Graph edges
            weights: Edge weights
            
        Returns:
            True if graph is connected, False otherwise
        """
        G = GraphUtils.create_graph_from_waypoint_matrix(offsets, edges, weights)
        return nx.is_connected(G)
    
    @staticmethod
    def find_shortest_path(offsets: np.ndarray, edges: np.ndarray, weights: np.ndarray,
                          source: int, target: int) -> Tuple[List[int], float]:
        """
        Find shortest path between two nodes.
        
        Args:
            offsets: Graph offsets
            edges: Graph edges
            weights: Edge weights
            source: Source node
            target: Target node
            
        Returns:
            Tuple of (path, total_weight)
        """
        G = GraphUtils.create_graph_from_waypoint_matrix(offsets, edges, weights)
        
        try:
            path = nx.shortest_path(G, source, target, weight='weight')
            total_weight = nx.shortest_path_length(G, source, target, weight='weight')
            return path, total_weight
        except nx.NetworkXNoPath:
            return [], float('inf')
    
    @staticmethod
    def get_node_degree(offsets: np.ndarray, edges: np.ndarray, node: int) -> int:
        """
        Get the degree of a specific node.
        
        Args:
            offsets: Graph offsets
            edges: Graph edges
            node: Node ID
            
        Returns:
            Degree of the node
        """
        if node >= len(offsets) - 1:
            return 0
        
        start_idx = offsets[node]
        end_idx = offsets[node + 1]
        return end_idx - start_idx
    
    @staticmethod
    def get_neighbors(offsets: np.ndarray, edges: np.ndarray, node: int) -> List[int]:
        """
        Get neighbors of a specific node.
        
        Args:
            offsets: Graph offsets
            edges: Graph edges
            node: Node ID
            
        Returns:
            List of neighbor node IDs
        """
        if node >= len(offsets) - 1:
            return []
        
        start_idx = offsets[node]
        end_idx = offsets[node + 1]
        return edges[start_idx:end_idx].tolist()
    
    @staticmethod
    def calculate_graph_density(offsets: np.ndarray, edges: np.ndarray) -> float:
        """
        Calculate the density of the graph.
        
        Args:
            offsets: Graph offsets
            edges: Graph edges
            
        Returns:
            Graph density (0.0 to 1.0)
        """
        num_nodes = len(offsets) - 1
        num_edges = len(edges) // 2  # Undirected graph
        
        max_edges = num_nodes * (num_nodes - 1) // 2
        return num_edges / max_edges if max_edges > 0 else 0.0
    
    @staticmethod
    def find_central_nodes(offsets: np.ndarray, edges: np.ndarray, 
                          weights: np.ndarray, top_k: int = 5) -> List[int]:
        """
        Find the most central nodes in the graph.
        
        Args:
            offsets: Graph offsets
            edges: Graph edges
            weights: Edge weights
            top_k: Number of top central nodes to return
            
        Returns:
            List of most central node IDs
        """
        G = GraphUtils.create_graph_from_waypoint_matrix(offsets, edges, weights)
        
        # Calculate betweenness centrality
        centrality = nx.betweenness_centrality(G, weight='weight')
        
        # Sort by centrality and return top k
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:top_k]]
    
    @staticmethod
    def validate_waypoint_matrix(offsets: np.ndarray, edges: np.ndarray, 
                               weights: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate a waypoint matrix for correctness.
        
        Args:
            offsets: Graph offsets
            edges: Graph edges
            weights: Edge weights
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check offsets
        if len(offsets) < 2:
            errors.append("Offsets must have at least 2 elements")
        
        if not np.all(offsets[1:] >= offsets[:-1]):
            errors.append("Offsets must be non-decreasing")
        
        if offsets[0] != 0:
            errors.append("First offset must be 0")
        
        if offsets[-1] != len(edges):
            errors.append("Last offset must equal length of edges")
        
        # Check edges
        if len(edges) != len(weights):
            errors.append("Edges and weights must have same length")
        
        num_nodes = len(offsets) - 1
        if np.any(edges >= num_nodes) or np.any(edges < 0):
            errors.append("Edge indices must be valid node indices")
        
        # Check weights
        if np.any(weights < 0):
            errors.append("Weights must be non-negative")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def create_random_graph(num_nodes: int, density: float = 0.3, 
                          weight_range: Tuple[float, float] = (1.0, 10.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a random graph in waypoint matrix format.
        
        Args:
            num_nodes: Number of nodes in the graph
            density: Graph density (0.0 to 1.0)
            weight_range: Range for edge weights
            
        Returns:
            Tuple of (offsets, edges, weights)
        """
        # Create random graph
        G = nx.erdos_renyi_graph(num_nodes, density)
        
        # Convert to waypoint matrix format
        offsets = [0]
        edges = []
        weights = []
        
        for node in range(num_nodes):
            neighbors = list(G.neighbors(node))
            # Only include neighbors with higher index to avoid duplicates
            neighbors = [n for n in neighbors if n > node]
            
            offsets.append(offsets[-1] + len(neighbors))
            edges.extend(neighbors)
            
            # Add random weights
            node_weights = np.random.uniform(weight_range[0], weight_range[1], len(neighbors))
            weights.extend(node_weights)
        
        return np.array(offsets), np.array(edges), np.array(weights)
    
    @staticmethod
    def visualize_graph(offsets: np.ndarray, edges: np.ndarray, weights: np.ndarray,
                       title: str = "Graph Visualization") -> None:
        """
        Visualize the graph (requires matplotlib).
        
        Args:
            offsets: Graph offsets
            edges: Graph edges
            weights: Edge weights
            title: Title for the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            G = GraphUtils.create_graph_from_waypoint_matrix(offsets, edges, weights)
            
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=500, alpha=0.9)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, alpha=0.5)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos)
            
            plt.title(title)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")
        except Exception as e:
            print(f"Error in visualization: {e}")
