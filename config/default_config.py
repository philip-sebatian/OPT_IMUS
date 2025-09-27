"""
Default configuration for the Optimus routing system.

This module contains default configuration parameters for various
components of the system.
"""

# Default optimization parameters
DEFAULT_OPTIMIZATION_CONFIG = {
    'time_limit': 30.0,              # Maximum solve time (seconds)
    'enable_split_delivery': True,    # Enable split delivery optimization
    'closest_depot_only': True,      # Always use closest depot for refilling
    'cost_tolerance': 0.01,          # Cost comparison tolerance
    'max_split_vehicles': 5,         # Maximum vehicles for split delivery
    'verbose': False,                # Verbose output
}

# Default vehicle parameters
DEFAULT_VEHICLE_CONFIG = {
    'min_capacity': 1,               # Minimum vehicle capacity
    'max_capacity': 100,             # Maximum vehicle capacity
    'default_capacity': 10,          # Default vehicle capacity
    'min_stock': 0,                  # Minimum initial stock
    'max_stock_ratio': 1.0,          # Maximum stock as ratio of capacity
}

# Default graph parameters
DEFAULT_GRAPH_CONFIG = {
    'min_nodes': 2,                  # Minimum number of nodes
    'max_nodes': 1000,               # Maximum number of nodes
    'min_edges_per_node': 1,         # Minimum edges per node
    'max_edges_per_node': 10,        # Maximum edges per node
    'min_weight': 0.1,               # Minimum edge weight
    'max_weight': 1000.0,            # Maximum edge weight
}

# Default depot parameters
DEFAULT_DEPOT_CONFIG = {
    'min_depots': 1,                 # Minimum number of depots
    'max_depots': 10,                # Maximum number of depots
    'default_depot_capacity': float('inf'),  # Default depot capacity
}

# Default delivery parameters
DEFAULT_DELIVERY_CONFIG = {
    'min_demand': 1,                 # Minimum delivery demand
    'max_demand': 100,               # Maximum delivery demand
    'min_deliveries': 1,             # Minimum number of deliveries
    'max_deliveries': 100,           # Maximum number of deliveries
}

# Default cost calculation parameters
DEFAULT_COST_CONFIG = {
    'distance_weight': 1.0,          # Weight for distance in cost calculation
    'time_weight': 0.1,              # Weight for time in cost calculation
    'refill_penalty': 0.0,           # Penalty for refill operations
    'split_penalty': 0.0,            # Penalty for split deliveries
}

# Default validation parameters
DEFAULT_VALIDATION_CONFIG = {
    'strict_validation': True,       # Enable strict input validation
    'warn_on_issues': True,          # Warn about potential issues
    'validate_graph_connectivity': True,  # Validate graph connectivity
    'validate_capacity_constraints': True,  # Validate capacity constraints
}

# Default logging parameters
DEFAULT_LOGGING_CONFIG = {
    'level': 'INFO',                 # Logging level
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': None,                    # Log file (None for console only)
    'max_file_size': 10 * 1024 * 1024,  # Max log file size (10MB)
    'backup_count': 5,               # Number of backup log files
}

# Default performance parameters
DEFAULT_PERFORMANCE_CONFIG = {
    'max_memory_usage': 1024 * 1024 * 1024,  # Max memory usage (1GB)
    'enable_caching': True,          # Enable result caching
    'cache_size': 1000,              # Maximum cache size
    'parallel_processing': True,     # Enable parallel processing
    'max_workers': None,             # Maximum number of workers (None for auto)
}

# Default visualization parameters
DEFAULT_VISUALIZATION_CONFIG = {
    'figure_size': (10, 8),          # Default figure size
    'node_size': 500,                # Default node size
    'edge_width': 1.0,               # Default edge width
    'font_size': 12,                 # Default font size
    'color_scheme': 'default',       # Default color scheme
    'show_labels': True,             # Show node labels
    'show_weights': False,           # Show edge weights
}

# Combined default configuration
DEFAULT_CONFIG = {
    'optimization': DEFAULT_OPTIMIZATION_CONFIG,
    'vehicle': DEFAULT_VEHICLE_CONFIG,
    'graph': DEFAULT_GRAPH_CONFIG,
    'depot': DEFAULT_DEPOT_CONFIG,
    'delivery': DEFAULT_DELIVERY_CONFIG,
    'cost': DEFAULT_COST_CONFIG,
    'validation': DEFAULT_VALIDATION_CONFIG,
    'logging': DEFAULT_LOGGING_CONFIG,
    'performance': DEFAULT_PERFORMANCE_CONFIG,
    'visualization': DEFAULT_VISUALIZATION_CONFIG,
}
