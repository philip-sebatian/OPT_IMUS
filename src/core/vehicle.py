"""
Vehicle state management for the Optimus routing system.

This module defines the VehicleState class which tracks the current state
of each vehicle including position, capacity, stock levels, and route history.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VehicleState:
    """
    Represents the current state of a vehicle in the routing system.
    
    Attributes:
        id: Unique identifier for the vehicle
        capacity: Maximum carrying capacity of the vehicle
        current_stock: Current amount of stock in the vehicle
        position: Current location of the vehicle
        total_cost: Total cost accumulated by this vehicle
        route: List of locations visited by this vehicle
    """
    
    id: int
    capacity: int
    current_stock: int
    position: int
    total_cost: float = 0.0
    route: Optional[List[int]] = None
    
    def __post_init__(self):
        """Initialize route if not provided."""
        if self.route is None:
            self.route = [self.position]
    
    def can_carry(self, amount: int) -> bool:
        """
        Check if the vehicle can carry the specified amount.
        
        Args:
            amount: Amount to check against capacity
            
        Returns:
            True if the vehicle can carry the amount, False otherwise
        """
        return amount <= self.capacity
    
    def has_stock(self, amount: int) -> bool:
        """
        Check if the vehicle has sufficient stock.
        
        Args:
            amount: Required amount of stock
            
        Returns:
            True if the vehicle has sufficient stock, False otherwise
        """
        return self.current_stock >= amount
    
    def can_deliver(self, amount: int) -> bool:
        """
        Check if the vehicle can deliver the specified amount.
        
        Args:
            amount: Amount to deliver
            
        Returns:
            True if the vehicle can deliver the amount, False otherwise
        """
        return self.has_stock(amount) and self.can_carry(amount)
    
    def refill(self) -> None:
        """Refill the vehicle to maximum capacity."""
        self.current_stock = self.capacity
    
    def deliver(self, amount: int) -> bool:
        """
        Deliver the specified amount and update stock.
        
        Args:
            amount: Amount to deliver
            
        Returns:
            True if delivery was successful, False otherwise
        """
        if not self.can_deliver(amount):
            return False
        
        self.current_stock -= amount
        return True
    
    def move_to(self, location: int) -> None:
        """
        Move the vehicle to a new location.
        
        Args:
            location: New location for the vehicle
        """
        self.position = location
        self.route.append(location)
    
    def add_cost(self, cost: float) -> None:
        """
        Add cost to the vehicle's total cost.
        
        Args:
            cost: Cost to add
        """
        self.total_cost += cost
    
    def get_utilization(self) -> float:
        """
        Get the current utilization of the vehicle.
        
        Returns:
            Utilization as a percentage (0.0 to 1.0)
        """
        if self.capacity == 0:
            return 0.0
        return self.current_stock / self.capacity
    
    def get_remaining_capacity(self) -> int:
        """
        Get the remaining capacity of the vehicle.
        
        Returns:
            Remaining capacity
        """
        return self.capacity - self.current_stock
    
    def is_empty(self) -> bool:
        """
        Check if the vehicle is empty.
        
        Returns:
            True if the vehicle has no stock, False otherwise
        """
        return self.current_stock == 0
    
    def is_full(self) -> bool:
        """
        Check if the vehicle is at full capacity.
        
        Returns:
            True if the vehicle is at full capacity, False otherwise
        """
        return self.current_stock == self.capacity
    
    def __str__(self) -> str:
        """String representation of the vehicle state."""
        return (f"Vehicle {self.id}: pos={self.position}, "
                f"stock={self.current_stock}/{self.capacity}, "
                f"cost={self.total_cost:.1f}")
    
    def __repr__(self) -> str:
        """Detailed string representation of the vehicle state."""
        return (f"VehicleState(id={self.id}, capacity={self.capacity}, "
                f"current_stock={self.current_stock}, position={self.position}, "
                f"total_cost={self.total_cost:.1f}, route={self.route})")
