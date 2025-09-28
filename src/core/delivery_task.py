"""
Delivery task management for the Optimus routing system.

This module defines the DeliveryTask class which represents individual
delivery requirements and tracks their completion status.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union


@dataclass
class DeliveryTask:
    """
    Represents a delivery task that can be split across multiple vehicles.
    
    Attributes:
        location: Target location for delivery
        demand: Amount to be delivered
        assigned_vehicle: ID of the vehicle assigned to this task (or list for split)
        completed: Whether the task has been completed
        split_deliveries: List of (vehicle_id, amount) tuples for split deliveries
    """
    
    location: int
    demand: int
    assigned_vehicle: Optional[Union[int, List[int]]] = None
    completed: bool = False
    split_deliveries: List[Tuple[int, int]] = field(default_factory=list)
    
    def is_split_delivery(self) -> bool:
        """
        Check if this task is a split delivery.
        
        Returns:
            True if the task is split across multiple vehicles, False otherwise
        """
        return len(self.split_deliveries) > 1
    
    def get_total_assigned(self) -> int:
        """
        Get the total amount assigned across all vehicles.
        
        Returns:
            Total amount assigned
        """
        return sum(amount for _, amount in self.split_deliveries)
    
    def is_fully_assigned(self) -> bool:
        """
        Check if the full demand has been assigned.
        
        Returns:
            True if fully assigned, False otherwise
        """
        return self.get_total_assigned() == self.demand
    
    def add_split_delivery(self, vehicle_id: int, amount: int) -> bool:
        """
        Add a split delivery assignment.
        
        Args:
            vehicle_id: ID of the vehicle
            amount: Amount to be delivered by this vehicle
            
        Returns:
            True if assignment was successful, False otherwise
        """
        if self.get_total_assigned() + amount > self.demand:
            return False  # Would exceed demand
        
        self.split_deliveries.append((vehicle_id, amount))
        return True
    
    def get_remaining_demand(self) -> int:
        """
        Get the remaining unassigned demand.
        
        Returns:
            Remaining demand
        """
        return self.demand - self.get_total_assigned()
    
    def get_vehicles_involved(self) -> List[int]:
        """
        Get list of vehicle IDs involved in this task.
        
        Returns:
            List of vehicle IDs
        """
        if self.split_deliveries:
            return [vehicle_id for vehicle_id, _ in self.split_deliveries]
        elif self.assigned_vehicle is not None:
            if isinstance(self.assigned_vehicle, list):
                return self.assigned_vehicle
            else:
                return [self.assigned_vehicle]
        return []
    
    def get_vehicle_amount(self, vehicle_id: int) -> int:
        """
        Get the amount assigned to a specific vehicle.
        
        Args:
            vehicle_id: ID of the vehicle
            
        Returns:
            Amount assigned to the vehicle, 0 if not assigned
        """
        for vid, amount in self.split_deliveries:
            if vid == vehicle_id:
                return amount
        return 0
    
    def mark_completed(self) -> None:
        """Mark the task as completed."""
        self.completed = True
    
    def is_valid(self) -> bool:
        """
        Check if the task is valid.
        
        Returns:
            True if valid, False otherwise
        """
        return (self.location >= 0 and 
                self.demand > 0 and 
                self.get_total_assigned() <= self.demand)
    
    def __str__(self) -> str:
        """String representation of the delivery task."""
        if self.split_deliveries:
            vehicles_str = ", ".join([f"V{v_id}({amount})" for v_id, amount in self.split_deliveries])
            return f"Task: {self.demand} units to location {self.location} [{vehicles_str}]"
        else:
            return f"Task: {self.demand} units to location {self.location}"
    
    def __repr__(self) -> str:
        """Detailed string representation of the delivery task."""
        return (f"DeliveryTask(location={self.location}, demand={self.demand}, "
                f"assigned_vehicle={self.assigned_vehicle}, completed={self.completed}, "
                f"split_deliveries={self.split_deliveries})")
