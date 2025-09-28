#!/usr/bin/env python3
"""
Unit tests for the VehicleState class.
"""

import unittest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from workspace_optimus.src.core.vehicle import VehicleState


class TestVehicleState(unittest.TestCase):
    """Test cases for VehicleState class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vehicle = VehicleState(
            id=0,
            capacity=10,
            current_stock=5,
            position=0
        )
    
    def test_initialization(self):
        """Test vehicle initialization."""
        self.assertEqual(self.vehicle.id, 0)
        self.assertEqual(self.vehicle.capacity, 10)
        self.assertEqual(self.vehicle.current_stock, 5)
        self.assertEqual(self.vehicle.position, 0)
        self.assertEqual(self.vehicle.total_cost, 0.0)
        self.assertEqual(self.vehicle.route, [0])
    
    def test_can_carry(self):
        """Test can_carry method."""
        self.assertTrue(self.vehicle.can_carry(5))
        self.assertTrue(self.vehicle.can_carry(10))
        self.assertFalse(self.vehicle.can_carry(11))
        self.assertFalse(self.vehicle.can_carry(15))
    
    def test_has_stock(self):
        """Test has_stock method."""
        self.assertTrue(self.vehicle.has_stock(5))
        self.assertTrue(self.vehicle.has_stock(3))
        self.assertFalse(self.vehicle.has_stock(6))
        self.assertFalse(self.vehicle.has_stock(10))
    
    def test_can_deliver(self):
        """Test can_deliver method."""
        self.assertTrue(self.vehicle.can_deliver(5))
        self.assertTrue(self.vehicle.can_deliver(3))
        self.assertFalse(self.vehicle.can_deliver(6))  # Not enough stock
        self.assertFalse(self.vehicle.can_deliver(11))  # Exceeds capacity
    
    def test_refill(self):
        """Test refill method."""
        self.vehicle.refill()
        self.assertEqual(self.vehicle.current_stock, 10)
        self.assertTrue(self.vehicle.is_full())
    
    def test_deliver(self):
        """Test deliver method."""
        # Successful delivery
        result = self.vehicle.deliver(3)
        self.assertTrue(result)
        self.assertEqual(self.vehicle.current_stock, 2)
        
        # Failed delivery (insufficient stock)
        result = self.vehicle.deliver(5)
        self.assertFalse(result)
        self.assertEqual(self.vehicle.current_stock, 2)  # Unchanged
    
    def test_move_to(self):
        """Test move_to method."""
        self.vehicle.move_to(5)
        self.assertEqual(self.vehicle.position, 5)
        self.assertEqual(self.vehicle.route, [0, 5])
    
    def test_add_cost(self):
        """Test add_cost method."""
        self.vehicle.add_cost(10.5)
        self.assertEqual(self.vehicle.total_cost, 10.5)
        
        self.vehicle.add_cost(5.0)
        self.assertEqual(self.vehicle.total_cost, 15.5)
    
    def test_get_utilization(self):
        """Test get_utilization method."""
        self.assertEqual(self.vehicle.get_utilization(), 0.5)  # 5/10
        
        self.vehicle.current_stock = 0
        self.assertEqual(self.vehicle.get_utilization(), 0.0)
        
        self.vehicle.current_stock = 10
        self.assertEqual(self.vehicle.get_utilization(), 1.0)
    
    def test_get_remaining_capacity(self):
        """Test get_remaining_capacity method."""
        self.assertEqual(self.vehicle.get_remaining_capacity(), 5)  # 10 - 5
        
        self.vehicle.current_stock = 0
        self.assertEqual(self.vehicle.get_remaining_capacity(), 10)
        
        self.vehicle.current_stock = 10
        self.assertEqual(self.vehicle.get_remaining_capacity(), 0)
    
    def test_is_empty(self):
        """Test is_empty method."""
        self.assertFalse(self.vehicle.is_empty())
        
        self.vehicle.current_stock = 0
        self.assertTrue(self.vehicle.is_empty())
    
    def test_is_full(self):
        """Test is_full method."""
        self.assertFalse(self.vehicle.is_full())
        
        self.vehicle.current_stock = 10
        self.assertTrue(self.vehicle.is_full())
    
    def test_string_representations(self):
        """Test string representations."""
        str_repr = str(self.vehicle)
        self.assertIn("Vehicle 0", str_repr)
        self.assertIn("pos=0", str_repr)
        self.assertIn("stock=5/10", str_repr)
        
        repr_str = repr(self.vehicle)
        self.assertIn("VehicleState", repr_str)
        self.assertIn("id=0", repr_str)
        self.assertIn("capacity=10", repr_str)


if __name__ == '__main__':
    unittest.main()
