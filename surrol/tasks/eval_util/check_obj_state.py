"""
Object State Checking Module
Provides foundational utility functions for detecting and validating object states in PyBullet.
"""
import pybullet as p
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class ObjectStateChecker:
    """
    Utility class for assessing physical interactions and spatial relationships of objects.
    """

    @staticmethod
    def check_contact(obj1_id: int, obj2_id: int) -> int:
        """
        Check for physical contact between two objects.

        Args:
            obj1_id: ID of the first object.
            obj2_id: ID of the second object.

        Returns:
            int: The number of active contact points between the objects.
        """
        contact_points = p.getContactPoints(bodyA=obj1_id, bodyB=obj2_id)
        return len(contact_points)

    @staticmethod
    def check_contact_with_force(obj1_id: int, obj2_id: int, min_force: float = 0.0) -> Dict[str, Any]:
        """
        Check for contact between two objects and evaluate the applied normal forces.

        Args:
            obj1_id: ID of the first object.
            obj2_id: ID of the second object.
            min_force: Minimum normal force threshold to register a valid contact.

        Returns:
            dict: Contains boolean contact status, maximum force, and contact point details.
        """
        contact_points = p.getContactPoints(bodyA=obj1_id, bodyB=obj2_id)

        if not contact_points:
            return {
                'in_contact': False,
                'max_force': 0.0,
                'num_contacts': 0,
                'contact_points': []
            }

        forces = [point[9] for point in contact_points]  # Normal force
        max_force = max(forces) if forces else 0.0

        return {
            'in_contact': max_force >= min_force,
            'max_force': max_force,
            'num_contacts': len(contact_points),
            'contact_points': contact_points
        }

    @staticmethod
    def get_all_contacts(obj_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all contact points within the environment or for a specific object.

        Args:
            obj_id: Optional; target object ID.

        Returns:
            List[Dict]: A list of dictionaries containing detailed contact point information.
        """
        if obj_id is not None:
            contact_points = p.getContactPoints(bodyA=obj_id) + p.getContactPoints(bodyB=obj_id)
        else:
            contact_points = p.getContactPoints()

        contacts_info = []
        for i, point in enumerate(contact_points):
            contacts_info.append({
                'id': i,
                'bodyA': point[1],
                'bodyB': point[2],
                'linkA': point[3],
                'linkB': point[4],
                'position': point[5],
                'normal': point[7],
                'distance': point[8],
                'force': point[9]
            })

        return contacts_info

    @staticmethod
    def get_object_height(obj_id: int) -> float:
        """
        Get the current height (Z-coordinate) of an object.

        Args:
            obj_id: Target object ID.

        Returns:
            float: The Z-coordinate of the object's base.
        """
        pos, _ = p.getBasePositionAndOrientation(obj_id)
        return pos[2]

    @staticmethod
    def get_object_position(obj_id: int) -> np.ndarray:
        """
        Get the current 3D position of an object.

        Args:
            obj_id: Target object ID.

        Returns:
            np.ndarray: The [x, y, z] coordinates of the object.
        """
        pos, _ = p.getBasePositionAndOrientation(obj_id)
        return np.array(pos)

    @staticmethod
    def get_object_orientation(obj_id: int) -> np.ndarray:
        """
        Get the current orientation of an object as a quaternion.

        Args:
            obj_id: Target object ID.

        Returns:
            np.ndarray: The quaternion [x, y, z, w].
        """
        _, orn = p.getBasePositionAndOrientation(obj_id)
        return np.array(orn)

    @staticmethod
    def get_object_velocity(obj_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the linear and angular velocity of an object.

        Args:
            obj_id: Target object ID.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing (linear_velocity, angular_velocity).
        """
        linear_vel, angular_vel = p.getBaseVelocity(obj_id)
        return np.array(linear_vel), np.array(angular_vel)

    @staticmethod
    def get_object_aabb(obj_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the Axis-Aligned Bounding Box (AABB) of an object.

        Args:
            obj_id: Target object ID.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing (min_bounds, max_bounds).
        """
        aabb_min, aabb_max = p.getAABB(obj_id)
        return np.array(aabb_min), np.array(aabb_max)

    @staticmethod
    def track_height_trajectory(obj_id: int, initial_height: float = None) -> Dict[str, Any]:
        """
        Track the vertical displacement of an object relative to its initial state.

        Args:
            obj_id: Target object ID.
            initial_height: The baseline height. If None, defaults to the current height.

        Returns:
            dict: Displacement metrics, including net change and boolean states for rising/falling.
        """
        current_height = ObjectStateChecker.get_object_height(obj_id)

        if initial_height is None:
            initial_height = current_height

        height_change = current_height - initial_height

        return {
            'current_height': current_height,
            'initial_height': initial_height,
            'height_change': height_change,
            'is_rising': height_change > 0,
            'rising_amount': max(0, height_change),
            'is_falling': height_change < 0,
            'falling_amount': abs(min(0, height_change))
        }

    @staticmethod
    def check_vertical_relationship(obj1_id: int, obj2_id: int,
                                    obj1_name: str = "Gauze",
                                    obj2_name: str = "Block") -> Dict[str, Any]:
        """
        Determine the spatial relationship between two objects, primarily along the Z-axis.

        Args:
            obj1_id: ID of the primary object.
            obj2_id: ID of the secondary object.
            obj1_name: String identifier for object 1 (used in relationship string).
            obj2_name: String identifier for object 2 (used in relationship string).

        Returns:
            dict: Assessment of overlap and vertical positioning.
        """
        # Retrieve AABBs
        aabb1_min, aabb1_max = ObjectStateChecker.get_object_aabb(obj1_id)
        aabb2_min, aabb2_max = ObjectStateChecker.get_object_aabb(obj2_id)

        # Retrieve base heights
        obj_height1 = ObjectStateChecker.get_object_height(obj1_id)
        obj_height2 = ObjectStateChecker.get_object_height(obj2_id)

        # Evaluate vertical offset
        z_overlap = obj_height1 - obj_height2

        # Check for planar overlap (X-Y plane)
        x_overlap = not (aabb1_min[0] > aabb2_max[0] or aabb1_max[0] < aabb2_min[0])
        y_overlap = not (aabb1_min[1] > aabb2_max[1] or aabb1_max[1] < aabb2_min[1])

        horizontal_overlap = x_overlap and y_overlap

        # Determine relative vertical placement
        if horizontal_overlap:
            if z_overlap < 0:
                relationship = f"{obj1_name} is below {obj2_name}"
                vertical_position = "below"
            elif z_overlap > 0:
                relationship = f"{obj1_name} is on {obj2_name}"
                vertical_position = "above"
            else:
                relationship = f"{obj1_name} and {obj2_name} overlap horizontally"
                vertical_position = "overlap"
        else:
            relationship = f"{obj1_name} and {obj2_name} do not overlap horizontally"
            vertical_position = "no_horizontal_overlap"

        return {
            'obj1_name': obj1_name,
            'obj2_name': obj2_name,
            'horizontal_overlap': bool(horizontal_overlap),
            'vertical_overlap': z_overlap,
            'vertical_position': vertical_position,
            'relationship': relationship
        }

    @staticmethod
    def check_stable_grasp(gripper_id: int, object_id: int,
                           min_contact_points: int = 2,
                           min_force: float = 0.1) -> Dict[str, Any]:
        """
        Evaluate whether the end effector has established a stable grasp on an object.

        Args:
            gripper_id: ID of the robotic gripper/end-effector.
            object_id: Target object ID.
            min_contact_points: Required number of contact points for stability.
            min_force: Required normal force threshold.

        Returns:
            dict: Boolean grasp status and supporting physical metrics.
        """
        contact_info = ObjectStateChecker.check_contact_with_force(gripper_id, object_id, min_force)

        object_pos = ObjectStateChecker.get_object_position(object_id)
        gripper_pos = ObjectStateChecker.get_object_position(gripper_id)

        distance = np.linalg.norm(object_pos - gripper_pos)

        is_grasped = (contact_info['in_contact'] and
                      contact_info['num_contacts'] >= min_contact_points)

        return {
            'is_grasped': is_grasped,
            'contact_info': contact_info,
            'distance': float(distance),
            'relative_position': (object_pos - gripper_pos).tolist()
        }

    @staticmethod
    def check_object_moved(obj_id: int, initial_position: np.ndarray,
                           movement_threshold: float = 0.01) -> Dict[str, Any]:
        """
        Verify if an object has displaced significantly from its starting position.

        Args:
            obj_id: Target object ID.
            initial_position: Baseline 3D coordinates.
            movement_threshold: The minimum distance required to qualify as movement.

        Returns:
            dict: Boolean movement flag and translation vectors.
        """
        current_pos = ObjectStateChecker.get_object_position(obj_id)
        displacement = np.linalg.norm(current_pos - initial_position)

        return {
            'has_moved': displacement > movement_threshold,
            'displacement': float(displacement),
            'current_position': current_pos.tolist(),
            'initial_position': initial_position.tolist(),
            'movement_vector': (current_pos - initial_position).tolist()
        }


# Singleton instance for global utility access
state_checker = ObjectStateChecker()

# Exported utility functions
check_contact = state_checker.check_contact
check_contact_with_force = state_checker.check_contact_with_force
get_object_height = state_checker.get_object_height
get_object_position = state_checker.get_object_position
track_height_trajectory = state_checker.track_height_trajectory
check_vertical_relationship = state_checker.check_vertical_relationship
check_stable_grasp = state_checker.check_stable_grasp