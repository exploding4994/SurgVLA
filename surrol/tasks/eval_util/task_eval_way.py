"""
Task Success Evaluation Module
Leverages the ObjectStateChecker to evaluate the completion status of various surgical manipulation tasks.
"""
from typing import Dict, Any, Optional
import numpy as np
from eval_util.check_obj_state import (
    check_contact_with_force,
    track_height_trajectory,
    get_object_height,
    get_object_position,
    check_stable_grasp,
    check_contact,
    check_vertical_relationship
)
from surrol.utils.pybullet_utils import (
    get_link_pose,
    wrap_angle
)
import pybullet as p


class TaskSuccessEvaluator:
    """
    Evaluator class mapped to specific surgical robotics primitives and tasks.
    """

    def __init__(self):
        """Initialize the evaluator and reset internal tracking states."""
        self.reset()

    def reset(self):
        """Clear all internal states for a fresh episode evaluation."""
        self._gauze_initial_height = None
        self._needle_initial_position = None
        self._task_states = {}

    def _get_state_key(self, prompt: str, *args) -> str:
        """Generate a unique tracking key for a specific task and set of entities."""
        obj_ids = '_'.join(str(arg) for arg in args if isinstance(arg, int))
        return f"{prompt}_{obj_ids}"

    def evaluate_pick_gauze(self, robot_id: int, gauze_id: int) -> Dict[str, Any]:
        """
        Evaluate the successful picking up of surgical gauze.

        Args:
            robot_id: ID of the robotic manipulator.
            gauze_id: ID of the surgical gauze object.

        Returns:
            dict: Evaluation results containing binary success flags and telemetry details.
        """
        state_key = self._get_state_key("pick_gauze", robot_id, gauze_id)

        # Initialize tracking records
        if state_key not in self._task_states:
            self._task_states[state_key] = {
                'initial_height': get_object_height(gauze_id),
                'contact_start_time': None,
                'rising_start_time': None
            }

        state = self._task_states[state_key]
        initial_height = state['initial_height']

        # Evaluate physical contact
        contact_info = check_contact(robot_id, gauze_id)

        # Monitor vertical displacement
        height_info = track_height_trajectory(gauze_id, initial_height)

        # Success criteria: active contact AND lifted above threshold
        GAUZE_LIFT_THRESHOLD = 0.01  # 10 mm
        is_success = (contact_info > 1 and
                     height_info['rising_amount'] > GAUZE_LIFT_THRESHOLD)

        result = {
            'success': is_success,
            'done': 1 if is_success else 0,
            'details': {
                'robot_gauze_contact_points_number': contact_info,
                'gauze_height_info': height_info,
                'lift_amount': height_info['rising_amount'],
                'lift_threshold': GAUZE_LIFT_THRESHOLD
            }
        }

        # Debugging output
        print(f"\n[Evaluator] Pick gauze:")
        print(f"  Contact: {contact_info}")
        print(f"  Height change: {height_info['height_change']:.4f}")
        print(f"  Lifted: {height_info['rising_amount']:.4f} > {GAUZE_LIFT_THRESHOLD}")
        print(f"  Success: {is_success}")

        return result

    def evaluate_grasp_needle(self, robot_id: int, needle_id: int) -> Dict[str, Any]:
        """
        Evaluate the acquisition of a stable grasp on a suture needle.

        Args:
            robot_id: ID of the robotic manipulator.
            needle_id: ID of the suture needle.

        Returns:
            dict: Evaluation results based on grasp stability constraints.
        """
        # Validate stable grasp parameters
        grasp_info = check_stable_grasp(robot_id, needle_id,
                                        min_contact_points=2,
                                        min_force=0.01)

        is_success = grasp_info['is_grasped']

        result = {
            'success': is_success,
            'done': 1 if is_success else 0,
            'details': {
                'grasp_info': grasp_info
            }
        }

        print(f"\n[Evaluator] Grasp needle:")
        print(f"  Grasped: {grasp_info['is_grasped']}")
        print(f"  Contact points: {grasp_info['contact_info']['num_contacts']}")
        print(f"  Force: {grasp_info['contact_info']['max_force']:.3f}")
        print(f"  Success: {is_success}")

        return result

    def evaluate_pick_needle(self, robot_id: int, needle_id: int, pad_id) -> Dict[str, Any]:
        """
        Evaluate the successful lifting of a suture needle away from a surface.

        Args:
            robot_id: ID of the robotic manipulator.
            needle_id: ID of the suture needle.
            pad_id: ID of the surface pad beneath the needle.

        Returns:
            dict: Evaluation results confirming the needle is lifted and disengaged from the pad.
        """
        state_key = self._get_state_key("pick_needle", robot_id, needle_id)

        # Initialize tracking records
        if state_key not in self._task_states:
            self._task_states[state_key] = {
                'initial_height': get_object_height(needle_id),
                'contact_start_time': None,
                'rising_start_time': None
            }

        state = self._task_states[state_key]
        initial_height = state['initial_height']

        # Evaluate contacts for both the robot and the underlying pad
        contact_info = check_contact(robot_id, needle_id)
        contact_info_pad = check_contact(needle_id, pad_id)

        # Monitor vertical displacement
        height_info = track_height_trajectory(needle_id, initial_height)

        # Success criteria: robot contact established, pad contact broken, lifted above threshold
        NEEDLE_LIFT_THRESHOLD = 0.02  # 20 mm

        is_success = (contact_info > 1 and contact_info_pad < 1 and
                      height_info['rising_amount'] > NEEDLE_LIFT_THRESHOLD)

        result = {
            'success': is_success,
            'done': 1 if is_success else 0,
            'details': {
                'robot_needle_contact_points_number': contact_info,
                'needle_height_info': height_info,
                'lift_amount': height_info['rising_amount'],
                'lift_threshold': NEEDLE_LIFT_THRESHOLD
            }
        }

        return result

    def evaluate_touch_kidney(self, robot_id: int, obj_id: int, state) -> Dict[str, Any]:
        """
        Evaluate target acquisition (touching a specific blood spot on a kidney model).

        Args:
            robot_id: ID of the robotic manipulator.
            obj_id: Target location/object ID on the kidney.
            state: Current state vector including end-effector position.

        Returns:
            dict: Evaluation results based on Euclidean proximity.
        """
        # Validate spatial proximity
        obj_position = get_object_position(obj_id)
        distance = np.linalg.norm(state[0:3] - obj_position)
        distance_THRESHOLD = 0.02
        is_success = (distance < distance_THRESHOLD)

        result = {
            'success': is_success,
            'distance': distance,
        }

        # Debugging output
        print(f"\n[Evaluator] Touch kidney:")
        print(f"  Distance: {distance}")
        print(f"  Success: {is_success}")

        return result

    def evaluate_clip_vein(self, robot_id: int, obj_id: int, obj_link1, state) -> Dict[str, Any]:
        """
        Evaluate the precise alignment and positioning for clipping a vein.

        Args:
            robot_id: ID of the robotic manipulator.
            obj_id: ID of the target vein structure.
            obj_link1: Specific link ID for the vein segment.
            state: Current state vector including position and orientation.

        Returns:
            dict: Evaluation results based on positional distance and yaw error.
        """
        # Calculate positional alignment
        obj_position = get_object_position(obj_id)
        distance = np.linalg.norm(state[0:3] - obj_position)

        current_yaw = state[3]
        target_yaw = -1.57  # Fixed target orientation for this scenario

        yaw_error = abs(current_yaw - target_yaw)
        yaw_tolerance = 0.1

        distance_THRESHOLD = 0.02
        is_success = (distance < distance_THRESHOLD) and (yaw_error < yaw_tolerance)

        result = {
            'success': is_success,
            'Distance': distance,
            'yaw_error': yaw_error
        }

        # Debugging output
        print(f"\n[Evaluator] Clip vein:")
        print(f"  Distance: {distance}")
        print(f"  Yaw error: {yaw_error}")
        print(f"  Success: {is_success}")
        print(f"  Object position: {obj_position}")

        return result

    def evaluate_place_gauze(self, gauze_id: int, obj_id: int) -> Dict[str, Any]:
        """
        Evaluate placing surgical gauze onto a specified block or tray.

        Args:
            gauze_id: ID of the surgical gauze.
            obj_id: ID of the target placement block/tray.

        Returns:
            dict: Evaluation results verifying spatial overlay and release.
        """
        # Validate vertical alignment and overlay
        position_info = check_vertical_relationship(gauze_id, obj_id)
        position = position_info['vertical_position']
        z = position_info['vertical_overlap']

        # Success criteria: Gauze is physically resting above the target
        is_success = (position == 'above' and z < 0.015)

        result = {
            'success': is_success,
            'done': 1 if is_success else 0,
            'details': {
                'horizontal_overlap': position_info['horizontal_overlap'],
                'vertical_overlap': position_info['vertical_overlap'],
                'relationship': position_info['relationship']
            }
        }

        print(f"\n[Evaluator] Place gauze:")
        print(f"  Horizontal overlap: {position_info['horizontal_overlap']}")
        print(f"  Vertical overlap: {position_info['vertical_overlap']}")
        print(f"  Relationship: {position_info['relationship']}")

        return result

    def evaluate_place_gauze2spleen(self, gauze_id: int, obj_id: int) -> Dict[str, Any]:
        """
        Evaluate placing surgical gauze onto a target coordinate on the spleen model.

        Args:
            gauze_id: ID of the surgical gauze.
            obj_id: Target placement ID on the spleen.

        Returns:
            dict: Evaluation results based on placement proximity.
        """
        # Validate Euclidean proximity
        obj_position = get_object_position(obj_id)
        state = get_object_position(gauze_id)
        distance = np.linalg.norm(state - obj_position)

        distance_THRESHOLD = 0.05
        is_success = (distance < distance_THRESHOLD)

        result = {
            'success': is_success,
            'done': 1 if is_success else 0,
            'distance': distance
        }

        # Debugging output
        print(f"\n[Evaluator] Place gauze to spleen:")
        print(f"  Distance: {distance}")
        print(f"  Success: {is_success}")

        return result

    def evaluate_move_to(self, robot_id: int, target_pos,
                        distance_threshold: float = 0.02) -> Dict[str, Any]:
        """
        Evaluate a simple point-to-point end-effector movement task.

        Args:
            robot_id: ID of the robotic manipulator.
            target_pos: Target 3D coordinates [x, y, z].
            distance_threshold: Acceptable margin of error for position arrival.

        Returns:
            dict: Evaluation results.
        """
        current_pos = get_object_position(robot_id)
        target = np.array(target_pos)
        distance = np.linalg.norm(current_pos - target)

        is_success = distance < distance_threshold

        result = {
            'success': is_success,
            'done': 1 if is_success else 0,
            'details': {
                'current_position': current_pos.tolist(),
                'target_position': target.tolist(),
                'distance': float(distance),
                'threshold': distance_threshold
            }
        }

        print(f"\n[Evaluator] Move to target:")
        print(f"  Distance: {distance:.4f} < {distance_threshold}")
        print(f"  Success: {is_success}")

        return result

    def evaluate(self, prompt: str, *args) -> Dict[str, Any]:
        """
        General routing function to trigger the appropriate evaluation method based on the NLP instruction.

        Args:
            prompt: Text description of the task instruction.
            *args: Variable arguments corresponding to required object IDs.

        Returns:
            dict: Evaluation results including the 'success' and 'done' status.
        """
        prompt_lower = prompt.lower()

        # Task 7
        if ('pick up the gauze and place it on the block' in prompt_lower) and len(args) >= 3:
            return self.evaluate_place_gauze(args[0], args[1])

        # Task 8
        elif ('pick up the gauze and place it on the red target point on the spleen' in prompt_lower) and len(args) >= 2:
            return self.evaluate_place_gauze2spleen(args[0], args[1])

        # Grasp needle tasks
        elif ('grasp the needle' in prompt_lower or
              'grasp needle' in prompt_lower) and len(args) >= 2:
            return self.evaluate_grasp_needle(args[0], args[1])

        # Task 2, 5
        elif ('pick up the needle' in prompt_lower or 'lift the needle' in prompt_lower or 'pick up the silver surgical needle on the tray' in prompt_lower) and len(args) >= 2:
            return self.evaluate_pick_needle(args[0], args[1], args[2])

        # Task 3
        elif ('touch the red blood spot on the kidney' in prompt_lower or 'contact the red blood spot on the kidney' in prompt_lower or 'position the end effector on the red spot on the kidney' in prompt_lower) and len(args) >= 2:
            return self.evaluate_touch_kidney(args[0], args[1], args[2])

        # Task 6
        elif ('clip the red blood point on the vein' in prompt_lower or 'clamp the bleeding point on the vessel' in prompt_lower) and len(args) >= 2:
            return self.evaluate_clip_vein(args[0], args[1], args[2], args[3])

        # Move to task
        elif 'move to' in prompt_lower and len(args) >= 2:
            robot_id = args[0]
            target_pos = args[1] if len(args) > 1 else None
            if target_pos is not None and isinstance(target_pos, (list, tuple, np.ndarray)):
                return self.evaluate_move_to(robot_id, target_pos)

        # Unknown or mismatched arguments
        else:
            print(f"\n[Evaluator] Unknown prompt or insufficient arguments: {prompt}")
            return {
                'success': False,
                'done': 0,
                'details': {'error': 'Unknown prompt or insufficient arguments'}
            }


# Initialize a global singleton evaluator
evaluator = TaskSuccessEvaluator()

# Exported main interface
def eval_success(prompt: str, *args) -> Dict[str, Any]:
    """
    Backwards-compatible wrapper function to execute the primary evaluator.
    """
    return evaluator.evaluate(prompt, *args)

def reset_evaluator():
    """Reset the global evaluator's tracking states."""
    evaluator.reset()