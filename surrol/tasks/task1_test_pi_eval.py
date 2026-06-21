import os
import time
import numpy as np
import pybullet as p
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
    wrap_angle
)
from surrol.const import ASSET_DIR_PATH
from surrol.robots.ecm import Ecm
import cv2
from surrol.utils.robotics import get_matrix_from_euler
from PIL import Image
import torch

# OpenPI imports
from openpi.training import config as _config
from openpi.policies import policy_config

# Set random seeds for reproducibility
SEED = 45
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


class OpenPIConfig:
    # OpenPI model configuration and checkpoint path
    lora_config_name: str = "pi05_my_dataset_lora"
    checkpoint_path: str = "path/to/your/openpi/checkpoint"


config = OpenPIConfig()


class GauzeRetrieve(PsmEnv):
    """
    Refer to Gym FetchPickAndPlace
    https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch/pick_and_place.py
    """
    POSE_TRAY = ((0.55, 0, 0.6781), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.681, 0.745))
    SCALING = 5.

    # Default joint positions for the ECM
    QPOS_ECM = (0, 0.9, 0.2, 0)

    def __init__(self, render_mode=None, cid=-1):
        super(GauzeRetrieve, self).__init__(render_mode, cid)
        self.ecm = None

    def _env_setup(self):
        super(GauzeRetrieve, self)._env_setup()
        self.has_object = True
        self._waypoint_goal = True

        # Robot setup
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False

        # Tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)
        p.changeVisualShape(obj_id, -1, rgbaColor=(225 / 255, 225 / 255, 225 / 255, 1))

        # Gauze
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
                            (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,
                             workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
                             workspace_limits[2][0] + 0.01),
                            (0, 0, 0, 1),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(0, 0, 0))
        self.obj_ids['rigid'].append(obj_id)
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], -1

        # Initialize ECM endoscope
        self.ecm = Ecm(
            (0.2, 0.0, 0.8),
            scaling=self.SCALING
        )
        self.ecm.reset_joint(self.QPOS_ECM)

    def get_ecm_image(self, image_width=640, image_height=480):
        """
        Retrieves RGB and depth images from the ECM camera perspective.
        """
        self.ecm.render_image()

        _, _, rgb_image, depth_image, mask = p.getCameraImage(
            width=image_width,
            height=image_height,
            viewMatrix=self.ecm.view_matrix,
            projectionMatrix=self.ecm.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        near, far = 0.02, 1.0
        depth = far * near / (far - (far - near) * depth_image)

        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        return rgb_image, depth, mask

    def _set_action(self, action: np.ndarray):
        action[3] = 0
        super(GauzeRetrieve, self)._set_action(action)

    def _sample_goal(self) -> np.ndarray:
        workspace_limits = self.workspace_limits1
        goal = np.array([workspace_limits[0].mean() + 0.02 * np.random.randn() * self.SCALING,
                         workspace_limits[1].mean() + 0.02 * np.random.randn() * self.SCALING,
                         workspace_limits[2][1] - 0.03 * self.SCALING])
        return goal.copy()

    def _sample_goal_callback(self):
        self._waypoints = [None, None, None, None]
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, 0., 0.5])
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, 0., 0.5])
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, 0., -0.5])
        self._waypoints[3] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.03) * self.SCALING, 0., -0.5])

        self._steps_per_waypoint = [20, 20, 5, 10]
        self._current_waypoint_index = 0
        self._step_in_waypoint = 0
        self._waypoint_start_state = None

    def _meet_contact_constraint_requirement(self):
        # Establish a rigid connection once the gripper is activated (contacted and closed)
        return self._activated >= 0

    def get_oracle_action(self, obs) -> np.ndarray:
        action = np.zeros(5)
        action[4] = -0.5
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.6
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0., waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4:
                self._waypoints[i] = None
            break

        return action

    def is_action_completed(self, obs, action, pos_tolerance=0.005, yaw_tolerance=0.1, gripper_tolerance=0.2):
        if self._current_waypoint_index >= len(self._waypoints):
            return True

        current_waypoint = self._waypoints[self._current_waypoint_index]

        if current_waypoint is None:
            self._current_waypoint_index += 1
            self._step_in_waypoint = 0
            return self.is_action_completed(obs, action, pos_tolerance, yaw_tolerance, gripper_tolerance)

        target_pos = current_waypoint[:3]
        target_yaw = current_waypoint[3]
        target_gripper = current_waypoint[4]

        current_pos = obs['observation'][:3]
        current_yaw = obs['observation'][5]
        current_gripper = obs['observation'][6]

        pos_error = np.linalg.norm(current_pos - target_pos)
        yaw_error = abs(current_yaw - target_yaw)
        gripper_error = abs(current_gripper - target_gripper)

        pos_completed = pos_error < pos_tolerance
        yaw_completed = yaw_error < yaw_tolerance
        gripper_completed = (gripper_error < gripper_tolerance or
                             (action is not None and abs(action[4] - target_gripper) < 0.1))

        completed = pos_completed and yaw_completed and gripper_completed

        if completed:
            print(f"Target waypoint {self._current_waypoint_index} reached!")
            self._waypoints[self._current_waypoint_index] = None
            self._current_waypoint_index += 1
            self._step_in_waypoint = 0
        else:
            print(f"  Waypoint {self._current_waypoint_index} incomplete - "
                  f"Pos error: {pos_error:.4f}, Yaw error: {yaw_error:.4f}, Gripper error: {gripper_error:.4f}")

        return completed

    def are_all_actions_completed(self):
        all_completed = all(waypoint is None for waypoint in self._waypoints)

        if all_completed:
            print("All waypoints completed!")
        else:
            remaining = sum(1 for waypoint in self._waypoints if waypoint is not None)
            print(f"Remaining waypoints: {remaining}")
            print(f"Current waypoint index: {self._current_waypoint_index}")

        return all_completed

    def move_end_effector_delta(self, delta_vector, num_steps=100, step_simulation=True):
        dx, dy, dz, drx, dry, drz, grip = delta_vector
        current_pose = self.psm1.get_current_position()
        delta_rot = get_matrix_from_euler([drx, dry, drz])

        for i in range(num_steps):
            alpha = (i + 1) / num_steps
            interp_rotation = current_pose[:3, :3] @ get_matrix_from_euler([drx * alpha, dry * alpha, drz * alpha])
            interp_position = current_pose[:3, 3] + np.array([dx, dy, dz]) * alpha

            target_pose = np.eye(4)
            target_pose[:3, :3] = interp_rotation
            target_pose[:3, 3] = interp_position

            self.psm1.move(target_pose)
            if step_simulation:
                p.stepSimulation()

        if grip < 0:
            self.psm1.close_jaw()
        else:
            self.psm1.move_jaw(np.deg2rad(40))

    def convert_7d_to_5d_standalone(self, openvla_action, scaling=1.0):
        dx, dy, dz, da, db, dc, grip = openvla_action
        delta_pos = np.array([dx, dy, dz]) / 0.01 / scaling
        delta_yaw = dc
        grip_action = grip

        if np.abs(delta_pos).max() > 1:
            delta_pos /= np.abs(delta_pos).max()
        delta_yaw = np.clip(delta_yaw, -0.4, 0.4)

        return np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, grip_action])


class CustomGauzePick(GauzeRetrieve):
    Gauze_PICKUP_HEIGHT_THRESHOLD = 0.01

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gripper_tip_link = 8
        self._gauze_grasped = False
        self._gauze_original_mass = None

    def reset(self):
        obs = super().reset()
        self._activated = -1
        self._gauze_id = self.obj_id

        dyn_info = p.getDynamicsInfo(self._gauze_id, -1)
        self._gauze_original_mass = dyn_info[0]
        self._gauze_grasped = False

        pos, _ = p.getBasePositionAndOrientation(self._gauze_id)
        self.gauze_initial_z = pos[2]

        p.setCollisionFilterPair(self.psm1.body, self._gauze_id, 6, -1, enableCollision=0)
        p.setCollisionFilterPair(self.psm1.body, self._gauze_id, 7, -1, enableCollision=0)

        return obs

    def step(self, action):
        gripper_cmd = action[4] if len(action) > 4 else -1.0
        is_gripper_closed = gripper_cmd < 0

        keys = p.getKeyboardEvents()
        if (ord('d') in keys and keys[ord('d')] & p.KEY_WAS_TRIGGERED) or is_gripper_closed:
            ee_pos = np.array(p.getLinkState(self.psm1.body, 8)[0])
            gauze_pos, _ = p.getBasePositionAndOrientation(self._gauze_id)

            dz = ee_pos[2] - gauze_pos[2]
            dxy = np.linalg.norm(ee_pos[:2] - gauze_pos[:2])

            Z_THRESHOLD = 0.015
            XY_THRESHOLD = 0.045

            print("\n" + "=" * 50)
            print(" DEBUG: Gauze Grasp Status")
            print(f"   Gripper command: {gripper_cmd:.3f}")
            print(f"   Is gripper closed? {is_gripper_closed}")
            print(f"   EE tip position: ({ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f})")
            print(f"   Gauze position:  ({gauze_pos[0]:.4f}, {gauze_pos[1]:.4f}, {gauze_pos[2]:.4f})")
            print(f"   dZ (vertical):   {dz:.4f} m")
            print(f"   dXY (horizontal):{dxy:.4f} m")
            print(f"   Thresholds: Z<{Z_THRESHOLD:.4f}, XY<{XY_THRESHOLD:.4f}")
            print(f"   Currently grasped? {self._gauze_grasped}")

            if self._gauze_grasped:
                print(" SUCCESS: Gauze is currently grasped!")
            else:
                if not is_gripper_closed:
                    print(" FAILURE REASON: Gripper is NOT closed.")
                elif dz >= Z_THRESHOLD:
                    print(f" FAILURE REASON: Too high! Need dZ < {Z_THRESHOLD:.4f}m, got {dz:.4f}m.")
                elif dxy >= XY_THRESHOLD:
                    print(f" FAILURE REASON: Too far horizontally! Need dXY < {XY_THRESHOLD:.4f}m, got {dxy:.4f}m.")
                else:
                    print(" UNEXPECTED: Should be grasped but isn't.")

            print("=" * 50 + "\n")

        obs, reward, done, info = super().step(action)

        ee_pos = np.array(p.getLinkState(self.psm1.body, 8)[0])
        gauze_pos, _ = p.getBasePositionAndOrientation(self._gauze_id)

        dz = (ee_pos[2] - gauze_pos[2])
        dxy = np.linalg.norm(ee_pos[:2] - gauze_pos[:2])

        Z_THRESHOLD = 0.015
        XY_THRESHOLD = 0.045

        if is_gripper_closed and (dz < Z_THRESHOLD) and (dxy < XY_THRESHOLD):
            self._gauze_grasped = True
            self._gauze_grasp_z = gauze_pos[2]
            print("Start grasping.")
            p.changeDynamics(self._gauze_id, -1, mass=0.0)

        elif not is_gripper_closed:
            self._gauze_grasped = False
            if hasattr(self, '_gauze_original_mass') and self._gauze_original_mass is not None:
                p.changeDynamics(self._gauze_id, -1, mass=self._gauze_original_mass)

        if self._gauze_grasped:
            offset_xy = np.array([0.0, 0.0])
            offset_z = -0.005

            target_x = ee_pos[0]
            target_y = ee_pos[1]
            ideal_z = ee_pos[2] + offset_z

            target_z = max(ideal_z, self._gauze_grasp_z)

            target_pos = [target_x, target_y, target_z]
            p.resetBasePositionAndOrientation(
                self._gauze_id,
                target_pos,
                [0, 0, 0, 1]
            )

        current_z = p.getBasePositionAndOrientation(self._gauze_id)[0][2]
        lifted = (current_z - self.gauze_initial_z) > self.Gauze_PICKUP_HEIGHT_THRESHOLD * self.SCALING
        is_success = self._gauze_grasped and lifted
        done = is_success

        info.update({
            'is_success': is_success,
            'gauze_grasped': self._gauze_grasped,
            'lifted': lifted,
            'gauze_height': current_z,
            'jaw_angle': obs['observation'][6] if 'observation' in obs else 0.0
        })

        return obs, reward, done, info


if __name__ == "__main__":
    device = "cuda:0"

    # 统一日志配置
    LOG_FILE = "surgvla_bench_results.txt"
    TASK_NAME = "Task 1"
    MODEL_NAME = "OpenPI"

    env = CustomGauzePick(render_mode='human')
    print("Created SurRoL Env.")

    print("Loading OpenPI policy...")
    cfg = _config.get_config(config.lora_config_name)
    policy = policy_config.create_trained_policy(cfg, config.checkpoint_path)
    print("OpenPI loaded!")

    obs = env.reset()
    total_success = 0
    num_episodes = 50

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")

        obs = env.reset()
        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            # ===== Detect 'F' key: trigger manual failure =====
            keys = p.getKeyboardEvents()
            if ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
                print(" 'F' pressed → Manual fail. Skipping to next episode.")
                done = True
                info = {'is_success': False}
                break

            rgb_img, depth_img, mask_img = env.get_ecm_image(image_width=224, image_height=224)

            # Gray + depth + mask processing
            depth_img = ((depth_img + 1) / (depth_img.max() + 1) * 255).astype(np.uint8)
            mask_img = ((mask_img + 1) / (mask_img.max() + 1) * 255).astype(np.uint8)
            gray = np.dot(rgb_img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            depth_array = np.array(depth_img, dtype=np.uint8)
            mask_array = np.array(mask_img, dtype=np.uint8)
            state = env._get_robot_state(idx=0)

            instruction = "pick up the gauze"
            print("Processing input and predicting action...")

            # Assume rgb_img is numpy.ndarray with shape (H, W, 3) and dtype uint8
            if isinstance(rgb_img, np.ndarray):
                if rgb_img.shape[2] == 3:
                    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
            else:
                pil_img = rgb_img

            observation = {
                "observation/image": pil_img,  # Ensure correct color space
                "observation/wrist_image": np.zeros_like(pil_img),  # Fill with zeros if this image is not needed
                "observation/state": state,
                "prompt": instruction,
            }

            start_time = time.time()

            result = policy.infer(observation)
            inference_time = time.time() - start_time

            action_chunk = result["actions"]  # Shape should be [8, 7] or [7,]
            print(f"Inference time: {inference_time:.3f}s")
            print(f"Raw action chunk shape: {action_chunk.shape}")

            # Ensure it is float32 numpy array
            if isinstance(action_chunk, np.ndarray):
                action_chunk = action_chunk.astype(np.float32)
            else:
                action_chunk = np.array(action_chunk, dtype=np.float32)

            # Send the first action (or the entire chunk depending on robot protocol)
            current_action = action_chunk[0] if action_chunk.ndim == 2 and action_chunk.shape[0] > 1 else action_chunk
            print(current_action)

            action = env.convert_7d_to_5d_standalone(current_action, env.SCALING)

            # Map continuous gripper predictions to discrete binary states
            action[4] = action[4] - 0.3
            if action[4] < 0:
                action[4] = -0.5
            else:
                action[4] = 0.5

            obs, reward, done, info = env.step(action)
            steps += 1

            print(f"Step {steps}: "
                  f"XY_dist={info.get('xy_dist', -1):.3f}, "
                  f"Z_offset={info.get('z_offset', -1):.3f}, "
                  f"Jaw={info.get('jaw_angle', -1):.2f} ({'open' if info.get('jaw_open', False) else 'closed'}), "
                  f"Placed={info.get('placed', False)}")

            if done:
                if info['is_success']:
                    total_success += 1
                    print(f"Episode {episode + 1}: Success!")
                else:
                    print(f"Episode {episode + 1}: Failed.")
                break

        if not done:
            print(f"Episode {episode + 1}: Reached max steps limit.")

        success_rate = total_success / num_episodes
        print(f"Current success rate: {total_success}/{num_episodes} = {success_rate:.2%}")

        # 统一日志写入: 记录每个Episode
        with open(LOG_FILE, 'a') as f:
            f.write(f"{TASK_NAME} | {MODEL_NAME} | Episode {episode + 1} Current Success Rate: {success_rate:.2%}\n")

    success_rate = total_success / num_episodes
    print(f"\n=== Final Result ===")
    print(f"Success rate: {total_success}/{num_episodes} = {success_rate:.2%}")

    # 统一日志写入: 记录最终结果
    with open(LOG_FILE, 'a') as f:
        f.write(f"{TASK_NAME} | {MODEL_NAME} | Final Success Rate: {success_rate:.2%}\n")

    env.close()