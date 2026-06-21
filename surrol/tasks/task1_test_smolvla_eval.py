import os
import torch

os.environ.pop('PYOPENGL_PLATFORM', None)
os.environ.pop('EGL_PLATFORM', None)
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import time
import numpy as np
import pybullet as p
import cv2
import torch
from PIL import Image
from torchvision import transforms

# Set random seeds for reproducibility
SEED = 45
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Import LeRobot
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# Import SurRoL
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import get_link_pose
from surrol.const import ASSET_DIR_PATH
from surrol.robots.ecm import Ecm
from surrol.utils.robotics import get_matrix_from_euler

# Configuration paths (consistent with training)
MODEL_PATH = "path/to/your/smolvla/model"
TOKENIZER_PATH = "path/to/your/smolvla/tokenizer"


class SmolVLAHelper:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device)

        # Load policy
        print("Loading SmolVLA model...")
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
        self.policy.to(self.device)
        self.policy.eval()

        # Use the framework's processor factory (key improvement)
        print("Loading preprocessor...")
        rename_map = {
            "observation.image": "observation.images.camera1",
            "observation.wrist_image": "observation.images.camera2"
        }

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint_path,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "rename_observations_processor": {"rename_map": rename_map},
            }
        )

        # Image preprocessing (ensure normalization to [0,1])
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        print(f" SmolVLA loaded on {device}")
        print(f" Preprocessor loaded with rename_map: {rename_map}")

    def _preprocess_image(self, image_bgr):
        """ Preprocess image to BCHW format in [0,1] range """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.image_transform(image_rgb)
        return tensor.unsqueeze(0).to(self.device)

    def predict_action(self, main_image, wrist_image, instruction, robot_state, debug=False):
        # 1. Prepare raw observations (using original key names from training)
        raw_obs = {
            "observation.image": self._preprocess_image(main_image),
            "observation.wrist_image": self._preprocess_image(wrist_image),
            "observation.state": torch.from_numpy(robot_state.astype(np.float32)).unsqueeze(0).to(self.device),
            "task": instruction
        }

        # 2. Use preprocessor to automatically handle all conversions (including tokenization)
        processed_obs = self.preprocessor(raw_obs)

        # 3. Debug output
        if debug:
            print("\n=== DEBUG: Processed Observation ===")
            for k, v in processed_obs.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")

        # 4. Inference - regenerate action chunk each time
        with torch.inference_mode(), torch.autocast(device_type=self.device.type):
            pred = self.policy.select_action(processed_obs)
            pred = pred[0]

        if self.postprocessor is not None:
            pred = self.postprocessor(pred)

        # 5. Convert to numpy
        pred = pred.cpu().numpy()
        return pred


class GauzeRetrieve(PsmEnv):
    """
    Refer to Gym FetchPickAndPlace
    https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch/pick_and_place.py
    """
    POSE_TRAY = ((0.55, 0, 0.6781), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.681, 0.745))
    SCALING = 5.

    QPOS_ECM = (0, 0.9, 0.2, 0)

    def __init__(self, render_mode=None, cid=-1):
        super(GauzeRetrieve, self).__init__(render_mode, cid)
        self.ecm = None

    def _env_setup(self):
        super(GauzeRetrieve, self)._env_setup()
        self.has_object = True
        self._waypoint_goal = True

        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False

        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)
        p.changeVisualShape(obj_id, -1, rgbaColor=(225 / 255, 225 / 255, 225 / 255, 1))

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

        self.ecm = Ecm(
            (0.2, 0.0, 0.8),
            scaling=self.SCALING
        )
        self.ecm.reset_joint(self.QPOS_ECM)

    def get_ecm_image(self, image_width=640, image_height=480):
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
    MODEL_NAME = "SmolVLA"

    env = CustomGauzePick(render_mode='human')
    print("Created SurRoL Env.")

    helper = SmolVLAHelper(MODEL_PATH, device)
    print(f" SmolVLA Helper initialized on {device} with correct input format")

    obs = env.reset()
    total_success = 0
    num_episodes = 50

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")

        obs = env.reset()
        helper.policy.reset()
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
            state = env._get_robot_state(idx=0)

            instruction = "pick up the gauze"
            prompt = f"In: What action should the robot take to {instruction}?\nOut:"
            print("Processing input and predicting action...")

            # Debug: only for the first step of the first episode
            debug = (episode == 0 and steps == 0)
            if debug:
                print(f"\n[DEBUG] Step {steps + 1}: Raw image shape: {rgb_img.shape}, State: {state}")

            action = helper.predict_action(
                main_image=rgb_img,
                wrist_image=rgb_img,  # Use the same image for the wrist view
                instruction=instruction,
                robot_state=state,
                debug=debug
            )

            if debug:
                print(f"[DEBUG] Predicted action (7D): {action}")

            print(action)
            action = env.convert_7d_to_5d_standalone(action, env.SCALING)

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