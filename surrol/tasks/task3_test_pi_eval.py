import os
import time
import random
import numpy as np
import pybullet as p
import cv2
import torch
from PIL import Image

from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import get_link_pose, wrap_angle
from surrol.const import ASSET_DIR_PATH
from surrol.robots.ecm import Ecm
from surrol.utils.robotics import get_matrix_from_euler

from eval_util.task_eval_way import eval_success, reset_evaluator

# OpenPI Imports
from openpi.training import config as _config
from openpi.policies import policy_config

# Set random seeds for reproducibility
SEED = 45
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


class OpenPIConfig:
    lora_config_name: str = "pi05_my_dataset_lora"
    checkpoint_path: str = "path/to/openpi/checkpoint"


config = OpenPIConfig()


class NeedlePick(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))
    SCALING = 5.

    QPOS_ECM = (0., 0.9, 0.2, 0.)

    def __init__(self, render_mode=None, cid=-1):
        super(NeedlePick, self).__init__(render_mode, cid)
        self.ecm = None

    def _env_setup(self):
        super(NeedlePick, self)._env_setup()
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

        self._contact_approx = False

        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)

        rd = (np.random.rand() - 0.5) * 0.1 * 3
        rd2 = (np.random.rand() - 0.5) * 0.1 * 2

        # Red sphere (representing a blood spot/target)
        obj_id = p.loadURDF(
            os.path.join(ASSET_DIR_PATH, 'sphere/sphere.urdf'),
            (workspace_limits[0].mean() + 0.11 + rd,
             workspace_limits[1].mean() + 0.13 + rd2,
             workspace_limits[2][0] + 0.07),
            p.getQuaternionFromEuler((0, 0, 0)),
            useFixedBase=True,
            globalScaling=4
        )

        p.changeVisualShape(obj_id, -1, rgbaColor=(1.0, 0.0, 0.0, 0.4))
        p.changeVisualShape(obj_id, -1, specularColor=(0.3, 0.3, 0.3))

        # Kidney
        obj_id2 = p.loadURDF(
            os.path.join(ASSET_DIR_PATH, 'keydney/keydney.urdf'),
            (workspace_limits[0].mean() + 0.5 + rd,
             workspace_limits[1].mean() + rd2,
             workspace_limits[2][0] - 0.14),
            p.getQuaternionFromEuler((np.pi / 2 + np.pi / 9, 0, np.pi / 2)),
            useFixedBase=True,
            globalScaling=2
        )

        psm_id = self.psm1.body
        keydney_id = obj_id2

        p.setCollisionFilterPair(psm_id, keydney_id, -1, -1, enableCollision=0)
        for i in range(p.getNumJoints(psm_id)):
            p.setCollisionFilterPair(psm_id, keydney_id, i, -1, enableCollision=0)

        p.changeVisualShape(obj_id2, -1, rgbaColor=(0.65, 0.45, 0.45, 1.0))
        p.changeVisualShape(obj_id2, -1, specularColor=(0.25, 0.25, 0.25))

        self.obj_ids['rigid'].append(obj_id)
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], -1

        self.ecm = Ecm(
            (0.2, 0., 0.8),
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

    def _sample_goal(self) -> np.ndarray:
        workspace_limits = self.workspace_limits1
        goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[1].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[2][1] - 0.04 * self.SCALING])
        print('Goal successfully set.')
        return goal.copy()

    def _sample_goal_callback(self):
        self._waypoints = [None, None]
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)

        self._waypoints[0] = np.array([pos_obj[0] + 0.02, pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0252 + 0.005) * self.SCALING, 0, -0.5])
        self._waypoints[1] = np.array([pos_obj[0] + 0.0, pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, 0, -0.5])

        self._steps_per_waypoint = [20, 20, 5, 10]
        self._current_waypoint_index = 0
        self._step_in_waypoint = 0
        self._waypoint_start_state = None

    def convert_7d_to_5d_standalone(self, openvla_action, scaling=1.0):
        dx, dy, dz, da, db, dc, grip = openvla_action
        delta_pos = np.array([dx, dy, dz]) / 0.01 / scaling
        delta_yaw = dc
        grip_action = grip

        if np.abs(delta_pos).max() > 1:
            delta_pos /= np.abs(delta_pos).max()
        delta_yaw = np.clip(delta_yaw, -0.4, 0.4)

        return np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, grip_action])


if __name__ == "__main__":
    # --- Unified Logging Configuration ---
    LOG_FILE = "surgvla_bench_results.txt"
    TASK_NAME = "Task 3"
    MODEL_NAME = "OpenPI"

    device = "cuda:0"
    env = NeedlePick(render_mode='human')
    print("Created SurRoL Environment.")

    print("Loading OpenPI policy...")
    cfg = _config.get_config(config.lora_config_name)
    policy = policy_config.create_trained_policy(cfg, config.checkpoint_path)
    print("OpenPI policy loaded!")

    total_success = 0
    num_episodes = 50

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        reset_evaluator()
        obs = env.reset()
        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            rgb_img, depth_img, mask_img = env.get_ecm_image(image_width=224, image_height=224)
            state = env._get_robot_state(idx=0)

            instruction = "contact the red blood spot on the kidney"
            prompt = f"In: What action should the robot take to {instruction}?\nOut:"
            print("Processing input and predicting action...")

            if isinstance(rgb_img, np.ndarray):
                if rgb_img.shape[2] == 3:
                    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
            else:
                pil_img = rgb_img

            observation = {
                "observation/image": pil_img,
                "observation/wrist_image": np.zeros_like(pil_img),
                "observation/state": state,
                "prompt": instruction,
            }

            start_time = time.time()
            result = policy.infer(observation)
            inference_time = time.time() - start_time

            action_chunk = result["actions"]
            print(f"Inference time: {inference_time:.3f}s")
            print(f"Raw action chunk shape: {action_chunk.shape}")

            if isinstance(action_chunk, np.ndarray):
                action_chunk = action_chunk.astype(np.float32)
            else:
                action_chunk = np.array(action_chunk, dtype=np.float32)

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
            done = False
            steps += 1

            # Evaluate success using the universal evaluation function
            robot_id = env.psm1.body
            obj_id = env.obj_id

            eval_result = eval_success(prompt, robot_id, obj_id, state)

            print(f"Step {steps}")
            print(f"Distance to target: {eval_result['distance']:.4f}")

            if eval_result['success']:
                done = True
                info['is_success'] = True
                total_success += 1
                print(f"Episode {episode + 1}: Success!")
                break

        if not done:
            print(f"Episode {episode + 1}: Reached max steps limit.")

        success_rate = total_success / (episode + 1)

        # Unified Logging
        with open(LOG_FILE, 'a') as f:
            f.write(
                f"{TASK_NAME} | {MODEL_NAME} | Episode {episode + 1}/{num_episodes} - Cumulative Success Rate: {success_rate:.2%}\n")

    final_rate = total_success / num_episodes

    print("\n" + "=" * 50)
    print(" FINAL RESULTS")
    print("=" * 50)
    print(f"Tissue Contact Task ({num_episodes} episodes): {total_success}/{num_episodes} = {final_rate:.2%}")
    print("=" * 50)

    # Unified Logging: Final Validation result
    with open(LOG_FILE, 'a') as f:
        f.write(f"{TASK_NAME} | {MODEL_NAME} | Final Validation Success Rate: {final_rate:.2%}\n")

    env.close()