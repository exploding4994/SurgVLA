import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
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

class VeinCoagulation(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))
    SCALING = 5.

    QPOS_ECM = (0., 0.9, 0.2, 0.)

    def __init__(self, render_mode=None, cid=-1):
        super(VeinCoagulation, self).__init__(render_mode, cid)
        self.ecm = None

    def _env_setup(self):
        super(VeinCoagulation, self)._env_setup()
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


        # 加载静脉模型
        obj_id = p.loadURDF(
            os.path.join(ASSET_DIR_PATH, 'Vein/静脉.urdf'),
            (workspace_limits[0].mean() + 0.03 + rd,
             workspace_limits[1].mean() - 0.7 + rd2,
             workspace_limits[2][0] + 0.06),
            p.getQuaternionFromEuler((np.pi / 2, 0, 0)),
            useFixedBase=True,
            globalScaling=2.0
        )
        # p.changeVisualShape(obj_id, -1, rgbaColor=(0.55, 0.2, 0.35, 0.9))
        # p.changeVisualShape(obj_id, -1, specularColor=(0.3, 0.3, 0.3))
        p.changeVisualShape(obj_id, -1, rgbaColor=(0.25, 0.35, 0.75, 0.9))
        p.changeVisualShape(obj_id, -1, specularColor=(0.2, 0.2, 0.3))


        # 红球（出血点）
        obj_id = p.loadURDF(
            os.path.join(ASSET_DIR_PATH, 'sphere/sphere.urdf'),
            (workspace_limits[0].mean()-0.02  + rd,
             workspace_limits[1].mean()  + rd2,
             workspace_limits[2][0] + 0.032),
            p.getQuaternionFromEuler((0, 0, 0)),
            useFixedBase=False,
            globalScaling=2
        )
        p.changeVisualShape(obj_id, -1, rgbaColor=(1.0, 0.0, 0.0, 0.6))
        p.changeVisualShape(obj_id, -1, specularColor=(0.3, 0.3, 0.3))

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
        print('had set goal')
        return goal.copy()

    def _sample_goal_callback(self):
        self._waypoints = [None, None, None]
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]

        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw
        yaw = abs(yaw)-np.pi

        # 第一个路径点：红球上方0.02米
        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + 0.02 * self.SCALING, yaw, 0.5])  # 夹爪张开

        # 第二个路径点：红球位置
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2]+0.04, yaw, 0.5])  # 夹爪下降

        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + 0.03, yaw, -0.5])  # 夹爪闭合夹住

        print(self.goal)

        self._steps_per_waypoint = [20, 20]
        self._current_waypoint_index = 0
        self._step_in_waypoint = 0
        self._waypoint_start_state = None

    def get_oracle_action(self, obs) -> np.ndarray:
        action = np.zeros(5)
        action[4] = -0.5
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.4
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < 1e-2:
                self._waypoints[i] = None
            break

        return action

    def _meet_contact_constraint_requirement(self):
        if self._contact_approx:
            return True
        else:
            pose = get_link_pose(self.obj_id, self.obj_link1)
            return pose[0][2] > self._waypoint_z_init + 0.005 * self.SCALING

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
            print(f"🎯 路径点 {self._current_waypoint_index} 完成!")
            self._waypoints[self._current_waypoint_index] = None
            self._current_waypoint_index += 1
            self._step_in_waypoint = 0
        else:
            print(f"  路径点 {self._current_waypoint_index} 未完成 - "
                  f"位置误差: {pos_error:.4f}, 角度误差: {yaw_error:.4f}, 夹爪误差: {gripper_error:.4f}")

        return completed

    def are_all_actions_completed(self):
        all_completed = all(waypoint is None for waypoint in self._waypoints)

        if all_completed:
            print("🎉 所有路径点已完成!")
        else:
            remaining = sum(1 for waypoint in self._waypoints if waypoint is not None)
            print(f"📊 剩余路径点: {remaining}")
            print(f"📊 当前路径点索引: {self._current_waypoint_index}")

        return all_completed


if __name__ == "__main__":
    env = VeinCoagulation(render_mode='human')

    for idx in range(1, 101):
        os.makedirs('rgb/Benchmark-raw-dataset/task5_2_vein_coagulation/rgb_images_long_224/' + str(idx),
                    exist_ok=True)
        os.makedirs('rgb/Benchmark-raw-dataset/task5_2_vein_coagulation/depth_images_long_224/' + str(idx),
                    exist_ok=True)
        os.makedirs('rgb/Benchmark-raw-dataset/task5_2_vein_coagulation/depth_images_long_224_32float/' + str(idx),
                    exist_ok=True)
        os.makedirs('rgb/Benchmark-raw-dataset/task5_2_vein_coagulation/mask_images_long_224/' + str(idx),
                    exist_ok=True)
        os.makedirs('rgb/Benchmark-raw-dataset/task5_2_vein_coagulation/state_data_long_224/' + str(idx),
                    exist_ok=True)

        obs = env.reset()
        frame_count = 0
        print(f"=== 开始Episode {idx} ===")

        env._current_waypoint_index = 0
        env._step_in_waypoint = 0

        while not env.are_all_actions_completed():
            frame_count = frame_count + 1

            action = env.get_oracle_action(obs)
            print(f"执行动作: {action}")
            env.is_action_completed(obs, action)

            obs, reward, done, info = env.step(action)
            print("---")

            rgb_img, depth_img, mask_img = env.get_ecm_image(image_width=224, image_height=224)

            depth_img_32 = depth_img.astype(np.float32)
            depth_img = ((depth_img + 1) / (depth_img.max() + 1) * 255).astype(np.uint8)
            mask_img = ((mask_img + 1) / (mask_img.max() + 1) * 255).astype(np.uint8)
            state = env._get_robot_state(idx=0)

            rgb_filename = f'rgb/Benchmark-raw-dataset/task5_2_vein_coagulation/rgb_images_long_224/' + str(
                idx) + f'/{frame_count:06d}.png'
            cv2.imwrite(rgb_filename, rgb_img)

            depth_filename = f'rgb/Benchmark-raw-dataset/task5_2_vein_coagulation/depth_images_long_224/' + str(
                idx) + f'/{frame_count:06d}.png'
            cv2.imwrite(depth_filename, depth_img)

            mask_filename = f'rgb/Benchmark-raw-dataset/task5_2_vein_coagulation/mask_images_long_224/' + str(
                idx) + f'/{frame_count:06d}.png'
            cv2.imwrite(mask_filename, mask_img)

            state_filename = f'rgb/Benchmark-raw-dataset/task5_2_vein_coagulation/state_data_long_224/' + str(
                idx) + f'/{frame_count:06d}.npy'
            np.save(state_filename, state)
            depth_filename32 = f'rgb/Benchmark-raw-dataset/task5_2_vein_coagulation/depth_images_long_224_32float/{idx}/{frame_count:06d}.npy'
            np.save(depth_filename32, depth_img_32)

            time.sleep(0.03)

        print(f"=== Episode {idx} 完成! 总帧数: {frame_count} ===\n")

    env.close()