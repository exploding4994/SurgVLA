import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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

class NeedlePick(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # reduce tip pad contact
    SCALING = 5.

    # 添加 ECM 的默认关节位置
    QPOS_ECM = (0., 0.9, 0.2, 0.)  # 可根据需要调整1111

    # TODO: grasp is sometimes not stable; check how to fix it

    def __init__(self, render_mode=None, cid=-1):
        super(NeedlePick, self).__init__(render_mode, cid)
        self.ecm = None  # 初始化为 None

    def _env_setup(self):
        super(NeedlePick, self)._env_setup()
        # np.random.seed(4)  # for experiment reproduce
        self.has_object = True
        self._waypoint_goal = True


        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        # physical interaction
        self._contact_approx = False

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 1

        # self._setup_soft_tissue(workspace_limits)

        # gauze
        # obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
        #                     (workspace_limits[0].mean() + 3*(np.random.rand() - 0.5) * 0.1,  # TODO: scaling
        #                      workspace_limits[1].mean() + 4*(np.random.rand() - 0.5) * 0.1,
        #                      workspace_limits[2][0] + 0.01),
        #                     (0, 0, 0, 1),
        #                     useFixedBase=False,
        #                     globalScaling=self.SCALING)
        # p.changeVisualShape(obj_id, -1, specularColor=(0, 0, 0))
        # self.obj_ids['rigid'].append(obj_id)  # 0
        # self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], -1

        # # needle
        yaw = (np.random.rand() - 0.5) * np.pi
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,  # TODO: scaling
                             workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
                             workspace_limits[2][0] + 0.01),
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1

        # --- 4. 初始化 ECM 内窥镜 ---
        self.ecm = Ecm(
            (0.2, 0., 0.8),  # ECM 基座位置
            scaling=self.SCALING
        )
        self.ecm.reset_joint(self.QPOS_ECM)  # 设置初始姿态

    def get_ecm_image(self, image_width=640, image_height=480):
        """
        从 ECM 相机视角获取 RGB 和深度图像
        """
        # 渲染图像（内部会设置 viewMatrix 和 projMatrix）
        self.ecm.render_image()

        # 获取图像
        _, _, rgb_image, depth_image, mask = p.getCameraImage(
            width=image_width,
            height=image_height,
            viewMatrix=self.ecm.view_matrix,
            projectionMatrix=self.ecm.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL  # 更高质量渲染renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # 转换深度图
        near, far = 0.02, 1.0
        depth = far * near / (far - (far - near) * depth_image)

        # OpenCV 格式：RGB -> BGR
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        return rgb_image, depth, mask

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[1].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[2][1] - 0.04 * self.SCALING])
        print('had set goal')
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        # super()._sample_goal_callback()
        self._waypoints = [None, None, None, None]  # four waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

        # # for physical deployment only
        # print(" -> Needle pose: {}, {}".format(np.round(pos_obj, 4), np.round(orn_obj, 4)))
        # qs = self.psm1.get_current_joint_position()
        # joint_positions = self.psm1.inverse_kinematics(
        #     (np.array(pos_obj) + np.array([0, 0, (-0.0007 + 0.0102)]) * self.SCALING,
        #      p.getQuaternionFromEuler([-90 / 180 * np.pi, -0 / 180 * np.pi, yaw])),
        #     self.psm1.EEF_LINK_INDEX)
        # self.psm1.reset_joint(joint_positions)
        # print("qs: {}".format(joint_positions))
        # print("Cartesian: {}".format(self.psm1.get_current_position()))
        # self.psm1.reset_joint(qs)

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        self._waypoints[3] = np.array([self.goal[0], self.goal[1],
                                       self.goal[2] + 0.02 * self.SCALING, yaw, -0.5])  # lift up  0.0102
        print(self.goal)

        self._steps_per_waypoint = [20, 20, 5, 10]
        self._current_waypoint_index = 0
        self._step_in_waypoint = 0
        self._waypoint_start_state = None

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # four waypoints executed in sequential order
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
        """
        检查当前动作是否完成
        """
        # 如果所有路径点都已完成，返回True
        if self._current_waypoint_index >= len(self._waypoints):
            return True

        # 获取当前路径点的目标状态
        current_waypoint = self._waypoints[self._current_waypoint_index]

        # 如果当前路径点已经是None，说明已完成，移动到下一个
        if current_waypoint is None:
            self._current_waypoint_index += 1
            self._step_in_waypoint = 0
            return self.is_action_completed(obs, action, pos_tolerance, yaw_tolerance, gripper_tolerance)

        # 提取目标状态
        target_pos = current_waypoint[:3]
        target_yaw = current_waypoint[3]
        target_gripper = current_waypoint[4]

        # 提取当前状态
        current_pos = obs['observation'][:3]
        current_yaw = obs['observation'][5]
        current_gripper = obs['observation'][6]

        # 计算误差
        pos_error = np.linalg.norm(current_pos - target_pos)
        yaw_error = abs(current_yaw - target_yaw)
        gripper_error = abs(current_gripper - target_gripper)

        # 检查是否完成
        pos_completed = pos_error < pos_tolerance
        yaw_completed = yaw_error < yaw_tolerance
        gripper_completed = (gripper_error < gripper_tolerance or
                             (action is not None and abs(action[4] - target_gripper) < 0.1))

        completed = pos_completed and yaw_completed and gripper_completed

        if completed:
            print(f"🎯 路径点 {self._current_waypoint_index} 完成!")
            # 标记当前路径点为已完成，并移动到下一个
            self._waypoints[self._current_waypoint_index] = None
            self._current_waypoint_index += 1
            self._step_in_waypoint = 0
        else:
            print(f"  路径点 {self._current_waypoint_index} 未完成 - "
                  f"位置误差: {pos_error:.4f}, 角度误差: {yaw_error:.4f}, 夹爪误差: {gripper_error:.4f}")

        return completed

    def are_all_actions_completed(self):
        """
        检查所有路径点是否都已完成
        """
        # 检查是否所有路径点都是None
        all_completed = all(waypoint is None for waypoint in self._waypoints)

        if all_completed:
            print("🎉 所有路径点已完成!")
        else:
            remaining = sum(1 for waypoint in self._waypoints if waypoint is not None)
            print(f"📊 剩余路径点: {remaining}")
            print(f"📊 当前路径点索引: {self._current_waypoint_index}")

        return all_completed


if __name__ == "__main__":
    # 创建环境
    env = NeedlePick(render_mode='human')

    for idx in range(1, 1001):
        # if idx in target_indices:
        # 创建保存数据的文件夹
        # idx = 2
        # os.makedirs('rgb/0330-1000-raw-dataset/task2_needle_pick/rgb_images_long_224/' + str(idx),
        #             exist_ok=True)
        # os.makedirs('rgb/0330-1000-raw-dataset/task2_needle_pick/depth_images_long_224/' + str(idx),
        #             exist_ok=True)
        # os.makedirs('rgb/0330-1000-raw-dataset/task2_needle_pick/depth_images_long_224_32float/' + str(idx),
        #             exist_ok=True)
        # os.makedirs('rgb/0330-1000-raw-dataset/task2_needle_pick/mask_images_long_224/' + str(idx),
        #             exist_ok=True)
        # os.makedirs('rgb/0330-1000-raw-dataset/task2_needle_pick/state_data_long_224/' + str(idx),
        #             exist_ok=True)

        obs = env.reset()
        frame_count = 0
        print(f"=== 开始Episode {idx} ===")

        # 重置路径点状态
        env._current_waypoint_index = 0
        env._step_in_waypoint = 0

        # 持续执行直到所有路径点完成
        while not env.are_all_actions_completed():
            frame_count = frame_count + 1

            # 获取专家动作
            action = env.get_oracle_action(obs)
            print(f"执行动作: {action}")

            # 执行动作前检查当前路径点是否完成
            env.is_action_completed(obs, action)

            obs, reward, done, info = env.step(action)
            print("---")

            # 获取 ECM 图像并保存
            rgb_img, depth_img, mask_img = env.get_ecm_image(image_width=224, image_height=224)

            depth_img_32 = depth_img.astype(np.float32)
            depth_img = ((depth_img + 1) / (depth_img.max() + 1) * 255).astype(np.uint8)
            mask_img = ((mask_img + 1) / (mask_img.max() + 1) * 255).astype(np.uint8)
            state = env._get_robot_state(idx=0)

            # # 保存各种数据...
            # rgb_filename = f'rgb/0330-1000-raw-dataset/task2_needle_pick/rgb_images_long_224/' + str(
            #     idx) + f'/{frame_count:06d}.png'
            # cv2.imwrite(rgb_filename, rgb_img)
            #
            # depth_filename = f'rgb/0330-1000-raw-dataset/task2_needle_pick/depth_images_long_224/' + str(
            #     idx) + f'/{frame_count:06d}.png'
            # cv2.imwrite(depth_filename, depth_img)
            #
            # mask_filename = f'rgb/0330-1000-raw-dataset/task2_needle_pick/mask_images_long_224/' + str(
            #     idx) + f'/{frame_count:06d}.png'
            # cv2.imwrite(mask_filename, mask_img)
            #
            # state_filename = f'rgb/0330-1000-raw-dataset/task2_needle_pick/state_data_long_224/' + str(
            #     idx) + f'/{frame_count:06d}.npy'
            # np.save(state_filename, state)
            # depth_filename32 = f'rgb/0330-1000-raw-dataset/task2_needle_pick/depth_images_long_224_32float/{idx}/{frame_count:06d}.npy'
            # np.save(depth_filename32, depth_img_32)

            # 添加小延迟以便观察
            time.sleep(0.01)

        print(f"=== Episode {idx} 完成! 总帧数: {frame_count} ===\n")

    env.close()
