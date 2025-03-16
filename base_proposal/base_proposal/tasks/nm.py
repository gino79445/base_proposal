# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import numpy as np
from base_proposal.tasks.base.basic import Task
from base_proposal.handlers.tiagodualWBhandler import TiagoDualWBHandler
from omni.isaac.core.objects.cone import VisualCone
from omni.isaac.core.prims import GeometryPrimView
from base_proposal.tasks.utils.pinoc_utils import PinTiagoIKSolver
from base_proposal.tasks.utils.motion_planner import MotionPlannerTiago
from base_proposal.tasks.utils import scene_utils
from base_proposal.tasks.utils import astar_utils
from base_proposal.tasks.utils import rrt_utils
from base_proposal.tasks.utils import get_features

# from omni.isaac.isaac_sensor import _isaac_sensor
from omni.isaac.sensor import _sensor as _isaac_sensor

# from omni.isaac.sensor import ContactSensor as _isaac_sensor

from omni.isaac.core.utils.semantics import add_update_semantics
from base_proposal.vlm.get_target import identify_object_in_image
from base_proposal.vlm.get_part import determine_part_to_grab
from base_proposal.vlm.get_answer import confirm_part_in_image
from base_proposal.vlm.get_base import determine_base
from base_proposal.vlm.get_affordance import determine_affordance
from base_proposal.annotation.annotation import get_base
import matplotlib.pyplot as plt
from pxr import UsdGeom
import cv2
import os
import time
import math

# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# from segment_anything import SamPredictor, sam_model_registry

from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp
from omni.isaac.core.utils.torch.rotations import euler_angles_to_quats, quat_diff_rad
from scipy.spatial.transform import Rotation
from omni.isaac.core.utils.extensions import enable_extension
from PIL import Image


# Base placement environment for fetching a target object among clutter
class NMTask(Task):
    def __init__(self, name, sim_config, env) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._device = self._cfg["sim_device"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._gamma = self._task_cfg["env"]["gamma"]
        self._max_episode_length = self._task_cfg["env"]["horizon"]

        self._randomize_robot_on_reset = self._task_cfg["env"][
            "randomize_robot_on_reset"
        ]

        # Get dt for integrating velocity commands and checking limit violations
        self._dt = torch.tensor(
            self._sim_config.task_config["sim"]["dt"]
            * self._sim_config.task_config["env"]["controlFrequencyInv"],
            device=self._device,
        )

        # Environment object settings: (reset() randomizes the environment)
        self._obstacle_names = self._task_cfg["env"]["obstacles"]
        # self._tabular_obstacle_mask = [False, True] # Mask to denote which objects are tabular (i.e. grasp objects can be placed on them)
        self._grasp_obj_names = self._task_cfg["env"]["target"]
        self._num_obstacles = min(
            self._task_cfg["env"]["num_obstacles"], len(self._obstacle_names)
        )
        self._num_grasp_objs = min(
            self._task_cfg["env"]["num_grasp_objects"], len(self._grasp_obj_names)
        )
        self._obstacles = []
        self._obstacles_dimensions = []
        self._grasp_objs = []
        self._grasp_objs_dimensions = []
        #  Contact sensor interface for collision detection:
        self._contact_sensor_interface = (
            _isaac_sensor.acquire_contact_sensor_interface()
        )

        self._move_group = self._task_cfg["env"]["move_group"]
        self._use_torso = self._task_cfg["env"]["use_torso"]
        # Position control. Actions are base SE2 pose (3) and discrete arm activation (2)
        # env specific limits
        self.max_rot_vel = torch.tensor(
            self._task_cfg["env"]["max_rot_vel"], device=self._device
        )
        self.max_base_xy_vel = torch.tensor(
            self._task_cfg["env"]["max_base_xy_vel"], device=self._device
        )

        # End-effector reaching settings
        self._goal_pos_threshold = self._task_cfg["env"]["goal_pos_thresh"]
        self._goal_ang_threshold = self._task_cfg["env"]["goal_ang_thresh"]

        self._collided = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.long
        )
        self._is_success = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.long
        )
        # self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        # self.collided = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        # self.sam.to(self._device)
        # self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        # self.mask_predictor = SamPredictor(self.sam)
        self.final_place = np.zeros((2), dtype=np.float32)

        self.instruction = self._task_cfg["env"]["instruction"]
        self.arm = self._task_cfg["env"]["move_group"]
        self.step_count = 0

        self.targets_position = self._task_cfg["env"]["targets_position"]
        self.targets_se3 = self._task_cfg["env"]["targets_se3"]
        self.num_se3 = self._task_cfg["env"]["num_se3"]
        # IK solver
        self._ik_solver = PinTiagoIKSolver(
            move_group=self._move_group,
            include_torso=self._use_torso,
            include_base=True,
            max_rot_vel=100.0,
        )  # No max rot vel
        self._motion_planner = MotionPlannerTiago(
            move_group=self._move_group,
            include_torso=self._use_torso,
            include_base=True,
            max_rot_vel=100.0,
        )

        # Handler for Tiago
        self.tiago_handler = TiagoDualWBHandler(
            move_group=self._move_group,
            use_torso=self._use_torso,
            sim_config=self._sim_config,
            num_envs=self._num_envs,
            device=self._device,
        )

        # RLTask.__init__(self, name, env)
        Task.__init__(self, name, env)

    def get_camera_intrinsics(self):
        # Get camera intrinsics for rendering
        return self.sd_helper.get_camera_intrinsics()

    def get_point_cloud(self, R, T, fx, fy, cx, cy):
        point_cloud = []
        # depth =self.sd_helper.get_groundtruth(["depth"], self.ego_viewport.get_viewport_window())["depth"]
        # rgb_data = self.sd_helper.get_groundtruth(["rgb"], self.ego_viewport.get_viewport_window())["rgb"]
        depth = self.get_depth_data()
        rgb_data = self.get_rgb_data()

        for i in range(rgb_data.shape[1]):
            for j in range(rgb_data.shape[0]):
                i = rgb_data.shape[1] - i
                j = rgb_data.shape[0] - j
                if (
                    i > 0
                    and j > 0
                    and i < rgb_data.shape[1]
                    and j < rgb_data.shape[0]
                    and depth[j, i] > 0
                ):
                    point = self.get_3d_point(
                        i,
                        j,
                        depth[rgb_data.shape[0] - j, rgb_data.shape[1] - i],
                        R,
                        T,
                        fx,
                        fy,
                        cx,
                        cy,
                    )
                    point_cloud.append(point)
        point_cloud = np.array(point_cloud)

        # save rgb image
        im = Image.fromarray(rgb_data)
        im.save("./data/end_rgb.png")
        # save the point cloud
        np.save("./data/point_cloud.npy", point_cloud)

    def set_up_scene(self, scene) -> None:
        import omni

        # if no data folder, create one
        if not os.path.exists("./data"):
            os.makedirs("./data")
        if self._task_cfg["env"]["plane"] == True:
            scene_utils.add_plane(
                name="building",
                prim_path=self.tiago_handler.default_zero_env_path,
                device=self._device,
            )
        self.tiago_handler.get_robot()
        if self._task_cfg["env"]["house"] == True:
            scene_utils.sence(
                name="building",
                prim_path=self.tiago_handler.default_zero_env_path,
                device=self._device,
            )
        # Spawn obstacles (from ShapeNet usd models):
        for i in range(self._num_obstacles):
            obst = scene_utils.spawn_obstacle(
                name=self._obstacle_names[i],
                prim_path=self.tiago_handler.default_zero_env_path,
                device=self._device,
            )
            self._obstacles.append(obst)  # Add to list of obstacles (Geometry Prims)
            # Optional: Add contact sensors for collision detection. Covers whole body by default
            omni.kit.commands.execute(
                "IsaacSensorCreateContactSensor",
                path="/Contact_Sensor",
                sensor_period=float(self._sim_config.task_config["sim"]["dt"]),
                parent=obst.prim_path,
            )
        # Spawn grasp objs (from YCB usd models):
        for i in range(self._num_grasp_objs):
            grasp_obj = scene_utils.spawn_grasp_object(
                name=self._grasp_obj_names[i],
                prim_path=self.tiago_handler.default_zero_env_path,
                device=self._device,
            )
            self._grasp_objs.append(
                grasp_obj
            )  # Add to list of grasp objects (Rigid Prims)
            # Optional: Add contact sensors for collision detection. Covers whole body by default
            omni.kit.commands.execute(
                "IsaacSensorCreateContactSensor",
                path="/Contact_Sensor",
                sensor_period=float(self._sim_config.task_config["sim"]["dt"]),
                parent=grasp_obj.prim_path,
            )
        # Goal visualizer
        goal_viz1 = VisualCone(
            prim_path=self.tiago_handler.default_zero_env_path + "/goal1",
            radius=0.05,
            height=0.05,
            color=np.array([1.0, 0.0, 0.0]),
        )
        goal_viz2 = VisualCone(
            prim_path=self.tiago_handler.default_zero_env_path + "/goal2",
            radius=0.05,
            height=0.05,
            color=np.array([0, 0.0, 1.0]),
        )

        super().set_up_scene(scene)
        self._robots = self.tiago_handler.create_articulation_view()
        scene.add(self._robots)
        # Contact sensor interface for robot collision detection

        self._goal_vizs1 = GeometryPrimView(
            prim_paths_expr="/World/envs/.*/goal1", name="goal_viz1"
        )
        scene.add(self._goal_vizs1)
        self._goal_vizs2 = GeometryPrimView(
            prim_paths_expr="/World/envs/.*/goal2", name="goal_viz2"
        )
        scene.add(self._goal_vizs2)

        # Enable object axis-aligned bounding box computations
        scene.enable_bounding_boxes_computations()
        # Add spawned objects to scene registry and store their bounding boxes:
        for obst in self._obstacles:
            scene.add(obst)
            self._obstacles_dimensions.append(
                scene.compute_object_AABB(obst.name)
            )  # Axis aligned bounding box used as dimensions
        for grasp_obj in self._grasp_objs:
            scene.add(grasp_obj)
            self._grasp_objs_dimensions.append(
                scene.compute_object_AABB(grasp_obj.name)
            )  # Axis aligned bounding box used as dimensions
        # Optional viewport for rendering in a separate viewer

        rgb = self.get_rgb_data()
        im = Image.fromarray(rgb)

        self.set_robot()

        # rgb_data = rgb.get_data()
        # im = Image.fromarray(rgb_data)
        # im.save("./data/start_rgb.png")

    # from omni.isaac.synthetic_utils import SyntheticDataHelper
    #        from omni.isaac.synthetic_utils import SyntheticDataHelper
    #       self.viewport_window = omni.kit.viewport_legacy.get_default_viewport_window()
    #      self.sd_helper = SyntheticDataHelper()
    #        sensor_names = [
    #            "rgb",
    #            "depth",
    #            "boundingBox2DTight",
    #            "boundingBox2DLoose",
    #            "instanceSegmentation",
    #            "semanticSegmentation",
    #            "boundingBox3D",
    #            "camera",
    #            "pose",
    #        ]
    #        self.sd_helper.initialize(sensor_names, viewport=self.viewport_window)
    #

    def post_reset(self):
        # reset that takes place when the isaac world is reset (typically happens only once)
        self.tiago_handler.post_reset()

    def get_pixel(self, point, R, T, fx, fy, cx, cy):
        point = R @ (point - T)
        X = point[1]
        Y = point[2]
        Z = point[0]

        u = (fx * X) / Z + cx
        v = (fy * Y) / Z + cy
        i = int(u)
        j = int(v)
        return i, j

    def get_3d_point(self, u, v, Z, R, T, fx, fy, cx, cy):
        # Retrieve camera parameters

        # Convert pixel coordinates to normalized camera coordinates
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        # Convert to camera coordinates (as column vector)
        point = np.array([[Z], [X], [Y]])
        # Apply the inverse transformation (R^-1 and translation)
        R_inv = np.linalg.inv(R)
        point_3d = R_inv @ point + T
        return point_3d

    def transform_points(self, points, old_origin, new_origin, old_vector, new_vector):
        old_vector = old_vector / np.linalg.norm(old_vector)
        new_vector = new_vector / np.linalg.norm(new_vector)

        angle = np.arctan2(old_vector[1], old_vector[0]) - np.arctan2(
            new_vector[1], new_vector[0]
        )
        # print(f"angle: {angle}")
        # print(f"angle: {angle * 180 / np.pi}")

        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

        translation_vector = new_origin - old_origin

        rotated_points = np.dot(points, rotation_matrix)
        final_points = rotated_points + translation_vector

        return final_points

    def get_path(self):
        return self.path

    def transform_path(self, path):
        transformed_path = []
        pre_angle = 0  # 初始角度
        pre_x, pre_y = 0, 0  # 初始座標

        for i, (cur_x, cur_y) in enumerate(path):
            if i == 0:
                x, y = cur_x - pre_x, cur_y - pre_y
                transformed_path.append([x, y])
            else:
                x, y = cur_x - pre_x, cur_y - pre_y

                cur_angle = np.arctan2(y, x)

                angle = cur_angle - pre_angle

                x_t = x * np.cos(-pre_angle) - y * np.sin(-pre_angle)
                y_t = x * np.sin(-pre_angle) + y * np.cos(-pre_angle)

                transformed_path.append([x_t, y_t])

                pre_angle = cur_angle

            pre_x, pre_y = cur_x, cur_y

        return transformed_path

    def angle_between(self, u, v):
        ux, uy = u
        vx, vy = v
        # 內積：u dot v = |u|*|v|*cos(theta)
        dot = ux * vx + uy * vy
        mag_u = np.sqrt(ux * ux + uy * uy)
        mag_v = np.sqrt(vx * vx + vy * vy)
        # 為避免浮點誤差超出 [-1,1]，我們夾住在 [-1,1]
        cos_theta = max(-1.0, min(1.0, dot / (mag_u * mag_v + 1e-12)))
        return np.arccos(cos_theta)

    def merge_small_angle_increments(self, local_increments, deg_threshold=5.0):
        rad_threshold = np.deg2rad(deg_threshold)
        merged = []

        for inc in local_increments:
            if not merged:
                merged.append(inc)
            else:
                last_inc = merged[-1]
                angle_diff = self.angle_between(last_inc, inc)
                if angle_diff < rad_threshold:
                    new_inc = [last_inc[0] + inc[0], last_inc[1] + inc[1]]
                    merged[-1] = new_inc
                else:
                    merged.append(inc)

        return merged

    def set_path(self, position):
        self.final_place = position
        cell_size = 0.05

        w = self.rgb_data.shape[1]
        h = self.rgb_data.shape[0]
        goal = position

        end = (int(goal[1] / cell_size) + 100, int(goal[0] / cell_size + 100))
        if self._task_cfg["env"]["build_global_map"]:
            start = (100, 100)
        else:
            robot_pos = self.tiago_handler.get_robot_obs()[0, :3]
            x, y, theta = robot_pos
            start = (int(y / cell_size) + 100, int(x / cell_size) + 100)
            star_delta = (start[0] - 100, start[1] - 100)
            start = (100, 100)
            end = (end[0] - star_delta[0], end[1] - star_delta[1])
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            # 以 start 為中心旋轉
            cx, cy = 100, 100
            px, py = end
            dx, dy = px - cx, py - cy
            new_x = cx + (dx * cos_theta - dy * sin_theta)
            new_y = cy + (dx * sin_theta + dy * cos_theta)
            end = (int(new_x), int(new_y))

            # end = (end[0] - self.curr_pos[0] + 100, end[1] - self.curr_pos[1] + 100)
        occupancy_2d_map = self.occupancy_2d_map
        path = []
        if self._task_cfg["env"]["build_global_map"]:
            occupancy_2d_map = np.zeros((200, 200), dtype=np.uint8)
            path = astar_utils.a_star2(occupancy_2d_map, start, end)
        else:
            path = astar_utils.a_star_rough(occupancy_2d_map, start, end)
            map = self.occupancy_2d_map.copy()
            for p in path:
                map[p[0], p[1]] = 200
                im = Image.fromarray(map)
                # save
                im.save("./data/path.png")
            # minus every element x , y by start x, y
            # path = [[p[0] - start[0] + 100, p[1] - start[1] + 100] for p in path]
        self.curr_pos = end

        # draw the path

        self.path = []
        for p in path:
            x = (p[1] - 100) * cell_size
            y = (p[0] - 100) * cell_size
            self.path.append([x, y])

        #        # transform the path to the robot coordinate
        #        robot_position = self.tiago_handler.get_robot_obs()[0, :3]
        #        R = np.array(
        #            [
        #                [np.cos(robot_position[2]), -np.sin(robot_position[2])],
        #                [np.sin(robot_position[2]), np.cos(robot_position[2])],
        #            ]
        #        )
        #        # R = np.linalg.inv(R)
        #        T = np.array([robot_position[0], robot_position[1]])
        #        print(R, T)
        #        # T = R @ T
        #        T_inv = -R @ T

        #        path = np.array([x, y])
        #        robot_points = R @ path + T_inv

        #        path = robot_points.tolist()
        #        x = path[0]
        #        y = path[1]

        #        self.path.append([x, y])
        #    # transform the path to the robot coordinate
        #    robot_position = self.tiago_handler.get_robot_obs()[0, :3]

        # theta = robot_position[2]
        # R_inv = np.array(
        #    [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        # )
        # T = np.array([robot_position[0], robot_position[1]])
        # T_inv = R_inv @ T
        ## R_inv = np.linalg.inv(R_inv)
        ## world_points = np.array(self.path)
        ## robot_points = (R_inv @ world_points) + T_inv
        # path = []
        # for p in self.path:
        #    path.append((R_inv @ np.array(p) - R_inv @ T).tolist())
        # print(f"Robot points: {path}")

        # self.path = path

        #        rotation_matrix = np.array(
        #            [
        #                [np.cos(robot_position[2]), -np.sin(robot_position[2])],
        #                [np.sin(robot_position[2]), np.cos(robot_position[2])],
        #            ]
        #        )
        #        translation_vector = np.array([robot_position[0], robot_position[1]])
        #        self.path = (
        #            rotation_matrix
        #            @ (np.array(self.path) + translation_vector[np.newaxis, :]).T
        #        ).T

        # self.path = self.transform_points(  # 將路徑轉換到機器人座標系
        #    np.array(self.path),  # 路徑
        #    np.array([0, 0]),  # 舊原點
        #    np.array([robot_position[0], robot_position[1]]),  # 新原點
        #    np.array([1, 0]),  # 舊向量
        #    np.array([np.cos(robot_position[2]), np.sin(robot_position[2])]),
        # )
        self.path = self.transform_path(self.path)
        self.path = self.merge_small_angle_increments(self.path, deg_threshold=3.0)
        return self.path

    def get_global_RT(self):
        R, T, fx, fy, cx, cy = self.retrieve_camera_params()
        camera_tf = torch.zeros((4, 4), device=self._device)
        camera_tf[:3, :3] = torch.tensor(R)
        camera_tf[:3, 3] = torch.tensor(T).squeeze()
        camera_rotate = torch.tensor(R)
        camera_translate = torch.tensor(T).squeeze()

        # last row
        camera_tf[-1, :] = torch.tensor([0, 0, 0, 1], device=self._device)

        # get the base position
        base_position = self.tiago_handler.get_robot_obs()[0, :3]

        base_rotate = torch.tensor(
            [
                [torch.cos(base_position[2]), -torch.sin(base_position[2]), 0],
                [torch.sin(base_position[2]), torch.cos(base_position[2]), 0],
                [0, 0, 1],
            ],
            dtype=torch.float64,
        )
        base_translate = torch.tensor([base_position[0], base_position[1], 0])
        base_invert_rotate = base_rotate.t()
        global_R = camera_rotate.float() @ base_invert_rotate.float()
        # global_T = camera_translate.float() + base_translate.float()
        global_T = (
            #   base_translate.float()
            camera_translate.float() @ base_invert_rotate.float()
            + base_translate.float()
        )

        #   global_T = (
        #       base_invert_rotate.float() @ camera_translate.float()
        #       + base_translate.float()
        #   )

        global_T = global_T.reshape(3, 1)
        global_R = global_R.numpy().astype(np.float64)
        global_T = global_T.numpy().astype(np.float64)
        return global_R, global_T

    def build_map(self, R, T, fx, fy, cx, cy):
        # 2d occupancy map
        self.occupancy_2d_map = np.zeros((200, 200), dtype=np.uint8)
        # open the occupancy_2d_map file

        if (
            os.path.exists("./data/occupancy_2d_map.npy")
            and self._task_cfg["env"]["build_global_map"]
        ):
            self.occupancy_2d_map = np.load("./data/occupancy_2d_map.npy")
            self.occupancy_2d_map = self.occupancy_2d_map.reshape(200, 200)
        # self.occupancy_2d_map.fill(255)
        depth = self.depth_data
        map_size = (200, 200)
        cell_size = 0.05
        #  for i in range(-10, 20):
        #      for j in range(-20, 20):
        #          self.occupancy_2d_map[100+j,100+i] = 0

        if depth.shape[0] == 0:
            return

        for i in range(self.rgb_data.shape[1]):
            for j in range(self.rgb_data.shape[0]):
                i = self.rgb_data.shape[1] - i
                j = self.rgb_data.shape[0] - j
                if (
                    i > 0
                    and j > 0
                    and i < self.rgb_data.shape[1]
                    and j < self.rgb_data.shape[0]
                    and depth[j, i] > 0
                ):
                    depth[j, i] = depth[j, i]
                    point = self.get_3d_point(
                        i,
                        j,
                        depth[self.rgb_data.shape[0] - j, self.rgb_data.shape[1] - i],
                        R,
                        T,
                        fx,
                        fy,
                        cx,
                        cy,
                    )
                    map_x = int(point[0] / cell_size)
                    map_x += int(map_size[0] / 2)
                    map_y = int(point[1] / cell_size)
                    map_y += int(map_size[1] / 2)
                    if 0 <= map_x < map_size[0] and 0 <= map_y < map_size[1]:
                        if point[2] > 0.1:
                            self.occupancy_2d_map[map_y, map_x] = 255
                        # else:
                        #    self.occupancy_2d_map[map_y, map_x] = 255

        # save the occupancy_2d_map numpy
        np.save("./data/occupancy_2d_map.npy", self.occupancy_2d_map)

        im = Image.fromarray(self.occupancy_2d_map)
        # flip up and down
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        # left_rotate
        im = im.transpose(Image.ROTATE_90)
        im.save("./data/occupancy_2d_map.png")

    def get_observations(self):
        # Handle any pending resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # perceptial data
        self.depth_data = self.get_depth_data()
        self.rgb_data = self.get_rgb_data()
        if self.depth_data.shape[0] != 0:
            # normalization
            depth_data = self.depth_data / 5
            depth_data = depth_data * 255.0
            depth_data = depth_data.astype("uint8")
            im = Image.fromarray(depth_data)
            im.save("./data/depth.png")
        # get the camera parameters
        R, T, fx, fy, cx, cy = self.retrieve_camera_params()
        if self._task_cfg["env"]["check_env"] == True:
            self.occupancy_2d_map = np.zeros((200, 200), dtype=np.uint8)
            return
        if self._task_cfg["env"]["build_global_map"]:
            self.occupancy_2d_map = np.zeros((200, 200), dtype=np.uint8)
            # make R matrix + T vector 4x4 matrix
            # camera_tf = torch.zeros((4, 4), device=self._device)
            # camera_tf[:3, :3] = torch.tensor(R)
            # camera_tf[:3, 3] = torch.tensor(T).squeeze()
            # camera_rotate = torch.tensor(R)
            # camera_translate = torch.tensor(T).squeeze()

            ## last row
            # camera_tf[-1, :] = torch.tensor([0, 0, 0, 1], device=self._device)

            ## get the base position
            # base_position = self.tiago_handler.get_robot_obs()[0, :3]

            # base_rotate = torch.tensor(
            #    [
            #        [torch.cos(base_position[2]), -torch.sin(base_position[2]), 0],
            #        [torch.sin(base_position[2]), torch.cos(base_position[2]), 0],
            #        [0, 0, 1],
            #    ],
            #    dtype=torch.float64,
            # )
            # base_translate = torch.tensor([base_position[0], base_position[1], 0])
            # base_invert_rotate = base_rotate.t()
            # global_R = camera_rotate.float() @ base_invert_rotate.float()
            ## global_T = camera_translate.float() + base_translate.float()
            # global_T = (
            #    #   base_translate.float()
            #    camera_translate.float() @ base_invert_rotate.float()
            #    + base_translate.float()
            # )

            ##   global_T = (
            ##       base_invert_rotate.float() @ camera_translate.float()
            ##       + base_translate.float()
            ##   )

            # global_T = global_T.reshape(3, 1)
            # global_R = global_R.numpy().astype(np.float64)
            # global_T = global_T.numpy().astype(np.float64)
            global_R, global_T = self.get_global_RT()
            self.build_map(global_R, global_T, fx, fy, cx, cy)

            return

        self.rgb_data = self.rgb_data.astype("uint8")
        # save the rgb image
        im = Image.fromarray(self.rgb_data)
        im.save("./data/rgb.png")

        # for box in self.bounding_box:
        #    if box['semanticLabel'] == 'target':
        #        x_min, y_min, x_max, y_max = int(box['x_min']), int(box['y_min']), int(box['x_max']), int(box['y_max'])
        #        cv2.rectangle(rgb, (x_min, y_min), (x_max, y_max), (220, 0, 0), 1)
        #        # save the rgb image
        #        im = Image.fromarray(rgb)
        #        im.save("./data/original.png")

        self.occupancy_2d_map = np.zeros((200, 200), dtype=np.uint8)
        # open the occupancy_2d_map png
        if os.path.exists("./data/occupancy_2d_map.npy"):
            self.occupancy_2d_map = np.load("./data/occupancy_2d_map.npy")
            self.occupancy_2d_map = self.occupancy_2d_map.reshape(200, 200)

        global_map = self.occupancy_2d_map
        robot_position = self.tiago_handler.get_robot_obs()[0, :3]
        robot_x, robot_y, theta = robot_position  # 世界座標的機器人位置

        import scipy.ndimage

        # === 🟢 定義地圖大小與網格間距 ===
        grid_size = 200  # 地圖大小 200x200
        cell_size = 0.05  # 每格代表 5cm (0.05m)

        # === 🟢 計算機器人位置對應的索引 (在 global_map 中的 pixel) ===
        robot_pixel_x = int((robot_x / cell_size) + grid_size // 2)
        robot_pixel_y = int((robot_y / cell_size) + grid_size // 2)

        # === 🔴 旋轉地圖到機器人座標 ===
        # scipy.ndimage 直接旋轉影像，不用手動計算旋轉矩陣

        robot_map = global_map.copy()
        # === 🔵 平移地圖，讓機器人居中 ===
        dx = grid_size // 2 - robot_pixel_x
        dy = grid_size // 2 - robot_pixel_y
        robot_map = np.roll(robot_map, shift=(dy, dx), axis=(0, 1))

        robot_map = scipy.ndimage.rotate(
            robot_map, np.rad2deg(theta), reshape=False, order=1
        )
        self.occupancy_2d_map = robot_map

        map = self.occupancy_2d_map.copy()

        # for i in range(-8, 8):
        #    for j in range(-8, 8):
        #        map[100 + j, 100 + i] = 255
        radius = 7
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                # ✅ 只檢查圓形內的點 (i, j)
                if i**2 + j**2 > radius**2:
                    continue  # 忽略圓外的格子
                map[100 + j, 100 + i] = 255

        im = Image.fromarray(map)
        # flip up and down
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        # left_rotate
        im = im.transpose(Image.ROTATE_90)
        im.save("./data/curr_2d_map.png")
        return

    def get_render(self):
        # Get ground truth viewport rgb image
        gt = self.sd_helper.get_groundtruth(
            ["rgb"],
            self.viewport_window,
            verify_sensor_init=False,
            wait_for_sensor_data=0,
        )
        return np.array(gt["rgb"])

    def get_motion_num(self):
        return len(self.motion_path)

    def set_new_base(self, x_scaled, y_scaled, theta_scaled):
        # NOTE: Actions are in robot frame but the handler is in world frame!
        # Get current base positions
        base_joint_pos = self.tiago_handler.get_robot_obs()[
            :, :3
        ]  # First three are always base positions
        base_tf = torch.zeros((4, 4), device=self._device)
        base_tf[:2, :2] = torch.tensor(
            [
                [torch.cos(base_joint_pos[0, 2]), -torch.sin(base_joint_pos[0, 2])],
                [torch.sin(base_joint_pos[0, 2]), torch.cos(base_joint_pos[0, 2])],
            ]
        )  # rotation about z axis
        base_tf[2, 2] = 1.0  # No rotation here
        base_tf[:, -1] = torch.tensor(
            [base_joint_pos[0, 0], base_joint_pos[0, 1], 0.0, 1.0]
        )  # x,y,z,1

        # Transform actions to world frame and apply to base
        action_tf = torch.zeros((4, 4), device=self._device)
        action_tf[:2, :2] = torch.tensor(
            [
                [torch.cos(theta_scaled[0]), -torch.sin(theta_scaled[0])],
                [torch.sin(theta_scaled[0]), torch.cos(theta_scaled[0])],
            ]
        )
        action_tf[2, 2] = 1.0  # No rotation here
        action_tf[:, -1] = torch.tensor([x_scaled[0], y_scaled[0], 0.0, 1.0])  # x,y,z,1
        return base_tf, action_tf

    def set_angle(self, theta):
        self.rot = theta

    def pre_physics_step(self, actions) -> None:
        # actions (num_envs, num_action)
        # Handle resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # robot_position = self.tiago_handler.get_robot_obs()[0, :3]
        # scene_utils.set_obj_pose(self._grasp_objs[0], robot_position)

        R, T, fx, fy, cx, cy = self.retrieve_camera_params()
        action = np.zeros(3)
        action = torch.tensor(action, dtype=torch.float, device=self._device).unsqueeze(
            dim=0
        )
        # self.tiago_handler.apply_base_actions(action)
        if actions == "start":
            action = np.zeros(3)
            self.tiago_handler.apply_base_actions(
                torch.tensor(action, dtype=torch.float, device=self._device).unsqueeze(
                    dim=0
                )
            )
            robot_base = self.tiago_handler.get_robot_obs()[0, :3]
            print(f"Robot base: {robot_base}")
            return
        if actions == "close_gripper":
            self.tiago_handler.close_gripper()
            return

        if self.flag == 1 and actions != "lift_object":
            pass
        # self.tiago_handler.close_gripper()
        if actions == "lift_object":
            # self.tiago_handler.close_gripper()
            self.tiago_handler.lift()
            return
        if actions == "set_angle":
            x = torch.tensor([0], device=self._device)
            y = torch.tensor([0], device=self._device)
            theta = torch.tensor([self.rot], device=self._device)
            base_tf, action_tf = self.set_new_base(x, y, theta)
            new_base_tf = torch.matmul(base_tf, action_tf)
            new_base_xy = new_base_tf[0:2, 3].unsqueeze(dim=0)
            new_base_theta = (
                torch.arctan2(new_base_tf[1, 0], new_base_tf[0, 0])
                .unsqueeze(dim=0)
                .unsqueeze(dim=0)
            )
            self.new_base_tf = new_base_tf
            self.new_base_xy = new_base_xy
            self.new_base_theta = new_base_theta
            self.path = []
            return

        if actions == "get_base":
            self.final_place = torch.tensor(
                self.path[self.pos_idx], device=self._device
            )
            x_scaled = torch.tensor([self.final_place[0]], device=self._device)
            y_scaled = torch.tensor([self.final_place[1]], device=self._device)
            theta_scaled = torch.atan2(y_scaled, x_scaled)
            base_tf, action_tf = self.set_new_base(x_scaled, y_scaled, theta_scaled)
            new_base_tf = torch.matmul(base_tf, action_tf)
            new_base_xy = new_base_tf[0:2, 3].unsqueeze(dim=0)
            new_base_theta = (
                torch.arctan2(new_base_tf[1, 0], new_base_tf[0, 0])
                .unsqueeze(dim=0)
                .unsqueeze(dim=0)
            )
            self.new_base_tf = new_base_tf
            self.new_base_xy = new_base_xy
            self.new_base_theta = new_base_theta
            return

        if self.check_robot_collisions():
            print("Collision detected")
            self._collided[0] = 1
        if self._collided[0] == 1:
            return

        if actions == "get_point_cloud":
            self.get_point_cloud(R, T, fx, fy, cx, cy)

            point_cloud = np.load("./data/point_cloud.npy")
            point_cloud = point_cloud.squeeze()
            point_cloud = point_cloud[::100]
            point_cloud = point_cloud[point_cloud[:, 2] > 0.1]
            point_cloud = point_cloud[point_cloud[:, 2] < 3]
            path = self._motion_planner.rrt_motion_plan_with_obstacles(
                self.start_q, self.end_q, point_cloud, max_iters=1000, step_size=0.3
            )
            self.motion_path = path
            return

        if actions == "move_arm":
            if self.ik_success:
                if self.path_num >= len(self.motion_path):
                    return
                base_positions = self.motion_path[self.path_num][0:3]

                # self.tiago_handler.set_base_positions(jnt_positions=torch.tensor(np.array([base_positions]),dtype=torch.float,device=self._device))
                self.tiago_handler.set_upper_body_positions(
                    jnt_positions=torch.tensor(
                        np.array(self.motion_path[self.path_num][4:]),
                        dtype=torch.float,
                        device=self._device,
                    )
                )
                self.path_num += 1
                return

        if self._is_success[0] == 1:
            return

        if actions == "forward":
            actions = np.zeros(3)
            actions[0] = self.max_base_xy_vel
            actions = torch.unsqueeze(
                torch.tensor(actions, dtype=torch.float, device=self._device), dim=0
            )
            self.tiago_handler.apply_base_actions(actions)
            return

        if actions == "right_rotate":
            actions = np.zeros(3)
            actions[2] = self.max_rot_vel
            actions = torch.unsqueeze(
                torch.tensor(actions, dtype=torch.float, device=self._device), dim=0
            )
            self.tiago_handler.apply_base_actions(actions)
            return

        if actions == "left_rotate":
            actions = np.zeros(3)
            actions[2] = -self.max_rot_vel
            actions = torch.unsqueeze(
                torch.tensor(actions, dtype=torch.float, device=self._device), dim=0
            )
            self.tiago_handler.apply_base_actions(actions)
            return

        # Move base
        if actions == "set_base":
            self.tiago_handler.set_base_positions(
                torch.hstack((self.new_base_xy, self.new_base_theta))
            )
            R, T, fx, fy, cx, cy = self.retrieve_camera_params()

            self.x_delta = self.new_base_xy[0, 0].cpu().numpy()
            self.y_delta = self.new_base_xy[0, 1].cpu().numpy()
            self.theta_delta = self.new_base_theta[0, 0].cpu().numpy()
            # theta_delta = theta
            self.pos_idx += 1
            if self.pos_idx == len(self.path) or len(self.path) == 0:
                self.pos_idx = 0
            inv_base_tf = torch.linalg.inv(self.new_base_tf)
            self._curr_goal_tf = torch.matmul(inv_base_tf, self._goal_tf)
        #            print(f"Goal position: {self._curr_goal_tf}")

        if actions == "turn_to_goal":
            # if torch.linalg.norm(self._curr_goal_tf[0:2,3]) < 0.01 :
            x_scaled = torch.tensor([0], device=self._device)
            y_scaled = torch.tensor([0], device=self._device)
            curr_goal_pos = self._curr_goal_tf[self.se3_idx, 0:3, 3]
            theta_scaled = torch.tensor(
                [torch.atan2(curr_goal_pos[1], curr_goal_pos[0])], device=self._device
            )

            base_tf, action_tf = self.set_new_base(x_scaled, y_scaled, theta_scaled)
            new_base_tf = torch.matmul(base_tf, action_tf)
            new_base_xy = new_base_tf[0:2, 3].unsqueeze(dim=0)
            new_base_theta = (
                torch.arctan2(new_base_tf[1, 0], new_base_tf[0, 0])
                .unsqueeze(dim=0)
                .unsqueeze(dim=0)
            )
            self.x_delta = new_base_xy[0, 0].cpu().numpy()
            self.y_delta = new_base_xy[0, 1].cpu().numpy()
            self.theta_delta = new_base_theta[0, 0].cpu().numpy()
            self.tiago_handler.set_base_positions(
                torch.hstack((new_base_xy, new_base_theta))
            )

            # Transform goal to robot frame
            inv_base_tf = torch.linalg.inv(new_base_tf)
            self._curr_goal_tf = torch.matmul(inv_base_tf, self._goal_tf)
            return

        if actions == "manipulate":
            self.obj_origin_pose = scene_utils.get_obj_pose(self._grasp_objs[0])

            success = False
            num = self.num_se3[self.se3_idx]
            for i in range(num):
                idx = self.se3_idx + i
                curr_goal_pos = self._curr_goal_tf[idx, 0:3, 3]
                curr_goal_quat = Rotation.from_matrix(
                    self._curr_goal_tf[idx, :3, :3]
                ).as_quat()[[3, 0, 1, 2]]

                success_list, base_positions_list = self._ik_solver.solve_ik_pos_tiago(
                    des_pos=curr_goal_pos.cpu().numpy(),
                    des_quat=curr_goal_quat,
                    # pos_threshold=self._goal_pos_threshold, angle_threshold=self._goal_ang_threshold, verbose=False, Rmin=[-0.0, -0.0, 0.965,-0.259],Rmax=[0.0, 0.0, 1, 0.259])
                    pos_threshold=self._goal_pos_threshold,
                    angle_threshold=self._goal_ang_threshold,
                    verbose=False,
                    Rmin=[-0.0, -0.0, 0.866, -0.5],
                    Rmax=[0.0, 0.0, 1, 0.5],
                )
                success = False
                for i in range(len(success_list)):
                    if success_list[i]:
                        x = int((base_positions_list[i][0]) / 0.05 + 100)
                        y = int((base_positions_list[i][1]) / 0.05 + 100)
                        success = success_list[i]
                        base_positions = base_positions_list[i]

                        if astar_utils.is_valid(y, x, self.occupancy_2d_map):
                            success = success_list[i]
                            base_positions = base_positions_list[i]
                            break
                    if success:
                        break
            self.se3_idx += 1
            self.ik_success = False
            self.path_num = 0
            self.motion_path = []
            if success:
                print("IK Success")

                self.ik_success = True
                theta = torch.arctan2(
                    torch.tensor(base_positions[3]), torch.tensor(base_positions[2])
                )
                theta += self.theta_delta

                self.tiago_handler.set_base_positions(
                    jnt_positions=torch.tensor(
                        np.array(
                            [
                                [
                                    base_positions[0] + self.x_delta,
                                    base_positions[1] + self.y_delta,
                                    theta,
                                ]
                            ]
                        ),
                        dtype=torch.float,
                        device=self._device,
                    )
                )

                self.tmp_x = base_positions[0] + self.x_delta
                self.tmp_y = base_positions[1] + self.y_delta
                self.tmp_theta = self.theta_delta

                start_arm = self.tiago_handler.get_upper_body_positions()
                # start_arm = np.array([0.25,1, 1.5707, 1.5707, 1, 1.5, -1.5707, 1.0])
                # start_arm = torch.tensor(start_arm).unsqueeze(0)

                start_base = self.tiago_handler.get_base_positions()
                start_base = np.array([0, 0, 1, 0])
                start_base = torch.tensor(start_base)
                start_base = start_base.unsqueeze(0)
                start_q = torch.hstack((start_base, start_arm))
                start_q = start_q.unsqueeze(0)
                start_q = start_q.cpu().numpy()
                start_q = start_q[0][0]
                self.start_q = start_q

                end_base = np.array(
                    [
                        base_positions[0],
                        base_positions[1],
                        base_positions[2],
                        base_positions[3],
                    ]
                )
                end_base = np.array([0, 0, 1, 0])
                end_arm = np.array([base_positions[4:]])
                end_base = torch.tensor(end_base)
                end_arm = torch.tensor(end_arm).squeeze(0)
                end_q = torch.hstack((end_base, end_arm))
                end_q = end_q.unsqueeze(0)
                end_q = end_q.cpu().numpy()
                end_q = end_q[0]
                self.end_q = end_q

                self.tiago_handler.set_upper_body_positions(
                    jnt_positions=torch.tensor(
                        np.array([base_positions[4:]]),
                        dtype=torch.float,
                        device=self._device,
                    )
                )
                return

        if actions == "return_arm":
            if self.ik_success:
                # self.attach_object(self._grasp_objs[0])
                # self.start_q = self.end_q
                # start_arm = np.array([0.25,1, 1.5707, 1.5707, 1, 1.5, -1.5707, 1.0])
                # start_arm = torch.tensor(start_arm).unsqueeze(0)

                self.end_q = self.start_q.copy()
                self.tiago_handler.set_upper_body_positions(
                    jnt_positions=torch.tensor(
                        np.array([self.end_q[4:]]),
                        dtype=torch.float,
                        device=self._device,
                    )
                )

                print("Return arm")
                self.tiago_handler.set_base_positions(
                    jnt_positions=torch.tensor(
                        np.array([[self.tmp_x, self.tmp_y, self.tmp_theta]]),
                        dtype=torch.float,
                        device=self._device,
                    )
                )
            return

        if actions == "check_success":
            curr_pose = scene_utils.get_obj_pose(self._grasp_objs[0])
            #
            print(curr_pose[0][2] - self.obj_origin_pose[0][2])
            if curr_pose[0][2] - self.obj_origin_pose[0][2] >= 0.1:
                # self._is_success[0] = 1
                print("check_success")

    def get_se3_transform(self, prim):
        # print(f"Prim: {prim}")
        # Check if the prim is valid and has a computed transform
        if not prim:
            print("Invalid Prim")
            return None

        # Access the prim's transform attributes
        xform = UsdGeom.Xformable(prim)
        if not xform:
            print("Prim is not Xformable")
            return None

        # Get the local transformation matrix (this is relative to the prim's parent)
        local_transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)

        return local_transform

    def reset_idx(self, env_ids):
        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        # reset dof values
        self.tiago_handler.reset(indices, randomize=self._randomize_robot_on_reset)

        # reset the scene objects (randomize), get target end-effector goal/grasp as well as oriented bounding boxes of all other objects
        self._goal = scene_utils.setup_tabular_scene(
            self._grasp_objs, self.targets_position, self.targets_se3, self._device
        )
        goal_num = self._goal.shape[0]
        self._goal_tf = torch.zeros((goal_num, 4, 4), device=self._device)
        goal_rots = Rotation.from_quat(self._goal[:, 3:])  # 使用所有 goal 的四元數
        self._goal_tf[:, :3, :3] = torch.tensor(
            goal_rots.as_matrix(), dtype=float, device=self._device
        )
        self._goal_tf[:, :3, -1] = torch.tensor(
            self._goal[:, :3], device=self._device
        )  # 設定每個 goal 的 x, y, z
        self._goal_tf[:, -1, -1] = 1.0  # 保持齊次變換矩陣的結構
        self._curr_goal_tf = self._goal_tf.clone()
        self._goals_xy_dist = torch.linalg.norm(
            self._goal[:, 0:2], dim=1
        )  # 計算每個 goal 到原點的 x, y 距離
        self.curr_pos = (100, 100)
        # Pitch visualizer by 90 degrees for aesthetics

        for i in range(goal_num):
            goal_viz_rot = goal_rots[i] * Rotation.from_euler(
                "xyz", [0, np.pi / 2.0, 0]
            )
            print(f"self._goal[i, :3]: {self._goal[i, :3]}")
            if i == 0:
                self._goal_vizs1.set_world_poses(
                    indices=indices,
                    positions=self._goal[i, :3].unsqueeze(dim=0),
                    orientations=torch.tensor(
                        goal_viz_rot.as_quat()[[3, 0, 1, 2]], device=self._device
                    ).unsqueeze(dim=0),
                )
            # if i == 1:
            #    self._goal_vizs2.set_world_poses(indices=indices, positions=self._goal[i, :3].unsqueeze(dim=0), orientations=torch.tensor(goal_viz_rot.as_quat()[[3, 0, 1, 2]], device=self._device).unsqueeze(dim=0))

        # bookkeeping
        self.step_count = 0
        self._is_success[env_ids] = 0
        self._collided[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.final_place = [0, 0]
        self.path = []
        self.pos_idx = 0
        self.start_q = []
        self.end_q = []
        self.path_num = 0
        self.rot = 0
        self.se3_idx = 0

        self.flag = 0

    def check_robot_collisions(self):
        # Check if the robot collided with an object
        # TODO: Parallelize
        for obst in self._obstacles:
            raw_readings = self._contact_sensor_interface.get_contact_sensor_raw_data(
                obst.prim_path + "/Contact_Sensor"
            )
            if raw_readings.shape[0]:
                for reading in raw_readings:
                    # str
                    if "Tiago" in str(
                        self._contact_sensor_interface.decode_body_name(
                            reading["body1"]
                        )
                    ):
                        return True  # Collision detected with some part of the robot
                    if "Tiago" in str(
                        self._contact_sensor_interface.decode_body_name(
                            reading["body0"]
                        )
                    ):
                        return True  # Collision detected with some part of the robot
        for grasp_obj in self._grasp_objs:
            #    if grasp_obj == self._curr_grasp_obj: continue # Important. Exclude current target object for collision checking

            raw_readings = self._contact_sensor_interface.get_contact_sensor_raw_data(
                grasp_obj.prim_path + "/Contact_Sensor"
            )
            if raw_readings.shape[0]:
                for reading in raw_readings:
                    if "Tiago" in str(
                        self._contact_sensor_interface.decode_body_name(
                            reading["body1"]
                        )
                    ):
                        return True  # Collision detected with some part of the robot
                    if "Tiago" in str(
                        self._contact_sensor_interface.decode_body_name(
                            reading["body0"]
                        )
                    ):
                        return True  # Collision detected with some part of the robot
        return False

    def calculate_metrics(self) -> None:
        # print(f"check_robot_collisions: {self.check_robot_collisions()}")
        if self.check_robot_collisions():  # TODO: Parallelize
            # Collision detected. Give penalty and no other rewards
            self._collided[0] = 1
            self._is_success[0] = 0  # Success isn't considered in this case
        #        data = self.sd_helper.get_groundtruth(["boundingBox2DTight"], self.ego_viewport.get_viewport_window())["boundingBox2DTight"]
        #        rgb = self.sd_helper.get_groundtruth(["rgb"], self.ego_viewport.get_viewport_window())["rgb"]
        rgb = self.get_rgb_data()
        im = Image.fromarray(rgb)
        im.save("./data/end.png")
        # for box in data:
        #    sem_label = box["semanticLabel"]
        #    if sem_label == "good_part":
        #        print(f"Detect {sem_label}")
        #        if self._is_success[0] == 1:
        #            print("Success")

    def is_done(self) -> None:
        # resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(pole_pos) > np.pi / 2, 1, resets)
        # resets = torch.zeros(self._num_envs, dtype=int, device=self._device)

        # reset if success OR collided OR if reached max episode length
        resets = self._is_success.clone()
        resets = torch.where(self._collided.bool(), 1, resets)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets
