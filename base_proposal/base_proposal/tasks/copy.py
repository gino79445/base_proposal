


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
from omni.isaac.isaac_sensor import _isaac_sensor
#from omni.isaac.sensor import _sensor as _isaac_sensor
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

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import SamPredictor, sam_model_registry
# from omni.isaac.core.utils.prims import get_prim_at_path
# from omni.isaac.core.utils.prims import create_prim
# from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp
from omni.isaac.core.utils.torch.rotations import euler_angles_to_quats, quat_diff_rad
from scipy.spatial.transform import Rotation
from PIL import Image


# Base placement environment for fetching a target object among clutter
class NMTask(Task):
    def __init__(
        self,
        name,
        sim_config,
        env
    ) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._device = self._cfg["sim_device"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        
        self._gamma = self._task_cfg["env"]["gamma"]
        self._max_episode_length = self._task_cfg["env"]["horizon"]
        
        self._randomize_robot_on_reset = self._task_cfg["env"]["randomize_robot_on_reset"]

        # Get dt for integrating velocity commands and checking limit violations
        self._dt = torch.tensor(self._sim_config.task_config["sim"]["dt"]*self._sim_config.task_config["env"]["controlFrequencyInv"],device=self._device)

        # Environment object settings: (reset() randomizes the environment)
        self._obstacle_names = self._task_cfg["env"]["obstacles"] 
        self._tabular_obstacle_mask = [False, True] # Mask to denote which objects are tabular (i.e. grasp objects can be placed on them)
        self._grasp_obj_names = ["pot","cup","mug","008_pudding_box", "010_potted_meat_can", "061_foam_brick"] # YCB models in usd format
        #self._grasp_obj_names = ["dishwasher"] # YCB models in usd format
        self._num_obstacles = min(self._task_cfg["env"]["num_obstacles"],len(self._obstacle_names))
        self._num_grasp_objs = min(self._task_cfg["env"]["num_grasp_objects"],len(self._grasp_obj_names))
        #self._obj_states = torch.zeros((6*(self._num_obstacles+self._num_grasp_objs-1),self._num_envs),device=self._device) # All grasp objs except the target object will be used in obj state (BBox)
        self._obstacles = []
        self._obstacles_dimensions = []
        self._grasp_objs = []
        self._grasp_objs_dimensions = []
        #  Contact sensor interface for collision detection:
        self._contact_sensor_interface = _isaac_sensor.acquire_contact_sensor_interface()

        # Choose num_obs and num_actions based on task
        # 6D goal/target object grasp pose + 6D bbox for each obstacle in the room. All grasp objs except the target object will be used in obj state
        # (3 pos + 4 quat + 6*(n-1)= 7 + )
        
        #self._num_observations = 7 + len(self._obj_states) + 1 
        #self._num_observations = 1 
        self._move_group = self._task_cfg["env"]["move_group"]
        self._use_torso = self._task_cfg["env"]["use_torso"]
        # Position control. Actions are base SE2 pose (3) and discrete arm activation (2)
        #self._num_actions = self._task_cfg["env"]["continous_actions"] + self._task_cfg["env"]["discrete_actions"]
        # env specific limits
        self._world_xy_radius = self._task_cfg["env"]["world_xy_radius"]
        self._action_xy_radius = self._task_cfg["env"]["action_xy_radius"]
        self._action_ang_lim = self._task_cfg["env"]["action_ang_lim"]
        # self.max_arm_vel = torch.tensor(self._task_cfg["env"]["max_rot_vel"], device=self._device)
        self.max_rot_vel = torch.tensor(self._task_cfg["env"]["max_rot_vel"], device=self._device)
        self.max_base_xy_vel = torch.tensor(self._task_cfg["env"]["max_base_xy_vel"], device=self._device)
        
        # End-effector reaching settings
        self._goal_pos_threshold = self._task_cfg["env"]["goal_pos_thresh"]
        self._goal_ang_threshold = self._task_cfg["env"]["goal_ang_thresh"]
        # For now, setting dummy goal:
        self._goals = torch.hstack((torch.tensor([[0.8,0.0,0.4+0.15]]),euler_angles_to_quats(torch.tensor([[0.19635, 1.375, 0.19635]]),device=self._device)))[0].repeat(self.num_envs,1)
        self._goal_tf = torch.zeros((4,4),device=self._device)
        self._goal_tf[:3,:3] = torch.tensor(Rotation.from_quat(np.array([self._goals[0,3+1],self._goals[0,3+2],self._goals[0,3+3],self._goals[0,3]])).as_matrix(),dtype=float,device=self._device) # Quaternion in scalar last format!!!
        self._goal_tf[:,-1] = torch.tensor([self._goals[0,0], self._goals[0,1], self._goals[0,2], 1.0],device=self._device) # x,y,z,1
        self._curr_goal_tf = self._goal_tf.clone()
        #self._goals_xy_dist = torch.linalg.norm(self._goals[:,0:2],dim=1)  # distance from origin

        # Reward settings
        self._collided = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._is_success = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        self.collided = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.sam.to(self._device)
       # self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.mask_predictor = SamPredictor(self.sam)
        self.final_place = np.zeros((2), dtype=np.float32) 
        self.base_x = 0
        self.base_y = 0
        self.base_theta = 0
        self.r = 0
        self.theta = 0
        self.phi = 0

        #Set the angle
        self.all_angle = [  -np.pi/2.5, -np.pi/4, 0, np.pi/4, np.pi/2.5]
        self.angle_list = []
        for i in range(5):
            if self._task_cfg["env"]["angle_list"][i] == True:
                self.angle_list.append(self.all_angle[i])
        times = self._task_cfg["env"]["times"]
        self.angle = []
        for a in self.angle_list:
            for i in range(times):
                self.angle.append(a)
        self.angle.append(0)
        self.successful_angle = [0,0,0,0,0,0]
        self.collision_angle = [0,0,0,0,0,0]

        self.angle_idx = 0
        self.instruction = self._task_cfg["env"]["instruction"]        
        self.position_list = []
        self.fail_list = []
        self.positions = self._task_cfg["env"]["positions"] 
        self.arm = self._task_cfg["env"]["move_group"]
        self.ik_success_num = 0
        self.step_count = 0
        # IK solver
        self._ik_solver = PinTiagoIKSolver(move_group=self._move_group, include_torso=self._use_torso, include_base=True, max_rot_vel=100.0) # No max rot vel
        self._motion_planner = MotionPlannerTiago(move_group=self._move_group, include_torso=self._use_torso, include_base=True, max_rot_vel=100.0)
        #self.base_ik_solver = PinTiagoIKSolver(move_group=self._move_group, include_torso=self._use_torso, include_base=True, max_rot_vel=100.0) # No max rot vel
        # Handler for Tiago
        self.tiago_handler = TiagoDualWBHandler(move_group=self._move_group, use_torso=self._use_torso, sim_config=self._sim_config, num_envs=self._num_envs, device=self._device)

        #RLTask.__init__(self, name, env)
        Task.__init__(self, name,  env)

    def get_camera_intrinsics(self):
        # Get camera intrinsics for rendering
        return self.sd_helper.get_camera_intrinsics()

    def get_point_cloud(self,R, T, fx, fy, cx, cy):
        point_cloud = []
        depth =self.sd_helper.get_groundtruth(["depth"], self.ego_viewport.get_viewport_window())["depth"]
        rgb_data = self.sd_helper.get_groundtruth(["rgb"], self.ego_viewport.get_viewport_window())["rgb"]
    
        # flipup and flip left
        
        #rgb_data = np.flipud(rgb_data)
        #rgb_data = np.fliplr(rgb_data)
        #depth = np.flipud(depth)
        #depth = np.fliplr(depth)

        for i in range(rgb_data.shape[1]):
            for j in range(rgb_data.shape[0]):
                i = rgb_data.shape[1] - i 
                j = rgb_data.shape[0] - j 
                if  i > 0 and j > 0 and i < rgb_data.shape[1] and j < rgb_data.shape[0] and depth[j,i] > 0:
                    point = self.get_3d_point(i, j, depth[rgb_data.shape[0] - j,rgb_data.shape[1] - i], R, T, fx, fy, cx, cy)
                    point_cloud.append(point)
        point_cloud = np.array(point_cloud)

        # save rgb image
        im = Image.fromarray(rgb_data)
        im.save("./data/end_rgb.png")
       # save the point cloud
        np.save("./data/point_cloud.npy", point_cloud)
    def set_up_scene(self, scene) -> None:
        import omni
        if self._task_cfg["env"]["plane"] == True:
            scene_utils.add_plane(name="simple", prim_path=self.tiago_handler.default_zero_env_path, device=self._device)
        self.tiago_handler.get_robot()
        if self._task_cfg["env"]["house"] == True:
            scene_utils.sence(name="simple", prim_path=self.tiago_handler.default_zero_env_path, device=self._device)
        # Spawn obstacles (from ShapeNet usd models):
        for i in range(self._num_obstacles):
            obst = scene_utils.spawn_obstacle(name=self._obstacle_names[i], prim_path=self.tiago_handler.default_zero_env_path, device=self._device)
            self._obstacles.append(obst) # Add to list of obstacles (Geometry Prims)
            # Optional: Add contact sensors for collision detection. Covers whole body by default
            omni.kit.commands.execute("IsaacSensorCreateContactSensor", path="/Contact_Sensor", sensor_period=float(self._sim_config.task_config["sim"]["dt"]),
                parent=obst.prim_path)
        # Spawn grasp objs (from YCB usd models):
        for i in range(self._num_grasp_objs):
            grasp_obj = scene_utils.spawn_grasp_object(name=self._grasp_obj_names[i], prim_path=self.tiago_handler.default_zero_env_path, device=self._device)
            self._grasp_objs.append(grasp_obj) # Add to list of grasp objects (Rigid Prims)
            # Optional: Add contact sensors for collision detection. Covers whole body by default
            omni.kit.commands.execute("IsaacSensorCreateContactSensor", path="/Contact_Sensor", sensor_period=float(self._sim_config.task_config["sim"]["dt"]),
                parent=grasp_obj.prim_path)
        # Goal visualizer
        goal_viz1 = VisualCone(prim_path=self.tiago_handler.default_zero_env_path+"/goal1",
                                radius=0.05,height=0.05,color=np.array([1.0,0.0,0.0]))
        goal_viz2 = VisualCone(prim_path=self.tiago_handler.default_zero_env_path+"/goal2",
                                radius=0.05,height=0.05,color=np.array([0,0.0,1.0]))

        super().set_up_scene(scene)
        self._robots = self.tiago_handler.create_articulation_view()
        scene.add(self._robots)
        self._goal_vizs1 = GeometryPrimView(prim_paths_expr="/World/envs/.*/goal1",name="goal_viz1")
        scene.add(self._goal_vizs1)
        self._goal_vizs2 = GeometryPrimView(prim_paths_expr="/World/envs/.*/goal2",name="goal_viz2")
        scene.add(self._goal_vizs2)
        
        # Enable object axis-aligned bounding box computations
        scene.enable_bounding_boxes_computations()
        # Add spawned objects to scene registry and store their bounding boxes:
        for obst in self._obstacles:
            scene.add(obst)
            self._obstacles_dimensions.append(scene.compute_object_AABB(obst.name)) # Axis aligned bounding box used as dimensions
        for grasp_obj in self._grasp_objs:
            scene.add(grasp_obj)
            self._grasp_objs_dimensions.append(scene.compute_object_AABB(grasp_obj.name)) # Axis aligned bounding box used as dimensions
        # Optional viewport for rendering in a separate viewer
        from omni.isaac.synthetic_utils import SyntheticDataHelper
        self.viewport_window = omni.kit.viewport_legacy.get_default_viewport_window()
        self.sd_helper = SyntheticDataHelper()
        sensor_names = [
            "rgb",
            "depth",
            "boundingBox2DTight",
            "boundingBox2DLoose",
            "instanceSegmentation",
            "semanticSegmentation",
            "boundingBox3D",
            "camera",
            "pose",
        ]
        self.sd_helper.initialize(sensor_names, viewport=self.viewport_window)
        
       

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
        # 將向量正規化
        old_vector = old_vector / np.linalg.norm(old_vector)
        new_vector = new_vector / np.linalg.norm(new_vector)
        
        # 計算旋轉角度
        #angle = np.arctan2(new_vector[1], new_vector[0]) - np.arctan2(old_vector[1], old_vector[0])
        angle = np.arctan2(old_vector[1], old_vector[0]) - np.arctan2(new_vector[1], new_vector[0])
        print(f"angle: {angle}")
        print(f"angle: {angle * 180 / np.pi}")
        
        # 創建旋轉矩陣
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        


        # 計算平移向量
        translation_vector = new_origin - old_origin
        
        # 首先平移點到新的原點
        
        #translated_points = points - old_origin
        
        # 旋轉點
        rotated_points = np.dot(points, rotation_matrix)
        final_points = rotated_points + translation_vector
        #final_points = rotated_points + new_origin
        
        # 再次平移點到新的位置
        #final_points = rotated_points + new_origin
        
        return final_points
 
    def get_base(self):
        return self.r, self.phi, self.theta

    def get_path(self):
        return self.path
    

    def transform_path(self,path):
        transformed_path = []
        pre_angle = 0  # 初始角度
        pre_x, pre_y = 0, 0  # 初始座標

        for i, (cur_x, cur_y) in enumerate(path):
            if i == 0:
                x, y = cur_x - pre_x, cur_y - pre_y
                transformed_path.append([x, y])
            else:
                # 計算當前點與上一點的相對座標
                x, y = cur_x - pre_x, cur_y - pre_y

                # 計算當前角度
                cur_angle = np.arctan2(y, x)

                # 計算角度變化量
                angle = cur_angle - pre_angle

                # 旋轉座標
                x_t = x * np.cos(-pre_angle) - y * np.sin(-pre_angle)
                y_t = x * np.sin(-pre_angle) + y * np.cos(-pre_angle)

                transformed_path.append([x_t, y_t])

                # 更新前一點的資訊
                pre_angle = cur_angle

            pre_x, pre_y = cur_x, cur_y

        return transformed_path

    def angle_between(self, u, v):
        """
        傳回向量 u 與向量 v 之間的夾角（弧度），範圍在 [0, π]。
        """
        ux, uy = u
        vx, vy = v
        # 內積：u dot v = |u|*|v|*cos(theta)
        dot = ux*vx + uy*vy
        mag_u = np.sqrt(ux*ux + uy*uy)
        mag_v = np.sqrt(vx*vx + vy*vy)
        # 為避免浮點誤差超出 [-1,1]，我們夾住在 [-1,1]
        cos_theta = max(-1.0, min(1.0, dot/(mag_u*mag_v + 1e-12)))
        return np.arccos(cos_theta)

    def merge_small_angle_increments(self,local_increments, deg_threshold=5.0):
        """
        把相鄰兩個增量的「方向差」小於 deg_threshold（單位：度）時，合併在一起。
        傳回合併後的局部增量列表。
        """
        rad_threshold = np.deg2rad(deg_threshold)
        merged = []

        for inc in local_increments:
            # 如果 merged 還是空的，直接放進去
            if not merged:
                merged.append(inc)
            else:
                last_inc = merged[-1]
                # 先判斷這兩個增量方向差
                angle_diff = self.angle_between(last_inc, inc)
                if angle_diff < rad_threshold:
                    # 方向幾乎相同，把向量相加
                    new_inc = [last_inc[0] + inc[0], last_inc[1] + inc[1]]
                    merged[-1] = new_inc
                else:
                    # 方向不一樣，另外開一個新的增量段
                    merged.append(inc)

        return merged

    def set_path(self, position):
        self.final_place = position
        cell_size = 0.05

        w = self.rgb_data.shape[1]
        h = self.rgb_data.shape[0]
        goal = position
        print(f"Goal: {goal}")

        start = (100, 100)
        end = (int(goal[1] / cell_size )+100, int(goal[0] / cell_size + 100))
        #end = (97, 130)

        print(f"Start: {start}, End: {end}")
        occupancy_2d_map = self.occupancy_2d_map
        path = astar_utils.a_star2(occupancy_2d_map, start, end)
        self.path = []
        for p in path:
            x = (p[1] - 100) * cell_size
            y = (p[0] - 100) * cell_size
            self.path.append([x, y])
        path = []
        self.path = self.transform_path(self.path)
        self.path = self.merge_small_angle_increments(self.path, deg_threshold=3.0)
        return self.path

    def build_local_map(self, R, T, fx, fy, cx, cy):

        # 2d occupancy map
        self.occupancy_2d_map = np.zeros((200, 200), dtype=np.uint8)
        #self.occupancy_2d_map.fill(255)
        depth = self.depth_data
        map_size = (200, 200)
        cell_size = 0.05
        for i in range(-10, 20):
            for j in range(-20, 20):
                self.occupancy_2d_map[100+j,100+i] = 0

        
        for i in range(self.rgb_data.shape[1]):
            for j in range(self.rgb_data.shape[0]):
                i = self.rgb_data.shape[1] - i 
                j = self.rgb_data.shape[0] - j 
                if  i > 0 and j > 0 and i < self.rgb_data.shape[1] and j < self.rgb_data.shape[0] and depth[j,i] > 0:
                    depth[j,i] = depth[j,i]  
                    point = self.get_3d_point(i, j, depth[self.rgb_data.shape[0] - j,self.rgb_data.shape[1] - i], R, T, fx, fy, cx, cy)
                    map_x = int(point[0] / cell_size)
                    map_x += int(map_size[0] / 2)
                    map_y = int(point[1] / cell_size)
                    map_y += int(map_size[1] / 2)
                    if 0 <= map_x < map_size[0] and 0 <= map_y < map_size[1]:
                        if point[2] < 0.1:
                            self.occupancy_2d_map[map_y, map_x] = 0
                        else:
                            self.occupancy_2d_map[map_y, map_x] = 255
        

        im = Image.fromarray(self.occupancy_2d_map)
        im.save("./data/occupancy_2d_map.png")


    def get_observations(self):
        # Handle any pending resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        if self.step_count >= 1:
            return        
        
        
        # perceptial data
        self.bounding_box = self.sd_helper.get_groundtruth(["boundingBox2DTight"], self.ego_viewport.get_viewport_window())["boundingBox2DTight"]
        self.rgb_data = self.sd_helper.get_groundtruth(["rgb"], self.ego_viewport.get_viewport_window())["rgb"]
        self.depth_data = self.sd_helper.get_groundtruth(["depth"], self.ego_viewport.get_viewport_window())["depth"]
        self.pose_data = self.sd_helper.get_groundtruth(["pose"], self.ego_viewport.get_viewport_window())["pose"]
        self.rgb_data = self.rgb_data[:,:,:3]
    
        # get the camera parameters
        R,T ,fx, fy, cx, cy = self.retrieve_camera_params()
        depth = self.depth_data
        self.rgb_data = self.rgb_data.astype('uint8')      
        # save the rgb image
        im = Image.fromarray(self.rgb_data)
        im.save("./data/rgb.png")


        rgb = self.rgb_data.copy()        
        for box in self.bounding_box:
            if box['semanticLabel'] == 'target':  
                x_min, y_min, x_max, y_max = int(box['x_min']), int(box['y_min']), int(box['x_max']), int(box['y_max'])
                cv2.rectangle(rgb, (x_min, y_min), (x_max, y_max), (220, 0, 0), 1)
                # save the rgb image        
                im = Image.fromarray(rgb)
                im.save("./data/original.png")
                im.save(f"./data/original_{self.angle_idx}.png")
        

        if self._task_cfg["env"]["check_env"] == True:
            return

        self.occupancy_2d_map = np.zeros((200, 200), dtype=np.uint8)
                
        #self.build_local_map(R, T, fx, fy, cx, cy)

        return 
#        base_pixel = get_base("./data/rgb.png", self.instruction,self.depth_data, self.occupancy_2d_map, R, T, fx, fy, cx, cy)
#        #print(f"base_pixel: {base_pixel}")
#        #base_pixel = [597, 373]
#        # transform the pixel to 3d point
#        w = self.rgb_data.shape[1]
#        h = self.rgb_data.shape[0]
#
#        base_point = self.get_3d_point( w - base_pixel[0], h - base_pixel[1], self.depth_data[base_pixel[1], base_pixel[0]], R, T, fx, fy, cx, cy)
#        #affordance_point = self.get_3d_point(w - affordance_center[1], h - affordance_center[0], self.depth_data[affordance_center[0], affordance_center[1]], R, T, fx, fy, cx, cy)
#        #base_point = [[2.57833435],[0.59395751],[0.0]]
#        #base_point = np.array(base_point)
#        print(f"base_point: {base_point}")
#        
#        self.final_place = np.zeros((2), dtype=np.float32)
#        self.final_place[0] = base_point[0]
#        self.final_place[1] = base_point[1]
#        x_scaled = torch.tensor([self.final_place[0]],device=self._device) 
#        y_scaled = torch.tensor([self.final_place[1]],device=self._device)
#        self.r = torch.sqrt(x_scaled**2 + y_scaled**2)
#        self.phi = torch.atan2(y_scaled,x_scaled)
#        self.theta = torch.atan2(y_scaled,x_scaled)
#        return 
#
#        
#        
#       
#
#        if len(self.positions) > 0:
#            print(f"angle_idx: {self.angle_idx}")
#            self.final_place[0] = self.positions[self.angle_idx-1][0]
#            self.final_place[1] = self.positions[self.angle_idx-1][1]
#            return
#
#       # # astar 
#        if self._task_cfg["env"]["astar_target"] == True:
#            start = (100, 100)
#            goal = scene_utils.get_goal_center()
#            end = (int(goal[1] / cell_size )+100, int(goal[0] / cell_size + 100))
#            # draw the circle around the goal
#            for i in range(1):
#                for j in range(1):
#                    x = end[0] + i
#                    y = end[1] + j
#                    if x >= 0 and x < 200 and y >= 0 and y < 200:
#                        self.occupancy_2d_map[x, y] = 100
#
#            occupancy_2d_map = self.occupancy_2d_map
#            path = astar_utils.a_star_target(occupancy_2d_map, start, end)
#            #path = astar_utils.a_star_rough(occupancy_2d_map, start, end)
#            if path is None:
#                print("No path found")
#            else:
#                for i in range(len(path)):
#                    pass
#            # save occupancy map
#            im = Image.fromarray(occupancy_2d_map)
#            im.save("./data/occupancy_2d_map.png")
#
#            self.final_place[0] = (path[-1][1] - 100) * cell_size
#            self.final_place[1] = (path[-1][0] - 100) * cell_size
#            return 
#
#
#        if self._task_cfg["env"]["rrt_target"] == True:
#            start = (100, 100)
#            goal = scene_utils.get_goal_center()
#            end = (int(goal[1] / cell_size )+100, int(goal[0] / cell_size + 100))
#            print(f"Start: {start}, End: {end}")
#            occupancy_2d_map = self.occupancy_2d_map
#            path = rrt_utils.rrt_star_target(occupancy_2d_map, start, end)
#            print(f"Path: {path}")
#            #remove the last 
#            path = path[:-1]
#
#            if path is None:
#                print("No path found")
#            else:
#                for i in range(len(path)):
#                    pass
#                    
#
#            self.final_place[0] = (path[-1][1] - 100) * cell_size
#            self.final_place[1] = (path[-1][0] - 100) * cell_size
#            return 
#
#        target_image = np.zeros((self.rgb_data.shape[0], self.rgb_data.shape[1], 3), dtype=np.uint8)
#        target_mask = np.zeros((self.rgb_data.shape[0], self.rgb_data.shape[1]), dtype=np.uint8)
#        target_x_min , target_y_min, target_x_max, target_y_max = 0, 0, 0, 0
#        for box in self.bounding_box:
#            if box['semanticLabel'] == 'target':
#                x_min, y_min, x_max, y_max = int(box['x_min']), int(box['y_min']), int(box['x_max']), int(box['y_max'])
#                target_x_min, target_y_min, target_x_max, target_y_max = x_min, y_min, x_max, y_max
#                #segment the target object
#                bbox = np.array([x_min, y_min, x_max, y_max])
#                self.mask_predictor.set_image(self.rgb_data)
#                target_mask, scores, logits = self.mask_predictor.predict(
#                    point_coords=None,
#                    point_labels=None,
#                    box= bbox,
#                    multimask_output=False
#                )
#                target_mask = target_mask[0]
#                y_min = max(0, y_min - 0) 
#                y_max = min(self.rgb_data.shape[0], y_max + 0)
#                x_min = max(0, x_min - 0)
#                x_max = min(self.rgb_data.shape[1], x_max + 0)
#                # only save the target in the image
#                target_image[y_min:y_max, x_min:x_max] = self.rgb_data[y_min:y_max, x_min:x_max]
#                # save the rgb image
#                im = Image.fromarray(target_image)
#                im.save("./data/target.png")
#                # save the mask
#                im = Image.fromarray(target_mask)
#                im.save("./data/target_mask.png")
#                break
#
#        # part to grab
#        path = "./data/original.png"
#        t = 0
#        N = 10
#        if self._task_cfg["env"]["astar"] == True or self._task_cfg["env"]["rrt"] == True:
#            N = 0
#            part_to_grab = ["knobs", "buttons"]
#        while t < N:
#            try:
#                part_to_grab = determine_part_to_grab(path, self.instruction)
#                break
#            except:
#                t += 1
#                time.sleep(20)
#                print("Error in determining part to grab")
#
#        # if duplicate parts, remove the duplicates
#        part_to_grab = list(set(part_to_grab))
#
#        part_to_grab_str = ""
#        if len(part_to_grab) > 1:
#            part_to_grab_str = " and ".join(part_to_grab)
#        else:
#            part_to_grab_str = part_to_grab[0] if len(part_to_grab) > 0 else "nothing"
#        if part_to_grab_str == "nothing":
#            print("No part to grab")
#            return
#        print(f"Part to grab: {part_to_grab_str}")
#
#
#        rgb = cv2.imread("./data/original.png")
#        affordance = np.zeros((self.rgb_data.shape[0], self.rgb_data.shape[1]), dtype=bool)
#        for part in part_to_grab:
#            i  = 0
#            box_threshold = 0.3
#            text_threshold = 0.25
#            while i < 1:
#                cmd = f"sudo docker run --gpus all -it --rm -v ./data:/opt/program/data/ groundingdino python grounding.py {part} --image_path ./data/target.png --box_threshold {box_threshold} --text_threshold {text_threshold}"
#                os.system(cmd)
#                # get the detected boxes
#                detected_boxes = np.load(f"./data/detected_boxes.npy")
#                #text_threshold -= 0.01
#                #box_threshold -= 0.01
#                if len(detected_boxes) > 1:
#                    break
#                i += 1
#            tmp = detected_boxes.copy()
#            # plot the detected boxes   
#            count = 0
#            indices_to_delete = []
#            for box in detected_boxes:
#                x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#                area = (x_max - x_min) * (y_max - y_min)
#                if area > (target_x_max - target_x_min) * (target_y_max - target_y_min) / 2:
#                    # remove the box
#                    indices_to_delete.append(count)
#                else:
#                    #affordance[y_min:y_max, x_min:x_max] = 1
#                    cv2.rectangle(rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#                count += 1
#        
#            detected_boxes = np.delete(detected_boxes, indices_to_delete, axis=0)
#            # save the rgb image
#            #im = Image.fromarray(rgb)
#            #im.save(f"./data/{part}_detected.png")
#        
#           # segment the detected object
#            masks = np.zeros((self.rgb_data.shape[0], self.rgb_data.shape[1]), dtype=bool)
#            affordance_tmp = np.zeros((self.rgb_data.shape[0], self.rgb_data.shape[1]), dtype=bool)
#            for mybox in detected_boxes:
#                mybox = np.array(mybox)
#                self.mask_predictor.set_image(self.rgb_data)
#                mask, scores, logits = self.mask_predictor.predict(
#                    point_coords=None,
#                    point_labels=None,
#                    box=mybox,
#                    multimask_output=False
#                )
#                affordance_tmp = np.logical_or(affordance_tmp, mask)        
#                break
#            affordance_tmp = affordance_tmp[0]
#            affordance = np.logical_or(affordance, affordance_tmp)
#            #break
#        K = 10
#        if affordance.sum() != 0:
#            im = Image.fromarray(affordance)
#            im.save("./data/target_mask.png")
#            K = 1
#
#        # get the features
#        cluster_points, cluster_labels , number_list = get_features.get_features(R, T, fx, fy, cx, cy,  self.depth_data, K)
#        mask = np.zeros((self.rgb_data.shape[0], self.rgb_data.shape[1]), dtype=np.uint8)
#        affordance = np.zeros((self.rgb_data.shape[0], self.rgb_data.shape[1]), dtype=bool)
#        print(f"Number list: {number_list}") 
#
#
#        affordance_list = []
#        for i in range(1):
#            t = 0
#            while t < 10:
#                try:
#                    affordance_num = determine_affordance("./data/clustered_image.png", part_to_grab_str, self.instruction, number_list)
#                    affordance_list.append(affordance_num)
#                    break
#                except:
#                    t += 1
#                    time.sleep(20)
#                    print("Error in determining affordance")
#            time.sleep(5)
#                
#        print(f"Affordance: {affordance_list}")
#        #select the most common affordance
#        affordance_num = max(set(affordance_list), key = affordance_list.count)   
#        print(f"Affordance: {affordance_num}")
#        points = cluster_points[cluster_labels == affordance_num-1]
#       # count = 0
#       # for point in points:
#       #     mask[point[0], point[1]] = 255
#       #     affordance[point[0], point[1]] = True
#       #     count += 1
#
#        # calculate the center of the affordance area
#        affordance_center = np.zeros((2), dtype=np.float32)
#        #count = 0
#        #for i in range(affordance.shape[1]):
#        #    for j in range(affordance.shape[0]):
#        #        if affordance[j,i] == 1:
#        #            affordance_center[0] += i
#        #            affordance_center[1] += j
#        #            count += 1
#        #affordance_center[0] /= count 
#        #affordance_center[1] /= count
# 
#        distance = np.linalg.norm(points - points.mean(axis=0), axis=1)
#        closest_point = points[np.argmin(distance)]
#        affordance_center = closest_point  
#        # get the affordance center in 3d
#        h = self.rgb_data.shape[0]
#        w = self.rgb_data.shape[1]
#        affordance_point = self.get_3d_point(w - affordance_center[1], h - affordance_center[0], self.depth_data[affordance_center[0], affordance_center[1]], R, T, fx, fy, cx, cy)
#        
#        
#
#        # 3d point
#        points = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.float32)
#        map_size = (200, 200) 
#        cell_size = 0.05  
#        occupancy_map = np.zeros((map_size[0], map_size[1],3), dtype=np.uint8)
#        # make occupancy map blue
#        occupancy_map[:,:] = [0, 0, 200]
#        # 2d occupancy map
#        occupancy_2d_map = np.zeros((map_size[0], map_size[1]), dtype=np.uint8)
#        affordance_x = 0
#        affordance_y = 0        
#
#        count = 0
#        target_height = 0
#        masks = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=bool)
#        for i in range(rgb.shape[1]):
#            for j in range(rgb.shape[0]):
#                i = rgb.shape[1] - i 
#                j = rgb.shape[0] - j 
#                if  i > 0 and j > 0 and i < rgb.shape[1] and j < rgb.shape[0] and depth[j,i] > 0:
#                    depth[j,i] = depth[j,i]  
#                    point = self.get_3d_point(i, j, depth[self.rgb_data.shape[0] - j,self.rgb_data.shape[1] - i], R, T, fx, fy, cx, cy)
#                    map_x = int(point[0] / cell_size)
#                    map_x += int(map_size[0] / 2)
#                    map_y = int(point[1] / cell_size)
#                    map_y += int(map_size[1] / 2)
#                    if 0 <= map_x < map_size[0] and 0 <= map_y < map_size[1]:
#                        
#                        if point[2] > 0.1:
#                            
#                            if target_mask[rgb.shape[0] - j, rgb.shape[1] - i] == 1:
#                                occupancy_map[map_y, map_x] = [200, 200,0]
#                                if point[2] > target_height:
#                                    target_height = point[2]
#                            else:
#                                occupancy_map[map_y, map_x] = [200, 0, 0]
#                            occupancy_2d_map[map_y, map_x] = 1
#                        else:
#                            occupancy_map[map_y, map_x] = [0, 200, 0]
#                        if point[2] < 0.01:
#                            masks[j,i] = 1
#                    
#                       # if affordance[rgb.shape[0] - j, rgb.shape[1] - i] == 1:
#                       #     count += 1
#                       #     affordance_x += map_x
#                       #     affordance_y += map_y
#        masks = np.flipud(masks)
#        masks = np.fliplr(masks)
#        target_height += 0.05              
#                                                        
#        map_x = int(0 / cell_size)
#        map_x += int(map_size[0] / 2)
#        map_y = int(0 / cell_size)
#        map_y += int(map_size[1] / 2)
#
#        
#            # the base area is 0.5m * 0.5m
#        for i in range(-5, 5):
#            for j in range(-5, 5):
#                occupancy_map[map_y+j, map_x+i] = [200, 200, 200]
#        # draw the affordance circle
##        if count > 0:
##            affordance_x /= count
##            affordance_y /= count
#                    
#
#
#
#        # astar 
#        start = (100, 100)
#        goal = scene_utils.get_goal_center()
#        end = (int(goal[1] / cell_size )+100, int(goal[0] / cell_size + 100))
#        occupancy_map[int(goal[1] / cell_size) + 100, int(goal[0] / cell_size) + 100] = [0, 255, 0]
#        self.occupancy_2d_map = occupancy_2d_map
#        #path = astar_utils.a_star(occupancy_2d_map, start, end)
#        #path = astar_utils.a_star_rough(occupancy_2d_map, start, end)
#        #if path is None:
#        #    print("No path found")
#        #else:
#        #    for i in range(len(path)):
#        #        pass
#        #        occupancy_map[path[i][0], path[i][1]] = [0, 0, 15]
#
#        #    self.final_place[0] = (path[-1][1] - 100) * cell_size
#        #    self.final_place[1] = (path[-1][0] - 100) * cell_size
#
#
#
#
#
#                        
#        h_size = 1000
#        w_size = 1000
#        occupancy_map = cv2.resize(occupancy_map, (1000, 1000), interpolation=cv2.INTER_NEAREST)
#        occupancy_map = np.rot90(occupancy_map)
#        occupancy_map = np.fliplr(occupancy_map)
#        occupancy_map = cv2.cvtColor(occupancy_map, cv2.COLOR_BGR2RGB)
#
#
#        affordance_x = int(affordance_point[0] / cell_size)
#        affordance_x += int(map_size[0] / 2)
#        affordance_y = int(affordance_point[1] / cell_size)
#        affordance_y += int(map_size[1] / 2)
#
#
#        # astar 
#        #goal = scene_utils.get_goal_center()
#        #end = (int(goal[1] / cell_size )+100, int(goal[0] / cell_size + 100))
#        if self._task_cfg["env"]["astar"] == True:
#            start = (100, 100)
#            end = (affordance_y, affordance_x)
#            occupancy_2d_map = self.occupancy_2d_map
#            path = astar_utils.a_star(occupancy_2d_map, start, end)
#            path = astar_utils.a_star_rough(occupancy_2d_map, start, end)
#
#            print(f"Path: {path}")
#            if path is None:
#                print("No path found")
#            else:
#                for i in range(len(path)):
#                    occupancy_map[path[i][0], path[i][1]] = [0, 0, 15]
#                    
#
#            self.final_place[0] = (path[-1][1] - 100) * cell_size
#            self.final_place[1] = (path[-1][0] - 100) * cell_size
#            return 
#        
#        if self._task_cfg["env"]["rrt"] == True:
#            start = (100, 100)
#            end = (affordance_y, affordance_x)
#            print(f"Start: {start}, End: {end}")
#            occupancy_2d_map = self.occupancy_2d_map
#            path = rrt_utils.rrt_star(occupancy_2d_map, start, end)
#            print(f"Path: {path}")
#            #remove the last 
#            path = path[:-1]
#
#            if path is None:
#                print("No path found")
#            else:
#                for i in range(len(path)):
#                    pass
#                    
#
#            self.final_place[0] = (path[-1][1] - 100) * cell_size
#            self.final_place[1] = (path[-1][0] - 100) * cell_size
#            return 
#        
#        # plot the points round the affordance center in the occupancy map
#        num_points = 18
#        num_points = 15
#        count = 0
#        counts = []
#        candidate_points = []
#        points = []
#        for i in range(num_points):
#            angle = 2 * np.pi / num_points * i
#            x = affordance_x + 12 * np.cos(angle)
#            y = affordance_y + 12 * np.sin(angle)
#            x = int(x)
#            y = int(y)
#            points.append([(x - 100) * cell_size, (y - 100) * cell_size, 0])
#            count += 1
#            # check if the point is on the obstacle
#            #if self.occupancy_2d_map[y, x] == 1: 
#            #    continue
#
#            if not astar_utils.is_valid_20(y, x, self.occupancy_2d_map):
#                continue
#            # 3d point in the world coordinate
#            candidate_points.append([(x - 100) * cell_size, (y - 100) * cell_size, 0])
#            occupancy_map = cv2.circle(occupancy_map, (int(1000 - y*5), int(1000 - x*5)), 3, (0, 135, 255), -1)
#            occupancy_map = cv2.putText(occupancy_map, str(count), (int(1000 - y*5), int(1000 - x*5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 200), 2)
#            counts.append(count)
#        # save the 2d occupancy map
#        im = Image.fromarray(self.occupancy_2d_map * 255)
#        im.save("./data/occupancy_2d_map.png")
#
#
#        
#        # plot the affordance center
#        occupancy_map = cv2.circle(occupancy_map, (int(1000  - affordance_y  * 5), int(1000  - affordance_x * 5)), 10, (255, 0, 255), 3)
#        occupancy_map = cv2.cvtColor(occupancy_map, cv2.COLOR_BGR2RGB)
#        im = Image.fromarray(occupancy_map)
#        im.save("./data/occupancy_map.png")
#        
#        
#        rgb = self.rgb_data.copy()
#        # masks is a boolean array
##        masks = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=bool)
#        # all true
#        #masks = np.logical_or(masks, True)
#        # execute the instruction "sudo docker run --gpus all -it --rm -v ./data:/opt/program/data/ groundingdino python grounding.py ground"
##        cmd = "sudo docker run --gpus all -it --rm -v ./data:/opt/program/data/ groundingdino python grounding.py floor"
##        os.system(cmd)
##        # get the detected boxes
##        detected_boxes = np.load('./data/detected_boxes.npy')
##        #print(detected_boxes)
##        for mybox in detected_boxes:
##            mybox = np.array(mybox)
##
##            self.mask_predictor.set_image(self.rgb_data)
##            mask, scores, logits = self.mask_predictor.predict(
##                point_coords=None,
##                point_labels=None,
##                box=mybox,
##                multimask_output=False
##            )
##            masks = np.logical_or(masks, mask)
##        
##            break
##
##        masks = masks[0]
##
#        count = 0
#        # get the affordance center in the 3d world coordinate
#        #affordance_depth = depth[int(affordance_center[1]), int(affordance_center[0])]
#        #affordance_point = self.get_3d_point(rgb.shape[1] - affordance_center[0], rgb.shape[0] - affordance_center[1], affordance_depth, R, T, fx, fy, cx, cy)
#        # Draw the affordance center in the rgb image
#        #cv2.circle(rgb, (int(affordance_center[0]), int(affordance_center[1])), 10, (255, 0, 0), -1)
#        #cv2.circle(rgb, (int(affordance_center[0]), int(affordance_center[1])), 10, (55, 0, 0), 2)
#
#
#
#   #     a, b = self.get_pixel(affordance_point, R, T, fx, fy, cx, cy)
#   #     b = rgb.shape[0] - b
#   #     a = rgb.shape[1] - a
#   #     cv2.circle(rgb, (a, b), 10, (255, 0, 255), 3)
#   #     cv2.putText(rgb, "affordance center", (a-5, b-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
#        count_list = []
#        point_list = []
#        for x,y,_ in candidate_points:
#            point = np.array([[x], [y], [0]])
#            a, b = self.get_pixel(point, R, T, fx, fy, cx, cy)
#            b = rgb.shape[0] - b
#            a = rgb.shape[1] - a
#
#            h_point = np.array([[x], [y], [target_height]])
#            a_h, b_h = self.get_pixel(h_point, R, T, fx, fy, cx, cy)
#            b_h = rgb.shape[0] - b_h
#            a_h = rgb.shape[1] - a_h
#            
#            # equation of the line passing through the affordance center and the candidate point in the 3d world coordinate
#            # get the vector of the line
#            vector_3d = np.array([x - affordance_point[0], y - affordance_point[1], target_height - affordance_point[2]])
#            direction_3d = vector_3d / 10
#            
#            t = 1
#            visible = True
#            
#            while t <= 10:
#                t_point = np.array([affordance_point[0] + t * direction_3d[0], affordance_point[1] + t * direction_3d[1], affordance_point[2] + t * direction_3d[2]])
#                a_t, b_t = self.get_pixel(t_point, R, T, fx, fy, cx, cy)
#                b_t = rgb.shape[0] - b_t
#                a_t = rgb.shape[1] - a_t
#                point = self.get_3d_point(rgb.shape[1] - a_t, rgb.shape[0] - b_t, depth[b_t, a_t], R, T, fx, fy, cx, cy)
#                if point[0] < t_point[0]: 
#                    visible = False
#                    break
#                    
#                t += 1
#            if a < rgb.shape[1] and b < rgb.shape[0] and a > 0 and b > 0 and masks[b,a] and visible:
#                #cv2.putText(rgb, str(count), (a-20, b-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
#                #cv2.circle(rgb, (a, b), 5, (255, 135, 0), -1)
#                point_list.append([a, b])
#                count_list.append(counts[count])
#                #cv2.line(rgb, (int(affordance_center[0]), int(affordance_center[1])), (a, b), (0, 135, 0), 1)
#                #cv2.putText(rgb, str(count), (a_h-5, b_h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
#                #cv2.circle(rgb, (a_h, b_h), 5, (255, 135, 0), -1)
#                #cv2.line(rgb, (int(affordance_center[0]), int(affordance_center[1])), (a_h, b_h), (0, 135, 0), 1)
#                #cv2.line(rgb, (a_h, b_h), (a, b), (135, 0, 135), 1)
#
#            count += 1
#        num_insertions = 15
#
#        # set the red circile points in per 1 meter in the rgb image from left 10m to right 10m
#        count = 1
#        w = -4
#        for i in range(0,32):
#            w += 0.25
#            h = 0
#            for j in range(0, 30):
#
#                point = np.array([[h], [w], [0]])
#                a, b = self.get_pixel(point, R, T, fx, fy, cx, cy)
#                # flip the image from top to bottom
#                b = rgb.shape[0] - b
#                # flip the image from left to right
#                a = rgb.shape[1] - a
#                #if a < rgb.shape[1] and b < rgb.shape[0] and a > 0 and b > 0 and masks[b,a]:
#                #    cv2.putText(rgb, str(count), (a-5, b-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
#                #    # plot the point
#                #    cv2.circle(rgb, (a, b), 5, (255, 135, 0), -1)
#                if j < 29: # 保證不會在最後一個點之後插值
#                    next_point = np.array([[h + 0.5], [w], [0]])
#                    next_a, next_b = self.get_pixel(next_point, R, T, fx, fy, cx, cy)
#                    next_b = rgb.shape[0] - next_b
#                    next_a = rgb.shape[1] - next_a
#
#                    # 根據插入點的數量進行插值
#                    for k in range(1, num_insertions + 1):
#                        interpolated_a = a + (next_a - a) * (k / (num_insertions + 1))
#                        interpolated_b = b + (next_b - b) * (k / (num_insertions + 1))
#                        if interpolated_a < rgb.shape[1] and interpolated_b < rgb.shape[0] and interpolated_a > 0 and interpolated_b > 0 and masks[int(interpolated_b),int(interpolated_a)]:
#                            cv2.circle(rgb, (int(interpolated_a), int(interpolated_b)), 1, (0, 155, 0), -1)
#                if i < 32:  # 保證不會在最後一列之後插值
#                    next_point_h = np.array([[h], [w + 0.5], [0]])
#                    next_a_h, next_b_h = self.get_pixel(next_point_h, R, T, fx, fy, cx, cy)
#                    next_b_h = rgb.shape[0] - next_b_h
#                    next_a_h = rgb.shape[1] - next_a_h
#
#                    # 橫向插值
#                    for k in range(1, num_insertions + 1):
#                        interpolated_a_h = a + (next_a_h - a) * (k / (num_insertions + 1))
#                        interpolated_b_h = b + (next_b_h - b) * (k / (num_insertions + 1))
#                        if interpolated_a_h < rgb.shape[1] and interpolated_b_h < rgb.shape[0] and interpolated_a_h > 0 and interpolated_b_h > 0 and masks[int(interpolated_b_h),int(interpolated_a_h)]:
#                            cv2.circle(rgb ,(int(interpolated_a_h), int(interpolated_b_h)), 1, (0, 155, 0), -1)
#                h += 0.25
#                count += 1
#        
#        i = 0 
#        for a, b in point_list:
#            cv2.circle(rgb, (a, b), 14, (255, 170, 0), -1)
#            cv2.circle(rgb, (a, b), 14, (2,77,207), 2)
#            text_width, text_height = cv2.getTextSize(f"{count_list[i]}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
#            cv2.putText(rgb, f"{count_list[i]}", (a - text_width // 2, b + text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2)
#
#           # cv2.circle(rgb, (a, b), 5, (255, 135, 0), -1)
#           # cv2.circle(final_image, (centroid[1], centroid[0]), 10, (255, 255, 255), -1)
#           # cv2.circle(final_image, (centroid[1], centroid[0]), 10, (0, 0, 255), 1)
#           # text_width, text_height = cv2.getTextSize(f"{number}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
#           # cv2.putText(final_image, f"{number}", (centroid[1] - text_width // 2, centroid[0] + text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#            i += 1
#         
#         
#        for box in self.bounding_box:
#            if box['semanticLabel'] == 'target':
#                x_min, y_min, x_max, y_max = int(box['x_min']), int(box['y_min']), int(box['x_max']), int(box['y_max'])
#                cv2.rectangle(rgb, (x_min, y_min), (x_max, y_max), (200, 0, 0), 1)
#                # save the rgb image        
#                im = Image.fromarray(rgb)
#                im.save("./data/result.png")
#        
#        
#        path = "./data/result.png"
#        ID_list = []
#        for i in range(5):
#            t = 0
#            while t < 10:
#                try:
#                    ID = determine_base(path, part_to_grab_str, count_list)
#                    ID_list.append(ID)
#                    break
#                except:
#                    t += 1
#                    time.sleep(10)
#                    print("Error in determining base")
#            time.sleep(5)
#        # Select the point in the ID_list that have the smallest sum of differences with the other points
#        min_sum_diff = 1000
#        for i in range(len(ID_list)):
#            sum_diff = 0
#            for j in range(len(ID_list)):
#                sum_diff += abs(ID_list[i] - ID_list[j])
#            if sum_diff < min_sum_diff:
#                min_sum_diff = sum_diff
#                ID = ID_list[i]
#        
#
#        print("ID_list", ID_list)
#        print("MID", ID)
#        print("count_list", count_list)
#        mid = 0
#        if self.arm == "arm_left":
#            first = ID
#            #if first > 1 and (first - 1) in count_list:
#            #    first -= 1
#            end = (ID + 4) % 18
#            if end not in count_list:
#                for i in range(1, 5):
#                    if (end - i) % 18 in count_list:
#                        end = (end - i) % 18
#                        break
#            print("end", end)
#            print("first", first)
#            mid = math.ceil((first + end) / 2)
#        if self.arm == "arm_right":
#            first = (ID + 14) % 18
#            if first not in count_list:
#                for i in range(1, 5):
#                    if (first + i) % 18 in count_list:
#                        first = (first + i) % 18
#                        break
#            end = ID
#            #if end < count_list[len(count_list) - 1] and (end + 1) in count_list:
#            #    end += 1
#            print("end", end)
#            print("first", first)
#            mid = math.floor((first + end) / 2)
#
#        print("mid", mid)
#        closest_point = 0
#        if mid in count_list:
#            closest_point = mid
#        if mid not in count_list:
#            # get the closest point to the mid point
#            min_diff = 1000
#            for i in range(len(count_list)):
#                diff = abs(count_list[i] - mid)
#                if diff < min_diff:
#                    min_diff = diff
#                    closest_point = count_list[i]
#        #ID = closest_point 
#        print("ID", ID)
#        idx = ID - 1    
#        print("idx", idx)
#        self.final_place[0] = points[idx][0]
#        self.final_place[1] = points[idx][1]
#        # calculate the 2d vector of the line passing through the affordance center and the candidate point in the plane
#        self.target_direction = np.array([points[idx][0] - affordance_point[0], points[idx][1] - affordance_point[1]])
#        # normalize the vector
#        self.target_direction = self.target_direction / np.linalg.norm(self.target_direction)
#        self.target_direction = np.squeeze(self.target_direction)
#        self.target_direction =  self.target_direction
#        print(f"target_direction: {self.target_direction}")
#        if self._task_cfg["env"]["course"] == True:
#            # get the point with the distance of 0.7 meter from the affordance center
#            affordance_point = np.array([affordance_point[0].item(), affordance_point[1].item()])
#            print(f"affordance_point: {affordance_point}")
#            base_point = affordance_point + 0.7 * self.target_direction
#            print(f"len target_direction: {np.linalg.norm(self.target_direction)}")
#            # print length of the vector base_point - affordance_point
#            print(f"len base_point - affordance_point: {np.linalg.norm(base_point - affordance_point)}")
#            print(f"base_point: {base_point}")
#            # check if the point is valid, if not, the distance - 0.05  and check again
#            
#            success = True
#            while not astar_utils.is_valid(int((base_point[1] - 100) / 0.05), int((base_point[0] - 100) / 0.05), self.occupancy_2d_map):
#                base_point = base_point - 0.05 * self.target_direction
#                if np.linalg.norm(base_point - affordance_point[0:2]) < 0.05:
#                    success = False
#                    break
#            if success:
#                self.final_place[0] = base_point[0]
#                self.final_place[1] = base_point[1]
#            return
#        
#
#
#
#        z = float(affordance_point[2])
#        base_list  = base_position(z)
#        #print(f"base_list: {base_list}")
#        affordance_point = [affordance_point[0].item(), affordance_point[1].item()]
#        affordance_point = np.array(affordance_point)
#
#        transformed_points = self.transform_points(base_list, np.array([0,0]), affordance_point, np.array([-1,0]) , self.target_direction)
#        #print(f"transformed_points: {transformed_points}")
#        base = [0,0]
#        for point in transformed_points:
#            print(f"point: {point}")
#            x = int((point[0].item() )/0.05 + 100)
#            y = int((point[1].item() )/0.05 + 100)
#            if astar_utils.is_valid(y, x, self.occupancy_2d_map):
#                base = [point[0].item(), point[1].item()]
#                break
#        self.final_place[0] = base[0]
#        self.final_place[1] = base[1]
#        return
#
          

    def get_render(self):
        # Get ground truth viewport rgb image
        gt = self.sd_helper.get_groundtruth(
            ["rgb"], self.viewport_window, verify_sensor_init=False, wait_for_sensor_data=0
        )
        return np.array(gt["rgb"])




    def get_motion_num(self):
        print(f"motion_path: {self.motion_path}")   
        return len(self.motion_path)


    def set_new_base(self,x_scaled,y_scaled,theta_scaled):

        # NOTE: Actions are in robot frame but the handler is in world frame!
        # Get current base positions
        base_joint_pos = self.tiago_handler.get_robot_obs()[:,:3] # First three are always base positions
        base_tf = torch.zeros((4,4),device=self._device)
        base_tf[:2,:2] = torch.tensor([[torch.cos(base_joint_pos[0,2]), -torch.sin(base_joint_pos[0,2])],[torch.sin(base_joint_pos[0,2]), torch.cos(base_joint_pos[0,2])]]) # rotation about z axis
        base_tf[2,2] = 1.0 # No rotation here
        base_tf[:,-1] = torch.tensor([base_joint_pos[0,0], base_joint_pos[0,1], 0.0, 1.0]) # x,y,z,1

        # Transform actions to world frame and apply to base
        action_tf = torch.zeros((4,4),device=self._device)
        action_tf[:2,:2] = torch.tensor([[torch.cos(theta_scaled[0]), -torch.sin(theta_scaled[0])],[torch.sin(theta_scaled[0]), torch.cos(theta_scaled[0])]])
        action_tf[2,2] = 1.0 # No rotation here
        action_tf[:,-1] = torch.tensor([x_scaled[0], y_scaled[0], 0.0, 1.0]) # x,y,z,1
        return base_tf, action_tf

    def set_angle(self,theta):
        self.rot  = theta


    def pre_physics_step(self, actions) -> None:
        # actions (num_envs, num_action)
        # Handle resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        
        R,T ,fx, fy, cx, cy = self.retrieve_camera_params()
      
        if actions == 'set_angle':
            x = torch.tensor([0],device=self._device)
            y = torch.tensor([0],device=self._device)
            theta = torch.tensor([self.rot],device=self._device)
            base_tf, action_tf = self.set_new_base(x,y,theta)
            new_base_tf = torch.matmul(base_tf,action_tf)
            new_base_xy = new_base_tf[0:2,3].unsqueeze(dim=0)
            new_base_theta = torch.arctan2(new_base_tf[1,0],new_base_tf[0,0]).unsqueeze(dim=0).unsqueeze(dim=0)
            self.new_base_tf = new_base_tf
            self.new_base_xy = new_base_xy
            self.new_base_theta = new_base_theta
            self.path = []
            return 

        if actions == 'get_base':

            self.final_place = torch.tensor(self.path[self.pos_idx],device=self._device)
            x_scaled = torch.tensor([self.final_place[0]],device=self._device) 
            y_scaled = torch.tensor([self.final_place[1]],device=self._device)
            theta_scaled = torch.atan2(y_scaled,x_scaled)
            base_tf, action_tf = self.set_new_base(x_scaled,y_scaled,theta_scaled)
            new_base_tf = torch.matmul(base_tf,action_tf)
            new_base_xy = new_base_tf[0:2,3].unsqueeze(dim=0)
            new_base_theta = torch.arctan2(new_base_tf[1,0],new_base_tf[0,0]).unsqueeze(dim=0).unsqueeze(dim=0)
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

            point_cloud = np.load('./data/point_cloud.npy')
            point_cloud = point_cloud.squeeze()
            point_cloud = point_cloud[::100]
            point_cloud = point_cloud[point_cloud[:, 2] > 0.1]
            point_cloud = point_cloud[point_cloud[:, 2] < 3]
            path = self._motion_planner.rrt_motion_plan_with_obstacles(self.start_q, self.end_q, point_cloud, max_iters=500, step_size=0.1)
            self.motion_path = path
            return

        if actions == 'move_arm':
            
                    
            if self.ik_success:
                
                if self.path_num >= len(self.motion_path):
                    return
                base_positions = self.motion_path[self.path_num][0:3]

                #self.tiago_handler.set_base_positions(jnt_positions=torch.tensor(np.array([base_positions]),dtype=torch.float,device=self._device))
                self.tiago_handler.set_upper_body_positions(jnt_positions=torch.tensor(np.array(self.motion_path[self.path_num][4:]),dtype=torch.float,device=self._device))
                self.path_num += 1
                return 

        if self._is_success[0] == 1:
            return



        
        if actions == 'forward':
            actions = np.zeros(3)
            actions[0] = self.max_base_xy_vel
            actions = torch.unsqueeze(torch.tensor(actions,dtype=torch.float,device=self._device),dim=0)
            self.tiago_handler.apply_base_actions(actions)
            return
       
        if actions == 'right_rotate':
            actions = np.zeros(3)
            actions[2] = self.max_rot_vel
            actions = torch.unsqueeze(torch.tensor(actions,dtype=torch.float,device=self._device),dim=0)
            self.tiago_handler.apply_base_actions(actions)
            return
       
        if actions == 'left_rotate':
            actions = np.zeros(3)
            actions[2] = -self.max_rot_vel
            actions = torch.unsqueeze(torch.tensor(actions,dtype=torch.float,device=self._device),dim=0)
            self.tiago_handler.apply_base_actions(actions)
            return

        
        # Move base
        if actions == 'set_base':
            self.tiago_handler.set_base_positions(torch.hstack((self.new_base_xy,self.new_base_theta)))
            self.x_delta= self.new_base_xy[0,0].cpu().numpy()
            self.y_delta = self.new_base_xy[0,1].cpu().numpy()
            self.theta_delta = self.new_base_theta[0,0].cpu().numpy()
            #theta_delta = theta
            self.pos_idx += 1
            if self.pos_idx == len(self.path) or len(self.path) == 0:
                self.pos_idx = 0
            inv_base_tf = torch.linalg.inv(self.new_base_tf)
            self._curr_goal_tf = torch.matmul(inv_base_tf,self._goal_tf)
#            print(f"Goal position: {self._curr_goal_tf}")


        if actions == 'manipulate':
          #if torch.linalg.norm(self._curr_goal_tf[0:2,3]) < 0.01 :
            #print(f"Goal position: {self._curr_goal_tf[0:3,3]}")
            #self._is_success[0] = 1 
            #curr_goal_pos = self._curr_goal_tf[0:3,3]
            #curr_goal_pos = self._curr_goal_tf[self.se3_idx,0:3,3]
            x_scaled = torch.tensor([0],device=self._device)
            y_scaled = torch.tensor([0],device=self._device)
            #theta_scaled = torch.tensor([torch.atan2(curr_goal_pos[1], curr_goal_pos[0])],device=self._device)
            theta_scaled = torch.tensor([0],device=self._device)

            base_tf, action_tf = self.set_new_base(x_scaled,y_scaled,theta_scaled)
            new_base_tf = torch.matmul(base_tf,action_tf)
            new_base_xy = new_base_tf[0:2,3].unsqueeze(dim=0)
            new_base_theta = torch.arctan2(new_base_tf[1,0],new_base_tf[0,0]).unsqueeze(dim=0).unsqueeze(dim=0)
            self.base_theta = torch.arctan2(new_base_tf[1,0],new_base_tf[0,0])
            self.tiago_handler.set_base_positions(torch.hstack((new_base_xy,new_base_theta)))
            
            # Transform goal to robot frame
            inv_base_tf = torch.linalg.inv(new_base_tf)
            self._curr_goal_tf = torch.matmul(inv_base_tf,self._goal_tf)

            #curr_goal_pos = self._curr_goal_tf[0:3,3]
            curr_goal_pos = self._curr_goal_tf[self.se3_idx,0:3,3]
            #curr_goal_quat = Rotation.from_matrix(self._curr_goal_tf[:3,:3]).as_quat()[[3, 0, 1, 2]]
            curr_goal_quat = Rotation.from_matrix(self._curr_goal_tf[self.se3_idx,:3,:3]).as_quat()[[3, 0, 1, 2]]
            self.se3_idx += 1

            success_list, base_positions_list = self._ik_solver.solve_ik_pos_tiago(des_pos=curr_goal_pos.cpu().numpy(), des_quat=curr_goal_quat,
                                        #pos_threshold=self._goal_pos_threshold, angle_threshold=self._goal_ang_threshold, verbose=False, Rmin=[-0.0, -0.0, 0.965,-0.259],Rmax=[0.0, 0.0, 1, 0.259])
                                        pos_threshold=self._goal_pos_threshold, angle_threshold=self._goal_ang_threshold, verbose=False, Rmin=[-0.0, -0.0, 0.866,-0.5],Rmax=[0.0, 0.0, 1, 0.5])
            success = False
            for i in range(len(success_list)):
                if success_list[i]:
                    x = int((base_positions_list[i][0] + self.base_x)/0.05 + 100)
                    y = int((base_positions_list[i][1] + self.base_y)/0.05 + 100)
                    success = success_list[i]
                    base_positions = base_positions_list[i]
                    
                    if astar_utils.is_valid(y, x, self.occupancy_2d_map):
                        success = success_list[i]
                        base_positions = base_positions_list[i]
                        break
            self.ik_success = False
            self.path_num = 0
            self.motion_path = []
            if success:


                self.ik_success = True
                theta = torch.arctan2(torch.tensor(base_positions[3]), torch.tensor(base_positions[2])) 
                theta += theta_scaled.item()
                theta += self.theta_delta
                
                #theta = theta.cpu().numpy()
                
               # x_scaled = torch.tensor([base_positions[0]],device=self._device)
               # y_scaled = torch.tensor([base_positions[1]],device=self._device)
               # theta_scaled = torch.tensor([theta],device=self._device)
               # #theta_scaled = torch.tensor([torch.arctan2(torch.tensor(base_positions[3]), torch.tensor(base_positions[2]))],device=self._device)
               # base_tf, action_tf = self.set_new_base(x_scaled,y_scaled,theta_scaled)
               # new_base_tf = torch.matmul(base_tf,action_tf)
               # new_base_xy = new_base_tf[0:2,3].unsqueeze(dim=0)
               # new_base_theta = torch.arctan2(new_base_tf[1,0],new_base_tf[0,0]).unsqueeze(dim=0).unsqueeze(dim=0)
               # inv_base_tf = torch.linalg.inv(new_base_tf)
               # self._curr_goal_tf = torch.matmul(inv_base_tf,self._goal_tf)
            
                self.tiago_handler.set_base_positions(jnt_positions=torch.tensor(np.array([[base_positions[0]+self.x_delta,base_positions[1]+self.y_delta,theta]]),dtype=torch.float,device=self._device))

                self.tmp_x = base_positions[0] + self.x_delta
                self.tmp_y = base_positions[1] + self.y_delta
                self.tmp_theta = self.theta_delta 
                
                print(f"base_positions.shape: {base_positions.shape}")
                start_arm = self.tiago_handler.get_upper_body_positions()
                start_base = self.tiago_handler.get_base_positions()
                start_base = np.array([0, 0, 1, 0])
                start_base = torch.tensor(start_base)
                start_base = start_base.unsqueeze(0)
                start_q = torch.hstack((start_base,start_arm))
                start_q = start_q.unsqueeze(0)
                start_q = start_q.cpu().numpy()
                start_q = start_q[0][0]
                print(f"start_q: {start_q}")
                self.start_q = start_q
                
                
                end_base= np.array([base_positions[0],base_positions[1], base_positions[2], base_positions[3]])
                end_base = np.array([0, 0, 1, 0])
                end_arm  = np.array([base_positions[4:]])
                end_base = torch.tensor(end_base)
                end_arm = torch.tensor(end_arm).squeeze(0)
                end_q = torch.hstack((end_base,end_arm))
                end_q = end_q.unsqueeze(0)
                end_q = end_q.cpu().numpy()
                end_q = end_q[0]
                self.end_q = end_q
                self.tiago_handler.set_upper_body_positions(jnt_positions=torch.tensor(np.array([base_positions[4:]]),dtype=torch.float,device=self._device))
                return

        if actions == 'return_arm':
            if self.ik_success:
                print("return arm")
                #self.start_q = self.end_q
                self.end_q = self.start_q.copy()
                self.tiago_handler.set_upper_body_positions(jnt_positions=torch.tensor(np.array([self.end_q[4:]]),dtype=torch.float,device=self._device))

                self.tiago_handler.set_base_positions(jnt_positions=torch.tensor(np.array([[self.tmp_x,self.tmp_y,self.tmp_theta]]),dtype=torch.float,device=self._device))

            return
                    #self.tiago_handler.set_base_positions(jnt_positions=torch.tensor(np.array([[base_positions[0]+x_delta,base_positions[1]+y_delta,theta]]),dtype=torch.float,device=self._device))
                    #self.tiago_handler.set_upper_body_positions(jnt_positions=torch.tensor(np.array([base_positions[4:]]),dtype=torch.float,device=self._device))
            #    self._is_success[0] = 1
            #    curr_goal_pos = self._curr_goal_tf[0:3,3]
            #    curr_goal_quat = Rotation.from_matrix(self._curr_goal_tf[:3,:3]).as_quat()[[3, 0, 1, 2]]
            #    success, ik_positions = self._ik_solver.solve_ik_pos_tiago(des_pos=curr_goal_pos.cpu().numpy(), des_quat=curr_goal_quat,
            #                            pos_threshold=self._goal_pos_threshold, angle_threshold=self._goal_ang_threshold, verbose=False)            
            #    #print(f"Success: {success}")
            #    if success:
            #        for box in self.bounding_box:
            #            if box['semanticLabel'] == 'cabinet_STEEL':
            #                self._num_ik_successes += 1
            #                break
            #        
            #        print(f"_num_ik_successes: {self._num_ik_successes}")
                    # set upper body positions
                    #self.tiago_handler.set_upper_body_positions(jnt_positions=torch.tensor(np.array([ik_positions]),dtype=torch.float,device=self._device))
                    #self.tiago_handler.set_base_positions(jnt_positions=torch.tensor(np.array([[ik_positions[0]+x,ik_positions[1]+y,ik_positions[3]*3.1415926/2+theta]]),dtype=torch.float,device=self._device))
                    #self.tiago_handler.set_upper_body_positions(jnt_positions=torch.tensor(np.array([ik_positions[4:]]),dtype=torch.float,device=self._device))

                

 

   # def pre_physics_step(self, actions) -> None:
   #     self.step_count += 1
   #     # actions (num_envs, num_action)
   #     # Handle resets
   #     reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
   #     if len(reset_env_ids) > 0:
   #         self.reset_idx(reset_env_ids)

   #     if torch.all(actions == 0):
   #         self.position_list.append([self.final_place[0], self.final_place[1]])
   #         print(f"final place: {self.position_list}")
   #         print("move gripper")
   #         curr_goal_pos = self._curr_goal_tf[0:3,3]
   #         curr_goal_quat = Rotation.from_matrix(self._curr_goal_tf[:3,:3]).as_quat()[[3, 0, 1, 2]]

   #        # success_list, base_positions_list = self.base_ik_solver.solve_ik_pos_tiago(des_pos=curr_goal_pos.cpu().numpy(), des_quat=curr_goal_quat,
   #        #                             pos_threshold=self._goal_pos_threshold, angle_threshold=self._goal_ang_threshold, verbose=False)
   #         success_list, base_positions_list = self._ik_solver.solve_ik_pos_tiago(des_pos=curr_goal_pos.cpu().numpy(), des_quat=curr_goal_quat,
   #                                     pos_threshold=self._goal_pos_threshold, angle_threshold=self._goal_ang_threshold, verbose=False, Rmin=[-0.0, -0.0, 0.965,-0.259],Rmax=[0.0, 0.0, 1, 0.259])
   #         success = False
   #         for i in range(len(success_list)):
   #             if success_list[i]:
   #                 x = int((base_positions_list[i][0] + self.base_x)/0.05 + 100)
   #                 y = int((base_positions_list[i][1] + self.base_y)/0.05 + 100)
   #                 success = success_list[i]
   #                 base_positions = base_positions_list[i]
   #                 
   #                 if astar_utils.is_valid(y, x, self.occupancy_2d_map):
   #                     success = success_list[i]
   #                     base_positions = base_positions_list[i]
   #                     break
   #             

   #         
   #         


   #         if success:
   #             print("ik success")
   #             self.ik_success_num += 1
   #             print(f"ik success num: {self.ik_success_num}")
   #             # success angle 
   #             angle = self.angle[self.angle_idx-1]
   #             print(f"angle: {angle}")
   #             if angle == -np.pi/2.5:
   #                 self.successful_angle[0] += 1
   #             if angle == -np.pi/4:
   #                 self.successful_angle[1] += 1
   #             if angle == 0:
   #                 self.successful_angle[2] += 1
   #             if angle == np.pi/4:
   #                 self.successful_angle[3] += 1
   #             if angle == np.pi/2.5:
   #                 self.successful_angle[4] += 1
   #             self._is_success[0] = 1
   #             theta = torch.arctan2(torch.tensor(base_positions[3]), torch.tensor(base_positions[2])) + self.base_theta
   #             self.tiago_handler.set_base_positions(jnt_positions=torch.tensor(np.array([[base_positions[0]+self.base_x,base_positions[1]+self.base_y,theta]]),dtype=torch.float,device=self._device))
   #             self.tiago_handler.set_upper_body_positions(jnt_positions=torch.tensor(np.array([base_positions[4:]]),dtype=torch.float,device=self._device))
   #         else:
   #             self.fail_list.append(self.angle_idx-1)
   #         print(f"successful_angle: {self.successful_angle}")
   #         print(f"fail_list: {self.fail_list}")
   #         return
   #             
   #     

   #     curr_goal_pos = self._curr_goal_tf[0:3,3]
   #     curr_goal_quat = Rotation.from_matrix(self._curr_goal_tf[:3,:3]).as_quat()[[3, 0, 1, 2]]
   #     
   #     x_scaled = torch.tensor([self.final_place[0]],device=self._device) 
   #     y_scaled = torch.tensor([self.final_place[1]],device=self._device)
   #     theta_scaled = torch.tensor([0],device=self._device)

   #     if torch.all(actions == 1):
   #         x_scaled = torch.tensor([0],device=self._device)
   #         y_scaled = torch.tensor([0],device=self._device)
   #         theta_scaled = torch.tensor([torch.atan2(curr_goal_pos[1], curr_goal_pos[0])],device=self._device)
   #         
   #           
   #     
   #     # NOTE: Actions are in robot frame but the handler is in world frame!
   #     # Get current base positions
   #     base_joint_pos = self.tiago_handler.get_robot_obs()[:,:3] # First three are always base positions
   #     
   #     base_tf = torch.zeros((4,4),device=self._device)
   #     base_tf[:2,:2] = torch.tensor([[torch.cos(base_joint_pos[0,2]), -torch.sin(base_joint_pos[0,2])],[torch.sin(base_joint_pos[0,2]), torch.cos(base_joint_pos[0,2])]]) # rotation about z axis
   #     base_tf[2,2] = 1.0 # No rotation here
   #     base_tf[:,-1] = torch.tensor([base_joint_pos[0,0], base_joint_pos[0,1], 0.0, 1.0]) # x,y,z,1

   #     # Transform actions to world frame and apply to base
   #     action_tf = torch.zeros((4,4),device=self._device)
   #     action_tf[:2,:2] = torch.tensor([[torch.cos(theta_scaled[0]), -torch.sin(theta_scaled[0])],[torch.sin(theta_scaled[0]), torch.cos(theta_scaled[0])]])
   #     action_tf[2,2] = 1.0 # No rotation here
   #     action_tf[:,-1] = torch.tensor([x_scaled[0], y_scaled[0], 0.0, 1.0]) # x,y,z,1

   #     new_base_tf = torch.matmul(base_tf,action_tf)
   #     new_base_xy = new_base_tf[0:2,3].unsqueeze(dim=0)
   #     new_base_theta = torch.arctan2(new_base_tf[1,0],new_base_tf[0,0]).unsqueeze(dim=0).unsqueeze(dim=0)
   #     self.base_x = new_base_tf[0,3]
   #     self.base_y = new_base_tf[1,3]
   #     self.base_theta = torch.arctan2(new_base_tf[1,0],new_base_tf[0,0])
   #     self.tiago_handler.set_base_positions(torch.hstack((new_base_xy,new_base_theta)))
   #     
   #     # Transform goal to robot frame
   #     inv_base_tf = torch.linalg.inv(new_base_tf)
   #     self._curr_goal_tf = torch.matmul(inv_base_tf,self._goal_tf)
   #     curr_goal_pos = self._curr_goal_tf[0:3,3]
   #     curr_goal_quat = Rotation.from_matrix(self._curr_goal_tf[:3,:3]).as_quat()[[3, 0, 1, 2]]

    
    def get_se3_transform(self,prim):
        print(f"Prim: {prim}")
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
        self.tiago_handler.reset(indices,randomize=self._randomize_robot_on_reset)
        angle = self.angle[self.angle_idx]
        self.angle_idx += 1
        # reset the scene objects (randomize), get target end-effector goal/grasp as well as oriented bounding boxes of all other objects
        self._goal= scene_utils.setup_tabular_scene(
                                self, self._obstacle_names[0] ,angle, self._obstacles, self._tabular_obstacle_mask[0:self._num_obstacles], self._grasp_objs,
                                self._obstacles_dimensions, self._grasp_objs_dimensions, self._world_xy_radius, self._device)
        goal_num = self._goal.shape[0]
        self._goal_tf = torch.zeros((goal_num, 4, 4), device=self._device)
        goal_rots = Rotation.from_quat(self._goal[:, 3:])  # 使用所有 goal 的四元數
        self._goal_tf[:, :3, :3] = torch.tensor(goal_rots.as_matrix(), dtype=float, device=self._device)
        self._goal_tf[:, :3, -1] = torch.tensor(self._goal[:, :3], device=self._device)  # 設定每個 goal 的 x, y, z
        self._goal_tf[:, -1, -1] = 1.0  # 保持齊次變換矩陣的結構
        self._curr_goal_tf = self._goal_tf.clone()
        self._goals_xy_dist = torch.linalg.norm(self._goal[:, 0:2], dim=1)  # 計算每個 goal 到原點的 x, y 距離
        # Pitch visualizer by 90 degrees for aesthetics
       # for i in range(goal_num):
       #     goal_viz_rot = goal_rots[i] * Rotation.from_euler("xyz", [0, np.pi / 2.0, 0])
       #     print(f"self._goal[i, :3]: {self._goal[i, :3]}")
       #     if i == 0:
       #         self._goal_vizs1.set_world_poses(indices=indices, positions=self._goal[i, :3].unsqueeze(dim=0), orientations=torch.tensor(goal_viz_rot.as_quat()[[3, 0, 1, 2]], device=self._device).unsqueeze(dim=0))   
       #     if i == 1:
       #         self._goal_vizs2.set_world_poses(indices=indices, positions=self._goal[i, :3].unsqueeze(dim=0), orientations=torch.tensor(goal_viz_rot.as_quat()[[3, 0, 1, 2]], device=self._device).unsqueeze(dim=0))
            

        #self._curr_obj_bboxes = self._obj_bboxes.clone()
        # self._goals[env_ids] = torch.hstack((goals_sample[:,:3],euler_angles_to_quats(goals_sample[:,3:6],device=self._device)))
        
#        self._goal_tf = torch.zeros((4,4),device=self._device)
#        goal_rot = Rotation.from_quat(np.array([self._goals[0,3+1],self._goals[0,3+2],self._goals[0,3+3],self._goals[0,3]])) # Quaternion in scalar last format!!!
#        self._goal_tf[:3,:3] = torch.tensor(goal_rot.as_matrix(),dtype=float,device=self._device)
#        self._goal_tf[:,-1] = torch.tensor([self._goals[0,0], self._goals[0,1], self._goals[0,2], 1.0],device=self._device) # x,y,z,1
#        self._curr_goal_tf = self._goal_tf.clone()
#        self._goals_xy_dist = torch.linalg.norm(self._goals[:,0:2],dim=1) # distance from origin
#        # print distances
#        print(f"Goal distances: {self._goals_xy_dist}")
#        # Pitch visualizer by 90 degrees for aesthetics
       # goal_viz_rot = goal_rot * Rotation.from_euler("xyz", [0,np.pi/2.0,0])
       # if self._task_cfg["env"]["check_env"] == True :
       #     self._goal_vizs.set_world_poses(indices=indices,positions=self._goals[:,:3],
       #     orientations=torch.tensor(goal_viz_rot.as_quat()[[3, 0, 1, 2]],device=self._device).unsqueeze(dim=0))
        #self._goal_vizs.set_world_poses(indices=indices,positions=self._goals[:,:3],
        #orientations=torch.tensor(goal_viz_rot.as_quat()[[3, 0, 1, 2]],device=self._device).unsqueeze(dim=0))

        # bookkeeping
        self.step_count = 0
        self.base_x = 0
        self.base_y = 0
        self.base_theta = 0
        self._is_success[env_ids] = 0
        self._collided[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.final_place = [0,0] 
        self.flag = 0
        self.collided[env_ids] = 0
        self.path = []
        self.pos_idx = 0
        self.start_q = []
        self.end_q = []
        self.path_num = 0
        self.rot =0 

        self.se3_idx = 0 




    def check_robot_collisions(self):
        # Check if the robot collided with an object
        # TODO: Parallelize
        for obst in self._obstacles:
            raw_readings = self._contact_sensor_interface.get_contact_sensor_raw_data(obst.prim_path + "/Contact_Sensor")
            if raw_readings.shape[0]:                
                for reading in raw_readings:
                   # print(f"sensors: {self._contact_sensor_interface.decode_body_name(reading['body0'])}")
                   # print(f"sensors: {self._contact_sensor_interface.decode_body_name(reading['body1'])}")
                    if "Tiago" in str(self._contact_sensor_interface.decode_body_name(reading["body1"])):
                        return True # Collision detected with some part of the robot
                    if "Tiago" in str(self._contact_sensor_interface.decode_body_name(reading["body0"])):
                        return True # Collision detected with some part of the robot
        #for grasp_obj in self._grasp_objs:
        #    if grasp_obj == self._curr_grasp_obj: continue # Important. Exclude current target object for collision checking

       #     raw_readings = self._contact_sensor_interface.get_contact_sensor_raw_data(grasp_obj.prim_path + "/Contact_Sensor")
            if raw_readings.shape[0]:
                for reading in raw_readings:
                    if "Tiago" in str(self._contact_sensor_interface.decode_body_name(reading["body1"])):
                        return True # Collision detected with some part of the robot
                    if "Tiago" in str(self._contact_sensor_interface.decode_body_name(reading["body0"])):
                        return True # Collision detected with some part of the robot
        return False
    

    def calculate_metrics(self) -> None:

        #print(f"check_robot_collisions: {self.check_robot_collisions()}")
        if(self.check_robot_collisions()): # TODO: Parallelize
            # Collision detected. Give penalty and no other rewards
            self._collided[0] = 1
            self._is_success[0] = 0 # Success isn't considered in this case
            print("Collision detected")
            # collision angle
            angle = self.angle[self.angle_idx-1]
            print(f"angle: {angle}")
            if angle == -np.pi/2.5:
                self.collision_angle[0] += 1
            if angle == -np.pi/4:
                self.collision_angle[1] += 1
            if angle == 0:
                self.collision_angle[2] += 1
            if angle == np.pi/4:
                self.collision_angle[3] += 1
            if angle == np.pi/2.5:
                self.collision_angle[4] += 1
        print(f"collision_angle: {self.collision_angle}") 
        data = self.sd_helper.get_groundtruth(["boundingBox2DTight"], self.ego_viewport.get_viewport_window())["boundingBox2DTight"]
        rgb = self.sd_helper.get_groundtruth(["rgb"], self.ego_viewport.get_viewport_window())["rgb"]
        im = Image.fromarray(rgb)
        im.save("./data/end.png")
        #for box in data:
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
