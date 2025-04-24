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


from abc import abstractmethod
import numpy as np
import torch

# from gym import spaces
# from mushroom_rl.utils.spaces import *
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.cloner import GridCloner
from base_proposal.tasks.utils.usd_utils import create_distant_light
from omni.kit.viewport.utility import get_active_viewport
import omni.kit.viewport.utility as vp_utils
import omni.replicator.core as rep

import omni.kit
import omni
from pxr import UsdGeom
from pxr import Gf
import math
from scipy.spatial.transform import Rotation

# Usd
from pxr import Usd
import omni.isaac.surface_gripper
from pxr import UsdPhysics, Gf


class Task(BaseTask):
    """This class provides a PyTorch RL-specific interface for setting up RL tasks.
    It includes utilities for setting up RL task related parameters,
    cloning environments, and data collection for RL algorithms.
    """

    def __init__(self, name, env, offset=None) -> None:
        """Initializes RL parameters, cloner object, and buffers.

        Args:
            name (str): name of the task.
            env (VecEnvBase): an instance of the environment wrapper class to register task.
            offset (Optional[np.ndarray], optional): offset applied to all assets of the task. Defaults to None.
        """

        super().__init__(name=name, offset=offset)

        self.test = self._cfg["test"]
        self._device = self._cfg["sim_device"]
        print("Task Device:", self._device)

        self.clip_obs = self._cfg["task"]["env"].get("clipObservations", np.Inf)
        self.clip_actions = self._cfg["task"]["env"].get("clipActions", np.Inf)
        self.rl_device = self._cfg.get("rl_device", "cuda:0")

        self.control_frequency_inv = self._cfg["task"]["env"].get(
            "controlFrequencyInv", 1
        )

        print("RL device: ", self.rl_device)

        self._env = env

        #    if not hasattr(self, "_num_agents"):
        #        self._num_agents = 1  # used for multi-agent environments
        #    if not hasattr(self, "_num_states"):
        #        self._num_states = 0

        #    # initialize data spaces (defaults to gym.Box or Mushroom Box)
        #    if not hasattr(self, "action_space"):
        #        self.action_space = Box(np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0)
        #    if not hasattr(self, "observation_space"):
        #        self.observation_space = Box(np.ones(self.num_observations) * -np.Inf, np.ones(self.num_observations) * np.Inf)
        #    if not hasattr(self, "state_space"):
        #        self.state_space = Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)

        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self.default_base_env_path)
        define_prim(self.default_zero_env_path)

        self.cleanup()

    def detach_object(self, obj):
        stage = omni.usd.get_context().get_stage()
        gripper_path = "/World/envs/env_0/TiagoDualHolo/gripper_left_left_finger_link"
        gripper_prim = stage.GetPrimAtPath(gripper_path)
        obj_prim = obj.prim
        joint_path = gripper_path + "/fixed_joint"
        if stage.GetPrimAtPath(joint_path):
            stage.RemovePrim(joint_path)

    def attach_object(self, obj):
        stage = omni.usd.get_context().get_stage()

        gripper_path = "/World/envs/env_0/TiagoDualHolo/gripper_left_left_finger_link"
        gripper_prim = stage.GetPrimAtPath(gripper_path)
        obj_prim = obj.prim

        # Transform
        gripper_xf = UsdGeom.Xformable(gripper_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        obj_xf = UsdGeom.Xformable(obj_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        gripper_xf.Orthonormalize()
        obj_xf.Orthonormalize()
        relative_xf = obj_xf * gripper_xf.GetInverse()

        local_pos = relative_xf.ExtractTranslation()
        quat_d = relative_xf.ExtractRotationQuat()
        local_rot = Gf.Quatf(quat_d.GetReal(), Gf.Vec3f(quat_d.GetImaginary()))
        local_rot.Normalize()

        # Create joint
        joint_path = gripper_path + "/fixed_joint"
        if not stage.GetPrimAtPath(joint_path):
            fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
            fixed_joint.CreateBody0Rel().SetTargets([gripper_path])
            fixed_joint.CreateBody1Rel().SetTargets([obj_prim.GetPath()])

            fixed_joint.CreateLocalPos0Attr().Set(local_pos)
            fixed_joint.CreateLocalRot0Attr().Set(local_rot)
            fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
            fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

    def distance_gripper_object(self, obj):
        gripper_path = "/World/envs/env_0/TiagoDualHolo/gripper_left_left_finger_link"
        gripper_prim = omni.usd.get_context().get_stage().GetPrimAtPath(gripper_path)
        obj_prim = obj.prim
        gripper_xf = UsdGeom.Xformable(gripper_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        obj_xf = UsdGeom.Xformable(obj_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        gripper_xf.Orthonormalize()
        obj_xf.Orthonormalize()
        relative_xf = obj_xf * gripper_xf.GetInverse()
        local_pos = relative_xf.ExtractTranslation()
        # dis = np.linalg.norm(local_pos)
        return local_pos

    def cleanup(self) -> None:
        """Prepares torch buffers for RL data collection."""

        # prepare tensors
        # self.obs_buf = torch.zeros((self._num_envs, self.num_observations), device=self._device, dtype=torch.float)
        # self.states_buf = torch.zeros((self._num_envs, self.num_states), device=self._device, dtype=torch.float)
        # self.rew_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self._num_envs, device=self._device, dtype=torch.long
        )
        self.progress_buf = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.long
        )
        self.extras = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

    def set_up_scene(self, scene) -> None:
        """Clones environments based on value provided in task config and applies collision filters to mask
            collisions across environments.

        Args:
            scene (Scene): Scene to add objects to.
        """

        super().set_up_scene(scene)

        collision_filter_global_paths = list()
        if self._sim_config.task_config["sim"].get("add_ground_plane", True):
            self._ground_plane_path = "/World/defaultGroundPlane"
            collision_filter_global_paths.append(self._ground_plane_path)
            scene.add_ground_plane(
                prim_path=self._ground_plane_path, color=np.array([0.87, 0.72, 0.53])
            )
        prim_paths = self._cloner.generate_paths("/World/envs/env", self._num_envs)
        self._env_pos = self._cloner.clone(
            source_prim_path="/World/envs/env_0", prim_paths=prim_paths
        )
        self._env_pos = torch.tensor(
            np.array(self._env_pos), device=self._device, dtype=torch.float
        )
        self._cloner.filter_collisions(
            self._env._world.get_physics_context().prim_path,
            "/World/collisions",
            prim_paths,
            collision_filter_global_paths,
        )
        self.set_initial_camera_params(
            camera_position=[-1, -1, 4], camera_target=[2, 0, 0]
        )
        if self._sim_config.task_config["sim"].get("add_distant_light", True):
            create_distant_light()

        # set the cube with the robot

    #        stage = omni.usd.get_context().get_stage()
    #        cube_path = "/World/envs/env_0/TiagoDualHolo/Cube"
    #        cube_prim = stage.DefinePrim(cube_path, "Cube")
    #        UsdGeom.XformCommonAPI(cube_prim).SetTranslate((1, 1, 1))
    #        UsdGeom.XformCommonAPI(cube_prim).SetScale((0.01, 0.01, 0.01))

    def set_initial_camera_params(
        self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]
    ):
        # viewport = omni.kit.viewport_legacy.get_default_viewport_window()
        # viewport = get_active_viewport()

        # viewport.set_camera_position("/OmniverseKit_Persp", camera_position[0], camera_position[1], camera_position[2], True)
        # viewport.set_camera_target("/OmniverseKit_Persp", camera_target[0], camera_target[1], camera_target[2], True)

        stage = omni.usd.get_context().get_stage()
        camera_path = "/World/envs/env_0/TiagoDualHolo/head_1_link/Camera"
        camera_prim = stage.DefinePrim(camera_path, "Camera")
        self.camera_prim = camera_prim
        camera = UsdGeom.Camera(camera_prim)
        camera.GetFocalLengthAttr().Set(10)
        camera.GetClippingRangeAttr().Set((0.1, 30))
        UsdGeom.XformCommonAPI(camera_prim).SetTranslate((0.2, 0, 0.2))
        # UsdGeom.XformCommonAPI(camera_prim).SetTranslate((-0.9, 0, 0))
        rotation = Gf.Vec3f(60, 0, 270)
        UsdGeom.XformCommonAPI(camera_prim).SetRotate(rotation)
        camera_world_transform = UsdGeom.Xformable(
            camera_prim
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        camera_world_transform.Orthonormalize()
        self.camera_world_position = camera_world_transform.ExtractTranslation()
        self.camera_world_rotation = camera_world_transform.ExtractRotationMatrix()

        # np
        self.camera_world_position = np.array(
            [
                [self.camera_world_position[0]],
                [self.camera_world_position[1]],
                [self.camera_world_position[2]],
            ]
        )

        RESOLUTION = (1280, 720)
        # EDIT:
        # rep_camera = rep.create.camera(camera)
        self.camera_path = camera_path
        # render_product = rep.create.render_product(camera_path, RESOLUTION)

        # rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        # distance_to_image_plane = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        #
        # distance_to_image_plane.attach(render_product)
        # rgb.attach(render_product)
        #
        # rep.orchestrator.step()

        # self.depth_data = distance_to_image_plane
        # self.rgb_data = rgb
        # print("depth_data: ", self.depth_data)
        #
        self.ego_viewport = get_active_viewport()
        self.ego_viewport.camera_path = str(camera_prim.GetPath())
        vp_window = vp_utils.create_viewport_window("viewport")

    def get_depth_data(self):
        """Retrieves observations from the environment."""
        # depth_data =np.zeros((1280, 720)).astype(np.float32)
        render_product = rep.create.render_product(self.camera_path, (1280, 720))
        distance_to_image_plane = rep.AnnotatorRegistry.get_annotator(
            "distance_to_image_plane"
        )
        distance_to_image_plane.attach(render_product)
        depth_data = distance_to_image_plane.get_data()

        return depth_data

    def get_rgb_data(self):
        """Retrieves observations from the environment."""
        # rep.orchestrator.step()
        render_product = rep.create.render_product(self.camera_path, (1280, 720))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach(render_product)
        rgb_data = rgb.get_data()
        # rgb_data = np.zeros((1280, 720, 3)).astype(np.uint8)

        return rgb_data

        # self.ego_viewport.get_viewport_window().set_active_camera(str(camera_prim.GetPath()))
        # viewport.set_camera_position("/OmniverseKit_Persp", camera_position[0], camera_position[1], camera_position[2], True)
        # viewport.set_camera_target("/OmniverseKit_Persp", camera_target[0], camera_target[1], camera_target[2], True)
        # Near Clipping Plane

    def retrieve_camera_params(self):
        stage = omni.usd.get_context().get_stage()
        # extrinsic parameters
        width = 1280
        height = 720
        aspect_ratio = width / height
        # get camera prim attached to viewport
        # viewport_window = omni.kit.viewport_legacy.get_default_viewport_window()
        # viewport_window = get_active_viewport()
        camera = stage.GetPrimAtPath(self.ego_viewport.get_active_camera())

        focal_length = camera.GetAttribute("focalLength").Get()
        horiz_aperture = camera.GetAttribute("horizontalAperture").Get()

        vert_aperture = height / width * horiz_aperture
        fov = 2 * math.atan(horiz_aperture / (2 * focal_length))

        focal_y = height * focal_length / vert_aperture
        focal_x = width * focal_length / horiz_aperture
        center_y = height * 0.5
        center_x = width * 0.5
        # print the attributes of the camera
        # print("attributes of the camera: ", camera.GetAttributes())
        rotate_attr = camera.GetAttribute("xformOp:rotateXYZ").Get()
        translate_attr = camera.GetAttribute("xformOp:translate").Get()

        # make them to numpy array
        rotate = np.array([rotate_attr[0], rotate_attr[1], rotate_attr[2]])
        # print("rotation: ", rotate)
        # print("translate: ", translate_attr)
        R = np.array(
            [
                [1, 0, 0],
                [0, np.cos(np.radians(-10)), -np.sin(np.radians(-10))],
                [0, np.sin(np.radians(-10)), np.cos(np.radians(-10))],
            ]
        )

        # rotate y axis
        R = np.array(
            [
                [np.cos(np.radians(-30)), 0, np.sin(np.radians(-30))],
                [0, 1, 0],
                [-np.sin(np.radians(-30)), 0, np.cos(np.radians(-30))],
            ]
        )
        #  R = np.array([[1, 0, 0],
        #                  [0, 1, 0],
        #                  [0, 0, 1]])
        #        # rotate z axis
        #        R = np.array([[np.cos(np.radians(-10)), -np.sin(np.radians(-10)), 0],
        #                        [np.sin(np.radians(-10)), np.cos(np.radians(-10)), 0],
        #                        [0, 0, 1]])

        # R =  np.array([[1,0,0],[0,1,0],[0,0,1]])

        # rotation_x, rotation_y, rotation_z = -10,0,-90

        # rotation = Rotation.from_euler('xyz', [rotation_x, rotation_y, rotation_z], degrees=True)
        # rotation_matrix = rotation.as_matrix()
        # R = np.array([[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2]],
        #                  [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2]],
        #                  [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2]]])
        # T = np.array([[0], [0], [-0.9]])
        # R = self.camera_world_rotation

        # R = self.camera_world_rotation
        T = self.camera_world_position
        T = np.array([[T[0][0]], [T[1][0]], [T[2][0] + 0.25]])
        # T = np.zeros((3,1))
        # T[0][0] = 0
        # T[1][0] = 0
        # T[2][0] = 0.3
        ##print("R: ", R)
        # print("T: ", T)
        # print("self.camera_world_position: ", self.camera_world_position)
        # print("self.camera_world_rotation: ", self.camera_world_rotation)
        # make rotaion to angle axis
        # rotation = Rotation.from_matrix(self.camera_world_rotation)
        # rotation = rotation.as_euler('xyz', degrees=True)
        # print("rotation: ", rotation)

        return R, T, focal_x, focal_y, center_x, center_y

    @property
    def default_base_env_path(self):
        """Retrieves default path to the parent of all env prims.

        Returns:
            default_base_env_path(str): Defaults to "/World/envs".
        """
        return "/World/envs"

    @property
    def default_zero_env_path(self):
        """Retrieves default path to the first env prim (index 0).

        Returns:
            default_zero_env_path(str): Defaults to "/World/envs/env_0".
        """
        return f"{self.default_base_env_path}/env_0"

    @property
    def num_envs(self):
        """Retrieves number of environments for task.

        Returns:
            num_envs(int): Number of environments.
        """
        return self._num_envs

    #
    #    @property
    #    def num_actions(self):
    #        """ Retrieves dimension of actions.
    #
    #        Returns:
    #            num_actions(int): Dimension of actions.
    #        """
    #        return self._num_actions
    #
    #    @property
    #    def num_observations(self):
    #        """ Retrieves dimension of observations.
    #
    #        Returns:
    #            num_observations(int): Dimension of observations.
    #        """
    #        return self._num_observations
    #
    #    @property
    #    def num_states(self):
    #        """ Retrieves dimesion of states.
    #
    #        Returns:
    #            num_states(int): Dimension of states.
    #        """
    #        return self._num_states
    #
    #    @property
    #    def num_agents(self):
    #        """ Retrieves number of agents for multi-agent environments.
    #
    #        Returns:
    #            num_agents(int): Dimension of states.
    #        """
    #        return self._num_agents
    #
    #    def get_states(self):
    #        """ API for retrieving states buffer, used for asymmetric AC training.
    #
    #        Returns:
    #            states_buf(torch.Tensor): States buffer.
    #        """
    #        return self.states_buf
    #
    #    def get_extras(self):
    #        """ API for retrieving extras data for RL.
    #
    #        Returns:
    #            extras(dict): Dictionary containing extras data.
    #        """
    #        return self.extras
    #
    def reset(self):
        """Flags all environments for reset."""
        self.reset_buf = torch.ones_like(self.reset_buf)

    #
    #    def pre_physics_step(self, actions):
    #        """ Optionally implemented by individual task classes to process actions.
    #
    #        Args:
    #            actions (torch.Tensor): Actions generated by RL policy.
    #        """
    #        pass

    def post_physics_step(self):
        """Processes RL required computations for observations, states, rewards, resets, and extras.
            Also maintains progress buffer for tracking step count per environment.

        Returns:
            obs_buf(torch.Tensor): Tensor of observation data.
            rew_buf(torch.Tensor): Tensor of rewards data.
            reset_buf(torch.Tensor): Tensor of resets/dones data.
            extras(dict): Dictionary of extras data.
        """

        self.progress_buf[:] += 1

        if self._env._world.is_playing():
            rgb, depth, occupancy_2d_map, robot_pos = self.get_observations()
            self.calculate_metrics()
            self.is_done()

        return rgb, depth, occupancy_2d_map, robot_pos, self.reset_buf
