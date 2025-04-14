# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import pinocchio as pin  # Optional: Needs to be imported before SimApp to avoid dependency issues
from omni.isaac.kit import SimulationApp

# from isaacsim import SimulationApp

import base_proposal
import numpy as np
import torch
import carb
import os

# from mushroom_rl.core import Environment, MDPInfo
# from mushroom_rl.utils.viewer import ImageViewer

RENDER_WIDTH = 1280  # 1600
RENDER_HEIGHT = 720  # 900
RENDER_DT = 1.0 / 60.0  # 60 Hz


class IsaacEnv:
    """This class provides a base interface for connecting RL policies with task implementations.
    APIs provided in this interface follow the interface in gym.Env and Mushroom Environemnt.
    This class also provides utilities for initializing simulation apps, creating the World,
    and registering a task.
    """

    def __init__(self, headless: bool, render: bool, sim_app_cfg_path: str) -> None:
        """Initializes RL and task parameters.

        Args:
            headless (bool): Whether to run training headless.
            render (bool): Whether to run simulation rendering (if rendering using the ImageViewer or saving the renders).
        """
        # Set isaac sim config file full path (if provided)

        if sim_app_cfg_path:
            sim_app_cfg_path = (
                os.path.dirname(base_proposal.__file__) + sim_app_cfg_path
            )
        self._simulation_app = SimulationApp(
            {
                #      "experience": sim_app_cfg_path,
                "headless": headless,
                "window_width": 1920,
                "window_height": 1080,
                "width": RENDER_WIDTH,
                "height": RENDER_HEIGHT,
            }
        )
        carb.settings.get_settings().set(
            "/persistent/omnihydra/useSceneGraphInstancing", True
        )
        self._run_sim_rendering = (
            (not headless) or render
        )  # tells the simulator to also perform rendering in addition to physics

        # Optional ImageViewer
        # self._viewer = ImageViewer([RENDER_WIDTH, RENDER_HEIGHT], RENDER_DT)
        self.sim_frame_count = 0
        self.action_step = 0

    def set_task(self, task, backend="torch", sim_params=None, init_sim=True) -> None:
        """Creates a World object and adds Task to World.
            Initializes and registers task to the environment interface.
            Triggers task start-up.

        Args:
            task (RLTask): The task to register to the env.
            backend (str): Backend to use for task. Can be "numpy" or "torch".
            sim_params (dict): Simulation parameters for physics settings. Defaults to None.
            init_sim (Optional[bool]): Automatically starts simulation. Defaults to True.
        """

        from omni.isaac.core.world import World

        self._device = "cpu"
        if sim_params and "use_gpu_pipeline" in sim_params:
            if sim_params["use_gpu_pipeline"]:
                self._device = sim_params["sim_device"]

        self._world = World(
            stage_units_in_meters=1.0,
            rendering_dt=RENDER_DT,
            backend=backend,
            sim_params=sim_params,
            device=self._device,
        )
        self._world.add_task(task)
        self._task = task
        self._num_envs = self._task.num_envs
        # assert (self._num_envs == 1), "Mushroom Env cannot currently handle running multiple environments in parallel! Set num_envs to 1"

        # self.observation_space = self._task.observation_space
        # self.action_space = self._task.action_space
        # self.num_states = self._task.num_states # Optional
        # self.state_space = self._task.state_space # Optional
        # gamma = self._task._gamma
        # horizon = self._task._max_episode_length

        # Create MDP info for mushroom

        if init_sim:
            self._world.reset()

    # def render(self) -> None:
    #     """Step the simulation renderer and display task render in ImageViewer."""

    #     self._world.render()
    #     # Get render from task
    #     task_render = self._task.get_render()
    #     # Display
    #     self._viewer.display(task_render)
    #     return

    def get_render(self):
        """Step the simulation renderer and return the render as per the task."""

        self._world.render()
        return self._task.get_render()

    def close(self) -> None:
        """Closes simulation."""

        self._simulation_app.close()
        return

    def seed(self, seed=-1):
        """Sets a seed. Pass in -1 for a random seed.

        Args:
            seed (int): Seed to set. Defaults to -1.
        Returns:
            seed (int): Seed that was set.
        """

        from omni.isaac.core.utils.torch.maths import set_seed

        return set_seed(seed)

    def wait(self):
        for _ in range(self._task.control_frequency_inv):
            self._world.step(render=self._run_sim_rendering)
            self.sim_frame_count += 1

    def render(self):
        for _ in range(self._task.control_frequency_inv):
            self._world.step(render=self._run_sim_rendering)
            self.sim_frame_count += 1

    def retrieve_camera_params(self):
        return self._task.retrieve_camera_params()

    def get_instruction(self):
        return self._task.get_instruction()

    def get_success_num(self):
        return self._task.get_success_num()

    def set_new_target_pose(self):
        return self._task.set_new_target_pose()

    def get_destination(self):
        return self._task.get_destination()

    def step(self, action):
        """Basic implementation for stepping simulation.
            Can be overriden by inherited Env classes
            to satisfy requirements of specific RL libraries. This method passes actions to task
            for processing, steps simulation, and computes observations, rewards, and resets.

        Args:
            action (numpy.ndarray): Action from policy.
        Returns:
            observation(numpy.ndarray): observation data.
            reward(numpy.ndarray): rewards data.
            done(numpy.ndarray): reset/done data.
            info(dict): Dictionary of extra data.
        """

        if action[0] == "start":
            self._task.pre_physics_step("start")
            self.render()

        if action[0] == "set_initial_base":
            self._task.pre_physics_step("set_initial_base")
            self.render()

        if action[0] == "turn_to_goal":
            print("Turning to goal")
            self._task.set_goal(action[1])
            self.render()
            self._task.pre_physics_step("turn_to_goal")
            self.render()

        if action[0] == "turn_to_se3":
            print("Turning to SE3")
            self._task.pre_physics_step("turn_to_se3")
            self.render()

        if "navigate" in action[0]:
            print("Navigating")
            position = np.array(action[1:])
            # self._task.set_path(position)
            goal = position[0][0:2]
            if action[0] == "navigateReach_astar":
                positions = self._task.set_path(goal, algo="astar", reached=True)
            elif action[0] == "navigateNear_rrt":
                positions = self._task.set_path(goal, algo="rrt", reached=False)
            elif action[0] == "navigateNear_astar":
                positions = self._task.set_path(goal, algo="astar", reached=False)
            elif action[0] == "navigateNear_astar_rough":
                positions = self._task.set_path(goal, algo="astar_rough", reached=False)
            elif action[0] == "navigateNear_rrt_rough":
                positions = self._task.set_path(goal, algo="rrt_rough", reached=False)

            for position in positions:
                task_actions = "get_base"
                self._task.pre_physics_step(task_actions)
                self.render()

                x_scaled = position[0]
                y_scaled = position[1]
                x_scaled = torch.tensor(
                    x_scaled, dtype=torch.float, device=self._device
                )
                y_scaled = torch.tensor(
                    y_scaled, dtype=torch.float, device=self._device
                )
                distance = torch.sqrt(x_scaled**2 + y_scaled**2)
                phi = torch.atan2(y_scaled, x_scaled)
                theta = torch.atan2(y_scaled, x_scaled)
                angle = phi
                theta_scaled = theta

                if angle < 0:
                    self.left_rotation(angle)
                else:
                    self.right_rotation(angle)
                self.forward(distance)
                self.render()

                theta_scaled = theta_scaled - angle

                if theta_scaled < 0:
                    self.left_rotation(theta_scaled)
                else:
                    self.right_rotation(theta_scaled)
                self.render()

                self._task.pre_physics_step("set_base")
                self.render()

        if action[0] == "rotate":
            self._task.set_angle(action[1][0])
            self._task.pre_physics_step("set_angle")
            self.render()

            angle = action[1][0]
            if angle < 0:
                self.left_rotation(angle)
            else:
                self.right_rotation(angle)
            self.render()

            self._task.pre_physics_step("set_base")
            self.render()

        if action[0] == "move_ee":
            self._task.pre_physics_step("move_ee")
            for i in range(100):
                self._task.pre_physics_step("wait")
                self.render()

            # self._task.pre_physics_step("get_point_cloud")
            # self.render()
            # motion_num = self._task.get_motion_num()
            # for i in range(motion_num):
            #    self._task.pre_physics_step("move_arm")
            #    self.render()
        if action[0] == "close_gripper":
            self.close_gripper()
            self.render()

        if action[0] == "lift":
            self.lift_object()
            self.render()

        if action[0] == "backward":
            print("Moving backward")
            self.backward()
            self.render()
            self._task.pre_physics_step("base_stop")
            self.render()
        if action[0] == "open_gripper":
            self._task.pre_physics_step("detach_object")
            self.render()
            self._task.pre_physics_step("open_gripper")
            self.render()

        if action[0] == "attach_object":
            self._task.pre_physics_step("attach_object")
            self.render()

        if action[0] == "lift_arm":
            # pass
            self._task.pre_physics_step("lift_arm")
            self.render()

        if action[0] == "reset_arm":
            self._task.pre_physics_step("reset_arm")
            self.render()

        if action[0] == "check_lift_success":
            self._task.pre_physics_step("check_lift_success")
            self.render()
        if action[0] == "check_place_success":
            self._task.pre_physics_step("check_place_success")
            self.render()
        if action[0] == "check_pull_success":
            self._task.pre_physics_step("check_pull_success")
            self.render()

        self.render()
        rgb, depth, occupancy_2d_map, robot_pos, resets = (
            self._task.post_physics_step()
        )  # buffers of obs, reward, dones and infos. Need to be squeezed
        # print("Resets: ",resets)

        done = resets[0].cpu().item()

        return rgb, depth, occupancy_2d_map, robot_pos, done

    def reset(self, state=None):
        """Resets the task and updates observations."""
        self._task.reset()
        self._world.step(render=self._run_sim_rendering)
        i = 0
        while i < 100:
            if i % 100 == 0:
                print("Waiting for reset")
            self.wait()
            i += 1
        self._task.get_observations()
        # try:
        #     self._task.get_observations()
        # except:
        #     print("Error in getting observations")
        #     pass
        self.move_gripper = False
        self.action_step = 0

    def stop(self):
        pass

    def shutdown(self):
        pass

    @property
    def num_envs(self):
        """Retrieves number of environments.

        Returns:
            num_envs(int): Number of environments.
        """
        return self._num_envs

    def left_rotation(self, angle, velocity=1):
        actions = np.zeros(5)
        velocity = self._task.max_rot_vel
        # angle = angle / (self._task._dt) * 3.14159265
        angle = angle / (self._task._dt)
        actions[:] = 2
        actions = torch.unsqueeze(
            torch.tensor(actions, dtype=torch.float, device=self._device), dim=0
        )
        actions = "left_rotate"
        if angle < 0:
            while self._simulation_app.is_running():
                angle += velocity
                if angle >= 0:
                    return True
                self._task.pre_physics_step(actions)
                self.render()

    def right_rotation(self, angle, velocity=1):
        actions = np.zeros(5)
        velocity = self._task.max_rot_vel
        # angle = angle / (self._task._dt) * 3.14159265
        angle = angle / (self._task._dt)
        actions[:] = 1
        actions = torch.unsqueeze(
            torch.tensor(actions, dtype=torch.float, device=self._device), dim=0
        )
        actions = "right_rotate"
        if angle > 0:
            while self._simulation_app.is_running():
                angle -= velocity
                if angle <= 0:
                    return True
                self._task.pre_physics_step(actions)
                self.render()

    def forward(self, distance, velocity=1):
        actions = np.zeros(5)
        velocity = self._task.max_base_xy_vel
        # distance = distance / (self._task._dt) * 4
        distance = distance / (self._task._dt)
        actions[:] = 0
        actions = torch.unsqueeze(
            torch.tensor(actions, dtype=torch.float, device=self._device), dim=0
        )
        actions = "forward"
        if distance > 0:
            while self._simulation_app.is_running():
                distance -= velocity
                if distance <= 0:
                    return True
                self._task.pre_physics_step(actions)
                self.render()

    def close_gripper(self):
        actions = "close_gripper"
        v = 0
        while self._simulation_app.is_running():
            if v >= 1:
                print("Gripper closed")
                break
            v += 0.01
            self._task.pre_physics_step(actions)
            self.render()

    def lift_object(self):
        actions = "lift_object"
        v = 0
        while self._simulation_app.is_running():
            if v >= 1:
                break
            v += 0.005
            self._task.pre_physics_step(actions)
            self.render()

    def backward(self):
        actions = "backward"
        v = 0
        while self._simulation_app.is_running():
            if v >= 1:
                break
            v += 0.005
            self._task.pre_physics_step(actions)
            self.render()
