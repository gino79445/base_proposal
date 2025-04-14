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
import hydra
from omegaconf import DictConfig

from base_proposal.utils.hydra_cfg.hydra_utils import *
from base_proposal.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from base_proposal.utils.task_util import initialize_task
from base_proposal.envs.env import IsaacEnv

# from base_proposal.policy.test
from base_proposal.policy.pivot import Policy as pivot_Policy
from base_proposal.policy.spaceAware_pivot import Policy as spaceAware_Policy

from dotenv import load_dotenv
import os

load_dotenv()  # 讀取 .env 檔案
api_key = os.getenv("API_KEY")  # 獲取 API Key


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    headless = cfg.headless
    render = cfg.render
    sim_app_cfg_path = cfg.sim_app_cfg_path

    env = IsaacEnv(headless=headless, render=render, sim_app_cfg_path=sim_app_cfg_path)
    task = initialize_task(cfg_dict, env)
    t = 0
    while env._simulation_app.is_running():
        if t % 100 == 0:
            env.reset()
            print(f"Step {t}")
        env.wait()
        if t == 200:
            break

        t += 1

    # env.step(["start"])
    # env.step(["set_initial_base", [0.7, -1]])
    # env.step(["rotate", [np.pi * 100]])
    # env.step(["rotate", [np.pi / 2]])
    # env.step(["rotate", [np.pi / 2]])
    # env.step(["rotate", [np.pi / 2]])

    time = 20
    current_time = 0

    while current_time < time:
        torch.cuda.empty_cache()

        current_time += 1
        global_position = [[1.47, 0.85], [2, -4.4]]
        # global_position = [[-0.2, -1.3], [0.15, 1.54]]
        global_position = [[1.85, -2.4], [1.64, -1]]
        # global_position = [[-0.2, -1.3], [0.15, 1.43]]
        # global_position = [[-0.58, -4.4]]
        instruction = env.get_instruction()

        policy = spaceAware_Policy(instruction)
        # policy = pivot_Policy(instruction)
        try:
            destinations = env.get_destination()
            global_position = [destination[0] for destination in destinations]
            # pick_and_place(
            #     env, policy, global_position, local_nav="pivot", algo="rrt_rough"
            # )
            # pick_and_place(env, policy, global_position, local_nav="None", algo="rrt")
            pick_and_place(
                env,
                policy,
                global_position,
                local_nav="spaceAware_pivot",
                algo="rrt_rough",
            )

        # pull(env, policy, global_position, local_nav="None", algo="astar")
        #   pull(env, policy, global_position, local_nav="pivot", algo="rrt_rough")
        # pull(
        #    env,
        #    policy,
        #    global_position,
        #    local_nav="spaceAware_pivot",
        #    algo="rrt_rough",
        # )
        except Exception as e:
            print(e)
            print("Error")

        env.step(["open_gripper"])
        env.step(["reset_arm"])
        success_num = env.get_success_num()
        env.set_new_target_pose()
        env.reset()
        policy.reset()
        # success_num success rate
        print(f"Success rate: {success_num}/{current_time}")

    env._simulation_app.close()


def pull(env, policy, global_position, local_nav="none", algo="astar"):
    policy.set_destination(global_position)
    R, T, fx, fy, cx, cy = env.retrieve_camera_params()
    policy.get_camera_params(R, T, fx, fy, cx, cy)
    rgb, depth, occupancy_2d_map, robot_pos, _ = env.step(["start"])
    rgb, depth, occupancy_2d_map, robot_pos, _ = env.step(["set_initial_base"])
    for global_pos in global_position:
        policy.get_observation(rgb, depth, occupancy_2d_map, robot_pos)
        pos = policy.global2local(global_pos)
        rgb, depth, occupancy_2d_map, robot_pos, terminal = env.step(
            [f"navigateNear_{algo}", pos]
        )

        ###   local navigate ###
        if local_nav == "pivot":
            policy.get_observation(rgb, depth, occupancy_2d_map, robot_pos)
            pos = policy.global2local(global_pos)
            rgb, depth, occupancy_2d_map, robot_pos, terminal = env.step(
                ["turn_to_goal", pos]
            )
            policy.get_observation(rgb, depth, occupancy_2d_map, robot_pos)
            action = policy.get_action()
            rgb, depth, occupancy_2d_map, robot_pos, terminal = env.step(action)
            policy.get_observation(rgb, depth, occupancy_2d_map, robot_pos)

        if local_nav == "spaceAware_pivot":
            # policy.get_observation(rgb, depth, occupancy_2d_map, robot_pos)
            # rough_action = policy.get_rough_action()
            # rgb, depth, occupancy_2d_map, robot_pos, terminal = env.step(rough_action)

            policy.get_observation(rgb, depth, occupancy_2d_map, robot_pos)
            pos = policy.global2local(global_pos)
            rgb, depth, occupancy_2d_map, robot_pos, terminal = env.step(
                ["turn_to_goal", pos]
            )

            policy.get_observation(rgb, depth, occupancy_2d_map, robot_pos)
            action = policy.get_action()
            rgb, depth, occupancy_2d_map, robot_pos, terminal = env.step(action)

            policy.get_observation(rgb, depth, occupancy_2d_map, robot_pos)

        rgb, depth, occupancy_2d_map, robot_pos, terminal = env.step(["turn_to_se3"])
        ###   local navigate ###
        env.step(["move_ee"])
        env.step(["close_gripper"])
        env.step(["backward"])
        env.step(["check_pull_success"])


def pick_and_place(env, policy, global_position, local_nav="none", algo="astar"):
    policy.set_destination(global_position)
    R, T, fx, fy, cx, cy = env.retrieve_camera_params()
    policy.get_camera_params(R, T, fx, fy, cx, cy)
    rgb, depth, occupancy_2d_map, robot_pos, _ = env.step(["start"])
    rgb, depth, occupancy_2d_map, robot_pos, _ = env.step(["set_initial_base"])
    action_step = 0
    for global_pos in global_position:
        policy.get_observation(rgb, depth, occupancy_2d_map, robot_pos)

        pos = policy.global2local(global_pos)
        rgb, depth, occupancy_2d_map, robot_pos, terminal = env.step(
            [f"navigateNear_{algo}", pos]
        )

        ###   local navigate ###
        if local_nav == "pivot":
            policy.get_observation(rgb, depth, occupancy_2d_map, robot_pos)
            pos = policy.global2local(global_pos)
            rgb, depth, occupancy_2d_map, robot_pos, terminal = env.step(
                ["turn_to_goal", pos]
            )
            policy.get_observation(rgb, depth, occupancy_2d_map, robot_pos)
            action = policy.get_action()
            rgb, depth, occupancy_2d_map, robot_pos, terminal = env.step(action)

        if local_nav == "spaceAware_pivot":
            # policy.get_observation(rgb, depth, occupancy_2d_map, robot_pos)
            # rough_action = policy.get_rough_action()
            # rgb, depth, occupancy_2d_map, robot_pos, terminal = env.step(rough_action)
            policy.get_observation(rgb, depth, occupancy_2d_map, robot_pos)
            pos = policy.global2local(global_pos)
            rgb, depth, occupancy_2d_map, robot_pos, terminal = env.step(
                ["turn_to_goal", pos]
            )
            policy.get_observation(rgb, depth, occupancy_2d_map, robot_pos)
            action = policy.get_action()
            rgb, depth, occupancy_2d_map, robot_pos, terminal = env.step(action)
        ###   local navigate ###

        rgb, depth, occupancy_2d_map, robot_pos, terminal = env.step(["turn_to_se3"])
        if action_step == 0:
            env.step(["move_ee"])
            env.step(["close_gripper"])
            env.step(["lift"])
            rgb, depth, occupancy_2d_map, robot_pos, terminal = env.step(
                ["check_lift_success"]
            )
            if not terminal:
                env.step(["attach_object"])
                rgb, depth, occupancy_2d_map, robot_pos, _ = env.step(["lift_arm"])
                action_step += 1
        else:
            env.step(["move_ee"])
            env.step(["open_gripper"])
            env.step(["reset_arm"])
            env.step(["rotate", [np.pi]])
            env.step(["check_place_success"])
            terminal = True
        if terminal:
            break


if __name__ == "__main__":
    parse_hydra_configs()
