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


import numpy as np
import hydra
from omegaconf import DictConfig

from base_proposal.utils.hydra_cfg.hydra_utils import *
from base_proposal.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from base_proposal.utils.task_util import initialize_task
from base_proposal.envs.env import IsaacEnv

# from base_proposal.policy.test
from base_proposal.policy.test import Policy

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

    policy = Policy()
    # print(policy.get_action())
    env = IsaacEnv(headless=headless, render=render, sim_app_cfg_path=sim_app_cfg_path)
    task = initialize_task(cfg_dict, env)
    env.reset()
    env.reset()
    episode_reward = 0
    t = 0
    while env._simulation_app.is_running():
        if t % 100 == 0:
            print(f"Step {t}")
        env.wait()
        if t == 0:
            break

        t += 1

    env.step(["start"])
    env.step(["navigate", [2.5, -1]])
    env.step(["turn_to_goal"])
    env.step(["manipulate"])
    env.step(["check_success"])
    env.step(["return_arm"])
    env.step(["navigate", [1.5, -2]])
    env.step(["turn_to_goal"])
    env.step(["manipulate"])

    # living room
    #    env.step(["navigate", [1.4, -0.1]])
    #    env.step(["rotate", [np.pi / 2]])
    #    env.step(["rotate", [np.pi / 2]])
    #    env.step(["rotate", [np.pi / 2]])
    #    env.step(["rotate", [np.pi / 2]])
    #    env.step(["navigate", [1.5, 0]])
    #    env.step(["rotate", [-np.pi / 2]])
    #    env.step(["navigate", [2, 0]])
    #    env.step(["rotate", [-np.pi / 2]])
    #    env.step(["navigate", [1.5, 0]])
    #    env.step(["rotate", [np.pi / 2]])
    #    env.step(["rotate", [np.pi / 2]])
    #    env.step(["rotate", [np.pi / 2]])
    #    env.step(["rotate", [np.pi / 2]])
    #    env.step(["navigate", [1.5, 0]])
    #    env.step(["rotate", [np.pi / 2]])
    #    env.step(["rotate", [np.pi / 2]])
    #    env.step(["rotate", [np.pi / 2]])
    #    env.step(["rotate", [np.pi / 2]])

    # env.step(["rotate", [np.pi / 2]])
    # env.step(["navigate", [1, 0.0]])
    # env.step(["rotate", [-np.pi / 2]])
    # env.step(["navigate", [1.4, 0.0]])
    # env.step(["rotate", [np.pi / 2]])
    # env.step(["rotate", [np.pi / 2]])
    # env.step(["rotate", [np.pi / 2]])
    # env.step(["rotate", [np.pi / 2]])
    # env.step(["navigate", [2, 0]])
    # env.step(["rotate", [-np.pi / 2]])
    # env.step(["navigate", [3.5, 0]])
    # env.step(["rotate", [-np.pi / 2]])
    # env.step(["navigate", [2, 0]])
    # env.step(["rotate", [np.pi / 2]])
    # env.step(["rotate", [np.pi / 2]])
    # env.step(["rotate", [np.pi / 2]])
    # env.step(["rotate", [np.pi / 2]])
    # env.step(["navigate", [1.5, 0]])
    # env.step(["rotate", [np.pi / 2]])
    # env.step(["rotate", [np.pi / 2]])
    # env.step(["rotate", [np.pi / 2]])
    # env.step(["rotate", [np.pi / 2]])

    # times = 25
    # t = 0

    #  while env._simulation_app.is_running():
    #      action = policy.get_action()
    #      terminal= env.step(action)

    #      if(render):
    #          env.render()

    #      if(terminal):
    #          env.reset()
    #          t += 1
    #      if(t==times):
    #          break
    env._simulation_app.close()


if __name__ == "__main__":
    parse_hydra_configs()
