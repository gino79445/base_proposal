import torch
import numpy as np
from base_proposal.tasks.utils import astar_utils
from base_proposal.vlm.parse_instruction import parse_instruction
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from base_proposal.annotation.course_move import get_rough_base
from base_proposal.affordance.get_affordance import get_affordance_point
from base_proposal.affordance.get_affordance import get_rough_affann


class Policy:
    def __init__(self, instruction, semantic_map=None):
        # self.rgb = cv2.imread("base_proposal/base_proposal/annotation/rgb.png")
        # self.depth = cv2.imread("base_proposal/base_proposal/annotation/depth.png", cv2.IMREAD_ANYDEPTH)
        # self.occupancy = cv2.imread("base_proposal/base_proposal/annotation/occupancy.png", cv2.IMREAD_ANYDEPTH)
        self.rgb = None
        self.depth = None
        self.occupancy = None
        self.destination = None
        self.des_idx = 0
        self.set_position_instruction(instruction)

    def get_camera_params(self, R, T, fx, fy, cx, cy):
        self.R = R
        self.T = T
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        return

    def set_position_instruction(self, instruction):
        # make the ellement with even index to be the position
        self.position = []
        self.target = []
        self.instruction = []
        print(f"Instruction: {instruction}")
        instruction = parse_instruction(instruction)
        for i in range(len(instruction)):
            if i % 2 == 0:
                self.position.append(instruction[i])
                self.target.append(instruction[i])
            else:
                self.instruction.append(instruction[i])
        print(f"Position: {self.position}")
        print(f"Instruction: {self.instruction}")

    def set_destination(self, destination):
        self.destination = destination

    def get_observation(self, rgb, depth, occupancy_2d_map, robot_pos):
        self.depth = depth
        self.rgb = rgb
        self.occupancy = occupancy_2d_map
        self.robot_pos = robot_pos

        self.get_map()

    def get_map(self):
        occupancy_map = self.occupancy.copy()
        destinations = self.destination

        # colorize the occupancy map
        occupancy_map = cv2.cvtColor(occupancy_map, cv2.COLOR_GRAY2BGR)
        cell_size = 0.05
        for i in range(len(destinations)):
            destination = self.global2local(destinations[i])
            destination = (
                int(destination[1] / cell_size) + 100,
                int(destination[0] / cell_size) + 100,
            )
            occupancy_map = cv2.circle(
                occupancy_map, (destination[1], destination[0]), 1, (0, 255, 0), -1
            )
        # resize the occupancy map to 200x200
        occupancy_map = cv2.resize(occupancy_map, (2000, 2000))
        im = Image.fromarray(occupancy_map)
        im.save("./data/target_2dmap.png")
        return occupancy_map

    def global2local(self, global_point):
        cell_size = 0.05
        end = (
            int(global_point[1] / cell_size) + 100,
            int(global_point[0] / cell_size) + 100,
        )

        robot_pos = self.robot_pos
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
        end = ((end[1] - 100) * cell_size, (end[0] - 100) * cell_size)
        return end

    def visibility(self):
        return True

    def manipulate(self):
        return True

    def reset(self):
        self.rgb = None
        self.depth = None
        self.occupancy = None
        self.destination = None
        self.des_idx = 0
        self.position = []
        self.instruction = []

    def get_rough_action(self):
        get_rough_affann(
            self.target[self.des_idx],
            self.instruction[self.des_idx],
            self.R,
            self.T,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.occupancy,
        )

        affann = cv2.imread("./data/rough_affann.png")

        rough_base = get_rough_base(
            self.occupancy,
            self.global2local(self.destination[self.des_idx]),
            self.instruction[self.des_idx],
            affann,
        )
        return ["navigateReach_astar", [rough_base[0], rough_base[1]]]

    def get_action(self):
        base_point, affordance_pixel = get_affordance_point(
            self.target[self.des_idx],
            self.instruction[self.des_idx],
            self.R,
            self.T,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.occupancy,
        )
        # base_point = get_base(
        #     self.occupancy,
        #     self.target[self.des_idx],
        #     self.instruction[self.des_idx],
        #     self.R,
        #     self.T,
        #     self.fx,
        #     self.fy,
        #     self.cx,
        #     self.cy,
        # )
        print(f"Base point: {base_point}")
        self.des_idx += 1
        return ["navigateNear_astar", [base_point[0], base_point[1]]]
