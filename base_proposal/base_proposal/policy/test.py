import torch
import numpy as np
from base_proposal.tasks.utils import astar_utils
from base_proposal.tasks.utils import rrt_utils
from base_proposal.tasks.utils import get_features
from base_proposal.vlm.get_target import identify_object_in_image
from base_proposal.vlm.get_part import determine_part_to_grab
from base_proposal.vlm.get_answer import confirm_part_in_image
from base_proposal.vlm.get_base import determine_base
from base_proposal.vlm.get_affordance import determine_affordance
from base_proposal.annotation.annotation import get_base
import matplotlib.pyplot as plt
import cv2
import os
import time
import math

# from base_proposal.tasks.utils import scene_utils

# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# from segment_anything import SamPredictor, sam_model_registry
from scipy.spatial.transform import Rotation
from PIL import Image


class Policy:
    def __init__(self):
        # self.rgb = cv2.imread("base_proposal/base_proposal/annotation/rgb.png")
        # self.depth = cv2.imread("base_proposal/base_proposal/annotation/depth.png", cv2.IMREAD_ANYDEPTH)
        # self.occupancy = cv2.imread("base_proposal/base_proposal/annotation/occupancy.png", cv2.IMREAD_ANYDEPTH)
        self.rgb = None
        self.depth = None
        self.occupancy = None
        self.destination = None
        self.des_idx = 0
        self.des_finish_idx = 0

    def get_camera_params(self, R, T, fx, fy, cx, cy):
        self.R = R
        self.T = T
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        return

    def set_destination(self, destination):
        self.destination = destination

    def get_observation(self, rgb, depth, occupancy_2d_map, robot_pos):
        self.depth = depth
        self.rgb = rgb
        self.occupancy = occupancy_2d_map
        self.robot_pos = robot_pos

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

    def get_action(self):
        if self.des_idx >= len(self.destination):
            return ["finish", []]
        if self.des_idx == self.des_finish_idx:
            self.des_finish_idx += 1
            destination = self.global2local(self.destination[self.des_idx])
            return ["navigate", [destination[0], destination[1]]]
        if self.visibility() and self.manipulate():
            self.des_idx += 1
            return ["manipulate"]

    # im = Image.fromarray(rgb)
    # im.save("./data/rgb1.png")

    # im = Image.fromarray(occupancy_2d_map)
    # im.save("./data/occupancy_2d_map1.png")


# def get_action(self):
#     return ["navigate", [1.3, -0.1, 0]]
