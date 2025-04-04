import torch
import numpy as np
from base_proposal.tasks.utils import astar_utils
from base_proposal.tasks.utils import rrt_utils
from base_proposal.tasks.utils import get_features
from base_proposal.vlm.get_target import identify_object_in_image
from base_proposal.vlm.get_part import determine_part_to_grab
from base_proposal.vlm.get_answer import confirm_part_in_image
from base_proposal.vlm.get_affordance import determine_affordance
from base_proposal.vlm.get_pos import determine_base
from base_proposal.annotation.annotation import get_base
from base_proposal.vlm.parse_instruction import parse_instruction
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
from base_proposal.annotation.pivot import get_base


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
        self.instruction = []
        print(f"Instruction: {instruction}")
        instruction = parse_instruction(instruction)
        for i in range(len(instruction)):
            if i % 2 == 0:
                self.position.append(instruction[i])
            else:
                self.instruction.append(instruction[i])
        print(f"Position: {self.position}")
        print(f"Instruction: {self.instruction}")

    def set_destination(self, destination):
        self.destination = destination

    def process_image(self, occupancy_2d_map):
        num_points = 18
        cell_size = 0.05
        destination = self.global2local(self.destination[0])
        count = 0
        counts = []
        candidate_points = []
        cell_size = 0.05
        scale = 10
        occupancy_map = occupancy_2d_map.copy()
        occupancy_map = np.flipud(occupancy_map)
        occupancy_map = np.rot90(occupancy_map)

        #
        # make occupancy map rgb
        occupancy_map = cv2.cvtColor(occupancy_map, cv2.COLOR_GRAY2BGR)
        # size 200 x 200 to 1000 x 1000
        occupancy_map = cv2.resize(occupancy_map, (200 * scale, 200 * scale))
        occupancy_map = cv2.circle(
            occupancy_map,
            (
                (199 - (int(destination[1] / cell_size) + 100)) * scale,
                (199 - (int(destination[0] / cell_size) + 100)) * scale,
            ),
            5,
            (0, 255, 0),
            -1,
        )
        candidate_points = []
        for i in range(num_points):
            angle = 2 * np.pi / num_points * i
            x = destination[1] + 0.75 * np.cos(angle)
            y = destination[0] + 0.75 * np.sin(angle)
            original_x = y
            original_y = x
            x = int(x / cell_size) + 100
            y = int(y / cell_size) + 100
            count += 1

            if not astar_utils.is_valid_des(x, y, occupancy_2d_map):
                candidate_points.append(None)
                continue
            x, y = 199 - y, 199 - x
            cv2.circle(occupancy_map, (y * scale, x * scale), 20, (255, 255, 255), -1)
            cv2.circle(occupancy_map, (y * scale, x * scale), 20, (0, 0, 255), 1)
            text_width, text_height = cv2.getTextSize(
                f"{count}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )[0]
            cv2.putText(
                occupancy_map,
                f"{count}",
                (y * scale - text_width // 2, x * scale + text_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
            counts.append(count)
            candidate_points.append((original_x, original_y))

        # print(candidate_points)
        # print(counts)
        occupancy_map = cv2.circle(
            occupancy_map, (100 * scale, 100 * scale), 20, (0, 0, 255), -1
        )
        crop_size = 300  # 半徑 100 pixel，總共 200x200
        x = 199 - (int(destination[0] / cell_size) + 100)
        y = 199 - (int(destination[1] / cell_size) + 100)
        x_min = max(0, x * scale - crop_size)
        x_max = min(occupancy_map.shape[1], x * 10 + crop_size)
        y_min = max(0, y * scale - crop_size)
        y_max = min(occupancy_map.shape[0], y * scale + crop_size)

        # 截取 200x200 的區域
        cropped_map = occupancy_map[x_min:x_max, y_min:y_max]
        cropped_map = cv2.resize(cropped_map, (1000, 1000))

        # 儲存裁剪後的影像
        im = Image.fromarray(cropped_map)
        im.save("./data/cropped_occupancy_map.png")

        im = Image.fromarray(occupancy_map)
        im.save("./data/occupancy_2d_map.png")

        return candidate_points, counts

    def get_observation(self, rgb, depth, occupancy_2d_map, robot_pos):
        self.depth = depth
        self.rgb = rgb
        self.occupancy = occupancy_2d_map
        self.robot_pos = robot_pos
        self.candidate_points, self.counts = self.process_image(occupancy_2d_map)

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
        # make rgb as bgr
        rgb = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        base_point = get_base(
            rgb,
            self.instruction[self.des_idx],
            self.depth,
            self.occupancy,
            self.R,
            self.T,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
        )
        print(f"Base point: {base_point}")
        self.des_idx += 1
        return ["navigateReach_astar", [base_point[0], base_point[1]]]

        # des_idx = determine_base(
        #    "./data/rgb.png",
        #    "./data/cropped_occupancy_map.png",
        #    "the handle of the mug",
        #    self.counts,
        # )

        # print(des_idx)
        # return ["navigateReach_astar", self.candidate_points[des_idx - 1]]
