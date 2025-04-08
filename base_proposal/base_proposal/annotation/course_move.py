import numpy as np
import cv2
import random
from base_proposal.tasks.utils import astar_utils
import os
from PIL import Image
from base_proposal.vlm.course_nav import get_point


def process_image(occupancy_2d_map, destination):
    num_points = 18
    cell_size = 0.05

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
        x = destination[1] + 1.2 * np.cos(angle)
        y = destination[0] + 1.2 * np.sin(angle)
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
    crop_size = 500  # 半徑 100 pixel，總共 200x200
    x = 199 - (int(destination[0] / cell_size) + 100)
    y = 199 - (int(destination[1] / cell_size) + 100)
    x_min = max(0, x * scale - crop_size)
    x_max = min(occupancy_map.shape[1], x * 10 + crop_size)
    y_min = max(0, y * scale - crop_size)
    y_max = min(occupancy_map.shape[0], y * scale + crop_size)

    # 截取 200x200 的區域
    cropped_map = occupancy_map[x_min:x_max, y_min:y_max]
    cropped_map = cv2.resize(cropped_map, (1000, 1000))

    im = Image.fromarray(cropped_map)
    im.save("./data/cropped_occupancy_map.png")

    im = Image.fromarray(occupancy_map)
    im.save("./data/occupancy_2d_map.png")
    return candidate_points, counts


def get_rough_base(occupancy_2d_map, destination, instruction):
    candidate_points, counts = process_image(occupancy_2d_map, destination)

    print(f"counts: {counts}")
    des_idx = get_point(
        "./data/rgb.png",
        "./data/cropped_occupancy_map.png",
        instruction,
        counts,
    )

    print(f"des_idx: {des_idx}")
    return candidate_points[des_idx[0] - 1]
