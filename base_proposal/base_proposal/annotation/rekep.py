import numpy as np
import cv2
from base_proposal.vlm.spaceAware_pivot import get_point
from base_proposal.tasks.utils import astar_utils
from base_proposal.affordance.get_affordance import get_affordance_point
from base_proposal.affordance.get_affordance import sample_from_mask_gaussian
from base_proposal.affordance.get_affordance import annotate_rgb
from base_proposal.affordance.get_affordance import get_affordance_direction_id


cell_size = 0.05
map_size = (203, 203)


def get_base(occupancy_2d_map, target, instruction, R, T, fx, fy, cx, cy, K=3):
    affordance_point, affordance_pixel = get_affordance_point(
        target, instruction, R, T, fx, fy, cx, cy, occupancy_2d_map
    )
    print(f"affordance_point: {affordance_point}")
    return (affordance_point[0], affordance_point[1])
