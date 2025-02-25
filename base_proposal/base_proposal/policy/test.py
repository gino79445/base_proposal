
import torch
import numpy as np
from base_proposal.tasks.utils.pinoc_utils import PinTiagoIKSolver
from base_proposal.tasks.utils.motion_planner import MotionPlannerTiago
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

#from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
#from segment_anything import SamPredictor, sam_model_registry
from scipy.spatial.transform import Rotation
from PIL import Image


class Policy:
    def __init__(self):
        #self.rgb = cv2.imread("base_proposal/base_proposal/annotation/rgb.png")
        #self.depth = cv2.imread("base_proposal/base_proposal/annotation/depth.png", cv2.IMREAD_ANYDEPTH)
        #self.occupancy = cv2.imread("base_proposal/base_proposal/annotation/occupancy.png", cv2.IMREAD_ANYDEPTH)
        self.rgb = None

    def get_action(self):
        return ["navigate", [1.3, -0.1, 0]]
