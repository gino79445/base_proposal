import torch
import cv2
import numpy as np
from torch.nn.functional import interpolate
import torch.hub
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from base_proposal.affordance.groundedSAM import detect_and_segment

from base_proposal.vlm.get_affordance import determine_affordance
import time


cell_size = 0.05  # meters
map_size = (203, 203)


class KeypointProposer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config["device"])
        # Load DINO-v2 model
        self.dinov2 = (
            torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            .eval()
            .to(self.device)
        )

    def _get_features(self, transformed_rgb, shape_info):
        patch_h = shape_info["patch_h"]
        patch_w = shape_info["patch_w"]
        img_tensors = (
            torch.from_numpy(transformed_rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        assert img_tensors.shape[1] == 3, (
            "Unexpected image shape, expected 3 color channels."
        )
        features_dict = self.dinov2.forward_features(img_tensors)
        raw_feature_grid = features_dict["x_norm_patchtokens"].reshape(
            1, patch_h, patch_w, -1
        )
        interpolated_feature_grid = (
            interpolate(
                raw_feature_grid.permute(0, 3, 1, 2), size=(720, 1280), mode="bilinear"
            )
            .permute(0, 2, 3, 1)
            .squeeze(0)
        )
        features_flat = interpolated_feature_grid.reshape(
            -1, interpolated_feature_grid.shape[-1]
        )
        return features_flat


def prepare_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    return image


def get_3d_point(u, v, Z, R, T, fx, fy, cx, cy):
    # Retrieve camera parameters

    # Convert pixel coordinates to normalized camera coordinates
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # Convert to camera coordinates (as column vector)
    point = np.array([[Z], [X], [Y]])
    # Apply the inverse transformation (R^-1 and translation)
    R_inv = np.linalg.inv(R)
    point_3d = R_inv @ point + T
    return point_3d


def get_pixel(point, R, T, fx, fy, cx, cy):
    point = R @ (point - T)
    X = point[1]
    Y = point[2]
    Z = point[0]

    u = (fx * X) / Z + cx
    v = (fy * Y) / Z + cy
    i = int(u)
    j = int(v)
    return i, j


def rotate_vector_z(vec, angle_deg):
    """Rotate a 3D vector around Z axis by given degree."""
    angle_rad = np.deg2rad(angle_deg)
    Rz = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1],
        ]
    )
    return Rz @ vec


def get_annotated_rgb1(rgb, goal, R, T, fx, fy, cx, cy, obstacle_map):
    annotated_image = rgb.copy()
    overlay = rgb.copy()
    origin_rgb = rgb.copy()
    overlay2 = rgb.copy()

    if isinstance(goal, tuple) or isinstance(goal, list):
        goal = np.array([[goal[0][0]], [goal[1][0]], [goal[2][0]]])
        print("goal:", goal)

    origin = np.array([[goal[0][0]], [goal[1][0]], [0.0]])  # shape: (3, 1)

    scale = 3
    step = 0.01
    import colorsys

    directions = list(range(0, 360, 30))  # 12 條箭頭
    color_map = {}
    hue_order = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]

    for i, angle in enumerate(directions):
        hue = i / len(directions)  # 分布在 HSV 色環上（0~1）
        hue = hue_order[i] / len(directions)  # 分布在 HSV 色環上（0~1）
        if 0.05 < hue < 0.12:
            hue -= 0.04
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        color_map[angle] = (int(b * 255), int(g * 255), int(r * 255))  # BGR
    depth = np.load("./data/depth.npy")

    annotated_angles = set()  # 記錄已經編號過的方向
    label = 0

    label_list = []
    pixel_list = []
    label_pixel = []
    angle_list = []
    for angle in directions:
        direction = np.array([[1], [0], [0]])  # 正前方
        rotated = rotate_vector_z(direction, angle)

        pt_prev = origin  # 初始點

        W = rgb.shape[1] - 1
        H = rgb.shape[0] - 1
        for s in np.arange(0.0, scale, step):
            tip = origin + rotated * s
            map_x = int(tip[0][0] / cell_size) + map_size[0] // 2
            map_y = int(tip[1][0] / cell_size) + map_size[1] // 2
            if (
                (map_x < 0)
                or (map_y < 0)
                or (map_x >= map_size[0])
                or (map_y >= map_size[1])
            ):
                continue

            pixel_x, pixel_y = get_pixel(tip, R, T, fx, fy, cx, cy)

            if (
                (H - pixel_y) < 0
                or (W - pixel_x) < 0
                or (H - pixel_y) >= H
                or (W - pixel_x) >= W
            ):
                continue
            point_3d = get_3d_point(
                pixel_x, pixel_y, depth[H - pixel_y, W - pixel_x], R, T, fx, fy, cx, cy
            )

            # if point_3d[2] < 0.02:
            if obstacle_map[map_y][map_x] < 1 and point_3d[2] < 0.03:
                ox, oy = get_pixel(pt_prev, R, T, fx, fy, cx, cy)
                tx, ty = get_pixel(tip, R, T, fx, fy, cx, cy)
                ox = W - ox
                oy = H - oy
                tx = W - tx
                ty = H - ty
                cv2.arrowedLine(
                    overlay, (ox, oy), (tx, ty), color_map[angle], 20, tipLength=0.02
                )
                cv2.arrowedLine(
                    overlay2, (ox, oy), (tx, ty), color_map[angle], 20, tipLength=0.02
                )
                if angle not in annotated_angles:
                    tx_ = ox + (tx - ox) * 4
                    ty_ = oy + (ty - oy) * 4
                    ox_ = ox + (tx - ox) * 3
                    oy_ = oy + (ty - oy) * 3
                    label_pixel.append((ox_, oy_, tx_, ty_))
                    angle_list.append(angle)
                    #     cv2.arrowedLine(
                    #         annotated_image,
                    #         (ox_, oy_),
                    #         (tx_, ty_),
                    #         (0, 0, 0),
                    #         7,
                    #         tipLength=5,
                    #     )
                    #     cv2.circle(annotated_image, (tx, ty), 17, (0, 0, 0), -1)  # 畫圓圈
                    #     cv2.circle(
                    #         annotated_image, (tx, ty), 17, color_map[angle], 2
                    #     )  # 畫圓圈
                    #     cv2.circle(overlay, (tx, ty), 17, color_map[angle], 2)  # 畫圓圈
                    #     L = chr(65 + label)
                    #     text_size = cv2.getTextSize(
                    #         str(L), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    #     )[0]
                    #     text_width, text_height = text_size

                    #     text_x = tx - text_width // 2
                    #     text_y = ty + text_height // 2

                    #     cv2.putText(
                    #         annotated_image,
                    #         str(L),
                    #         (text_x, text_y),
                    #         cv2.FONT_HERSHEY_SIMPLEX,
                    #         0.7,
                    #         color_map[angle],
                    #         2,
                    #         cv2.LINE_AA,
                    #     )
                    #     cv2.putText(
                    #         overlay,
                    #         str(L),
                    #         (text_x, text_y),
                    #         cv2.FONT_HERSHEY_SIMPLEX,
                    #         0.7,
                    #         (0, 0, 0),
                    #         2,
                    #         cv2.LINE_AA,
                    #     )
                    label_list.append(label)
                    pixel_list.append((tx, ty))
                    annotated_angles.add(angle)  # 記錄這個方向已經加過編號了

            pt_prev = tip  # 更新前一點
        label += 1

        for i, (ox_, oy_, tx_, ty_) in enumerate(label_pixel):
            L = label_list[i]
            angle = angle_list[i]
            tx2 = ox_ + (tx_ - ox_) * 8
            ty2 = oy_ + (ty_ - oy_) * 8
            ox2 = ox_ + (tx_ - ox_) * 7
            oy2 = oy_ + (ty_ - oy_) * 7

            cv2.arrowedLine(
                annotated_image,
                (ox2, oy2),
                (tx2, ty2),
                (0, 0, 0),
                6,
                tipLength=8,
            )
            cv2.arrowedLine(
                overlay,
                (ox2, oy2),
                (tx2, ty2),
                (0, 0, 0),
                6,
                tipLength=8,
            )
            cv2.circle(annotated_image, (tx_, ty_), 17, (0, 0, 0), -1)
            cv2.circle(overlay, (tx_, ty_), 17, (0, 0, 0), -1)
            cv2.circle(annotated_image, (tx_, ty_), 17, color_map[angle], 2)
            cv2.circle(overlay, (tx_, ty_), 17, color_map[angle], 2)
            text_size = cv2.getTextSize(str(L), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_width, text_height = text_size
            text_x = tx_ - text_width // 2
            text_y = ty_ + text_height // 2

            cv2.putText(
                annotated_image,
                str(L),
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color_map[angle],
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                str(L),
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    cv2.addWeighted(overlay2, 0.5, origin_rgb, 0.5, 0, origin_rgb)
    cv2.addWeighted(overlay, 0.5, annotated_image, 0.7, 0, annotated_image)

    gx, gy = get_pixel(goal, R, T, fx, fy, cx, cy)
    gx = rgb.shape[1] - gx
    gy = rgb.shape[0] - gy

    cv2.imwrite("./data/annotated_rgb_line.png", origin_rgb)
    cv2.imwrite("./data/annotated_rgb1.png", annotated_image)
    return label_list, pixel_list


def get_affordance_direction_id(
    rgb, goal, instruction, R, T, fx, fy, cx, cy, obstacle_map
):
    IDs = []
    # run 3 times select the most common
    for i in range(3):
        ID = -1
        labels, pixels = get_annotated_rgb1(
            rgb, goal, R, T, fx, fy, cx, cy, obstacle_map
        )
        if len(labels) == 0:
            print("No labels found.")

        else:
            t = 0
            while True:
                try:
                    ID = determine_affordance(
                        "./data/annotated_rgb1.png", instruction, labels
                    )
                    break
                except Exception as e:
                    t += 1
                    print(f"Error: {e}, retrying {t} time(s)...")
                    if t > 3:
                        break
        if isinstance(ID, list):
            IDs.append(ID[0])  # or apply a strategy to select one
        else:
            IDs.append(ID)

    ID = max(set(IDs), key=IDs.count)
    print(f"IDs : {IDs}, ID : {ID}")
    num_ID = IDs.count(ID)
    if num_ID < 2:
        for i in range(len(IDs)):
            if abs(IDs[i] - ID) > 1:
                return -1, None
    H = rgb.shape[0] - 1
    W = rgb.shape[1] - 1
    if ID != -1:
        pixel = pixels[labels.index(ID)]
        depth = np.load("./data/depth.npy")
        point_3d = get_3d_point(
            pixel[0], pixel[1], depth[H - pixel[1], W - pixel[0]], R, T, fx, fy, cx, cy
        )
    else:
        pixel = None
        point_3d = None

    return ID, pixel


def get_annotated_rgb2(rgb, pixel):
    if pixel is None:
        print("No pixel found.")
        cv2.imwrite("./data/annotated_image.png", rgb)
        return
    label = "A"
    annotated_image = rgb.copy()

    cv2.circle(annotated_image, (pixel[0], pixel[1]), 17, (0, 0, 0), -1)  # 畫圓圈
    cv2.circle(annotated_image, (pixel[0], pixel[1]), 17, (0, 165, 255), 2)  # 畫圓圈
    text_size = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_width, text_height = text_size
    text_x = pixel[0] - text_width // 2
    text_y = pixel[1] + text_height // 2

    cv2.putText(
        annotated_image,
        str(label),
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 120, 255),
        2,
        cv2.LINE_AA,
    )
    # cv2.imwrite("./data/annotated_rgb2.png", annotated_image)
    cv2.imwrite("./data/annotated_image.png", annotated_image)


def annotate_rgb(
    rgb, goal, actions, instruction, R, T, fx, fy, cx, cy, obstacle_map, pixel=None
):
    annotated_image = rgb.copy()
    overlay = rgb.copy()
    get_annotated_rgb1(rgb, goal, R, T, fx, fy, cx, cy, obstacle_map)  # 這邊會畫出箭頭
    rgb = cv2.imread("./data/annotated_rgb_line.png")
    get_annotated_rgb2(rgb, pixel)

    # annotated_image = cv2.imread("./data/annotated_image.png")
    # === 畫點位與 index ===
    #   actions_3d = []
    #   for action in actions:
    #       map_x = action[0]
    #       map_y = action[1]
    #       x = (map_x - 100) * 0.05
    #       y = (map_y - 100) * 0.05
    #       actions_3d.append((x, y, 0))

    #   for i, action in enumerate(actions_3d):
    #       if isinstance(action, tuple) or isinstance(action, list):
    #           action = np.array([[action[0]], [action[1]], [0.0]])
    #       pixel_x, pixel_y = get_pixel(action, R, T, fx, fy, cx, cy)
    #       if (
    #           pixel_x < 0
    #           or pixel_y < 0
    #           or pixel_x >= rgb.shape[1]
    #           or pixel_y >= rgb.shape[0]
    #       ):
    #           continue
    #       depth = np.load("./data/depth.npy")
    #       point_3d = get_3d_point(
    #           pixel_x,
    #           pixel_y,
    #           depth[rgb.shape[0] - pixel_y, rgb.shape[1] - pixel_x],
    #           R,
    #           T,
    #           fx,
    #           fy,
    #           cx,
    #           cy,
    #       )
    #       pixel_x = rgb.shape[1] - pixel_x
    #       pixel_y = rgb.shape[0] - pixel_y
    #       # get poit 3d
    #       if point_3d[2] > 0.03:
    #           continue

    #       cv2.circle(annotated_image, (pixel_x, pixel_y), 15, (255, 255, 255), -1)
    #       cv2.circle(annotated_image, (pixel_x, pixel_y), 15, (225, 0, 0), 2)
    #       text_width, text_height = cv2.getTextSize(
    #           f"{i}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    #       )[0]
    #       cv2.putText(
    #           annotated_image,
    #           f"{i}",
    #           (pixel_x - text_width // 2, pixel_y + text_height // 2),
    #           cv2.FONT_HERSHEY_SIMPLEX,
    #           0.6,
    #           (0, 100, 150),
    #           2,
    #       )
    #   # save
    #   cv2.imwrite("./data/annotated_image.png", annotated_image)

    # === 處理 goal 格式 ===
    #  if isinstance(goal, tuple) or isinstance(goal, list):
    #      goal = np.array([[goal[0][0]], [goal[1][0]], [goal[2][0]]])
    #      print("goal:", goal)
    #  goal = np.vstack((goal, np.array([[0.0]])))  # shape: (3, 1)
    #  # 假設 goal 已經是 (3, 1) 形狀的 np.array 了
    #  goal = np.array([[goal[0][0]], [goal[1][0]], [0.0]])  # shape: (3, 1)
    #  origin = goal
    #  scale = 1000
    #  # scale = 2
    #  angle_list = list(range(0, 360, 30))  # 每 30 度

    #  for i, angle in enumerate(angle_list):
    #      # 建立方向向量
    #      direction = np.array([[scale], [0], [0]])
    #      rotated_direction = rotate_vector_z(direction, angle)
    #      tip = origin + rotated_direction

    #      # 投影到像素座標
    #      ox, oy = get_pixel(origin, R, T, fx, fy, cx, cy)
    #      tx, ty = get_pixel(tip, R, T, fx, fy, cx, cy)

    #      # 轉換成 OpenCV 座標系（左上為 (0,0)）
    #      H, W = rgb.shape[:2]
    #      ox = W - ox
    #      oy = H - oy
    #      tx = W - tx
    #      ty = H - ty

    #      # 產生不同顏色（HSV → BGR）
    #      import colorsys

    #      hue = i / len(angle_list)
    #      r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    #      color = (int(b * 255), int(g * 255), int(r * 255))  # OpenCV 用 BGR

    #      # 畫箭頭
    #      cv2.arrowedLine(overlay, (ox, oy), (tx, ty), color, 3, tipLength=0.2)
    #      dx = tx - ox
    #      dy = ty - oy
    #      arrow_length = (dx**2 + dy**2) ** 0.5
    #      unit_dx = dx / arrow_length
    #      unit_dy = dy / arrow_length
    #      text_offset = 30
    #      tx = tx + int(unit_dx * text_offset)
    #      ty = ty + int(unit_dy * text_offset)

    #      # ====== 加上置中編號文字 ======
    #      label = f"{i}"

    #      cv2.circle(annotated_image, (tx, ty), 15, (0, 0, 0), -1)  # 畫圓圈
    #      cv2.circle(overlay, (tx, ty), 15, (0, 0, 0), -1)  # 畫圓圈
    #      cv2.circle(annotated_image, (tx, ty), 15, color, 2)  # 畫圓圈
    #      cv2.circle(overlay, (tx, ty), 15, color, 2)  # 畫圓圈
    #      text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    #      text_width, text_height = text_size

    #      text_x = tx - text_width // 2
    #      text_y = ty + text_height // 2

    #      cv2.putText(
    #          annotated_image,
    #          label,
    #          (text_x, text_y),
    #          cv2.FONT_HERSHEY_SIMPLEX,
    #          0.6,
    #          color,
    #          2,
    #          cv2.LINE_AA,
    #      )
    #      cv2.putText(
    #          overlay,
    #          label,
    #          (text_x, text_y),
    #          cv2.FONT_HERSHEY_SIMPLEX,
    #          0.6,
    #          (0, 0, 0),
    #          2,
    #          cv2.LINE_AA,
    #      )

    # === 畫正前方 + 左右 ±30° 的射線（避開障礙）===
    #    H, W = rgb.shape[:2]
    #
    #    origin = np.array([[0], [0], [0]])
    #    origin = np.array([[goal[0][0]], [goal[1][0]], [0.0]])  # shape: (3, 1)
    #    directions = [0, -45, -30, -15, 0, 15, 30, 45]  # degrees: center, left, right
    #    # directions = []  # degrees: center, left, right
    #
    #    scale = 3
    #    step = 0.01
    #    color_map = {
    #        0: (0, 0, 255),  # 紅色（純紅）
    #        45: (180, 0, 255),  # 紫色（偏洋紅，更不像藍）
    #        15: (225, 225, 80),  # 水綠色（與黃和青都拉開）
    #        -30: (0, 255, 0),  # 綠色（鮮綠）
    #        -15: (0, 255, 255),  # 黃色（經典亮黃）
    #        30: (0, 128, 255),  # 橘色（亮橘偏黃，避免偏藍）
    #        -45: (255, 100, 0),  # 藍色（深藍橘感，更易區分）
    #    }
    #    import colorsys
    #
    #    directions = list(range(0, 360, 30))  # 12 條箭頭
    #    color_map = {}
    #    hue_order = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
    #
    #    for i, angle in enumerate(directions):
    #        hue = i / len(directions)  # 分布在 HSV 色環上（0~1）
    #        hue = hue_order[i] / len(directions)  # 分布在 HSV 色環上（0~1）
    #        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    #        color_map[angle] = (int(b * 255), int(g * 255), int(r * 255))  # BGR
    #    depth = np.load("./data/depth.npy")
    #    annotated_angles = set()  # 記錄已經編號過的方向
    #    label = 0
    #
    #    for angle in directions:
    #        direction = np.array([[1], [0], [0]])  # 正前方
    #        rotated = rotate_vector_z(direction, angle)
    #
    #        pt_prev = origin  # 初始點
    #
    #        W = rgb.shape[1]
    #        H = rgb.shape[0]
    #        for s in np.arange(0.0, scale, step):
    #            tip = origin + rotated * s
    #
    #            pixel_x, pixel_y = get_pixel(tip, R, T, fx, fy, cx, cy)
    #
    #            if (
    #                (H - pixel_y) < 0
    #                or (W - pixel_x) < 0
    #                or (H - pixel_y) >= H
    #                or (W - pixel_x) >= W
    #            ):
    #                continue
    #            point_3d = get_3d_point(
    #                pixel_x, pixel_y, depth[H - pixel_y, W - pixel_x], R, T, fx, fy, cx, cy
    #            )
    #            if point_3d[2] < 0.01:
    #                ox, oy = get_pixel(pt_prev, R, T, fx, fy, cx, cy)
    #                tx, ty = get_pixel(tip, R, T, fx, fy, cx, cy)
    #                ox = W - ox
    #                oy = H - oy
    #                tx = W - tx
    #                ty = H - ty
    #                cv2.arrowedLine(
    #                    overlay, (ox, oy), (tx, ty), color_map[angle], 20, tipLength=0.02
    #                )
    #                if angle not in annotated_angles:
    #                    cv2.circle(annotated_image, (tx, ty), 16, (0, 0, 0), -1)  # 畫圓圈
    #                    cv2.circle(overlay, (tx, ty), 16, (0, 0, 0), -1)  # 畫圓圈
    #                    cv2.circle(
    #                        annotated_image, (tx, ty), 16, color_map[angle], 2
    #                    )  # 畫圓圈
    #                    cv2.circle(overlay, (tx, ty), 16, color_map[angle], 2)  # 畫圓圈
    #                    text_size = cv2.getTextSize(
    #                        str(label), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    #                    )[0]
    #                    text_width, text_height = text_size
    #
    #                    text_x = tx - text_width // 2
    #                    text_y = ty + text_height // 2
    #
    #                    cv2.putText(
    #                        annotated_image,
    #                        str(label),
    #                        (text_x, text_y),
    #                        cv2.FONT_HERSHEY_SIMPLEX,
    #                        0.7,
    #                        color_map[angle],
    #                        2,
    #                        cv2.LINE_AA,
    #                    )
    #                    cv2.putText(
    #                        overlay,
    #                        str(label),
    #                        (text_x, text_y),
    #                        cv2.FONT_HERSHEY_SIMPLEX,
    #                        0.7,
    #                        (0, 0, 0),
    #                        2,
    #                        cv2.LINE_AA,
    #                    )
    #                    annotated_angles.add(angle)  # 記錄這個方向已經加過編號了
    #
    #            pt_prev = tip  # 更新前一點
    #        label += 1
    #
    # 檢查是否撞到障礙
    # for s in np.arange(0.2, scale, step):
    #     tip = origin + rotated * s

    #     # map 座標 (map 是 200x200)
    #     map_x = int(tip[0][0] / 0.05 + 100)
    #     map_y = int(tip[1][0] / 0.05 + 100)
    #     if 0 <= map_x < 200 and 0 <= map_y < 200:
    #         if obstacle_map[map_y, map_x] != 0:
    #             break  # 撞到障礙，停
    #     else:
    #         break  # 超出地圖邊界

    # # 畫箭頭（origin → tip）
    # ox, oy = get_pixel(origin, R, T, fx, fy, cx, cy)
    # tx, ty = get_pixel(tip, R, T, fx, fy, cx, cy)
    # ox = W - ox
    # oy = H - oy
    # tx = W - tx
    # ty = H - ty
    # cv2.arrowedLine(
    #     overlay, (ox, oy), (tx, ty), color_map[angle], 20, tipLength=0.02
    # )

    # 混合 overlay 到原圖
    cv2.addWeighted(overlay, 0.3, annotated_image, 0.7, 0, annotated_image)

    # cv2.imwrite("./data/annotated_image.png", annotated_image)


def get_features(R, T, fx, fy, cx, cy, depth_image, K, map=None):
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    proposer = KeypointProposer(config)

    if map is not None:
        # map to rgb
        map = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
    image_path = "./data/rgb.png"
    mask_path = "./data/mask.png"
    original_image = cv2.imread(image_path)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask_image, 127, 1, cv2.THRESH_BINARY)

    origin_h, origin_w = original_image.shape[:2]
    processed_image = prepare_image(image_path)

    shape_info = {
        "img_h": processed_image.shape[0],
        "img_w": processed_image.shape[1],
        "patch_h": processed_image.shape[0] // 14,
        "patch_w": processed_image.shape[1] // 14,
    }
    features_flat = (
        proposer._get_features(processed_image, shape_info).detach().cpu().numpy()
    )
    normalized_features = normalize(features_flat, norm="l2", axis=1)
    # point = self.get_3d_point(i, j, depth[self.rgb_data.shape[0] - j,self.rgb_data.shape[1] - i], R, T, fx, fy, cx, cy)
    masked_features = normalized_features[mask.flatten().astype(bool)]

    kmeans = KMeans(n_clusters=K, random_state=42)
    cluster_labels = kmeans.fit_predict(masked_features)

    colors = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [0, 255, 255],
            [255, 0, 255],
            [0, 128, 128],
            [128, 0, 0],
            [0, 128, 128],
            [128, 0, 128],
            [128, 128, 0],
            [128, 128, 128],
            [192, 192, 192],
            [255, 165, 0],
            [255, 20, 147],
            [0, 191, 255],
            [0, 255, 127],
            [255, 105, 180],
            [255, 228, 181],
            [240, 230, 140],
            [255, 228, 225],
        ]
    )

    # colors = np.random.randint(0, 255, (10, 3))
    color_mask = np.zeros((origin_h, origin_w, 3), dtype=np.uint8)
    color_mask[mask.astype(bool)] = colors[cluster_labels]

    final_image = original_image.copy()

    # Update to find closest valid point within the mask for each centroid
    cluster_points = np.column_stack(np.where(mask))
    number = 1
    centroid_points = []
    number_list = []
    for i in range(kmeans.n_clusters):
        # Extract all points in the current cluster
        points = cluster_points[cluster_labels == i]
        distance = np.linalg.norm(points - points.mean(axis=0), axis=1)
        closest_point = points[np.argmin(distance)]
        centroid = closest_point

        # Compute the mean (centroid) of these points
        # centroid = points.mean(axis=0).astype(int)
        Continue = False
        for point in centroid_points:
            i, j = point
            P = get_3d_point(
                origin_w - j, origin_h - i, depth_image[i, j], R, T, fx, fy, cx, cy
            )
            Q = get_3d_point(
                origin_w - centroid[1],
                origin_h - centroid[0],
                depth_image[centroid[0], centroid[1]],
                R,
                T,
                fx,
                fy,
                cx,
                cy,
            )
            # calculate the distance between the centroid and the point
            distance = np.linalg.norm(P - Q)
            if distance < 0.08:
                Continue = True
                break
        if Continue:
            number += 1
            continue
        centroid_points.append(centroid)
        number_list.append(number)
        y, x = centroid
        cv2.circle(final_image, (x, y), 15, (255, 255, 255), -1)
        cv2.circle(final_image, (x, y), 15, (0, 0, 255), 1)
        text_width, text_height = cv2.getTextSize(
            f"{number}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )[0]
        cv2.putText(
            final_image,
            f"{number}",
            (x - text_width // 2, y + text_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        if map is not None:
            # transform the pixel coordinates to map coordinates
            for point in centroid_points:
                point_3d = get_3d_point(
                    origin_w - point[1],
                    origin_h - point[0],
                    depth_image[point[0], point[1]],
                    R,
                    T,
                    fx,
                    fy,
                    cx,
                    cy,
                )
                # convert to map point
                map_x = int(point_3d[0] / cell_size) + map_size[0] // 2
                map_y = int(point_3d[1] / cell_size) + map_size[1] // 2
                # draw the occupancy point on map
                cv2.circle(map, (map_x, map_y), 1, (255, 0, 255), -1)

        number += 1
    # save the map
    if map is not None:
        cv2.imwrite("./data/map_clustter.png", map)

    cv2.imwrite("./data/clustered_image.png", final_image)
    return cluster_points, cluster_labels, number_list


def get_rough_affann(target, instruction, R, T, fx, fy, cx, cy, map):
    depth = np.load("./data/depth.npy")
    rgb = cv2.imread("./data/rgb.png")
    detect_and_segment("./data/rgb.png", target)
    cluster_points, cluster_labels, number_list = get_features(
        R, T, fx, fy, cx, cy, depth, 20, map
    )

    w = rgb.shape[1]
    h = rgb.shape[0]
    map_rgb = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
    mask = cv2.imread("./data/mask.png", cv2.IMREAD_GRAYSCALE)
    # convert 2d object mask to 3d point
    mask_points = np.column_stack(np.where(mask))
    mask_points_list = []
    for point in mask_points:
        Z = depth[point[0], point[1]]
        point_3d = get_3d_point(w - point[1], h - point[0], Z, R, T, fx, fy, cx, cy)
        mask_points_list.append(point_3d)
        # convert to map point
        map_x = int(point_3d[0] / cell_size) + map_size[0] // 2
        map_y = int(point_3d[1] / cell_size) + map_size[1] // 2

        # draw the occupancy point on map
        map_rgb[map_y, map_x] = (255, 255, 0)

    mask_points_mean = np.mean(mask_points_list, axis=0)
    np.save("./data/rough_mask_points_mean.npy", mask_points_mean)
    cv2.imwrite("./data/rough_affann.png", map_rgb)


def get_affordance_point(target, instruction, R, T, fx, fy, cx, cy, map, destination):
    depth = np.load("./data/depth.npy")
    rgb = cv2.imread("./data/rgb.png")
    detect_success = detect_and_segment("./data/rgb.png", target)
    cluster_points, cluster_labels, number_list = get_features(
        R, T, fx, fy, cx, cy, depth, 20, map
    )

    times = 0
    while True:
        try:
            affordance_num = determine_affordance(
                "./data/clustered_image.png",
                instruction,
                number_list,
            )
            break
        except ValueError:
            print("Invalid affordance number. Please try again.")
            time.sleep(1)
            times += 1
            if times > 10:
                affordance_num = np.random.randint(0, len(number_list))
                # raise ValueError("Invalid affordance number. Please try again.")
                break

    print(f"Affordance num: {affordance_num}")
    points = cluster_points[cluster_labels == affordance_num - 1]

    count = 0
    affordance = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=bool)
    for point in points:
        affordance[point[0], point[1]] = True
        count += 1

    affordance_center = np.zeros((2), dtype=np.float32)
    count = 0
    for i in range(affordance.shape[1]):
        for j in range(affordance.shape[0]):
            if affordance[j, i] == 1:
                affordance_center[0] += i
                affordance_center[1] += j
                count += 1
    affordance_center[0] /= count
    affordance_center[1] /= count

    distance = np.linalg.norm(points - points.mean(axis=0), axis=1)
    closest_point = points[np.argmin(distance)]
    affordance_center = closest_point
    # announce the center in rgb image
    # rgb = cv2.imread("./data/rgb.png")

    cv2.circle(
        rgb,
        (int(affordance_center[1]), int(affordance_center[0])),
        10,
        (255, 0, 255),
        -1,
    )

    # if the depth of the center is outlier, then set it to the mean depth
    depth_center = depth[int(affordance_center[0]), int(affordance_center[1])]
    # z score
    depth_mean = np.mean(depth)
    depth_std = np.std(depth)
    depth_z = (depth_center - depth_mean) / depth_std
    depth_z = np.abs(depth_z)
    if depth_z > 1:
        depth_center = depth_mean

    # save the image
    cv2.imwrite("./data/affrgb.png", rgb)
    # get the affordance center in 3d
    h = rgb.shape[0]
    w = rgb.shape[1]
    affordance_point = get_3d_point(
        w - affordance_center[1],
        h - affordance_center[0],
        depth[affordance_center[0], affordance_center[1]],
        R,
        T,
        fx,
        fy,
        cx,
        cy,
    )
    # draw the affordance center on depth image
    # cvt map to rgb
    map_rgb = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
    # scale to 1000 x 1000
    # map_rgb = cv2.resize(map_rgb, (2000, 2000))

    # draw the mask on map
    mask = cv2.imread("./data/mask.png", cv2.IMREAD_GRAYSCALE)
    # convert 2d object mask to 3d point
    mask_points = np.column_stack(np.where(mask))

    mask_points_list = []
    if detect_success:
        for point in mask_points:
            Z = depth[point[0], point[1]]
            point_3d = get_3d_point(w - point[1], h - point[0], Z, R, T, fx, fy, cx, cy)
            mask_points_list.append(point_3d)

            # convert to map point
            map_x = int(point_3d[0] / cell_size) + map_size[0] // 2
            map_y = int(point_3d[1] / cell_size) + map_size[1] // 2

            # draw the occupancy point on map
            map_rgb[map_y, map_x] = (0, 255, 255)

        # caculate the mean of the mask points
        mask_points_list = np.array(mask_points_list)
        mask_points_mean = np.mean(mask_points_list, axis=0)
        np.save("./data/mask_points_mean.npy", mask_points_mean)
    else:
        mask_points_mean = np.zeros((3), dtype=np.float32)
        mask_points_mean[0] = destination[0]
        mask_points_mean[1] = destination[1]
        mask_points_mean[2] = 0
        mask_points_mean = np.array(mask_points_mean).reshape(3, 1)
        np.save("./data/mask_points_mean.npy", mask_points_mean)

    cv2.imwrite("./data/affann.png", map_rgb)

    return (
        (affordance_point[0], affordance_point[1]),
        (
            affordance_center[0],
            affordance_center[1],
        ),
    )


def sample_from_mask_gaussian(
    center, target, sigma, R, T, fx, fy, cx, cy, map, num_samples=1
):
    torch.cuda.empty_cache()
    mask = cv2.imread("./data/mask.png", cv2.IMREAD_GRAYSCALE)
    mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1]

    H, W = mask.shape
    cx, cy = center

    # 建立整張圖的網格座標
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # 計算高斯權重（以中心點為中心）
    gauss = np.exp(-((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / (2 * sigma**2))

    # 把非 mask 區域的權重設為 0
    gauss[mask == 0] = 0

    # 如果所有權重都為 0（例如中心不在 mask 附近），就 return None
    if np.sum(gauss) == 0:
        return None

    # Normalize 成機率分布
    probs = gauss.flatten() / np.sum(gauss)

    # 所有像素點 index
    all_indices = np.arange(H * W)

    # 根據權重進行抽樣
    sampled_indices = np.random.choice(all_indices, size=num_samples, p=probs)
    sampled_coords = np.array([(idx // W, idx % W) for idx in sampled_indices])

    rgb = cv2.imread("./data/rgb.png")

    number_list = []
    coordinates = []
    depth = np.load("./data/depth.npy")
    for i in range(sampled_coords.shape[0]):
        y, x = sampled_coords[i]
        coordinates.append((y, x))
        number_list.append(i)
        # daw the  point on the image like above
        cv2.circle(rgb, (x, y), 12, (255, 255, 255), -1)
        cv2.circle(rgb, (x, y), 12, (0, 0, 255), 1)
        text_width, text_height = cv2.getTextSize(
            f"{i}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )[0]
        cv2.putText(
            rgb,
            f"{i}",
            (x - text_width // 2, y + text_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

    cv2.imwrite("./data/sample.png", rgb)
    times = 0
    while True:
        try:
            affordance_num = determine_affordance(
                "./data/sample.png",
                target,
                number_list,
            )
            break
        except ValueError:
            print("Invalid affordance number. Please try again.")
            time.sleep(1)
            times += 1
            if times > 10:
                # random select the affordance_num
                affordance_num = np.random.randint(0, len(number_list))
                # raise ValueError("Invalid affordance number. Please try again.")
                break
    print(f"Affordance num: {affordance_num}")

    coordinates = coordinates[affordance_num]

    # get the 3d point
    depth = np.load("./data/depth.npy")
    point_3d = get_3d_point(
        W - coordinates[1],
        H - coordinates[0],
        depth[coordinates[0], coordinates[1]],
        R,
        T,
        fx,
        fy,
        cx,
        cy,
    )
    # convert to map point
    map_x = int(point_3d[0] / cell_size) + map_size[0] // 2
    map_y = int(point_3d[1] / cell_size) + map_size[1] // 2
    # draw the occupancy point on map
    map_color = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
    cv2.circle(map_color, (map_x, map_y), 1, (255, 0, 255), -1)
    cv2.imwrite("./data/sample_map.png", map_color)

    return point_3d, coordinates
