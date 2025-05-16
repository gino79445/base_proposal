import numpy as np
import cv2
from base_proposal.vlm.sage import get_point
from base_proposal.tasks.utils import astar_utils
from base_proposal.affordance.get_affordance import get_affordance_point
from base_proposal.affordance.get_affordance import sample_from_mask_gaussian
from base_proposal.affordance.get_affordance import annotate_rgb
from base_proposal.affordance.get_affordance import get_affordance_direction_id


cell_size = 0.05
map_size = (203, 203)


def process_2d_map(occupancy_2d_map):
    occupancy_2d_map = occupancy_2d_map.copy()
    occupancy_2d_map = np.flipud(occupancy_2d_map)
    occupancy_2d_map = np.rot90(occupancy_2d_map)
    # occupancy_2d_map = cv2.cvtColor(occupancy_2d_map, cv2.COLOR_GRAY2BGR)
    scale = 10
    occupancy_2d_map = cv2.resize(
        occupancy_2d_map, (map_size[0] * scale, map_size[1] * scale)
    )
    return occupancy_2d_map


def sample_gaussian_actions_on_map(
    center,
    std_dev,
    num_samples,
    image_size,
    obstacle_map,
    preferred_mean=None,
    bias_sigma=0.2,
    dist_sigma=0.1,
    alpha=0.2,
    cell_size=0.05,
    map_size=(200, 200),
):
    all_candidates = []
    heatmap = np.zeros(map_size, dtype=np.float32)
    w, h = image_size
    preferred_dist = 0.7
    R = 1.2
    from scipy.stats import norm

    total_sampled = 0
    N = 1000
    t = 0
    while t < 5:
        while total_sampled < 10000:
            x = np.clip(
                np.random.normal(center[0], std_dev), center[0] - R, center[0] + R
            )
            y = np.clip(
                np.random.normal(center[1], std_dev), center[1] - R, center[1] + R
            )
            dist_to_goal = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

            if dist_to_goal < 0.4 or dist_to_goal > R:
                total_sampled += 1
                continue

            prob_dis = norm(loc=preferred_dist, scale=dist_sigma).cdf(
                dist_to_goal + 0.05
            ) - norm(loc=preferred_dist, scale=dist_sigma).cdf(dist_to_goal - 0.05)

            if preferred_mean is not None:
                semantic_dist = np.linalg.norm(
                    [x - preferred_mean[0], y - preferred_mean[1]]
                )
                prob_semantic = norm(loc=0, scale=bias_sigma).cdf(
                    semantic_dist + 0.05
                ) - norm(loc=0, scale=bias_sigma).cdf(semantic_dist - 0.05)
            else:
                prob_semantic = 1.0

            log_prob_dis = np.log(prob_dis + 1e-6)
            log_prob_semantic = np.log(prob_semantic + 1e-6)
            log_weight = alpha * log_prob_dis + (1 - alpha) * log_prob_semantic
            weight = np.exp(log_weight)
            # weight = alpha * prob_dis + (1 - alpha) * prob_semantic

            map_x = int(x / cell_size) + map_size[0] // 2
            map_y = int(y / cell_size) + map_size[1] // 2

            if (
                0 <= map_x < map_size[0]
                and 0 <= map_y < map_size[1]
                and astar_utils.is_valid_des(map_y, map_x, obstacle_map)
            ):
                heatmap[map_y, map_x] = max(heatmap[map_y, map_x], weight)
                if len(all_candidates) < N:
                    all_candidates.append(((map_x, map_y), weight))
                else:
                    break

            total_sampled += 1
        if len(all_candidates) > 1:
            break
        else:
            R += 0.2
            t += 1

        # 按照機率排序並取前 num_samples 個
        # all_candidates.sort(key=lambda x: -x[1])
        # actions = [pt for pt, _ in all_candidates[:num_samples]]
    positions, weights = zip(*[(pt, w) for pt, w in all_candidates])

    weights = np.array(weights)
    probs = weights / np.sum(weights)  # 正規化成機率

    # 根據機率做不重複抽樣
    num_to_sample = min(num_samples, len(positions))
    chosen_idx = np.random.choice(
        len(positions), size=num_to_sample, replace=False, p=probs.flatten()
    )

    # 選中的點
    actions = [positions[i] for i in chosen_idx]

    # 若有 preferred_mean，畫圈圈
    if preferred_mean is not None:
        map_img = obstacle_map.copy()
        map_img = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
        map_x = int(preferred_mean[0] / cell_size) + map_size[0] // 2
        map_y = int(preferred_mean[1] / cell_size) + map_size[1] // 2
        cv2.circle(map_img, (map_x, map_y), 1, (0, 255, 0), -1)
        cv2.imwrite("./data/mean_map.png", map_img)

    # 繪製 heatmap 疊加到地圖上
    normalized_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    colored_heatmap = cv2.applyColorMap(
        normalized_heatmap.astype(np.uint8), cv2.COLORMAP_JET
    )

    if len(obstacle_map.shape) == 2:
        base_map = cv2.cvtColor(obstacle_map, cv2.COLOR_GRAY2BGR)
    else:
        base_map = obstacle_map.copy()

    colored_heatmap = cv2.resize(
        colored_heatmap, (base_map.shape[1], base_map.shape[0])
    )
    overlayed = cv2.addWeighted(base_map, 0.5, colored_heatmap, 0.5, 0)

    # 調整方向與大小
    overlayed = np.flipud(overlayed)
    overlayed = np.rot90(overlayed)
    overlayed = cv2.resize(
        overlayed,
        (map_size[0] * 10, map_size[1] * 10),
        interpolation=cv2.INTER_NEAREST,
    )

    # 裁切出中心區域
    crop_size = 400
    center_px = (
        int(map_size[1] - 1 - (int(center[0] / cell_size) + map_size[1] // 2)) * 10,
        int(map_size[0] - 1 - (int(center[1] / cell_size) + map_size[0] // 2)) * 10,
    )
    x_min = int(max(0, center_px[1] - crop_size))
    x_max = int(min(overlayed.shape[1], center_px[1] + crop_size))
    y_min = int(max(0, center_px[0] - crop_size))
    y_max = int(min(overlayed.shape[0], center_px[0] + crop_size))
    cropped_map = overlayed[y_min:y_max, x_min:x_max]
    cropped_map = cv2.resize(cropped_map, (2000, 2000))

    # 儲存圖片
    import time

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"./heatmap/w_distribution_alpha_map_{timestamp}.png"
    cv2.imwrite(filename, cropped_map)

    base_map = np.flipud(base_map)
    base_map = np.rot90(base_map)
    base_map = cv2.resize(
        base_map, (map_size[0] * 10, map_size[1] * 10), interpolation=cv2.INTER_NEAREST
    )
    # 裁切出中心區域
    x_min = int(max(0, center_px[1] - crop_size))
    x_max = int(min(base_map.shape[1], center_px[1] + crop_size))
    y_min = int(max(0, center_px[0] - crop_size))
    y_max = int(min(base_map.shape[0], center_px[0] + crop_size))
    cropped_map = base_map[y_min:y_max, x_min:x_max]
    cropped_map = cv2.resize(cropped_map, (2000, 2000))
    # 儲存圖片
    cv2.imwrite("./heatmap/base_map.png", cropped_map)

    return actions


# def sample_gaussian_actions_on_map(
#    center,
#    std_dev,
#    num_samples,
#    image_size,
#    obstacle_map,
#    preferred_mean=None,
#    bias_sigma=0.2,
#    alpha=0.2,
# ):
#    actions = []
#    w, h = image_size
#    i = 0
#    preferred_dist = 0.7
#    dist_sigma = 0.1
#    R = 1
#    heatmap = np.zeros(map_size, dtype=np.float32)
#    while i < num_samples:
#        t = 0
#        while True:
#            t += 1
#            x = np.clip(
#                np.random.normal(center[0], std_dev), center[0] - R, center[0] + R
#            )
#            y = np.clip(
#                np.random.normal(center[1], std_dev), center[1] - R, center[1] + R
#            )
#
#            dist_to_goal = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
#            if dist_to_goal < 0.4 or dist_to_goal > R:
#                continue
#
#            #   weight_to_goal = np.exp(
#            #       -0.5 * ((dist_to_goal - preferred_dist) / dist_sigma) ** 2
#            #   )
#            from scipy.stats import norm
#
#            dis_cdf = norm(loc=preferred_dist, scale=dist_sigma)
#            prob_dis = dis_cdf.cdf(dist_to_goal + 0.05) - dis_cdf.cdf(
#                dist_to_goal - 0.05
#            )
#            if preferred_mean is not None:
#                semantic_mean = np.linalg.norm(
#                    [x - preferred_mean[0], y - preferred_mean[1]]
#                )
#                mean_cdf = norm(loc=0, scale=bias_sigma)
#                prob_semantic = mean_cdf.cdf(semantic_mean + 0.05) - mean_cdf.cdf(
#                    semantic_mean - 0.05
#                )
#            else:
#                prob_semantic = 1.0
#
#            weight = alpha * prob_dis + (1 - alpha) * prob_semantic
#
#            # weight_to_mean = 1.0
#            # if preferred_mean is not None:
#            #     dist_to_mean = np.linalg.norm(
#            #         [x - preferred_mean[0], y - preferred_mean[1]]
#            #     )
#            #     weight_to_mean = np.exp(-0.5 * (dist_to_mean / bias_sigma) ** 2)
#            # # alpha = 0.5
#            # weight = alpha * weight_to_goal + (1 - alpha) * weight_to_mean
#
#            map_x = int(x / cell_size) + map_size[0] // 2
#            map_y = int(y / cell_size) + map_size[1] // 2
#
#            if (
#                0 <= map_x < map_size[0]
#                and 0 <= map_y < map_size[1]
#                and astar_utils.is_valid_des(map_y, map_x, obstacle_map)
#            ):
#                heatmap[map_y, map_x] = max(heatmap[map_y, map_x], weight)
#                if np.random.rand() < weight:
#                    actions.append((map_x, map_y))
#                    #    print(
#                    #        f"weight: {weight}, "
#                    #        f"weight_to_goal: {prob_dis}, "
#                    #        f"weight_to_mean: {prob_semantic}, "
#                    #    )
#                    i += 1
#                    break
#
#            if t > 10000:
#                if R > 2:
#                    i += 1
#                    break
#                R += 0.2
#                dist_sigma += 0.2
#
#    if preferred_mean is not None:
#        map = obstacle_map.copy()
#        map = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
#        map_x = int(preferred_mean[0] / cell_size) + map_size[0] // 2
#        map_y = int(preferred_mean[1] / cell_size) + map_size[1] // 2
#        cv2.circle(map, (map_x, map_y), 1, (0, 255, 0), -1)  # draw the mean on the map
#        # save the map with the mean
#        cv2.imwrite("./data/mean_map.png", map)
#
#    normalized_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
#    colored_heatmap = cv2.applyColorMap(
#        normalized_heatmap.astype(np.uint8), cv2.COLORMAP_JET
#    )
#
#    # 將 obstacle_map 處理成 3 通道灰階圖
#    if len(obstacle_map.shape) == 2:
#        base_map = cv2.cvtColor(obstacle_map, cv2.COLOR_GRAY2BGR)
#    else:
#        base_map = obstacle_map.copy()
#
#    # 縮放 heatmap 和 map 一樣大
#    colored_heatmap = cv2.resize(
#        colored_heatmap, (base_map.shape[1], base_map.shape[0])
#    )
#
#    # 疊加 heatmap 到 map 上
#    overlayed = cv2.addWeighted(base_map, 0.5, colored_heatmap, 0.5, 0)
#    # reshape  map
#    overlayed = np.flipud(overlayed)
#    overlayed = np.rot90(overlayed)
#    overlayed = cv2.resize(
#        overlayed,
#        (map_size[0] * 10, map_size[1] * 10),
#        interpolation=cv2.INTER_NEAREST,
#    )
#
#    # crop the map
#    crop_size = 400
#    center = (
#        int(map_size[1] - 1 - (int(center[0] / cell_size) + map_size[1] // 2)) * 10,
#        int(map_size[0] - 1 - (int(center[1] / cell_size) + map_size[0] // 2)) * 10,
#    )
#    x_min = int(max(0, center[1] - crop_size))
#    x_max = int(min(overlayed.shape[1], center[1] + crop_size))
#    y_min = int(max(0, center[0] - crop_size))
#    y_max = int(min(overlayed.shape[0], center[0] + crop_size))
#    cropped_map = overlayed[y_min:y_max, x_min:x_max]
#    cropped_map = cv2.resize(cropped_map, (2000, 2000))
#
#    import time
#
#    timestamp = time.strftime("%Y%m%d-%H%M%S")
#    filename = f"./heatmap/w_distribution_alpha_map_{timestamp}.png"
#    cv2.imwrite(filename, cropped_map)
#    return actions
#


def rotate_vector_2d(v, angle_deg):
    """Rotate a 2D vector by degrees."""
    angle_rad = np.deg2rad(angle_deg)
    rot = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    return rot @ v


def find_arrow_stop_point(start, vector, annotated_image):
    length = int(np.linalg.norm(vector))
    unit_vector = vector / length

    for i in range(length):
        x = int(start[0] + unit_vector[0] * i)
        y = int(start[1] + unit_vector[1] * i)

        # 確保不越界
        if (
            x < 0
            or y < 0
            or x >= annotated_image.shape[1]
            or y >= annotated_image.shape[0]
        ):
            break

        b, g, r = annotated_image[y, x]
        if (abs(b - 255) < 10 and abs(g - 255) < 10 and abs(r - 255) < 10) or (
            abs(b - 0) < 10 and abs(g - 255) < 10 and abs(r - 255) < 10
        ):
            # 碰到白色就停
            return (x, y)
    return (int(start[0] + vector[0]), int(start[1] + vector[1]))


def get_affordance_direction_pos(goal, direction_id, occupancy_2d_map):
    occupancy_2d_map = occupancy_2d_map.copy()
    directions = list(range(0, 360, 30))  # 0° 到 330°，每 30°
    if direction_id != -1:
        angle = directions[direction_id]
    else:
        return None
    scale = 200
    step = 1
    start = (goal[0], goal[1])
    start = (
        int(start[0] / cell_size) + map_size[0] // 2,
        int(start[1] / cell_size) + map_size[1] // 2,
    )
    for s in np.arange(0.0, scale, step):
        vector = np.array(
            [s * np.cos(np.deg2rad(angle)), s * np.sin(np.deg2rad(angle))]
        )
        tx, ty = start[0] + vector[0], start[1] + vector[1]
        if (
            tx < 0
            or ty < 0
            or tx >= occupancy_2d_map.shape[1]
            or ty >= occupancy_2d_map.shape[0]
            or occupancy_2d_map[int(ty), int(tx)] != 0
        ):
            continue
        else:
            print(f"Direction {angle}°: ({tx}, {ty})")
            cv2.circle(occupancy_2d_map, (int(tx), int(ty)), 3, (0, 255, 0), -1)
            break
    cv2.imwrite("./data/affordance_direction.png", occupancy_2d_map)
    return (tx, ty)


def annotate_map(image, destination, actions, direction_id=0, occupancy_2d_map=None):
    annotated_image = image.copy()
    overlay = np.zeros_like(annotated_image)
    mask_points_mean = np.load("./data/mask_points_mean.npy")
    mask_x = int(mask_points_mean[0] / cell_size) + map_size[0] // 2
    mask_y = int(mask_points_mean[1] / cell_size) + map_size[1] // 2
    mask_x, mask_y = (map_size[1] - 1 - mask_y) * 10, (map_size[0] - 1 - mask_x) * 10
    goal = (
        (map_size[1] - 1 - (int(destination[1] / cell_size) + map_size[1] // 2)) * 10,
        (map_size[0] - 1 - (int(destination[0] / cell_size) + map_size[0] // 2)) * 10,
    )
    direction_2d = get_affordance_direction_pos(
        mask_points_mean, direction_id, occupancy_2d_map
    )
    if direction_2d is not None:
        scale = 1000
        step = 1

        # for i, (tx_, ty_) in enumerate(point_pos_list):
        #     L = chr(label_list[i] + 65)
        #     angle = angle_list[i]
        #     cv2.circle(annotated_image, (tx_, ty_), 17, (0, 0, 0), -1)
        #     cv2.circle(overlay, (tx_, ty_), 17, (0, 0, 0), -1)
        #     cv2.circle(annotated_image, (tx_, ty_), 17, color_map[angle], 2)
        #     cv2.circle(overlay, (tx_, ty_), 17, color_map[angle], 2)
        #     text_size = cv2.getTextSize(str(L), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        #     text_width, text_height = text_size
        #     text_x = tx_ - text_width // 2
        #     text_y = ty_ + text_height // 2

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

        #    cv2.addWeighted(overlay, 0.4, annotated_image, 1, 0, annotated_image)
        direction_2d = (
            int(map_size[1] - 1 - direction_2d[1]) * 10,
            int(map_size[0] - 1 - direction_2d[0]) * 10,
        )

        # draw the arrow
        vector = np.array([direction_2d[0] - mask_x, direction_2d[1] - mask_y])
        norm = np.linalg.norm(vector)
        if norm == 0:
            norm = 1
        vector = vector / norm
        arrow_length = 200
        center = (mask_x, mask_y)
        axes = (arrow_length, arrow_length)  # 扇形半徑
        angle = np.degrees(np.arctan2(vector[1], vector[0]))  # 箭頭方向的角度

        # cv2.ellipse(
        #    overlay,
        #    center,
        #    axes,
        #    0,  # ellipse rotation
        #    0,
        #    360,
        #    (0, 180, 0),  # orange color
        #    -1,  # -1 表示填滿扇形
        # )
        start_angle = angle - 60  # 左右各45度
        end_angle = angle + 60

        cv2.ellipse(
            overlay,
            center,
            axes,
            0,  # ellipse rotation
            start_angle,
            end_angle,
            (0, 120, 255),  # orange color
            -1,  # -1 表示填滿扇形
        )

        end = (1000, 999)
        start = (1000, 1000)
        start = (mask_x, mask_y)
        end = (mask_x, mask_y + 1)
        v = np.array([end[0] - start[0], end[1] - start[1]])
        norm = np.linalg.norm(v)
        if norm == 0:
            norm = 1
        v = v / norm
        color_map = {}
        hue_order = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]

        import colorsys

        directions = list(range(-180, -540, -30))  # 0° 到 330°，每 30°
        for i, angle in enumerate(directions):
            hue = i / len(directions)  # 分布在 HSV 色環上（0~1）
            hue = hue_order[i] / len(directions)  # 分布在 HSV 色環上（0~1）
            if 0.05 < hue < 0.12:
                hue -= 0.04
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            color_map[angle] = (int(b * 255), int(g * 255), int(r * 255))  # BGR
        origin_map = image.copy()
        for i, angle in enumerate(directions):
            rotated = rotate_vector_2d(v, angle)
            pt_prev = start

            for s in np.arange(0.0, scale, step):
                tip = start + rotated * s
                # not in white area and yellow area
                if (
                    tip[0] < 0
                    or tip[1] < 0
                    or tip[0] >= origin_map.shape[1]
                    or tip[1] >= origin_map.shape[0]
                ):
                    continue
                b, g, r = origin_map[int(tip[1]), int(tip[0])]
                if abs(b - 0) < 10 and abs(g - 0) < 10 and abs(r - 0) < 10:
                    cv2.arrowedLine(
                        overlay,
                        (int(pt_prev[0]), int(pt_prev[1])),
                        (int(tip[0]), int(tip[1])),
                        color_map[angle],
                        15,
                        tipLength=0.001,
                    )
                # if first_found:
                #     point_pos_list.append((int(tip[0]), int(tip[1])))
                #     label_list.append(i)
                #     angle_list.append(angle)
                # first_found = False

                pt_prev = tip  # 更新前一點
        cv2.arrowedLine(
            annotated_image,
            (mask_x, mask_y),
            (
                int(mask_x + vector[0] * arrow_length),
                int(mask_y + vector[1] * arrow_length),
            ),
            (0, 120, 255),
            2,
            tipLength=0.1,
        )
        cv2.addWeighted(overlay, 0.4, annotated_image, 1, 0, annotated_image)
        cv2.circle(
            annotated_image, (direction_2d[0], direction_2d[1]), 17, (0, 0, 0), -1
        )
        cv2.circle(
            annotated_image, (direction_2d[0], direction_2d[1]), 17, (0, 120, 255), 3
        )
        text_size = cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_width, text_height = text_size
        text_x = direction_2d[0] - text_width // 2
        text_y = direction_2d[1] + text_height // 2

        cv2.putText(
            annotated_image,
            "A",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 120, 255),
            2,
        )
    else:
        scale = 1000
        step = 1
        end = (1000, 999)
        start = (1000, 1000)
        start = (mask_x, mask_y)
        end = (mask_x, mask_y + 1)
        v = np.array([end[0] - start[0], end[1] - start[1]])
        norm = np.linalg.norm(v)
        if norm == 0:
            norm = 1
        v = v / norm
        color_map = {}
        hue_order = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]

        import colorsys

        directions = list(range(-180, -540, -30))  # 0° 到 330°，每 30°
        for i, angle in enumerate(directions):
            hue = i / len(directions)  # 分布在 HSV 色環上（0~1）
            hue = hue_order[i] / len(directions)  # 分布在 HSV 色環上（0~1）
            if 0.05 < hue < 0.12:
                hue -= 0.04
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            color_map[angle] = (int(b * 255), int(g * 255), int(r * 255))  # BGR
        origin_map = image.copy()
        for i, angle in enumerate(directions):
            rotated = rotate_vector_2d(v, angle)
            pt_prev = start

            for s in np.arange(0.0, scale, step):
                tip = start + rotated * s
                # not in white area and yellow area
                if (
                    tip[0] < 0
                    or tip[1] < 0
                    or tip[0] >= origin_map.shape[1]
                    or tip[1] >= origin_map.shape[0]
                ):
                    continue
                b, g, r = origin_map[int(tip[1]), int(tip[0])]
                if abs(b - 0) < 10 and abs(g - 0) < 10 and abs(r - 0) < 10:
                    cv2.arrowedLine(
                        overlay,
                        (int(pt_prev[0]), int(pt_prev[1])),
                        (int(tip[0]), int(tip[1])),
                        color_map[angle],
                        15,
                        tipLength=0.001,
                    )
                # if first_found:
                #     point_pos_list.append((int(tip[0]), int(tip[1])))
                #     label_list.append(i)
                #     angle_list.append(angle)
                # first_found = False

                pt_prev = tip  # 更新前一點
        cv2.addWeighted(overlay, 0.4, annotated_image, 1, 0, annotated_image)

    for i, (x, y) in enumerate(actions):
        x, y = (map_size[1] - y) * 10, (map_size[0] - x) * 10
        cv2.circle(annotated_image, (x, y), 18, (255, 255, 255), -1)
        cv2.circle(annotated_image, (x, y), 18, (225, 0, 0), 3)
        text_width, text_height = cv2.getTextSize(
            f"{i}", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )[0]
        cv2.putText(
            annotated_image,
            f"{i}",
            (x - text_width // 2, y + text_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 100, 150),
            2,
        )

    overlay = annotated_image.copy()

    cv2.addWeighted(overlay, 0.3, annotated_image, 0.7, 0, annotated_image)

    # 額外標記
    crop_size = 400
    # cv2.circle(annotated_image, goal, 5, (255, 0, 255), -1)
    cv2.circle(annotated_image, (1000, 1000), 20, (255, 0, 0), -1)

    # 儲存圖
    cv2.imwrite("./data/annotated_map.png", annotated_image)

    x_min = max(0, goal[1] - crop_size)
    x_max = min(annotated_image.shape[1], goal[1] + crop_size)
    y_min = max(0, goal[0] - crop_size)
    y_max = min(annotated_image.shape[0], goal[0] + crop_size)

    cropped_map = annotated_image[x_min:x_max, y_min:y_max]
    cropped_map = cv2.resize(cropped_map, (2000, 2000))

    return cropped_map


def update_gaussian_distribution(
    best_actions_positions,
    bias_sigma,
    dist_sigma,
    destination=None,
    max_drift=1.2,
    std_dev_decay=0.8,
):
    if not best_actions_positions:
        return None, bias_sigma

    mean_x = np.mean([x for x, y in best_actions_positions])
    mean_y = np.mean([y for x, y in best_actions_positions])
    mean_x = (mean_x - map_size[0] // 2) * cell_size
    mean_y = (mean_y - map_size[1] // 2) * cell_size
    if destination is not None:
        dx = destination[0]
        dy = destination[1]
        mean_x = np.clip(mean_x, dx - max_drift, dx + max_drift)
        mean_y = np.clip(mean_y, dy - max_drift, dy + max_drift)
    bias_sigma = max(0.01, std_dev_decay * bias_sigma)
    # dist_sigma = max(0.01, std_dev_decay * dist_sigma)

    return (mean_x, mean_y), bias_sigma, dist_sigma


def get_dynamic_alpha(i, total_iterations=6, max_alpha=0.6):
    shift = total_iterations * 0.35  # 提早轉折點
    k = 12 / total_iterations  # 增加斜率，加速轉變
    sigmoid = 1 / (1 + np.exp(-k * (i - shift)))
    return sigmoid * max_alpha


def get_base(
    occupancy_2d_map, target, instruction, R, T, fx, fy, cx, cy, destination, K=3
):
    iterations = 3
    parallel = 1
    final_actions = []
    std_dev = 1.0

    for p in range(parallel):
        num_samples = 20
        preferred_mean = None
        bias_sigma = 0.2
        dist_sigma = 0.1
        alpha = 0.0

        affordance_point, affordance_pixel = get_affordance_point(
            target, instruction, R, T, fx, fy, cx, cy, occupancy_2d_map, destination
        )

        rgb = cv2.imread("./data/rgb.png")
        mask_points_mean = np.load("./data/mask_points_mean.npy")
        mask_points_mean = np.array(mask_points_mean[0:3])
        img_map = cv2.imread("./data/affann.png")
        map_img = process_2d_map(img_map)
        destination = affordance_point
        center_sigma = 100
        num_samples_center = 15

        affordance_direction_id, affordance_direction_pixel = (
            get_affordance_direction_id(
                rgb,
                mask_points_mean,
                instruction,
                R,
                T,
                fx,
                fy,
                cx,
                cy,
                occupancy_2d_map,
            )
        )

        for i in range(iterations):
            print(f"bias_sigma: {bias_sigma:.2f} dist_sigma: {dist_sigma:.2f}")
            alpha = get_dynamic_alpha(i, iterations, max_alpha=0.5)
            alpha = 0.5
            print(f"Iteration {i + 1}/{iterations}, alpha: {alpha:.2f}")
            while True:
                actions = sample_gaussian_actions_on_map(
                    center=destination,
                    std_dev=std_dev,
                    num_samples=num_samples,
                    image_size=map_size,
                    obstacle_map=occupancy_2d_map,
                    preferred_mean=preferred_mean,
                    bias_sigma=bias_sigma,
                    dist_sigma=dist_sigma,
                    alpha=alpha,
                )
                if len(actions) >= 0:
                    break
                print("Not enough actions, resampling...")
                if False:
                    destination, affordance_pixel = get_affordance_point(
                        target, instruction, R, T, fx, fy, cx, cy, occupancy_2d_map
                    )
                else:
                    destination, affordance_pixel = sample_from_mask_gaussian(
                        (affordance_pixel[1], affordance_pixel[0]),
                        target,
                        center_sigma,
                        R,
                        T,
                        fx,
                        fy,
                        cx,
                        cy,
                        occupancy_2d_map,
                        num_samples=num_samples_center,
                    )
                    center_sigma += 10

            annotated_image = annotate_map(
                map_img.copy(),
                destination,
                actions,
                direction_id=affordance_direction_id,
                occupancy_2d_map=occupancy_2d_map,
            )

            annotate_rgb(
                rgb,
                mask_points_mean,
                actions,
                instruction,
                R,
                T,
                fx,
                fy,
                cx,
                cy,
                occupancy_2d_map,
                affordance_direction_pixel,
            )
            cv2.imwrite(f"./data/annotated_map_{i + 1}.png", annotated_image)
            t = 0
            while True:
                try:
                    result = get_point(
                        "./data/annotated_image.png",
                        f"./data/annotated_map_{i + 1}.png",
                        instruction,
                        f"{K} candidate base positions",
                    )
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    t += 1
                    if t > 10:
                        raise e
                    print("Retrying...")

            print(f"Best Action Index -> {result}")
            # idx have to in range of actions
            result = [idx for idx in result if idx < len(actions)]
            best_actions_positions = [actions[idx] for idx in result]
            preferred_mean, bias_sigma, dist_sigma = update_gaussian_distribution(
                best_actions_positions, bias_sigma, dist_sigma, destination
            )

        for action in best_actions_positions:
            final_actions.append(action)

    preferred_mean, bias_sigma, dist_sigma = update_gaussian_distribution(
        final_actions, bias_sigma, dist_sigma, destination
    )
    # get the preferred mean map
    preferred_mean_map = occupancy_2d_map.copy()
    preferred_mean_map = cv2.cvtColor(preferred_mean_map, cv2.COLOR_GRAY2BGR)
    map_x = int(preferred_mean[0] / cell_size) + map_size[0] // 2
    map_y = int(preferred_mean[1] / cell_size) + map_size[1] // 2
    cv2.circle(
        preferred_mean_map, (map_x, map_y), 1, (0, 255, 0), -1
    )  # draw the mean on the map
    cv2.imwrite("./data/preferred_mean_map.png", preferred_mean_map)
    while True:
        final_actions = sample_gaussian_actions_on_map(
            center=destination,
            std_dev=std_dev,
            num_samples=20,
            image_size=map_size,
            obstacle_map=occupancy_2d_map,
            preferred_mean=preferred_mean,
            bias_sigma=bias_sigma,
            dist_sigma=dist_sigma,
            alpha=alpha,
        )
        if len(final_actions) >= 0:
            break
        print("Not enough actions, resampling...")
        destination, affordance_pixel = sample_from_mask_gaussian(
            (affordance_pixel[1], affordance_pixel[0]),
            target,
            center_sigma,
            R,
            T,
            fx,
            fy,
            cx,
            cy,
            occupancy_2d_map,
            num_samples=num_samples_center,
        )

    final_img = annotate_map(
        map_img.copy(),
        destination,
        final_actions,
        direction_id=affordance_direction_id,
        occupancy_2d_map=occupancy_2d_map,
    )
    annotate_rgb(
        rgb,
        mask_points_mean,
        final_actions,
        instruction,
        R,
        T,
        fx,
        fy,
        cx,
        cy,
        occupancy_2d_map,
        affordance_direction_pixel,
    )
    cv2.imwrite("./data/final_annotated_map.png", final_img)

    t = 0
    while True:
        try:
            result = get_point(
                "./data/annotated_image.png",
                "./data/final_annotated_map.png",
                instruction,
                "5 candidate base positions",
            )
            break
        except Exception as e:
            print(f"Error: {e}")
            t += 1
            if t > 10:
                raise e
            print("Retrying...")
    print(f"Final Best Action Index -> {result}")

    # if the idx >= 10 then remove the idx
    result = [idx for idx in result if idx < len(final_actions)]
    final_actions = [final_actions[idx] for idx in result]

    mean_x = np.mean([x for x, y in final_actions])
    mean_y = np.mean([y for x, y in final_actions])
    # remove 2 outliers most far from the mean
    distances = [
        np.sqrt((x - mean_x) ** 2 + (y - mean_y) ** 2) for x, y in final_actions
    ]

    sorted_indices = np.argsort(distances)
    if len(final_actions) > 2:
        # remove the 2 farthest points
        final_actions = [
            final_actions[i] for i in sorted_indices[: len(final_actions) - 2]
        ]

    elif len(final_actions) == 2:
        final_actions = [
            final_actions[i] for i in sorted_indices[: len(final_actions) - 1]
        ]
    final_actions = np.array(final_actions)
    base = np.mean(final_actions, axis=0)
    # make it as int
    base = np.round(base).astype(int)
    # check if the base is colliding with the obstacle
    map_x = base[0]
    map_y = base[1]
    if astar_utils.is_valid_des(map_y, map_x, occupancy_2d_map):
        print("Base Position is valid")
    else:
        print("Base Position is invalid")
        # get the closest point to the base
        distances = [
            np.sqrt((x - base[0]) ** 2 + (y - base[1]) ** 2) for x, y in final_actions
        ]
        min_index = np.argmin(distances)
        base = final_actions[min_index]
        print(f"Base Position is invalid, use the closest point {base}")

    print(f"Final Base Position -> {base}")
    base = (
        (base[0] - map_size[0] // 2) * cell_size,
        (base[1] - map_size[1] // 2) * cell_size,
    )

    return base
