import numpy as np
import cv2
from base_proposal.vlm.spaceAware_pivot import get_point
from base_proposal.tasks.utils import astar_utils
from base_proposal.affordance.get_affordance import get_affordance_point
from base_proposal.affordance.get_affordance import sample_from_mask_gaussian
from base_proposal.affordance.get_affordance import get_annotated_image


def process_2d_map(occupancy_2d_map):
    occupancy_2d_map = occupancy_2d_map.copy()
    occupancy_2d_map = np.flipud(occupancy_2d_map)
    occupancy_2d_map = np.rot90(occupancy_2d_map)
    # occupancy_2d_map = cv2.cvtColor(occupancy_2d_map, cv2.COLOR_GRAY2BGR)
    scale = 10
    occupancy_2d_map = cv2.resize(occupancy_2d_map, (200 * scale, 200 * scale))
    return occupancy_2d_map


def sample_gaussian_actions_on_map(
    center,
    std_dev,
    num_samples,
    image_size,
    obstacle_map,
    preferred_mean=None,
    bias_sigma=0.2,
    alpha=0.2,
):
    actions = []
    w, h = image_size
    i = 0
    preferred_dist = 0.7
    dist_sigma = 0.1
    R = 1.5

    while i < num_samples:
        t = 0
        while True:
            t += 1
            x = np.clip(
                np.random.normal(center[0], std_dev), center[0] - R, center[0] + R
            )
            y = np.clip(
                np.random.normal(center[1], std_dev), center[1] - R, center[1] + R
            )

            dist_to_goal = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            if dist_to_goal < 0.4 or dist_to_goal > R:
                continue

            weight_to_goal = np.exp(
                -0.5 * ((dist_to_goal - preferred_dist) / dist_sigma) ** 2
            )

            weight_to_mean = 1.0
            if preferred_mean is not None:
                dist_to_mean = np.linalg.norm(
                    [x - preferred_mean[0], y - preferred_mean[1]]
                )
                weight_to_mean = np.exp(-0.5 * (dist_to_mean / bias_sigma) ** 2)
            # alpha = 0.5
            weight = alpha * weight_to_goal + (1 - alpha) * weight_to_mean
            map_x = int(x / 0.05) + 100
            map_y = int(y / 0.05) + 100

            if (
                0 <= map_x < 200
                and 0 <= map_y < 200
                and astar_utils.is_valid_des(map_y, map_x, obstacle_map)
            ):
                if np.random.rand() < weight:
                    actions.append((map_x, map_y))
                    i += 1
                    break

            if t > 10000:
                if R > 2:
                    i += 1
                    break
                R += 0.2
                dist_sigma += 0.2

    if preferred_mean is not None:
        map = obstacle_map.copy()
        map = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
        map_x = int(preferred_mean[0] / 0.05) + 100
        map_y = int(preferred_mean[1] / 0.05) + 100
        cv2.circle(map, (map_x, map_y), 1, (0, 255, 0), -1)  # draw the mean on the map
        # save the map with the mean
        cv2.imwrite("./data/mean_map.png", map)

    return actions


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


def annotate_image(image, goal, actions, best_actions=[]):
    annotated_image = image.copy()
    mask_points_mean = np.load("./data/mask_points_mean.npy")
    mask_x = int(mask_points_mean[0] / 0.05) + 100
    mask_y = int(mask_points_mean[1] / 0.05) + 100
    mask_x, mask_y = (199 - mask_y) * 10, (199 - mask_x) * 10

    for i, (x, y) in enumerate(actions):
        color = (0, 0, 255)
        thickness = 1
        x, y = (199 - y) * 10, (199 - x) * 10
        # cv2.arrowedLine(
        #    annotated_image, (x, y), (mask_x, mask_y), color, thickness, tipLength=0.1
        # )
        cv2.circle(annotated_image, (x, y), 15, (255, 255, 255), -1)
        cv2.circle(annotated_image, (x, y), 15, (225, 0, 0), 2)
        text_width, text_height = cv2.getTextSize(
            f"{i}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )[0]
        cv2.putText(
            annotated_image,
            f"{i}",
            (x - text_width // 2, y + text_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 100, 150),
            2,
        )

    overlay = annotated_image.copy()

    #   for i, (x, y) in enumerate(actions):
    #       color = (0, 0, 255)
    #       thickness = 1
    #       x, y = (199 - y) * 10, (199 - x) * 10
    #       cv2.arrowedLine(overlay, (1000, 1000), (x, y), color, thickness, tipLength=0.1)

    # === 新增：從 (1000, 1000) 朝 goal 畫出三條箭頭 ===
    start = (1000, 1000)
    vector = np.array([goal[0] - start[0], goal[1] - start[1]])
    vector = vector / np.linalg.norm(vector) * 100  # 控制箭頭長度

    angle_list = [0]
    color_map = {
        0: (255, 0, 255),  # center - magenta
        30: (0, 255, 0),  # left - green
        -30: (0, 165, 255),  # right - orange
    }

    for angle in angle_list:
        rotated = rotate_vector_2d(vector, angle)
        end_point = (int(start[0] + rotated[0]), int(start[1] + rotated[1]))
        cv2.arrowedLine(overlay, start, end_point, color_map[angle], 8, tipLength=0.2)

    # 加上 overlay 到圖上
    cv2.addWeighted(overlay, 0.5, annotated_image, 0.5, 0, annotated_image)

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
    destination=None,
    max_drift=1.5,
    std_dev_decay=0.9,
):
    if not best_actions_positions:
        return None, bias_sigma

    mean_x = np.mean([x for x, y in best_actions_positions])
    mean_y = np.mean([y for x, y in best_actions_positions])
    mean_x = (mean_x - 100) * 0.05
    mean_y = (mean_y - 100) * 0.05
    if destination is not None:
        dx = destination[0]
        dy = destination[1]
        mean_x = np.clip(mean_x, dx - max_drift, dx + max_drift)
        mean_y = np.clip(mean_y, dy - max_drift, dy + max_drift)
    bias_sigma = max(0.1, std_dev_decay * bias_sigma)

    return (mean_x, mean_y), bias_sigma


def get_dynamic_alpha(i, total_iterations=3, max_alpha=0.6):
    shift = total_iterations / 2
    k = 6 / total_iterations  # 控制斜率，值越大越陡
    sigmoid = 1 / (1 + np.exp(-k * (i - shift)))
    return sigmoid * max_alpha


def get_base(occupancy_2d_map, target, instruction, R, T, fx, fy, cx, cy, K=3):
    # affordance_point, affordance_pixel = get_affordance_point(
    #     target, instruction, R, T, fx, fy, cx, cy, occupancy_2d_map
    # )
    # destination = affordance_point
    # goal = (
    #     (199 - (int(destination[1] / 0.05) + 100)) * 10,
    #     (199 - (int(destination[0] / 0.05) + 100)) * 10,
    # )

    iterations = 4
    parallel = 1
    final_actions = []
    std_dev = 1.0

    for p in range(parallel):
        num_samples = 15
        preferred_mean = None
        bias_sigma = 0.2
        alpha = 0.0
        affordance_point, affordance_pixel = get_affordance_point(
            target, instruction, R, T, fx, fy, cx, cy, occupancy_2d_map
        )
        rgb = cv2.imread("./data/rgb.png")

        mask_points_mean = np.load("./data/mask_points_mean.npy")
        mask_points_mean = np.array(mask_points_mean[0:2])
        get_annotated_image(rgb, mask_points_mean, R, T, fx, fy, cx, cy)
        img_map = cv2.imread("./data/affann.png")
        map_img = process_2d_map(img_map)
        destination = affordance_point
        goal = (
            (199 - (int(destination[1] / 0.05) + 100)) * 10,
            (199 - (int(destination[0] / 0.05) + 100)) * 10,
        )
        center_sigma = 100
        num_samples_center = 15
        for i in range(iterations):
            alpha = get_dynamic_alpha(i, iterations, max_alpha=0.8)
            #  actions = sample_gaussian_actions_on_map(
            #      center=destination,
            #      std_dev=std_dev,
            #      num_samples=num_samples,
            #      image_size=(200, 200),
            #      obstacle_map=occupancy_2d_map,
            #      preferred_mean=preferred_mean,
            #      bias_sigma=bias_sigma,
            #      alpha=alpha,
            #  )
            while True:
                actions = sample_gaussian_actions_on_map(
                    center=destination,
                    std_dev=std_dev,
                    num_samples=num_samples,
                    image_size=(200, 200),
                    obstacle_map=occupancy_2d_map,
                    preferred_mean=preferred_mean,
                    bias_sigma=bias_sigma,
                    alpha=alpha,
                )
                if len(actions) >= 5:
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

            annotated_image = annotate_image(map_img.copy(), goal, actions)
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
            preferred_mean, std_dev = update_gaussian_distribution(
                best_actions_positions, std_dev, destination
            )
            #      destination, affordance_pixel = sample_from_mask_gaussian(
            #          (affordance_pixel[1], affordance_pixel[0]),
            #          target,
            #          center_sigma,
            #          R,
            #          T,
            #          fx,
            #          fy,
            #          cx,
            #          cy,
            #          occupancy_2d_map,
            #          num_samples=num_samples_center,
            #      )
            center_sigma -= 20
            num_samples_center -= 2
        #    final_actions.extend(best_actions_positions)

        for action in best_actions_positions:
            final_actions.append(action)

    preferred_mean, bias_sigma = update_gaussian_distribution(
        final_actions, bias_sigma, destination
    )
    # get the preferred mean map
    preferred_mean_map = occupancy_2d_map.copy()
    preferred_mean_map = cv2.cvtColor(preferred_mean_map, cv2.COLOR_GRAY2BGR)
    map_x = int(preferred_mean[0] / 0.05) + 100
    map_y = int(preferred_mean[1] / 0.05) + 100
    cv2.circle(
        preferred_mean_map, (map_x, map_y), 1, (0, 255, 0), -1
    )  # draw the mean on the map
    cv2.imwrite("./data/preferred_mean_map.png", preferred_mean_map)
    # final_actions = sample_gaussian_actions_on_map(
    #    center=destination,
    #    std_dev=std_dev,
    #    num_samples=10,
    #    image_size=(200, 200),
    #    obstacle_map=occupancy_2d_map,
    #    preferred_mean=preferred_mean,
    #    bias_sigma=0.1,
    #    alpha=0.3,
    # )
    while True:
        final_actions = sample_gaussian_actions_on_map(
            center=destination,
            std_dev=std_dev,
            num_samples=15,
            image_size=(200, 200),
            obstacle_map=occupancy_2d_map,
            preferred_mean=preferred_mean,
            bias_sigma=0.1,
            alpha=0.4,
        )
        if len(final_actions) >= 5:
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
        center_sigma += 20

    final_img = annotate_image(map_img.copy(), goal, final_actions)
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
    base = (base[0] - 100) * 0.05, (base[1] - 100) * 0.05

    return base
