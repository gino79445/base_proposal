import numpy as np
import cv2
from base_proposal.vlm.spaceAware_pivot import get_point
from base_proposal.tasks.utils import astar_utils


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
    dist_sigma = 0.2
    R = 1.2

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
            alpha = 0.5
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

            if t > 1000:
                R += 0.2
                break
        # draw the mean on the map
    if preferred_mean is not None:
        map = obstacle_map.copy()
        map = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)
        map_x = int(preferred_mean[0] / 0.05) + 100
        map_y = int(preferred_mean[1] / 0.05) + 100
        cv2.circle(map, (map_x, map_y), 1, (0, 255, 0), -1)  # draw the mean on the map
        # save the map with the mean
        cv2.imwrite("./data/mean_map.png", map)

    return actions


def annotate_image(image, goal, actions, best_actions=[]):
    annotated_image = image.copy()
    for i, (x, y) in enumerate(actions):
        color = (0, 0, 255)
        thickness = 1
        x, y = (199 - y) * 10, (199 - x) * 10

        # cv2.arrowedLine(annotated_image, goal, (x, y), color, thickness, tipLength=0.2)
        cv2.arrowedLine(
            annotated_image, (1000, 1000), (x, y), color, thickness, tipLength=0.2
        )
        cv2.circle(annotated_image, (x, y), 12, (255, 255, 255), -1)
        cv2.circle(annotated_image, (x, y), 12, (0, 0, 255), 1)
        text_width, text_height = cv2.getTextSize(
            f"{i}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )[0]
        cv2.putText(
            annotated_image,
            f"{i}",
            (x - text_width // 2, y + text_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    crop_size = 300
    cv2.circle(annotated_image, goal, 5, (255, 0, 255), -1)
    cv2.circle(annotated_image, (1000, 1000), 20, (255, 0, 0), -1)
    # save the annotated image
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
    max_drift=1,
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


def get_base(occupancy_2d_map, destination, instruction, K=3):
    img_map = cv2.imread("./data/affann.png")
    map_img = process_2d_map(img_map)
    image_size = map_img.shape[:2][::-1]

    goal = (
        (199 - (int(destination[1] / 0.05) + 100)) * 10,
        (199 - (int(destination[0] / 0.05) + 100)) * 10,
    )
    iterations = 3
    parallel = 3
    final_actions = []
    std_dev = 1.0

    for p in range(parallel):
        num_samples = 15
        preferred_mean = None
        bias_sigma = 0.2
        alpha = 0.8
        for i in range(iterations):
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
            annotated_image = annotate_image(map_img.copy(), goal, actions)
            cv2.imwrite(f"./data/annotated_map_{i + 1}.png", annotated_image)
            result = get_point(
                "./data/rgb.png", f"./data/annotated_map_{i + 1}.png", instruction, K
            )
            best_actions_positions = [actions[idx] for idx in result]
            preferred_mean, std_dev = update_gaussian_distribution(
                best_actions_positions, std_dev, destination
            )
            alpha -= 0.3
        #    final_actions.extend(best_actions_positions)

        for action in best_actions_positions:
            final_actions.append(action)

    preferred_mean, bias_sigma = update_gaussian_distribution(
        final_actions, bias_sigma, destination
    )
    final_actions = sample_gaussian_actions_on_map(
        center=destination,
        std_dev=std_dev,
        num_samples=15,
        image_size=(200, 200),
        obstacle_map=occupancy_2d_map,
        preferred_mean=preferred_mean,
        bias_sigma=bias_sigma,
        alpha=0.2,
    )
    final_img = annotate_image(map_img.copy(), goal, final_actions)
    cv2.imwrite("./data/final_annotated_map.png", final_img)
    result = get_point(
        "./data/affrgb.png", "./data/final_annotated_map.png", instruction, 1
    )
    print(f"Final Best Action Index -> {result}")
    base = final_actions[result[0]]
    base = (base[0] - 100) * 0.05, (base[1] - 100) * 0.05
    return base
