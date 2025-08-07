import numpy as np
import cv2
import random
from base_proposal.vlm.pivot import get_point
from base_proposal.tasks.utils import astar_utils


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


def sample_gaussian_actions(
    mean,
    std_dev,
    num_samples,
    image_size,
    depth_image,
    obstacle_map,
    R,
    T,
    fx,
    fy,
    cx,
    cy,
):
    actions = []
    w = image_size[0]
    h = image_size[1]
    i = 0
    time = 0
    while i < num_samples:
        x = int(np.clip(np.random.normal(mean[0], std_dev), 0, image_size[0] - 1))
        y = int(np.clip(np.random.normal(mean[1], std_dev), 0, image_size[1] - 1))

        point_3d = get_3d_point(w - x, h - y, depth_image[y, x], R, T, fx, fy, cx, cy)
        x_map = int(point_3d[0] / 0.05 + 100)
        y_map = int(point_3d[1] / 0.05 + 100)
        time += 1
        if time > 100:
            break

        if (
            astar_utils.is_valid(y_map, x_map, obstacle_map)
            and x >= 20
            and x < w - 20
            and y >= 20
            and y < h - 20
        ):
            actions.append((x, y))
            i += 1
            time = 0
    cv2.imwrite(f"./data/obstacle_map.png", obstacle_map)
    return actions


def annotate_image(image, robot_base, actions, best_actions=[]):
    annotated_image = image.copy()

    for i, (x, y) in enumerate(actions):
        color = (0, 0, 255)
        thickness = 1

        cv2.arrowedLine(
            annotated_image, robot_base, (x, y), color, thickness, tipLength=0.2
        )

        cv2.circle(annotated_image, (x, y), 14, (255, 255, 255), -1)
        cv2.circle(annotated_image, (x, y), 14, (0, 0, 255), 1)

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

    return annotated_image


def update_gaussian_distribution(best_actions_positions, std_dev, std_dev_decay=1):
    if not best_actions_positions:
        return None, std_dev

    new_mean_x = int(np.mean([x for x, y in best_actions_positions]))
    new_mean_y = int(np.mean([y for x, y in best_actions_positions]))

    new_std_dev = max(10, std_dev_decay * std_dev)  # 避免 std_dev 太小
    return (new_mean_x, new_mean_y), new_std_dev


def get_base(
    image_path, instruction, depth_image, obstacle_map, R, T, fx, fy, cx, cy, K=3
):
    # 讀取圖片
    image = cv2.imread(image_path)
    iterations = 3
    parrallel = 3
    final_actions = []
    for p in range(parrallel):
        image_size = image.shape[:2][::-1]
        robot_base = (image_size[0] // 2, image_size[1] - 0)
        initial_mean = robot_base
        num_samples = 15
        std_dev = 300
        for i in range(iterations):
            actions = sample_gaussian_actions(
                initial_mean,
                std_dev,
                num_samples,
                image_size,
                depth_image,
                obstacle_map,
                R,
                T,
                fx,
                fy,
                cx,
                cy,
            )

            annotated_image = annotate_image(image, robot_base, actions)
            cv2.imwrite(f"./data/annotated_image_{i + 1}.png", annotated_image)
            result = get_point(f"./data/annotated_image_{i + 1}.png", instruction, K)
            best_indices = result
            best_actions_positions = [actions[idx] for idx in best_indices]
            initial_mean, std_dev = update_gaussian_distribution(
                best_actions_positions, std_dev
            )
            num_samples -= 4
        for action in best_actions_positions:
            final_actions.append(action)

    initial_mean, std_dev = update_gaussian_distribution(final_actions, std_dev)
    final_actions = sample_gaussian_actions(
        initial_mean,
        std_dev,
        3,
        image_size,
        depth_image,
        obstacle_map,
        R,
        T,
        fx,
        fy,
        cx,
        cy,
    )

    annotated_image = annotate_image(image, robot_base, final_actions)
    # write the image
    cv2.imwrite(f"./data/final_annotated_image.png", annotated_image)
    result = get_point(f"./data/final_annotated_image.png", instruction, 1)
    print(f"Final Best Action Index -> {result}")
    position = final_actions[result[0]]
    return position
