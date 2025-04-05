import numpy as np
import cv2
import random
from base_proposal.vlm.pivot import get_point
from base_proposal.tasks.utils import astar_utils


def process_2d_map(occupancy_2d_map):
    occupancy_2d_map = occupancy_2d_map.copy()
    occupancy_2d_map = np.flipud(occupancy_2d_map)
    occupancy_2d_map = np.rot90(occupancy_2d_map)
    occupancy_2d_map = cv2.cvtColor(occupancy_2d_map, cv2.COLOR_GRAY2BGR)
    scale = 10
    occupancy_2d_map = cv2.resize(occupancy_2d_map, (200 * scale, 200 * scale))
    return occupancy_2d_map


def sample_gaussian_actions_on_map(
    mean, std_dev, num_samples, image_size, obstacle_map
):
    actions = []
    w, h = image_size
    i = 0
    time = 0
    while i < num_samples:
        x = np.clip(np.random.normal(mean[0], std_dev), -10, 10)
        y = np.clip(np.random.normal(mean[1], std_dev), -10, 10)

        x = int(x / 0.05) + 100
        y = int(y / 0.05) + 100

        if 0 <= x < 200 and 0 <= y < 200:
            if astar_utils.is_valid(y, x, obstacle_map):
                actions.append((x, y))
                i += 1
                time = 0
            else:
                time += 1
        else:
            time += 1

        if time > 100:
            break
    return actions


def annotate_image(image, robot_base, actions, best_actions=[]):
    annotated_image = image.copy()
    for i, (x, y) in enumerate(actions):
        color = (0, 0, 255)
        thickness = 1
        x, y = (199 - y) * 10, (199 - x) * 10

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


def update_gaussian_distribution(best_actions_positions, std_dev, std_dev_decay=0.8):
    if not best_actions_positions:
        return None, std_dev
    new_mean_x = int(np.mean([x for x, y in best_actions_positions]))
    new_mean_y = int(np.mean([y for x, y in best_actions_positions]))
    new_std_dev = max(10, std_dev_decay * std_dev)
    return (new_mean_x, new_mean_y), new_std_dev


def get_base(occupancy_2d_map, destination, instruction, K=3):
    map_img = process_2d_map(occupancy_2d_map)
    image_size = map_img.shape[:2][::-1]

    robot_base = (
        (199 - (int(destination[1] / 0.05) + 100)) * 10,
        (199 - (int(destination[0] / 0.05) + 100)) * 10,
    )
    iterations = 3
    parallel = 3
    final_actions = []
    initial_mean = destination
    std_dev = 0.8
    num_samples = 15

    for p in range(parallel):
        for i in range(iterations):
            actions = sample_gaussian_actions_on_map(
                initial_mean, std_dev, num_samples, (200, 200), occupancy_2d_map
            )
            annotated_image = annotate_image(map_img.copy(), robot_base, actions)
            cv2.imwrite(f"./data/annotated_map_{i + 1}.png", annotated_image)
            result = get_point(f"./data/annotated_map_{i + 1}.png", instruction, K)
            best_actions_positions = [actions[idx] for idx in result]
            initial_mean, std_dev = update_gaussian_distribution(
                best_actions_positions, std_dev
            )
            num_samples -= 4
            final_actions.extend(best_actions_positions)

    initial_mean, std_dev = update_gaussian_distribution(final_actions, std_dev)
    final_actions = sample_gaussian_actions_on_map(
        initial_mean, std_dev, 3, image_size, occupancy_2d_map
    )
    final_img = annotate_image(map_img.copy(), robot_base, final_actions)
    cv2.imwrite(f"./data/final_annotated_map.png", final_img)
    result = get_point(f"./data/final_annotated_map.png", instruction, 1)
    print(f"Final Best Action Index -> {result}")
    return final_actions[result[0]]
