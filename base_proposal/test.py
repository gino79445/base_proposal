import numpy as np
import random
import yaml
import os
from scipy.spatial.transform import Rotation as R


def se3_to_matrix_object(pos, euler_rx_ry_rz):
    rot = R.from_euler("xyz", euler_rx_ry_rz)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = pos
    return T


def se3_to_matrix_ee(pos, euler_rz_ry_rx):
    rot = R.from_euler("zyx", euler_rz_ry_rx)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = pos
    return T


def matrix_to_se3_ee(T):
    pos = np.round(T[:3, 3], 4).tolist()
    rot = np.round(R.from_matrix(T[:3, :3]).as_euler("zyx"), 4).tolist()
    return [pos, rot]


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    else:
        return obj


def generate_yaml(
    object_names: list,
    yaml_path: str,
    output_dir: str,
    count: int = 20,
    random_seed: int = 42,  # ✅ 加入 seed 參數
):
    os.makedirs(output_dir, exist_ok=True)

    # ✅ 固定 random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    zones = [
        {"name": "long_table", "x": (0.0, -0.2), "y": (-2.0, -1.2), "z": 0.57},
        {"name": "tv_table", "x": (1.75, 1.85), "y": (-2.6, -2.3), "z": 0.99},
        {"name": "big_table", "x": (1.42, 1.5), "y": (0.8, 0.9), "z": 0.99},
        {"name": "cabinet", "x": (-0.35, -0.6), "y": (-4.5, -4.3), "z": 1.01},
    ]

    for i in range(count):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        new_positions = {}
        new_rotations = {}
        relative_dest = []

        original_positions = {
            name: data["targets_position"][data["target"].index(name)][0][:2]
            for name in data["target"]
        }

        if "destination" in data and len(data["destination"]) > 0:
            relative_dest = []
            for i_dest, dest in enumerate(data["destination"]):
                if i_dest >= len(data["target"]):
                    break
                from_name = data["target"][i_dest]
                from_pos = original_positions[from_name]
                if len(dest) == 1:
                    dx = dest[0][0] - from_pos[0]
                    dy = dest[0][1] - from_pos[1]
                    relative_dest.append([[dx, dy]])

        for obj_name in object_names:
            if obj_name not in data["target"]:
                continue

            obj_idx = data["target"].index(obj_name)
            ee_se3_list = data["targets_se3"][obj_idx]
            obj_pos, obj_rot = data["targets_position"][obj_idx]
            T_obj = se3_to_matrix_object(obj_pos, obj_rot)

            relative_transforms = []
            for ee_se3 in ee_se3_list:
                T_ee = se3_to_matrix_ee(*ee_se3)
                T_rel = np.linalg.inv(T_obj) @ T_ee
                relative_transforms.append(T_rel)

            zone = random.choice(zones)
            x = round(random.uniform(*sorted(zone["x"])), 4)
            y = round(random.uniform(*sorted(zone["y"])), 4)
            z = zone["z"]

            if zone["name"] == "long_table":
                rz = round(random.uniform(np.pi / 2, 3 * np.pi / 4), 4)
            elif zone["name"] == "tv_table":
                rz = round(random.uniform(0, np.pi / 2), 4)
            elif zone["name"] == "big_table":
                rz = round(random.uniform(-np.pi / 2, np.pi / 2), 4)
            elif zone["name"] == "cabinet":
                rz = round(random.uniform(np.pi / 2, 3 * np.pi / 4), 4)

            new_pos = [x, y, z]
            new_rot = [0, 0, rz]
            T_new_obj = se3_to_matrix_object(new_pos, new_rot)

            new_ee_se3_list = []
            for T_rel in relative_transforms:
                T_new_ee = T_new_obj @ T_rel
                new_ee_se3_list.append(matrix_to_se3_ee(T_new_ee))

            data["targets_position"][obj_idx] = [new_pos, new_rot]
            data["targets_se3"][obj_idx] = new_ee_se3_list
            new_positions[obj_name] = new_pos
            new_rotations[obj_name] = rz

        updated_dest = []
        for i_dest, rel in enumerate(relative_dest):
            if i_dest >= len(data["target"]):
                continue
            from_name = data["target"][i_dest]
            from_pos = new_positions.get(from_name, original_positions[from_name])
            rz = new_rotations.get(from_name, 0.0)

            Rz = np.array([[np.cos(rz), -np.sin(rz)], [np.sin(rz), np.cos(rz)]])

            if len(rel) == 1:
                offset = np.array(rel[0])
                rotated = Rz @ offset
                updated_dest.append(
                    [
                        [
                            round(from_pos[0] + rotated[0], 4),
                            round(from_pos[1] + rotated[1], 4),
                        ]
                    ]
                )

        data["destination"] = updated_dest
        data = convert_numpy(data)

        output_path = os.path.join(output_dir, f"{i + 1:02d}_dataset.yaml")
        with open(output_path, "w") as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)

    print(f"✅ 已產出 {count} 筆資料到：{output_dir}")


generate_yaml(
    object_names=["black_mug"],
    yaml_path="base_proposal/cfg/env/pickplace_mug_shelf.yaml",
    output_dir="base_proposal/cfg/env/pickplace_mug_shelf",
    count=20,
    random_seed=1,  # ✅ 固定這個值就能重現結果
)
