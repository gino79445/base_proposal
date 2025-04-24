import numpy as np
import math
import random


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0


def distance(node1, node2):
    return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)


# def is_valid(x, y, Map):
#    # 檢查點是否在地圖內，並且是可通行的
#    for i in range(-6, 6):
#        for j in range(-6, 6):
#            if x + i < 0 or y + j < 0 or x + i >= len(Map) or y + j >= len(Map[0]) or Map[x + i][y + j] != 0:
#                return False
#    return True


def is_valid_des(x, y, map, radius=9):
    r_int = math.ceil(radius)
    for i in range(-r_int, r_int + 1):
        for j in range(-r_int, r_int + 1):
            # ✅ 只檢查圓形內的點 (i, j)
            if i**2 + j**2 > radius**2:
                continue  # 忽略圓外的格子

            # ✅ 檢查是否超出邊界或為障礙物
            if (
                x + i < 0
                or y + j < 0
                or x + i >= len(map)
                or y + j >= len(map[0])
                or map[x + i][y + j] != 0
            ):
                return False
    return True


def is_valid(x, y, map, radius=8):
    r_int = math.ceil(radius)
    for i in range(-r_int, r_int + 1):
        for j in range(-r_int, r_int + 1):
            # ✅ 只檢查圓形內的點 (i, j)
            if i**2 + j**2 > radius**2:
                continue  # 忽略圓外的格子

            # ✅ 檢查是否超出邊界或為障礙物
            if (
                x + i < 0
                or y + j < 0
                or x + i >= len(map)
                or y + j >= len(map[0])
                or map[x + i][y + j] != 0
            ):
                return False
    return True


def get_nearest_node(node_list, random_node):
    nearest_node = node_list[0]
    min_dist = distance(nearest_node, random_node)
    for node in node_list:
        dist = distance(node, random_node)
        if dist < min_dist:
            nearest_node = node
            min_dist = dist
    return nearest_node


def steer(from_node, to_node, step_size):
    angle = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
    new_x = from_node.x + step_size * math.cos(angle)
    new_y = from_node.y + step_size * math.sin(angle)
    new_node = Node(int(new_x), int(new_y))
    new_node.parent = from_node
    return new_node


def get_nearby_nodes(node_list, new_node, radius):
    nearby_nodes = []
    for node in node_list:
        if distance(node, new_node) <= radius:
            nearby_nodes.append(node)
    return nearby_nodes


def choose_parent(nearby_nodes, new_node, Map):
    if not nearby_nodes:
        return None
    min_cost = float("inf")
    best_parent = None
    for node in nearby_nodes:
        temp_cost = node.cost + distance(node, new_node)
        if temp_cost < min_cost and is_valid(new_node.x, new_node.y, Map):
            min_cost = temp_cost
            best_parent = node
    new_node.cost = min_cost
    return best_parent


def rewire(nearby_nodes, new_node, Map):
    for node in nearby_nodes:
        new_cost = new_node.cost + distance(new_node, node)
        if new_cost < node.cost and is_valid(node.x, node.y, Map):
            node.parent = new_node
            node.cost = new_cost


def extract_path(end_node):
    path = []
    current = end_node
    while current is not None:
        path.append((current.x, current.y))
        current = current.parent
    return path[::-1]


def rrt_star_rough(
    Map,
    start,
    goal,
    max_iter=6000,
    step_size=2,
    goal_sample_rate=0.15,
    radius=15,
    R=0.8,
):
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])
    node_list = [start_node]

    for _ in range(max_iter):
        if random.random() < goal_sample_rate:
            random_node = goal_node
        else:
            random_x = random.randint(0, Map.shape[0] - 1)
            random_y = random.randint(0, Map.shape[1] - 1)
            random_node = Node(random_x, random_y)

        nearest_node = get_nearest_node(node_list, random_node)
        new_node = steer(nearest_node, random_node, step_size)

        if is_valid(new_node.x, new_node.y, Map):
            nearby_nodes = get_nearby_nodes(node_list, new_node, radius)
            best_parent = choose_parent(nearby_nodes, new_node, Map)

            if best_parent:
                new_node.parent = best_parent
                node_list.append(new_node)
                rewire(nearby_nodes, new_node, Map)

            if (
                distance(new_node, goal_node) * 0.05 <= R + 0.1
                and distance(new_node, goal_node) * 0.05 >= R - 0.1
            ):
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + distance(new_node, goal_node)
                node_list.append(goal_node)
                return extract_path(goal_node)

    return None


def rrt_star_target(
    Map, start, goal, max_iter=1000, step_size=2, goal_sample_rate=0.2, radius=15, R=0.8
):
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])
    node_list = [start_node]

    for _ in range(max_iter):
        if random.random() < goal_sample_rate:
            random_node = goal_node
        else:
            random_x = random.randint(0, Map.shape[0] - 1)
            random_y = random.randint(0, Map.shape[1] - 1)
            random_node = Node(random_x, random_y)

        nearest_node = get_nearest_node(node_list, random_node)
        new_node = steer(nearest_node, random_node, step_size)

        if is_valid(new_node.x, new_node.y, Map):
            nearby_nodes = get_nearby_nodes(node_list, new_node, radius)
            best_parent = choose_parent(nearby_nodes, new_node, Map)

            if best_parent:
                new_node.parent = best_parent
                node_list.append(new_node)
                rewire(nearby_nodes, new_node, Map)

            if (
                distance(new_node, goal_node) * 0.05 <= R + 0.1
                and distance(new_node, goal_node) * 0.05 >= R - 0.1
                and is_valid_des(new_node.x, new_node.y, Map, radius=9)
            ):
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + distance(new_node, goal_node)
                node_list.append(goal_node)
                return extract_path(goal_node)

    return None


## 測試地圖
# Map = np.zeros((100, 100))
# Map[30:70, 40:60] = 1  # 障礙物
#
# start = (10, 10)
# goal = (90, 90)
#
# path = rrt_star(Map, start, goal)
# if path:
#    print("找到的路徑:", path)
# else:
#    print("無法找到路徑")
