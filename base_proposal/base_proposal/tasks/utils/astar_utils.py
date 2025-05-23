import heapq
import numpy as np
import math


# 定義節點
class Node:
    def __init__(self, x, y, g, h, parent=None):
        self.x = x  # 節點的 X 座標 (左上角的 X)
        self.y = y  # 節點的 Y 座標 (左上角的 Y)
        self.g = g  # 從起點到當前節點的距離
        self.h = h  # 啟發函數，即從當前節點到終點的預估距離
        self.parent = parent  # 父節點
        self.f = g + h  # 總的評估函數 f = g + h

    # 定義小於運算符，用於優先隊列
    def __lt__(self, other):
        return self.f < other.f


# 曼哈頓距離（作為啟發函數）
def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def euclidean_distance(x1, y1, x2, y2):
    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance


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


def is_valid2(x, y, map, radius=8.5):
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


def a_star(map, start, end):
    open_list = []
    closed_list = set()

    start_node = Node(
        start[0], start[1], 0, manhattan_distance(start[0], start[1], end[0], end[1])
    )
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        # 如果到達終點（檢查左上角是否對應終點左上角）
        if (
            euclidean_distance(current_node.x, current_node.y, end[0], end[1]) * 0.05
            <= 0.05
        ):
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        closed_list.add((current_node.x, current_node.y))

        # 定義上下左右的鄰居移動
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for move in neighbors:
            new_x, new_y = current_node.x + move[0], current_node.y + move[1]

            # 檢查機器人 2x2 區域是否可以移動到新位置
            if is_valid(new_x, new_y, map) and (new_x, new_y) not in closed_list:
                new_g = current_node.g + 1
                new_h = manhattan_distance(new_x, new_y, end[0], end[1])
                new_node = Node(new_x, new_y, new_g, new_h, current_node)

                # 檢查是否在開放列表中，且找到更短的路徑

                if not any(
                    node.x == new_x and node.y == new_y and node.g <= new_g
                    for node in open_list
                ):
                    heapq.heappush(open_list, new_node)

    return None


def a_star_rough(map, start, end, R=0.8):
    open_list = []
    closed_list = set()

    start_node = Node(
        start[0], start[1], 0, manhattan_distance(start[0], start[1], end[0], end[1])
    )
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        # 如果到達目標附近（距離小於 0.7 公尺）
        if (
            euclidean_distance(current_node.x, current_node.y, end[0], end[1]) * 0.05
            <= R + 0.1
            and euclidean_distance(current_node.x, current_node.y, end[0], end[1])
            * 0.05
            >= R - 0.1
            and is_valid_des(current_node.x, current_node.y, map)
        ):
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]  # 返回從起點到終點的路徑

        closed_list.add((current_node.x, current_node.y))

        # 定義上下左右的鄰居移動
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for move in neighbors:
            new_x, new_y = current_node.x + move[0], current_node.y + move[1]

            # 檢查機器人 5x5 區域是否可以移動到新位置
            if is_valid(new_x, new_y, map) and (new_x, new_y) not in closed_list:
                new_g = current_node.g + 1  # 距離增加1
                new_h = manhattan_distance(new_x, new_y, end[0], end[1])
                new_node = Node(new_x, new_y, new_g, new_h, current_node)

                # 檢查是否在開放列表中，且找到更短的路徑
                if not any(
                    node.x == new_x and node.y == new_y and node.g <= new_g
                    for node in open_list
                ):
                    heapq.heappush(open_list, new_node)

    return None  # 如果無法找到路徑


# 測試地圖 (使用 numpy array)
# map = np.array([
#    [0, 0, 1, 0, 0, 0],
#    [0, 0, 1, 0, 0, 0],
#    [0, 0, 0, 0, 1, 0],
#    [1, 1, 0, 0, 1, 0],
#    [0, 0, 0, 0, 0, 0],
#    [0, 0, 0, 1, 1, 0],
# ])
#
## 起點和終點分別是機器人占用的 2x2 區域的左上角
# start = (0, 0)
# end = (4, 4)
#
## 執行 A* 演算法
# path = a_star(map, start, end)
# if path:
#    print("找到的路徑:", path)
# else:
#    print("無法找到路徑")
#
