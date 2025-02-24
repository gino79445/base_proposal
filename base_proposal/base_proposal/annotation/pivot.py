import numpy as np
import cv2
import random
from base_proposal.vlm.pivot import get_point 
from base_proposal.tasks.utils import astar_utils


def get_3d_point( u, v, Z, R, T, fx, fy, cx, cy):
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

def sample_gaussian_actions(mean, std_dev, num_samples, image_size, depth_image, obstacle_map, R, T, fx, fy, cx, cy):
    """
    根據 isotropic Gaussian 分佈生成候選動作點
    """
    actions = []
    w = image_size[0]
    h = image_size[1]
    i = 0
    time = 0
    while i < num_samples:
        x = int(np.clip(np.random.normal(mean[0], std_dev), 0, image_size[0] - 1))
        y = int(np.clip(np.random.normal(mean[1], std_dev), 0, image_size[1] - 1))

  #      point_3d = get_3d_point(w - x, h - y, depth_image[y, x], R, T, fx, fy, cx, cy)
  #      x_map = int(point_3d[0]/0.05 + 100)
  #      y_map = int(point_3d[1]/0.05 + 100)
        time += 1
        if time > 100:
            break

        if x >= 20 and x < w -20 and y >= 20 and y < h-20:
            actions.append((x, y))
            i += 1
            time = 0
  #  cv2.imwrite(f"/home/gino79445/Desktop/Research/base_proposal/base_proposal/data/obstacle_map.png", obstacle_map)
    return actions

def annotate_image(image, robot_base, actions, best_actions=[]):
    """
    在圖片上標示候選動作點，並使用箭頭指示方向，最佳點用不同顏色標示
    """
    annotated_image = image.copy()

    for i, (x, y) in enumerate(actions):
        color = (0, 0, 255)  # 預設為紅色箭頭
        thickness = 1

        #if i in best_actions:
        #    color = (255, 0, 0)  # 藍色箭頭
        #    thickness = 2

        # 畫箭頭（從機器人基底位置出發）
        cv2.arrowedLine(annotated_image, robot_base, (x, y), color, thickness, tipLength=0.2)

        # 畫標記圓圈
        cv2.circle(annotated_image, (x, y), 14, (255, 255, 255), -1)
        cv2.circle(annotated_image, (x, y), 14, (0, 0, 255), 1)

        # 標記數字
        text_width, text_height = cv2.getTextSize(f"{i}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.putText(annotated_image, f"{i}", (x - text_width // 2, y + text_height // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return annotated_image

def update_gaussian_distribution(best_actions_positions, std_dev, std_dev_decay=0.8):
    """
    使用選擇的最佳動作更新 Gaussian 分佈（讓其更集中）
    """
    if not best_actions_positions:
        return None, std_dev  # 如果沒有選擇最佳點，則不更新

    # 計算新的均值（使用5個最佳點的平均）
    new_mean_x = int(np.mean([x for x, y in best_actions_positions]))
    new_mean_y = int(np.mean([y for x, y in best_actions_positions]))
    
    new_std_dev = max(10, std_dev_decay * std_dev)  # 避免 std_dev 太小
    return (new_mean_x, new_mean_y), new_std_dev

def get_base(image_path, instruction, depth_image, obstacle_map, R, T, fx, fy, cx, cy, K=3):
    # 讀取圖片
    image = cv2.imread(image_path)
    iterations = 3  # 迭代次數
    parrallel = 3
    final_actions = []
    for p in range(parrallel):
        image_size = image.shape[:2][::-1]  # 圖片大小（寬度, 高度）
        robot_base = (image_size[0] // 2, image_size[1] - 0)  # 機器人基底位置
        initial_mean = robot_base  # 初始均值從機器人位置開始
        num_samples = 15  # 每次迭代生成的候選點數量
        std_dev = 200
        for i in range(iterations):
            actions = sample_gaussian_actions(initial_mean, std_dev, num_samples, image_size, depth_image, obstacle_map, R, T, fx, fy, cx, cy)

            # 選擇 5 個最佳點（使用標記的數字來選擇）
            #best_indices = random.sample(range(num_samples), 3)  # 隨機選 5 個最佳標記
            #best_actions_positions = [actions[idx] for idx in best_indices]  # 透過數字獲取位置

            # 標記圖片
            annotated_image = annotate_image(image, robot_base, actions)
            # write the image
            cv2.imwrite(f"/home/gino79445/Desktop/Research/base_proposal/base_proposal/data/annotated_image_{i+1}.png", annotated_image)
            result = get_point(f"/home/gino79445/Desktop/Research/base_proposal/base_proposal/data/annotated_image_{i+1}.png", instruction, K)
            best_indices = result
            #print(actions)
            #print(best_indices)
            best_actions_positions =  [actions[idx] for idx in best_indices] 
            #print(f"Iteration {i+1}: Best Actions Indices -> {best_indices}")
            #print(f"Iteration {i+1}: Best Actions Positions -> {best_actions_positions}")
            # 更新 Gaussian 分佈
            initial_mean, std_dev = update_gaussian_distribution(best_actions_positions, std_dev)
            num_samples -= 4
        for action in best_actions_positions:
            final_actions.append(action)

    initial_mean, std_dev = update_gaussian_distribution(final_actions, std_dev)
    final_actions = sample_gaussian_actions(initial_mean, std_dev, 3, image_size, depth_image, obstacle_map, R, T, fx, fy, cx, cy)

    # 標記最終圖片
    annotated_image = annotate_image(image, robot_base, final_actions)
    # write the image
    cv2.imwrite(f"/home/gino79445/Desktop/Research/base_proposal/base_proposal/data/final_annotated_image.png", annotated_image)
    result = get_point(f"/home/gino79445/Desktop/Research/base_proposal/base_proposal/data/final_annotated_image.png", instruction, 1)
    print(f"Final Best Action Index -> {result}")
    position = final_actions[result[0]]
    return position


#r = get_base("/home/gino79445/Desktop/Research/base_proposal/base_proposal/data/rgb.png", "Take out the cookies from the cabinet.", 3)
#print(r)
## 讀取圖片
#image = cv2.imread("/home/gino79445/Desktop/Research/base_proposal/base_proposal/data/rgb.png")
#iterations = 3  # 迭代次數
#parrallel = 3
#final_actions = []
#for p in range(parrallel):
#    image_size = image.shape[:2][::-1]  # 圖片大小（寬度, 高度）
#    robot_base = (image_size[0] // 2, image_size[1] - 0)  # 機器人基底位置
#    initial_mean = robot_base  # 初始均值從機器人位置開始
#    num_samples = 15  # 每次迭代生成的候選點數量
#    std_dev = 200  # 初始標準差
#    for i in range(iterations):
#        actions = sample_gaussian_actions(initial_mean, std_dev, num_samples, image_size)
#
#        # 選擇 5 個最佳點（使用標記的數字來選擇）
#        #best_indices = random.sample(range(num_samples), 3)  # 隨機選 5 個最佳標記
#        #best_actions_positions = [actions[idx] for idx in best_indices]  # 透過數字獲取位置
#
#        # 標記圖片
#        annotated_image = annotate_image(image, robot_base, actions)
#        # write the image
#        cv2.imwrite(f"/home/gino79445/Desktop/Research/base_proposal/base_proposal/data/annotated_image_{i+1}.png", annotated_image)
#        result = get_point(f"/home/gino79445/Desktop/Research/base_proposal/base_proposal/data/annotated_image_{i+1}.png", "Take out the cookies from the cabinet.", 3)
#        best_indices = result
#        best_actions_positions =  [actions[idx] for idx in best_indices] 
#        print(f"Iteration {i+1}: Best Actions Indices -> {best_indices}")
#        print(f"Iteration {i+1}: Best Actions Positions -> {best_actions_positions}")
#        # 更新 Gaussian 分佈
#        initial_mean, std_dev = update_gaussian_distribution(best_actions_positions, std_dev)
#        num_samples -= 4
#
#    actions = sample_gaussian_actions(initial_mean, std_dev, 1, image_size)
#    final_actions.append(actions[0])
#
## 標記最終圖片
#annotated_image = annotate_image(image, robot_base, final_actions)
## write the image
#cv2.imwrite(f"/home/gino79445/Desktop/Research/base_proposal/base_proposal/data/final_annotated_image.png", annotated_image)
#result = get_point(f"/home/gino79445/Desktop/Research/base_proposal/base_proposal/data/final_annotated_image.png", "Take out the cookies from the cabinet.", 1)
#print(f"Final Best Action Index -> {result}")
# 顯示圖片
#cv2.imshow("Annotated Image", annotated_image)
#cv2.waitKey(0)
#
#cv2.destroyAllWindows()

