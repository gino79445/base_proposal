import numpy as np
import pinocchio as pin
import os
import random
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d


class MotionPlannerTiago:
    def __init__(
            self,
            urdf_name: str = "tiago_dual_holobase.urdf",
            move_group: str = "arm_left", # Can be 'arm_right' or 'arm_left'
            include_torso: bool = False, # Use torso in the IK solution
            include_base: bool = False, # Use base in the IK solution
            max_rot_vel: float = 1.0472, # Maximum rotational velocity of all joints
            baseXY_range: np.ndarray = np.array([[-5,5],[-5,5]]), # Range of baseXY
        ) -> None:
        
        # Settings
        self.damp = 1e-10 # Damping coefficient to avoid singularities
        self._include_torso = include_torso
        self._include_base = include_base
        self.max_rot_vel = max_rot_vel
        self.baseXY_range = baseXY_range

        # Load URDF
        urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/" + urdf_name
        self.model = pin.buildModelFromUrdf(urdf_file)

        # End-effector selection
        name_end_effector = "gripper_" + move_group[4:] + "_grasping_frame"

        # Define joints of interest
        jointsOfInterest = [move_group+'_1_joint', move_group+'_2_joint',
                            move_group+'_3_joint', move_group+'_4_joint', move_group+'_5_joint',
                            move_group+'_6_joint', move_group+'_7_joint']
        if self._include_torso:
            jointsOfInterest = ['torso_lift_joint'] + jointsOfInterest
        if self._include_base:
            jointsOfInterest = ['X', 'Y', 'R'] + jointsOfInterest

        print('[IK INFO]: Using joints:', jointsOfInterest)

        remove_ids = []
        for jnt in jointsOfInterest:
            if self.model.existJointName(jnt):
                remove_ids.append(self.model.getJointId(jnt))
            else:
                print('[IK WARNING]: joint ' + str(jnt) + ' does not belong to the model!')

        jointIdsToExclude = np.delete(np.arange(0, self.model.njoints), remove_ids)
        reference_configuration = pin.neutral(self.model)
        if not self._include_torso:
            reference_configuration[26] = 0.25 

        self.model = pin.buildReducedModel(self.model, jointIdsToExclude[1:].tolist(), reference_configuration=reference_configuration)
        print(self.model)
        #for joint in self.model.joints:
        #    print(joint)
        assert (len(self.model.joints)==(len(jointsOfInterest)+1)), "[IK Error]: Joints != nDoFs"
        self.model_data = self.model.createData()
        # frame name
        #for frame in self.model.frames:
        #    print(frame.name)

        # Joint limits
        self.joint_pos_min = np.array([-1.1780972451, -1.1780972451, -0.785398163397, -0.392699081699, -2.09439510239, -1.41371669412, -2.09439510239])
        self.joint_pos_max = np.array([+1.57079632679, +1.57079632679, +3.92699081699, +2.35619449019, +2.09439510239, +1.41371669412, +2.09439510239])

        if self._include_torso:
            self.joint_pos_min = np.hstack((np.array([0.0]), self.joint_pos_min))
            self.joint_pos_max = np.hstack((np.array([0.35]), self.joint_pos_max))

        if self._include_base:
            self.joint_pos_min = np.hstack((-0, -0, 0, -0, self.joint_pos_min))
            self.joint_pos_max = np.hstack((0, 0, 1, 0, self.joint_pos_max))

        self.joint_pos_mid = (self.joint_pos_max + self.joint_pos_min) / 2.0
        self.id_EE = self.model.getFrameId(name_end_effector)

    def is_valid(self, q):
        return np.all(q >= self.joint_pos_min) and np.all(q <= self.joint_pos_max )

    def sample_random_q(self):
        q = np.random.uniform(self.joint_pos_min, self.joint_pos_max)
        return q
    def forward_kinematics(self, q):
        """ 計算正向運動學以獲取末端執行器 (EEF) 的位置 """
        pin.forwardKinematics(self.model, self.model_data, q)  # 更新運動學
        pin.updateFramePlacements(self.model, self.model_data)  # 確保 Frame 資訊更新
        base_pose = self.model_data.oMf[self.model.getFrameId("base_link")] 
        eef_pose = self.model_data.oMf[self.id_EE]  # 取得 EEF 的世界座標
        eef_pose = base_pose.act(eef_pose)  # 將 EEF 的座標轉換到 base_link 的座標
        return np.array(eef_pose.translation), np.array([eef_pose.rotation])


    def voxelize_point_cloud(self, point_cloud, voxel_size=0.05):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
        voxels = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        return voxels, voxel_size

    def is_collision_point(self, q, point_cloud, safety_threshold=0.5):
        """ 直接用 Point Cloud 進行碰撞檢測，不使用 Voxel Grid """
        pin.forwardKinematics(self.model, self.model_data, q)
        pin.updateFramePlacements(self.model, self.model_data)

        # 建立 KDTree 加速最近鄰搜索
        kdtree = KDTree(point_cloud)

        for frame in self.model.frames:
            if frame.type == pin.FrameType.BODY:
                link_pose = self.model_data.oMf[frame.parent]
                link_pos = link_pose.translation

                # 查找點雲中最近的點
                dist, _ = kdtree.query(link_pos)
                if dist < safety_threshold:
                    print(f"Collision detected at {link_pos}, distance: {dist}")
                    return True  # 發生碰撞

        return False  # 無碰撞


    def is_collision(self, q, voxel_grid, voxel_size):
        pin.forwardKinematics(self.model, self.model_data, q)
        pin.updateFramePlacements(self.model, self.model_data)

        voxel_set = {tuple(v) for v in voxel_grid}  

        for frame in self.model.frames:
            if frame.type == pin.FrameType.BODY:
                link_pose = self.model_data.oMf[frame.parent]
                link_pos = link_pose.translation
                voxel_index = tuple(np.round(link_pos / voxel_size).astype(int))
                # make the link is bigger than the voxel


                if voxel_index in voxel_set:
                    print("Collision detected at:", link_pos)
                    return True  # 發生碰撞 (考慮膨脹後的障礙物)

        return False




 #   def rrt_motion_plan_with_obstacles(self, start_q, goal_q, point_cloud ,max_iters=1000, step_size=0.1):
 #       """ 使用 RRT 方法規劃運動路徑，並避開障礙物 """

 #       # clip the start and goal configuration
 #       start_q = np.clip(start_q, self.joint_pos_min, self.joint_pos_max)
 #       goal_q = np.clip(goal_q, self.joint_pos_min, self.joint_pos_max)

 #       if not self.is_valid(start_q) or not self.is_valid(goal_q):
 #           print("Invalid start or goal configuration")
 #           return None

 #       tree = {tuple(start_q): None}
 #       nodes = [start_q]

 #       for _ in range(max_iters):
 #           rand_q = self.sample_random_q() if random.random() > 0.1 else goal_q
 #           nearest_q = min(nodes, key=lambda n: np.linalg.norm(n - rand_q))
 #           direction = rand_q - nearest_q
 #           if np.linalg.norm(direction) < 1e-6:
 #               continue  # 避免添加過於接近的點
 #          # new_q = nearest_q + step_size * (direction / (np.linalg.norm(direction) + 1e-6))
 #           new_q = nearest_q + step_size * (direction / (np.linalg.norm(direction) ))
 #           if np.linalg.norm(direction) > 1e-6:
 #               new_q = nearest_q + step_size * (direction / np.linalg.norm(direction))
 #           else:
 #               continue  # 避免 NaN

 #           #  這裡檢查 **整個機械手臂是否碰撞**
 #           if self.is_valid(new_q) and not self.is_collision_point(new_q, point_cloud):
 #           #if self.is_valid(new_q) :
 #               nodes.append(new_q)
 #               tree[tuple(new_q)] = nearest_q

 #               if np.linalg.norm(new_q - goal_q) < step_size:
 #                   return self.extract_path(tree, new_q, point_cloud)
 #       return None


#    def extract_path(self, tree, end_q):
#        path = [end_q]
#        while tuple(path[-1]) in tree and tree[tuple(path[-1])] is not None:
#            path.append(tree[tuple(path[-1])])
#        return path[::-1]

   # def extract_path(self, tree, end_q, point_cloud):
   #     path = [end_q]
   #     while tuple(path[-1]) in tree and tree[tuple(path[-1])] is not None:
   #         path.append(tree[tuple(path[-1])])

   #     path = np.array(path[::-1])
   #     if path.size == 0:
   #         return None

   #     # 生成時間參數 t
   #     t = np.linspace(0, 1, len(path))
   #     smoothed_path = []

   #     for joint_idx in range(path.shape[1]):
   #         cs = CubicSpline(t, path[:, joint_idx])
   #         t_fine = np.linspace(0, 1, len(path) * 5)  # 讓插值後的數據更密
   #         smoothed_path.append(cs(t_fine))

   #     smoothed_path = np.array(smoothed_path).T
   #     smoothed_path = gaussian_filter1d(smoothed_path, sigma=1.0, axis=0)

   #     #  **新增：對平滑後的路徑逐點檢查碰撞**
   #     safe_path = []
   #     for q in smoothed_path:
   #         if not self.is_collision_point(q, point_cloud):
   #             safe_path.append(q)
   #         else:
   #             print(f"Warning: Smoothed path has a collision at {q}")

   #     return safe_path if len(safe_path) > 0 else None

    def extract_path(self, tree, end_q, point_cloud):
        path = [end_q]
        while tuple(path[-1]) in tree and tree[tuple(path[-1])] is not None:
            path.append(tree[tuple(path[-1])])

        path = np.array(path[::-1])
        if path.size == 0:
            return None

        t = np.linspace(0, 1, len(path))
        smoothed_path = []

        for joint_idx in range(path.shape[1]):
            cs = CubicSpline(t, path[:, joint_idx])
            t_fine = np.linspace(0, 1, len(path) * 20)  # 增加插值密度
            smoothed_path.append(cs(t_fine))

        smoothed_path = np.array(smoothed_path).T
        smoothed_path = gaussian_filter1d(smoothed_path, sigma=0.5, axis=0)  # 降低 sigma

        #  **確保平滑後的軌跡無碰撞**
        safe_path = []
        for q in smoothed_path:
            if not self.is_collision_point(q, point_cloud, safety_threshold=0.15):  # 加強碰撞檢查
                safe_path.append(q)
            else:
                print(f"Warning: Smoothed path has a collision at {q}")

        return safe_path if len(safe_path) > 0 else None

    
    def refine_goal_approach(self, path, point_cloud, goal_q, step_size=0.05, max_iters=10):
        if path is None or len(path) < 2:
            return path  # 無可調整的情況

        path = np.array(path)
        refined_path = path.copy()

        for _ in range(max_iters):
            last_q = refined_path[-1]
            direction = goal_q - last_q
            norm_dir = np.linalg.norm(direction)

            if norm_dir < 1e-3:
                break  # 如果已經足夠接近則停止

            new_q = last_q + step_size * (direction / norm_dir)

            if self.is_valid(new_q) and not self.is_collision_point(new_q, point_cloud, safety_threshold=0.1):
                refined_path[-1] = new_q  # 只調整最後一個點
            else:
                break  # 避免進入障礙物區域

        return refined_path


    def rrt_motion_plan_with_obstacles(self, start_q, goal_q, point_cloud, max_iters=1000, step_size=0.1):
        """ 使用 RRT 方法規劃運動路徑，並避開障礙物，最後微調終點 """
        
        start_q = np.clip(start_q, self.joint_pos_min, self.joint_pos_max)
        goal_q = np.clip(goal_q, self.joint_pos_min, self.joint_pos_max)

        if not self.is_valid(start_q) or not self.is_valid(goal_q):
            print("Invalid start or goal configuration")
            return None

        tree = {tuple(start_q): None}
        nodes = [start_q]

        for _ in range(max_iters):
            #  **增加 goal_q 被選中的機率**
            rand_q = self.sample_random_q() if random.random() > 0.3 else goal_q  
            nearest_q = min(nodes, key=lambda n: np.linalg.norm(n - rand_q))

            direction = rand_q - nearest_q
            norm_dir = np.linalg.norm(direction)

            if norm_dir < 1e-6:
                continue  # 避免過近的點

            new_q = nearest_q + step_size * (direction / norm_dir)

            if self.is_valid(new_q) and not self.is_collision_point(new_q, point_cloud, safety_threshold=0.01):
                nodes.append(new_q)
                tree[tuple(new_q)] = nearest_q

                # **當新點已經足夠接近 goal_q，則開始提取路徑**
                if np.linalg.norm(new_q - goal_q) < step_size:
                    path = self.extract_path(tree, new_q, point_cloud)
                    
                    # **最後微調終點，使其更貼近 goal_q**
                    if path is not None:
                        refined_path = self.refine_goal_approach(path, point_cloud, goal_q, step_size=0.05, max_iters=10)
                        # np to list
                        refined_path = refined_path.tolist()
                        return refined_path

        return None



#    def extract_path(self, tree, end_q):
#            path = [end_q]
#            while tuple(path[-1]) in tree and tree[tuple(path[-1])] is not None:
#                path.append(tree[tuple(path[-1])])
#            path = np.array(path[::-1])
#            t = np.linspace(0, 1, len(path))
#            smoothed_path = []
#            for joint_idx in range(path.shape[1]):
#                cs = CubicSpline(t, path[:, joint_idx])
#                t_fine = np.linspace(0, 1, len(path) * 5)
#                smoothed_path.append(cs(t_fine))
#            smoothed_path = np.array(smoothed_path).T
#            smoothed_path = gaussian_filter1d(smoothed_path, sigma=1.0, axis=0)
#            # np to list
#            smoothed_path = smoothed_path.tolist()
#            return smoothed_path
 #   def extract_path(self, tree, end_q):
 #       path = [end_q]
 #       while tuple(path[-1]) in tree and tree[tuple(path[-1])] is not None:
 #           path.append(tree[tuple(path[-1])])
 #       
 #       path = path[::-1]  # Reverse the path to start from the initial point
 #       refined_path = []

 #       for i in range(len(path) - 1):
 #           start = path[i]
 #           end = path[i + 1]
 #           
 #           # Interpolate 5 extra points
 #           for j in range(6):  # 6 points total (5 interpolated + 1 original endpoint)
 #               alpha = j / 5.0
 #               interpolated_point = (1 - alpha) * start + alpha * end
 #               refined_path.append(interpolated_point)

 #       return refined_path


# Main function to demonstrate path planning with obstacle avoidance
def main():
    # Generate synthetic point cloud (replace with real RGBD data)
    #point_cloud = np.random.uniform(-1, 1, (100, 3))
    # read point cloud from file.npy
    point_cloud = np.load('./data/point_cloud.npy')
    point_cloud = point_cloud.squeeze()
 
     # Initialize planner
    planner = MotionPlannerTiago( include_torso=True, include_base=True, max_rot_vel=100, baseXY_range=np.array([[-5,5],[-5,5]]))

    # Sample start and goal configurations
    start_q = np.array([ 0.0000000e+00 ,0.0000000e+00 ,0.0000000e+00 ,0.0000000e+00 ,2.4995229e-01 ,-1.1780972e+00 ,1.3705798e+00 ,4.3613971e-05 ,-3.9259380e-01 ,1.5708807e+00 ,1.4137164e+00 ,-4.6996109e-05])
    goal_q = np.array([ 0.          ,0.          ,0.          ,0.          ,0.          ,1.32986133 ,-0.28366732 ,1.97406002 ,1.67248146 ,-0.60182133 ,-1.13218905 ,1.73224318])
#start_q: [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
#  2.4995227e-01 -1.1780974e+00  1.3705800e+00  4.3575033e-05
# -3.9259374e-01  1.5708808e+00  1.4137164e+00 -4.6973015e-05]
#end_q: [ 0.          0.          0.          0.          0.02323479  1.24236415
# -0.29066146  2.06582184  1.73989306 -0.65348198 -1.11607726  1.86842996]
    start_q = np.array([ 0.0000000e+00 ,0.0000000e+00 ,0.0000000e+00 ,0.0000000e+00 ,2.4995227e-01 ,-1.1780974e+00 ,1.3705800e+00 ,4.3575033e-05 ,-3.9259374e-01 ,1.5708808e+00 ,1.4137164e+00 ,-4.6973015e-05])
    goal_q = np.array([ 0.          ,0.          ,0.          ,0.          ,0.02323479 ,1.24236415 ,-0.29066146 ,2.06582184 ,1.73989306 ,-0.65348198 ,-1.11607726 ,1.86842996])
    
#    start_q: [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00
#  2.4995229e-01 -1.1780974e+00  1.3705797e+00  4.3623684e-05
# -3.9259377e-01  1.5708807e+00  1.4137164e+00 -4.6960027e-05]
#end_q: [ 0.          0.          0.          0.          0.21110718  1.44172127
#  0.36238697  2.073912    0.28613534 -0.00275988 -0.32972351  0.49309001]
    start_q = np.array([ 0.0000000e+00 ,0.0000000e+00 ,0.0000000e+00 ,0.0000000e+00 ,2.4995229e-01 ,-1.1780974e+00 ,1.3705797e+00 ,4.3623684e-05 ,-3.9259377e-01 ,1.5708807e+00 ,1.4137164e+00 ,-4.6960027e-05])
    goal_q = np.array([ 0.          ,0.          ,1.          ,0.          ,0.21110718 ,1.44172127 ,0.36238697 ,2.073912    ,0.28613534 ,-0.00275988 ,-0.32972351 ,0.49309001])

#start_q: [ 0.0000000e+00  0.0000000e+00  1.0000000e+00  0.0000000e+00
#  2.4995109e-01 -1.1780974e+00  1.3705610e+00  4.2650507e-05
# -3.9259169e-01  1.5708798e+00  1.4137164e+00 -4.8371166e-05]
#end_q: [ 0.          0.          1.          0.          0.0529959   1.4225562
#  0.10628298  1.80618009  0.54230212 -0.23579322 -0.63474094  0.54368636]

    start_q = np.array([ 0.0000000e+00 ,0.0000000e+00 ,1.0000000e+00 ,0.0000000e+00 ,2.4995109e-01 ,-1.1780974e+00 ,1.3705610e+00 ,4.2650507e-05 ,-3.9259169e-01 ,1.5708798e+00 ,1.4137164e+00 ,-4.8371166e-05])
    goal_q = np.array([ 0.          ,0.          ,1.          ,0.          ,0.0529959   ,1.4225562 ,0.10628298 ,1.80618009 ,0.54230212 ,-0.23579322 ,-0.63474094 ,0.54368636])
#start_q: [ 0.0000000e+00  0.0000000e+00  1.0000000e+00  0.0000000e+00
#  2.4995109e-01 -1.1780974e+00  1.3705610e+00  4.2650507e-05
# -3.9259169e-01  1.5708798e+00  1.4137164e+00 -4.8371166e-05]
#end_q: [ 0.          0.          1.          0.          0.13731615  1.37519676
#  0.2714935   1.74783371  0.46460181 -0.38435673 -0.52358791  0.57542086]

    start_q = np.array([ 0.0000000e+00 ,0.0000000e+00 ,1.0000000e+00 ,0.0000000e+00 ,2.4995109e-01 ,-1.1780974e+00 ,1.3705610e+00 ,4.2650507e-05 ,-3.9259169e-01 ,1.5708798e+00 ,1.4137164e+00 ,-4.8371166e-05])
    goal_q = np.array([ 0.          ,0.          ,1.          ,0.          ,0.13731615  ,1.37519676 ,0.2714935   ,1.74783371 ,0.46460181 ,-0.38435673 ,-0.52358791 ,0.57542086])



#start_q: [ 0.          0.          1.          0.          0.24935882  0.24892128
#  1.5704275   1.5728476   0.5484485   0.9977282  -1.4118348   1.0007526 ]
#end_q: [ 0.          0.          1.          0.          0.35        1.39164308
# -0.25110264  3.53171315  0.43457094 -1.18888397  1.41371669  0.87400178]
    start_q = np.array([ 0.          ,0.          ,1.          ,0.          ,0.24935882 ,0.24892128 ,1.5704275   ,1.5728476   ,0.5484485   ,0.9977282  ,-1.4118348   ,1.0007526 ])
    goal_q = np.array([ 0.          ,0.          ,1.          ,0.          ,0.35        ,1.39164308 ,-0.25110264  ,3.53171315  ,0.43457094 ,-1.18888397  ,1.41371669  ,0.87400178])

    print("Sampled Random Q:", planner.sample_random_q())
    #start_q = planner.sample_random_q()
    #goal_q = planner.sample_random_q()


    # Convert point cloud to voxel grid
    voxel_grid, voxel_size = planner.voxelize_point_cloud(point_cloud, voxel_size=0.1)

    # Plan avoiding obstacles
    point_cloud = point_cloud[::100]
    point_cloud = point_cloud[point_cloud[:, 2] > 0]
    point_cloud = point_cloud[point_cloud[:, 0] <5]
    path = planner.rrt_motion_plan_with_obstacles(start_q, goal_q, point_cloud, max_iters=1000, step_size=0.1)
    #print("Path found:", path)
    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(voxel_grid[:, 0], voxel_grid[:, 1], voxel_grid[:, 2], c='red', alpha=0.5, s=1)
    # if the height of the points is bigger than 0, plot the points
    voxel_grid_obs = voxel_grid[voxel_grid[:, 2] > 1]
    voxel_grid_obs = voxel_grid[voxel_grid[:, 0] <10]
    ax.scatter(voxel_grid_obs[:, 0], voxel_grid_obs[:, 1], voxel_grid_obs[:, 2], c='blue', alpha=0.5, s=1)

    # downsample the point cloud
    #point_cloud = point_cloud[::10]

    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='red', alpha=0.5, s=1)

    if path:
        path = np.array(path)
        # remove the points before the last point
        #path = np.array([path[0] , path[-1]])
        print("Path length:", len(path))
        print("Path:", path)


        eef_positions = [planner.forward_kinematics(q)[0] for q in path]
        eef_positions = np.array(eef_positions)
        # get base 
        
        ax.scatter(eef_positions[:, 0], eef_positions[:, 1], eef_positions[:, 2], c='green', alpha=0.5, s=2)

        #ax.scatter(eef_positions[:, 2], eef_positions[:, 0], eef_positions[:, 1], c='green', alpha=0.5, s=2)
        # voxel grid 
        voxel_eef = np.round(eef_positions / voxel_size).astype(int) 
        voxel_eef = np.array([voxel for voxel in voxel_eef if tuple(voxel) in voxel_grid])

    
        
        #ax.scatter(voxel_eef[:, 0], voxel_eef[:, 1], voxel_eef[:, 2], c='green', alpha=0.5, s=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_box_aspect([1,1,1])

    ax.legend()
    plt.show()
#if __name__ == "__main__":
#    main()
