# from typing import Optional
# import numpy as np
import os
import numpy as np
import torch
from pxr import Gf

# from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.physx.scripts import utils

# from pxr import UsdGeom
from base_proposal.utils.files import get_usd_path
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.torch.rotations import euler_angles_to_quats
from omni.isaac.core.utils.semantics import add_update_semantics
from pxr import UsdGeom
import omni.isaac.core.utils.physics as physics_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
goal_center = torch.tensor([0.0, 0.0, 0.0], device=device)


def get_goal_center():
    return goal_center


def get_se3_transform(prim):
    # Check if the prim is valid and has a computed transform
    if not prim:
        print("Invalid Prim")
        return None

    # Access the prim's transform attributes
    xform = UsdGeom.Xformable(prim)
    if not xform:
        print("Prim is not Xformable")
        return None

    # Get the local transformation matrix (this is relative to the prim's parent)
    local_transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)

    return local_transform


def add_plane(name, prim_path, device):
    object_usd_path = os.path.join(get_usd_path(), "Props", name, "plane.usd")
    add_reference_to_stage(object_usd_path, prim_path + name)
    obj = GeometryPrim(
        prim_path=prim_path + name,
        name=name,
        position=torch.tensor([0.0, 0.0, 0.0], device=device),
        orientation=torch.tensor(
            [0.707106, 0.707106, 0.0, 0.0], device=device
        ),  # Shapenet model may be downward facing. Rotate in X direction by 90 degrees,
        scale=[
            1.5,
            0.01,
            1.5,
        ],  # Has to be scaled down to metres. Default usd units for these objects is cms
        # collision=True
    )
    # Enable tight collision approximation
    # obj.set_collision_approximation("convexDecomposition")


def sence(name, prim_path, device):
    # Spawn Shapenet obstacle model from usd path
    object_usd_path = os.path.join(get_usd_path(), "Props", name, "house.usd")
    if not os.path.exists(object_usd_path):
        print("Could not find object at path: ", object_usd_path)
        return None
    add_reference_to_stage(object_usd_path, prim_path + "/obstacle/" + name)

    pose = torch.tensor([0.0, 0, 0.8], device=device)
    obj = GeometryPrim(
        prim_path=prim_path + "/obstacle/" + name,
        name=name,
        position=pose,
        # orientation= torch.tensor([0.707106, 0.707106, 0.0, 0.0], device=device), # Shapenet model may be downward facing. Rotate in X direction by 90 degrees,
        scale=[
            4,
            4,
            3.5,
        ],  # Has to be scaled down to metres. Default usd units for these objects is cms
        # collision=True
    )
    # Enable tight collision approximation
    # obj.set_collision_approximation("convexDecomposition")

    return obj


def spawn_obstacle(name, prim_path, device):
    # Spawn Shapenet obstacle model from usd path

    object_usd_path = os.path.join(
        get_usd_path(), "Props", "Shapenet", name, "models", "model_normalized.usd"
    )
    add_reference_to_stage(object_usd_path, prim_path + "/obstacle/" + name)
    prim = get_prim_at_path(prim_path + "/obstacle/" + name)

    obj = GeometryPrim(
        prim_path=prim_path + "/obstacle/" + name,
        name=name,
        position=torch.tensor([0.0, 0.0, 0.0], device=device),
        orientation=torch.tensor(
            [1, 0, 0.0, 0.0], device=device
        ),  # Shapenet model may be downward facing. Rotate in X direction by 90 degrees,
        scale=[
            1.2,
            1.2,
            1.2,
        ],  # Has to be scaled down to metres. Default usd units for these objects is cms
        # collision=True,
    )
    # Enable tight collision approximation

    # obj.set_collision_approximation("convexDecomposition")
    #
    RigidPrim.__init__(
        obj,
        prim_path=prim_path + "/obstacle/" + name,
        name=obj.name,
        position=torch.tensor([0.0, 0.0, 0.0], device=device),
        orientation=torch.tensor(
            [1, 0, 0.0, 0.0], device=device
        ),  # Shapenet model may be downward facing. Rotate in X direction by 90 degrees,
        scale=[
            1.2,
            1.2,
            1.2,
        ],  # Has to be scaled down to metres. Default usd units for these objects is cms
        # visible=visible,
        mass=99999999,
        # linear_velocity=linear_velocity,
        # angular_velocity=angular_velocity,
    )

    return obj


def spawn_grasp_object(name, prim_path, device):
    # Spawn YCB object model from usd path
    object_usd_path = os.path.join(
        get_usd_path(), "Props", "YCB", "Axis_Aligned", name + ".usd"
    )
    add_reference_to_stage(object_usd_path, prim_path + "/grasp_obj/ycb_" + name)
    prim = get_prim_at_path(prim_path + "/grasp_obj/ycb_" + name)

    obj = GeometryPrim(
        prim_path=prim_path + "/grasp_obj/ycb_" + name,
        name=name,
        position=torch.tensor([0.0, 0.0, 0.0], device=device),
        orientation=torch.tensor(
            [0.707106, -0.707106, 0.0, 0.0], device=device
        ),  # YCB model may be downward facing. Rotate in X direction by -90 degrees,
        scale=[
            0.01,
            0.01,
            0.01,
        ],  # Has to be scaled down to metres. Default usd units for these objects is cms
        collision=True,
    )
    # Enable tight collision approximation
    obj.set_collision_approximation("convexDecomposition")

    if name == "PLACE":
        return obj
    print("Adding object: ", name)
    density = 0.01 if "heavy" in name else -1

    RigidPrim.__init__(
        obj,  # Add Rigid prim attributes since it can move
        prim_path=prim_path + "/grasp_obj/ycb_" + name,
        name=name,
        position=torch.tensor([100.0, 100.0, 100.0], device=device),
        orientation=torch.tensor(
            [0.707106, -0.707106, 0.0, 0.0], device=device
        ),  # YCB model may be downward facing. Rotate in X direction by -90 degrees,
        scale=[
            0.01,
            0.01,
            0.01,
        ],  # Has to be scaled down to metres. Default usd units for these objects is cms
        density=density,
    )
    # utils.setRigidBody(obj.prim, "convexDecomposition", False)
    # print(prim)
    # print("Transform: ", transform)
    return obj


def get_obj_pose(obj):
    pose = obj.get_world_pose()
    return pose


def set_obj_pose(obj, pose):
    # pose = obj.get_world_pose()
    # print(pose)

    pose[0] = pose[0] + 0.5
    pose[2] = 0.5
    obj.set_world_pose(
        position=torch.tensor(pose[0:3], device=device),
        orientation=euler_angles_to_quats(torch.tensor([[0, 0, 0]], device=device))[0],
    )
    # make object not rigid

    # prim = get_prim_at_path(obj.prim_path)
    # physics_utils.set_rigid_body_enabled(False, prim)


def setup_tabular_scene(grasp_objs, targets_position, targets_se3, obstacles, device):
    # Randomly arrange the objects in the environment. Ensure no overlaps, collisions!
    # Grasp objects will be placed on a random tabular obstacle.
    # Returns: target grasp object, target grasp/goal, all object's oriented bboxes
    # TODO: Add support for circular tables

    # set obstacles in (0,0,0)
    for idx, obj in enumerate(obstacles):
        obj.set_world_pose(
            position=torch.tensor([0.0, 0.0, 0.0], device=device),
            orientation=euler_angles_to_quats(torch.tensor([[0, 0, 0]], device=device))[
                0
            ],
        )

    goal_pose = []
    for idx, obj in enumerate(grasp_objs):
        obj.set_world_pose(
            position=torch.tensor(targets_position[idx][0], device=device),
            orientation=euler_angles_to_quats(
                torch.tensor(targets_position[idx][1], device=device)
            ).to(device),
        )

        se3_list = targets_se3[idx]
        for se3 in se3_list:
            pose = torch.hstack(
                (
                    torch.tensor(se3[0], dtype=torch.float, device=device),
                    euler_angles_to_quats(
                        torch.tensor(se3[1], dtype=torch.float, device=device)
                    ).to(device),
                )
            )
            goal_pose.append(pose)

        #    goal_pose = []
    #   for obj in grasp_objs:
    #       if obj.name == "nothing":
    #           # obj.set_world_pose(
    #           #     position=torch.tensor([1.5000000, -2, 0.3], device=device),
    #           #     orientation=euler_angles_to_quats(
    #           #         torch.tensor([[-torch.pi / 2, 0, torch.pi]], device=device)
    #           #     )[0],
    #           # )
    #           pose = torch.hstack(
    #               (
    #                   torch.tensor([1.3, -1.3, 0.6], dtype=torch.float, device=device),
    #                   euler_angles_to_quats(
    #                       torch.tensor(
    #                           [[-torch.pi / 2, -torch.pi / 2, 0]],
    #                           dtype=torch.float,
    #                           device=device,
    #                       )
    #                   )[0],
    #               )
    #           )
    #       else:
    #           #  obj.set_world_pose(
    #           #      position=torch.tensor([1.5, 0.8, 0.75], device=device),
    #           #      orientation=euler_angles_to_quats(
    #           #          torch.tensor([[-torch.pi / 2, 0, torch.pi]], device=device)
    #           #      )[0],
    #           #  )
    #           pose = torch.hstack(
    #               (
    #                   torch.tensor([1.45, 0.58, 0.82], dtype=torch.float, device=device),
    #                   euler_angles_to_quats(
    #                       torch.tensor(
    #                           # [[-torch.pi / 2, -torch.pi / 2, 0]],
    #                           [[0, 0, torch.pi / 2]],
    #                           dtype=torch.float,
    #                           device=device,
    #                       )
    #                   )[0],
    #               )
    #           )

    # goal_pose.append(pose)
    # tranlate list to tensor
    goal_pose = torch.stack(goal_pose)
    # print type
    return goal_pose


#    object_positions, object_yaws, objects_dimensions = [], [], []
#    obst_aabboxes, grasp_obj_aabboxes = [], []
#    robot_radius = 0.45 # metres. To exclude circle at origin where the robot (Tiago) is
#
#    # Choose one tabular obstacle to place grasp objects on
#    tab_index = np.random.choice(np.nonzero(tabular_obstacle_mask)[0])
#    # Place tabular obstacle at random location on the ground plane
#    tab_xyz_size = obstacles_dimensions[tab_index][1] - obstacles_dimensions[tab_index][0]
#    tab_z_to_ground = - obstacles_dimensions[tab_index][0,2]
#    # polar co-ords
#    #tab_r = np.random.uniform(robot_radius+np.max(tab_xyz_size[0:2]),world_xy_radius) # taking max xy size margin from robot
#    tab_r = 100
#    tab_phi = np.random.uniform(-np.pi/2,-np.pi/2)
#
#    tab_phi = -np.pi # Place tabular obstacle on the right side of the robot
#    tab_x, tab_y = tab_r*np.cos(tab_phi), tab_r*np.sin(tab_phi)
#    tab_position = [tab_x,tab_y,tab_z_to_ground]
#    obstacles[tab_index].set_world_pose(position=torch.tensor(tab_position,dtype=torch.float,device=device),
#                                    orientation=torch.tensor([0.707106, 0.707106, 0.0, 0.0], device=device)) # Shapenet model: Rotate in X direction by 90 degrees
#    # Don't add a random orientation to tabular obstacle yet. We will add it after placing the grasp objects on it
#
#    # Place all grasp objects on the tabular obstacle (without overlaps)
#    for idx, _ in enumerate(grasp_objs):
#        grasp_obj_z_to_ground = - grasp_objs_dimensions[idx][0,2]
#
#        while(1): # Be careful about infinite loops!
#            # Add random orientation (yaw) to object
#            grasp_obj_yaw = np.random.uniform(-np.pi,np.pi) # random yaw
#            grasp_objs[idx].set_world_pose(position= torch.tensor([0.0, 0.0, 0.0], device=device),
#                        orientation=euler_angles_to_quats(torch.tensor([[-torch.pi/2,0,grasp_obj_yaw]],device=device))[0]) # YCB needs X -90 deg rotation
#            # compute new AxisAligned bbox
#            self._scene._bbox_cache.Clear()
#            grasp_obj_aabbox = self._scene.compute_object_AABB(grasp_objs[idx].name)
#
#            # Place object at height of tabular obstacle and in the x-y range of the tabular obstacle
#            grasp_obj_x = tab_x + np.random.uniform((-tab_xyz_size[0]-grasp_obj_aabbox[0,0])/2.0, (tab_xyz_size[0]-grasp_obj_aabbox[1,0])/2.0)
#            grasp_obj_y = tab_y + np.random.uniform((-tab_xyz_size[1]-grasp_obj_aabbox[0,1])/2.0, (tab_xyz_size[1]-grasp_obj_aabbox[1,1])/2.0)
#            grasp_obj_z = tab_xyz_size[2] + grasp_obj_z_to_ground +0.01 # Place on top of tabular obstacle
#            grasp_obj_position = [grasp_obj_x,grasp_obj_y,grasp_obj_z]
#            grasp_objs[idx].set_world_pose(position=torch.tensor(grasp_obj_position,dtype=torch.float,device=device),
#                                           orientation=euler_angles_to_quats(torch.tensor([[-torch.pi/2,0,grasp_obj_yaw]],dtype=torch.float,device=device))[0])  # YCB needs X -90 deg rotation
#            # compute new AxisAligned bbox
#            self._scene._bbox_cache.Clear()
#            grasp_obj_aabbox = self._scene.compute_object_AABB(grasp_objs[idx].name)
#            # Check for overlap with all existing grasp objects
#            overlap = False
#            for other_aabbox in grasp_obj_aabboxes: # loop over existing AAbboxes
#                grasp_obj_range = Gf.Range3d(Gf.Vec3d(grasp_obj_aabbox[0,0],grasp_obj_aabbox[0,1],grasp_obj_aabbox[0,2]),Gf.Vec3d(grasp_obj_aabbox[1,0],grasp_obj_aabbox[1,1],grasp_obj_aabbox[1,2]))
#                other_obj_range = Gf.Range3d(Gf.Vec3d(other_aabbox[0,0],other_aabbox[0,1],other_aabbox[0,2]),Gf.Vec3d(other_aabbox[1,0],other_aabbox[1,1],other_aabbox[1,2]))
#                intersec = Gf.Range3d.GetIntersection(grasp_obj_range, other_obj_range)
#                if (not intersec.IsEmpty()):
#                    overlap = True # Failed. Try another pose
#                    break
#            if (overlap):
#                continue # Failed. Try another pose
#            else:
#                # Success. Add this valid AAbbox to the list
#                grasp_obj_aabboxes.append(grasp_obj_aabbox)
#                # Store grasp object position, orientation (yaw), dimensions
#                object_positions.append(grasp_obj_position)
#                object_yaws.append(grasp_obj_yaw)
#                objects_dimensions.append(grasp_objs_dimensions[idx])
#                break
#
#    # Now add a random orientation to the tabular obstacle and move all the grasp objects placed on it accordingly
#    tab_yaw = np.random.uniform(-np.pi,np.pi) # random yaw
#    obstacles[tab_index].set_world_pose(orientation=euler_angles_to_quats(torch.tensor([[torch.pi/2,0,tab_yaw]],device=device))[0])
#    for idx, _ in enumerate(grasp_objs):
#        object_yaws[idx] += tab_yaw # Add orientation that was just added to tabular obstacle
#        if (object_yaws[idx] < -np.pi): object_yaws[idx] + 2*np.pi, # ensure within -pi to pi
#        if (object_yaws[idx] >  np.pi): object_yaws[idx] - 2*np.pi, # ensure within -pi to pi
#        # modify x-y positions of grasp objects accordingly
#        curr_rel_x, curr_rel_y = object_positions[idx][0] - tab_position[0], object_positions[idx][1] - tab_position[1] # Get relative co-ords
#        modify_x, modify_y = curr_rel_x*np.cos(tab_yaw) - curr_rel_y*np.sin(tab_yaw), curr_rel_x*np.sin(tab_yaw) + curr_rel_y*np.cos(tab_yaw)
#        new_x, new_y = modify_x + tab_position[0], modify_y + tab_position[1]
#        object_positions[idx] = [new_x, new_y, object_positions[idx][2]] # new x and y but z is unchanged
#        grasp_objs[idx].set_world_pose(position=torch.tensor(object_positions[idx],dtype=torch.float,device=device),
#                                       orientation=euler_angles_to_quats(torch.tensor([[-torch.pi/2,0,object_yaws[idx]]],device=device))[0])
#    # Store tabular obstacle position, orientation, dimensions and AABBox
#    object_positions.append(tab_position)
#    object_yaws.append(tab_yaw)
#    objects_dimensions.append(obstacles_dimensions[tab_index])
#    self._scene._bbox_cache.Clear()
#    obst_aabboxes.append(self._scene.compute_object_AABB(obstacles[tab_index].name))
#
#    # Now we need to place all the other obstacles (without overlaps):
#    for idx, _ in enumerate(obstacles):
#        if (idx == tab_index): continue # Skip this since we have already placed tabular obstacle
#
#        obst_xyz_size = obstacles_dimensions[idx][1] - obstacles_dimensions[idx][0]
#        obst_z_to_ground = - obstacles_dimensions[idx][0,2]
#        first = False
#        if idx == 0:
#            first = True
#        while(1): # Be careful about infinite loops!
#            # Place obstacle at random position and orientation on the ground plane
#            # polar co-ords
#
#            #obst_r = np.random.uniform(robot_radius+np.max(obst_xyz_size[0:2]),world_xy_radius) # taking max xy size margin from robot
#            obst_r = 150
#            obst_phi = np.random.uniform(-np.pi,np.pi)
#            obst_x, obst_y = obst_r*np.cos(obst_phi), obst_r*np.sin(obst_phi)
#            obst_position = [obst_x,obst_y,obst_z_to_ground]
#            room_yaw = np.random.uniform(-np.pi,np.pi) # random yaw
#            if first:
#                obst_phi = 0 # Place first obstacle on the right side of the robot
#                obst_r = world_xy_radius
#                obst_r = 2
#                obst_x, obst_y = obst_r*np.cos(obst_phi), obst_r*np.sin(obst_phi)
#                obst_position = [obst_x,obst_y,obst_z_to_ground]
#                #room_yaw = np.random.uniform(-np.pi,np.pi) # random yaw
#                #room_yaw = np.random.uniform(-np.pi/2,-np.pi) # random yaw
#                room_yaw = -np.pi/2 + target_angle
#                obst_pos = obst_position
#                angle = [0,0,0]
#                obstacle_pose = torch.hstack(( torch.tensor(obst_pos,dtype=torch.float,device=device),
#                        euler_angles_to_quats(torch.tensor([[0,np.pi/2,room_yaw]],dtype=torch.float,device=device))[0] ))
#                global goal_center
#                goal_center = obstacle_pose[0:3]
#                if target == 'cabinet':
#                    obst_pos = [obst_pos[0]-0.4*np.cos(room_yaw+np.pi/2), obst_pos[1]-0.4*np.sin(room_yaw+np.pi/2), obst_pos[2]+0.5]
#                    angle = [0,np.pi/2+np.pi/4,room_yaw-np.pi/2]
#                    #angle = [0,np.pi/2+np.pi/4,room_yaw-np.pi/1.2]
#                    #angle = [0,np.pi/2,room_yaw-np.pi/2]
#                    goal_center = obstacle_pose[0:3]
#                if target == 'stove':
#                    # mid
#                    #obst_pos = [obst_pos[0]-0.38*np.cos(room_yaw+np.pi/2), obst_pos[1]-0.38*np.sin(room_yaw+np.pi/2), obst_pos[2]+0.85]
#                    # left
#                    #obst_pos = [obst_pos[0]-0.47*np.cos(room_yaw+np.pi/3.5), obst_pos[1]-0.47*np.sin(room_yaw+np.pi/3.5), obst_pos[2]+0.85]
#                    # right
#                    obst_pos = [obst_pos[0]-0.47*np.cos(room_yaw+np.pi/1.45), obst_pos[1]-0.47*np.sin(room_yaw+np.pi/1.45), obst_pos[2]+0.85]
#                    angle = [0,np.pi/2+np.pi/4,room_yaw-np.pi/2]
#                    goal_center = obstacle_pose[0:3]
#                if target == 'bookshelf':
#                    obst_pos = [obst_pos[0]-0.28*np.cos(room_yaw+np.pi/2), obst_pos[1]-0.28*np.sin(room_yaw+np.pi/2), obst_pos[2]+0.7]
#                    angle = [0,np.pi/2+np.pi/4,room_yaw-np.pi/2]
#                    goal_center = obstacle_pose[0:3]
#
#                if target == 'cabinet_clutter' or target == 'cabinet_clutter1' or target == 'cabinet_clutter2':
#                    obst_pos = [obst_pos[0]+0.2*np.cos(room_yaw+np.pi/2), obst_pos[1]+0.2*np.sin(room_yaw+np.pi/2), obst_pos[2]-0.3]
#                    #obst_pos = [obst_pos[0]+0.13*np.cos(room_yaw+np.pi/2), obst_pos[1]+0.13*np.sin(room_yaw+np.pi/2), obst_pos[2]-0.3]
#                    angle = [0,np.pi/2+np.pi/4,room_yaw-np.pi/2]
#                    goal_center = [obst_pos[0]+0.4*np.cos(room_yaw+np.pi/2), obst_pos[1]+0.4*np.sin(room_yaw+np.pi/2)]
#
#
#                if target == 'oven_cluster':
#                    #obst_pos = [obst_pos[0]-0.2*np.cos(room_yaw+np.pi/2), obst_pos[1]-0.2*np.sin(room_yaw+np.pi/2), obst_pos[2]+0.65]
#                    obst_pos = [obst_pos[0]+0.63*np.cos(room_yaw+np.pi/2), obst_pos[1]+0.63*np.sin(room_yaw+np.pi/2), obst_pos[2]+0.75]
#                    angle = [np.pi/2,np.pi/2+np.pi/4,room_yaw-np.pi/2]
#                    goal_center = [obst_pos[0]+0.4*np.cos(room_yaw+np.pi/2), obst_pos[1]+0.4*np.sin(room_yaw+np.pi/2)]
#
#                if target == 'light':
#                    obst_pos = [obst_pos[0]+1.2*np.cos(room_yaw+np.pi/1.5), obst_pos[1]+1.2*np.sin(room_yaw+np.pi/1.5), obst_pos[2]+0.88]
#                    #angle = [0,np.pi/2+np.pi/2,room_yaw-np.pi/2]
#                    angle = [np.pi/2,np.pi/2+np.pi/2,room_yaw-np.pi/2]
#                    goal_center = [obst_pos[0]+0.3*np.cos(room_yaw+np.pi/1.5), obst_pos[1]+0.3*np.sin(room_yaw+np.pi/1.5)]
#
#
#                if target == 'lamp':
#                    obst_pos = [obst_pos[0]-0.4*np.cos(room_yaw+np.pi/2), obst_pos[1]-0.4*np.sin(room_yaw+np.pi/2), obst_pos[2]+0.5]
#                    angle = [0,np.pi/2+np.pi/4,room_yaw-np.pi/2]
#
#                if target == 'oven':
#                    # mid
#                    obst_pos = [obst_pos[0]-0.35*np.cos(room_yaw+np.pi/2), obst_pos[1]-0.35*np.sin(room_yaw+np.pi/2), obst_pos[2]+0.7]
#                    angle = [0,np.pi/2+np.pi/2,room_yaw-np.pi/2]
#                    goal_center = obstacle_pose[0:3]
#
#                if target == 'cabinet2_cluster':
#                    obst_pos = [obst_pos[0]+0.25*np.cos(room_yaw+np.pi/2), obst_pos[1]+0.25*np.sin(room_yaw+np.pi/2), obst_pos[2]-0.1]
#                    angle = [0,np.pi/2+np.pi/4,room_yaw-np.pi/2]
#                    goal_center = [obst_pos[0]+0.5*np.cos(room_yaw+np.pi/1.4), obst_pos[1]+0.5*np.sin(room_yaw+np.pi/1.4)]
#
#                if target == 'mug_cluster':
#                    obst_pos = [obst_pos[0]-0.1*np.cos(room_yaw+np.pi/2), obst_pos[1]-0.1*np.sin(room_yaw+np.pi/2), obst_pos[2]-0.3]
#                    angle = [0,np.pi/2+np.pi/4,room_yaw-np.pi/2]
#                    goal_center = obstacle_pose[0:3]
#
#                obstacle_pose = torch.hstack(( torch.tensor(obst_pos,dtype=torch.float,device=device),
#                        euler_angles_to_quats(torch.tensor([angle],dtype=torch.float,device=device))[0] ))
#            break
#
#            obstacles[idx].set_world_pose(position=torch.tensor(obst_position,device=device),
#                                          orientation=euler_angles_to_quats(torch.tensor([[torch.pi/2,0,room_yaw]],device=device))[0])
# compute new AxisAligned bbox
#           self._scene._bbox_cache.Clear()
#           obst_aabbox = self._scene.compute_object_AABB(obstacles[idx].name)
#           # Check for overlap with all existing grasp objects
#           overlap = False
#           for other_aabbox in obst_aabboxes: # loop over existing AAbboxes
#               obst_range = Gf.Range3d(Gf.Vec3d(obst_aabbox[0,0],obst_aabbox[0,1],obst_aabbox[0,2]),Gf.Vec3d(obst_aabbox[1,0],obst_aabbox[1,1],obst_aabbox[1,2]))
#               other_obst_range = Gf.Range3d(Gf.Vec3d(other_aabbox[0,0],other_aabbox[0,1],other_aabbox[0,2]),Gf.Vec3d(other_aabbox[1,0],other_aabbox[1,1],other_aabbox[1,2]))
#               intersec = Gf.Range3d.GetIntersection(obst_range, other_obst_range)
#               if (not intersec.IsEmpty()):
#                   overlap = True # Failed. Try another pose
#                   break
#           if (overlap):
#               continue # Failed. Try another pose
#           else:
#               # Success. Add this valid AAbbox to the list
#               obst_aabboxes.append(obst_aabbox)
#               # Store obstacle position, orientation (yaw) and dimensions
#               object_positions.append(obst_position)
#               object_yaws.append(room_yaw)
#               objects_dimensions.append(obstacles_dimensions[idx])
#               break

#   # All objects placed in the scene!
#   # Pick one object to be the grasp object and compute its grasp:
#   goal_obj_index = np.random.randint(len(grasp_objs))
#   # For now, generating only top grasps: no roll, pitch 90, same yaw as object
#   goal_roll = 0.0 # np.random.uniform(-np.pi,np.pi)
#   goal_pitch = np.pi/2.0 # np.random.uniform(0,np.pi/2.0)
#   goal_yaw = object_yaws[goal_obj_index]
#   goal_position = np.array(object_positions[goal_obj_index])
#   goal_position[2] = (grasp_obj_aabboxes[goal_obj_index][1,2] + np.random.uniform(0.05,0.20)) # Add (random) z offset to object top (5 to 20 cms)
#   goal_pose = torch.hstack(( torch.tensor(goal_position,dtype=torch.float,device=device),
#                       euler_angles_to_quats(torch.tensor([[goal_roll,goal_pitch,goal_yaw]],dtype=torch.float,device=device))[0] ))

#
#   # Remove the goal object from obj_positions and yaws list (for computing oriented bboxes)
#   del object_positions[goal_obj_index], object_yaws[goal_obj_index]
#
#   # Compute oriented bounding boxes for all remaining objects
#   for idx in range(len(object_positions)):
#       bbox_tf = np.zeros((3,3))
#       bbox_tf[:2,:2] = np.array([[np.cos(object_yaws[idx]), -np.sin(object_yaws[idx])],[np.sin(object_yaws[idx]), np.cos(object_yaws[idx])]])
#       bbox_tf[:,-1] = np.array([object_positions[idx][0], object_positions[idx][1], 1.0]) # x,y,1
#       min_xy_vertex = np.array([[objects_dimensions[idx][0,0],objects_dimensions[idx][0,1],1.0]]).T
#       max_xy_vertex = np.array([[objects_dimensions[idx][1,0],objects_dimensions[idx][1,1],1.0]]).T
#       new_min_xy_vertex = (bbox_tf @ min_xy_vertex)[0:2].T.squeeze()
#       new_max_xy_vertex = (bbox_tf @ max_xy_vertex)[0:2].T.squeeze()
#       z_top_to_ground = object_positions[idx][2] + objects_dimensions[idx][1,2] # z position plus distance to object top
#       # Oriented bbox: x0, y0, x1, y1, z_to_ground, yaw
#       oriented_bbox = torch.tensor([ new_min_xy_vertex[0], new_min_xy_vertex[1],
#                                      new_max_xy_vertex[0], new_max_xy_vertex[1],
#                                           z_top_to_ground,     object_yaws[idx], ] ,dtype=torch.float,device=device)
#       if idx == 0:
#           object_oriented_bboxes = oriented_bbox
#       else:
#           object_oriented_bboxes = torch.vstack(( object_oriented_bboxes, oriented_bbox ))


#    return obstacle_pose


# def setup_tabular_scene(self, target ,target_angle ,obstacles, tabular_obstacle_mask, grasp_objs, obstacles_dimensions, grasp_objs_dimensions, world_xy_radius, device):
#    # Randomly arrange the objects in the environment. Ensure no overlaps, collisions!
#    # Grasp objects will be placed on a random tabular obstacle.
#    # Returns: target grasp object, target grasp/goal, all object's oriented bboxes
#    # TODO: Add support for circular tables
#    object_positions, object_yaws, objects_dimensions = [], [], []
#    obst_aabboxes, grasp_obj_aabboxes = [], []
#    robot_radius = 0.45 # metres. To exclude circle at origin where the robot (Tiago) is
#
#    # Choose one tabular obstacle to place grasp objects on
#    tab_index = np.random.choice(np.nonzero(tabular_obstacle_mask)[0])
#    # Place tabular obstacle at random location on the ground plane
#    tab_xyz_size = obstacles_dimensions[tab_index][1] - obstacles_dimensions[tab_index][0]
#    tab_z_to_ground = - obstacles_dimensions[tab_index][0,2]
#    # polar co-ords
#    #tab_r = np.random.uniform(robot_radius+np.max(tab_xyz_size[0:2]),world_xy_radius) # taking max xy size margin from robot
#    tab_r = 100
#    tab_phi = np.random.uniform(-np.pi/2,-np.pi/2)
#
#    tab_phi = -np.pi # Place tabular obstacle on the right side of the robot
#    tab_x, tab_y = tab_r*np.cos(tab_phi), tab_r*np.sin(tab_phi)
#    tab_position = [tab_x,tab_y,tab_z_to_ground]
#    obstacles[tab_index].set_world_pose(position=torch.tensor(tab_position,dtype=torch.float,device=device),
#                                    orientation=torch.tensor([0.707106, 0.707106, 0.0, 0.0], device=device)) # Shapenet model: Rotate in X direction by 90 degrees
#    # Don't add a random orientation to tabular obstacle yet. We will add it after placing the grasp objects on it
#
#    # Place all grasp objects on the tabular obstacle (without overlaps)
#    for idx, _ in enumerate(grasp_objs):
#        grasp_obj_z_to_ground = - grasp_objs_dimensions[idx][0,2]
#
#        while(1): # Be careful about infinite loops!
#            # Add random orientation (yaw) to object
#            grasp_obj_yaw = np.random.uniform(-np.pi,np.pi) # random yaw
#            grasp_objs[idx].set_world_pose(position= torch.tensor([0.0, 0.0, 0.0], device=device),
#                        orientation=euler_angles_to_quats(torch.tensor([[-torch.pi/2,0,grasp_obj_yaw]],device=device))[0]) # YCB needs X -90 deg rotation
#            # compute new AxisAligned bbox
#            self._scene._bbox_cache.Clear()
#            grasp_obj_aabbox = self._scene.compute_object_AABB(grasp_objs[idx].name)
#
#            # Place object at height of tabular obstacle and in the x-y range of the tabular obstacle
#            grasp_obj_x = tab_x + np.random.uniform((-tab_xyz_size[0]-grasp_obj_aabbox[0,0])/2.0, (tab_xyz_size[0]-grasp_obj_aabbox[1,0])/2.0)
#            grasp_obj_y = tab_y + np.random.uniform((-tab_xyz_size[1]-grasp_obj_aabbox[0,1])/2.0, (tab_xyz_size[1]-grasp_obj_aabbox[1,1])/2.0)
#            grasp_obj_z = tab_xyz_size[2] + grasp_obj_z_to_ground +0.01 # Place on top of tabular obstacle
#            grasp_obj_position = [grasp_obj_x,grasp_obj_y,grasp_obj_z]
#            grasp_objs[idx].set_world_pose(position=torch.tensor(grasp_obj_position,dtype=torch.float,device=device),
#                                           orientation=euler_angles_to_quats(torch.tensor([[-torch.pi/2,0,grasp_obj_yaw]],dtype=torch.float,device=device))[0])  # YCB needs X -90 deg rotation
#            # compute new AxisAligned bbox
#            self._scene._bbox_cache.Clear()
#            grasp_obj_aabbox = self._scene.compute_object_AABB(grasp_objs[idx].name)
#            # Check for overlap with all existing grasp objects
#            overlap = False
#            for other_aabbox in grasp_obj_aabboxes: # loop over existing AAbboxes
#                grasp_obj_range = Gf.Range3d(Gf.Vec3d(grasp_obj_aabbox[0,0],grasp_obj_aabbox[0,1],grasp_obj_aabbox[0,2]),Gf.Vec3d(grasp_obj_aabbox[1,0],grasp_obj_aabbox[1,1],grasp_obj_aabbox[1,2]))
#                other_obj_range = Gf.Range3d(Gf.Vec3d(other_aabbox[0,0],other_aabbox[0,1],other_aabbox[0,2]),Gf.Vec3d(other_aabbox[1,0],other_aabbox[1,1],other_aabbox[1,2]))
#                intersec = Gf.Range3d.GetIntersection(grasp_obj_range, other_obj_range)
#                if (not intersec.IsEmpty()):
#                    overlap = True # Failed. Try another pose
#                    break
#            if (overlap):
#                continue # Failed. Try another pose
#            else:
#                # Success. Add this valid AAbbox to the list
#                grasp_obj_aabboxes.append(grasp_obj_aabbox)
#                # Store grasp object position, orientation (yaw), dimensions
#                object_positions.append(grasp_obj_position)
#                object_yaws.append(grasp_obj_yaw)
#                objects_dimensions.append(grasp_objs_dimensions[idx])
#                break
#
#    # Now add a random orientation to the tabular obstacle and move all the grasp objects placed on it accordingly
#    tab_yaw = np.random.uniform(-np.pi,np.pi) # random yaw
#    obstacles[tab_index].set_world_pose(orientation=euler_angles_to_quats(torch.tensor([[torch.pi/2,0,tab_yaw]],device=device))[0])
#    for idx, _ in enumerate(grasp_objs):
#        object_yaws[idx] += tab_yaw # Add orientation that was just added to tabular obstacle
#        if (object_yaws[idx] < -np.pi): object_yaws[idx] + 2*np.pi, # ensure within -pi to pi
#        if (object_yaws[idx] >  np.pi): object_yaws[idx] - 2*np.pi, # ensure within -pi to pi
#        # modify x-y positions of grasp objects accordingly
#        curr_rel_x, curr_rel_y = object_positions[idx][0] - tab_position[0], object_positions[idx][1] - tab_position[1] # Get relative co-ords
#        modify_x, modify_y = curr_rel_x*np.cos(tab_yaw) - curr_rel_y*np.sin(tab_yaw), curr_rel_x*np.sin(tab_yaw) + curr_rel_y*np.cos(tab_yaw)
#        new_x, new_y = modify_x + tab_position[0], modify_y + tab_position[1]
#        object_positions[idx] = [new_x, new_y, object_positions[idx][2]] # new x and y but z is unchanged
#        grasp_objs[idx].set_world_pose(position=torch.tensor(object_positions[idx],dtype=torch.float,device=device),
#                                       orientation=euler_angles_to_quats(torch.tensor([[-torch.pi/2,0,object_yaws[idx]]],device=device))[0])
#    # Store tabular obstacle position, orientation, dimensions and AABBox
#    object_positions.append(tab_position)
#    object_yaws.append(tab_yaw)
#    objects_dimensions.append(obstacles_dimensions[tab_index])
#    self._scene._bbox_cache.Clear()
#    obst_aabboxes.append(self._scene.compute_object_AABB(obstacles[tab_index].name))
#
#    # Now we need to place all the other obstacles (without overlaps):
#    for idx, _ in enumerate(obstacles):
#        if (idx == tab_index): continue # Skip this since we have already placed tabular obstacle
#
#        obst_xyz_size = obstacles_dimensions[idx][1] - obstacles_dimensions[idx][0]
#        obst_z_to_ground = - obstacles_dimensions[idx][0,2]
#        first = False
#        if idx == 0:
#            first = True
#        while(1): # Be careful about infinite loops!
#            # Place obstacle at random position and orientation on the ground plane
#            # polar co-ords
#
#            #obst_r = np.random.uniform(robot_radius+np.max(obst_xyz_size[0:2]),world_xy_radius) # taking max xy size margin from robot
#            obst_r = 150
#            obst_phi = np.random.uniform(-np.pi,np.pi)
#            obst_x, obst_y = obst_r*np.cos(obst_phi), obst_r*np.sin(obst_phi)
#            obst_position = [obst_x,obst_y,obst_z_to_ground]
#            room_yaw = np.random.uniform(-np.pi,np.pi) # random yaw
#            if first:
#                obst_phi = 0 # Place first obstacle on the right side of the robot
#                obst_r = world_xy_radius
#                obst_r = 2
#                obst_x, obst_y = obst_r*np.cos(obst_phi), obst_r*np.sin(obst_phi)
#                obst_position = [obst_x,obst_y,obst_z_to_ground]
#                #room_yaw = np.random.uniform(-np.pi,np.pi) # random yaw
#                #room_yaw = np.random.uniform(-np.pi/2,-np.pi) # random yaw
#                room_yaw = -np.pi/2 + target_angle
#                obst_pos = obst_position
#                angle = [0,0,0]
#                obstacle_pose = torch.hstack(( torch.tensor(obst_pos,dtype=torch.float,device=device),
#                        euler_angles_to_quats(torch.tensor([[0,np.pi/2,room_yaw]],dtype=torch.float,device=device))[0] ))
#                global goal_center
#                goal_center = obstacle_pose[0:3]
#                if target == 'cabinet':
#                    obst_pos = [obst_pos[0]-0.4*np.cos(room_yaw+np.pi/2), obst_pos[1]-0.4*np.sin(room_yaw+np.pi/2), obst_pos[2]+0.5]
#                    angle = [0,np.pi/2+np.pi/4,room_yaw-np.pi/2]
#                    #angle = [0,np.pi/2+np.pi/4,room_yaw-np.pi/1.2]
#                    #angle = [0,np.pi/2,room_yaw-np.pi/2]
#                    goal_center = obstacle_pose[0:3]
#                if target == 'stove':
#                    # mid
#                    #obst_pos = [obst_pos[0]-0.38*np.cos(room_yaw+np.pi/2), obst_pos[1]-0.38*np.sin(room_yaw+np.pi/2), obst_pos[2]+0.85]
#                    # left
#                    #obst_pos = [obst_pos[0]-0.47*np.cos(room_yaw+np.pi/3.5), obst_pos[1]-0.47*np.sin(room_yaw+np.pi/3.5), obst_pos[2]+0.85]
#                    # right
#                    obst_pos = [obst_pos[0]-0.47*np.cos(room_yaw+np.pi/1.45), obst_pos[1]-0.47*np.sin(room_yaw+np.pi/1.45), obst_pos[2]+0.85]
#                    angle = [0,np.pi/2+np.pi/4,room_yaw-np.pi/2]
#                    goal_center = obstacle_pose[0:3]
#                if target == 'bookshelf':
#                    obst_pos = [obst_pos[0]-0.28*np.cos(room_yaw+np.pi/2), obst_pos[1]-0.28*np.sin(room_yaw+np.pi/2), obst_pos[2]+0.7]
#                    angle = [0,np.pi/2+np.pi/4,room_yaw-np.pi/2]
#                    goal_center = obstacle_pose[0:3]
#
#                if target == 'cabinet_clutter' or target == 'cabinet_clutter1' or target == 'cabinet_clutter2':
#                    obst_pos = [obst_pos[0]+0.2*np.cos(room_yaw+np.pi/2), obst_pos[1]+0.2*np.sin(room_yaw+np.pi/2), obst_pos[2]-0.3]
#                    #obst_pos = [obst_pos[0]+0.13*np.cos(room_yaw+np.pi/2), obst_pos[1]+0.13*np.sin(room_yaw+np.pi/2), obst_pos[2]-0.3]
#                    angle = [0,np.pi/2+np.pi/4,room_yaw-np.pi/2]
#                    goal_center = [obst_pos[0]+0.4*np.cos(room_yaw+np.pi/2), obst_pos[1]+0.4*np.sin(room_yaw+np.pi/2)]
#
#
#                if target == 'oven_cluster':
#                    #obst_pos = [obst_pos[0]-0.2*np.cos(room_yaw+np.pi/2), obst_pos[1]-0.2*np.sin(room_yaw+np.pi/2), obst_pos[2]+0.65]
#                    obst_pos = [obst_pos[0]+0.63*np.cos(room_yaw+np.pi/2), obst_pos[1]+0.63*np.sin(room_yaw+np.pi/2), obst_pos[2]+0.75]
#                    angle = [np.pi/2,np.pi/2+np.pi/4,room_yaw-np.pi/2]
#                    goal_center = [obst_pos[0]+0.4*np.cos(room_yaw+np.pi/2), obst_pos[1]+0.4*np.sin(room_yaw+np.pi/2)]
#
#                if target == 'light':
#                    obst_pos = [obst_pos[0]+1.2*np.cos(room_yaw+np.pi/1.5), obst_pos[1]+1.2*np.sin(room_yaw+np.pi/1.5), obst_pos[2]+0.88]
#                    #angle = [0,np.pi/2+np.pi/2,room_yaw-np.pi/2]
#                    angle = [np.pi/2,np.pi/2+np.pi/2,room_yaw-np.pi/2]
#                    goal_center = [obst_pos[0]+0.3*np.cos(room_yaw+np.pi/1.5), obst_pos[1]+0.3*np.sin(room_yaw+np.pi/1.5)]
#
#
#                if target == 'lamp':
#                    obst_pos = [obst_pos[0]-0.4*np.cos(room_yaw+np.pi/2), obst_pos[1]-0.4*np.sin(room_yaw+np.pi/2), obst_pos[2]+0.5]
#                    angle = [0,np.pi/2+np.pi/4,room_yaw-np.pi/2]
#
#                if target == 'oven':
#                    # mid
#                    obst_pos = [obst_pos[0]-0.35*np.cos(room_yaw+np.pi/2), obst_pos[1]-0.35*np.sin(room_yaw+np.pi/2), obst_pos[2]+0.7]
#                    angle = [0,np.pi/2+np.pi/2,room_yaw-np.pi/2]
#                    goal_center = obstacle_pose[0:3]
#
#                if target == 'cabinet2_cluster':
#                    obst_pos = [obst_pos[0]+0.25*np.cos(room_yaw+np.pi/2), obst_pos[1]+0.25*np.sin(room_yaw+np.pi/2), obst_pos[2]-0.1]
#                    angle = [0,np.pi/2+np.pi/4,room_yaw-np.pi/2]
#                    goal_center = [obst_pos[0]+0.5*np.cos(room_yaw+np.pi/1.4), obst_pos[1]+0.5*np.sin(room_yaw+np.pi/1.4)]
#
#                if target == 'mug_cluster':
#                    obst_pos = [obst_pos[0]-0.1*np.cos(room_yaw+np.pi/2), obst_pos[1]-0.1*np.sin(room_yaw+np.pi/2), obst_pos[2]-0.3]
#                    angle = [0,np.pi/2+np.pi/4,room_yaw-np.pi/2]
#                    goal_center = obstacle_pose[0:3]
#
#                obstacle_pose = torch.hstack(( torch.tensor(obst_pos,dtype=torch.float,device=device),
#                        euler_angles_to_quats(torch.tensor([angle],dtype=torch.float,device=device))[0] ))
#
#
#            obstacles[idx].set_world_pose(position=torch.tensor(obst_position,device=device),
#                                          orientation=euler_angles_to_quats(torch.tensor([[torch.pi/2,0,room_yaw]],device=device))[0])
#            # compute new AxisAligned bbox
#            self._scene._bbox_cache.Clear()
#            obst_aabbox = self._scene.compute_object_AABB(obstacles[idx].name)
#            # Check for overlap with all existing grasp objects
#            overlap = False
#            for other_aabbox in obst_aabboxes: # loop over existing AAbboxes
#                obst_range = Gf.Range3d(Gf.Vec3d(obst_aabbox[0,0],obst_aabbox[0,1],obst_aabbox[0,2]),Gf.Vec3d(obst_aabbox[1,0],obst_aabbox[1,1],obst_aabbox[1,2]))
#                other_obst_range = Gf.Range3d(Gf.Vec3d(other_aabbox[0,0],other_aabbox[0,1],other_aabbox[0,2]),Gf.Vec3d(other_aabbox[1,0],other_aabbox[1,1],other_aabbox[1,2]))
#                intersec = Gf.Range3d.GetIntersection(obst_range, other_obst_range)
#                if (not intersec.IsEmpty()):
#                    overlap = True # Failed. Try another pose
#                    break
#            if (overlap):
#                continue # Failed. Try another pose
#            else:
#                # Success. Add this valid AAbbox to the list
#                obst_aabboxes.append(obst_aabbox)
#                # Store obstacle position, orientation (yaw) and dimensions
#                object_positions.append(obst_position)
#                object_yaws.append(room_yaw)
#                objects_dimensions.append(obstacles_dimensions[idx])
#                break
#
#    # All objects placed in the scene!
#    # Pick one object to be the grasp object and compute its grasp:
#    goal_obj_index = np.random.randint(len(grasp_objs))
#    # For now, generating only top grasps: no roll, pitch 90, same yaw as object
#    goal_roll = 0.0 # np.random.uniform(-np.pi,np.pi)
#    goal_pitch = np.pi/2.0 # np.random.uniform(0,np.pi/2.0)
#    goal_yaw = object_yaws[goal_obj_index]
#    goal_position = np.array(object_positions[goal_obj_index])
#    goal_position[2] = (grasp_obj_aabboxes[goal_obj_index][1,2] + np.random.uniform(0.05,0.20)) # Add (random) z offset to object top (5 to 20 cms)
#    goal_pose = torch.hstack(( torch.tensor(goal_position,dtype=torch.float,device=device),
#                        euler_angles_to_quats(torch.tensor([[goal_roll,goal_pitch,goal_yaw]],dtype=torch.float,device=device))[0] ))
#
#
#    # Remove the goal object from obj_positions and yaws list (for computing oriented bboxes)
#    del object_positions[goal_obj_index], object_yaws[goal_obj_index]
#
#    # Compute oriented bounding boxes for all remaining objects
#    for idx in range(len(object_positions)):
#        bbox_tf = np.zeros((3,3))
#        bbox_tf[:2,:2] = np.array([[np.cos(object_yaws[idx]), -np.sin(object_yaws[idx])],[np.sin(object_yaws[idx]), np.cos(object_yaws[idx])]])
#        bbox_tf[:,-1] = np.array([object_positions[idx][0], object_positions[idx][1], 1.0]) # x,y,1
#        min_xy_vertex = np.array([[objects_dimensions[idx][0,0],objects_dimensions[idx][0,1],1.0]]).T
#        max_xy_vertex = np.array([[objects_dimensions[idx][1,0],objects_dimensions[idx][1,1],1.0]]).T
#        new_min_xy_vertex = (bbox_tf @ min_xy_vertex)[0:2].T.squeeze()
#        new_max_xy_vertex = (bbox_tf @ max_xy_vertex)[0:2].T.squeeze()
#        z_top_to_ground = object_positions[idx][2] + objects_dimensions[idx][1,2] # z position plus distance to object top
#        # Oriented bbox: x0, y0, x1, y1, z_to_ground, yaw
#        oriented_bbox = torch.tensor([ new_min_xy_vertex[0], new_min_xy_vertex[1],
#                                       new_max_xy_vertex[0], new_max_xy_vertex[1],
#                                            z_top_to_ground,     object_yaws[idx], ] ,dtype=torch.float,device=device)
#        if idx == 0:
#            object_oriented_bboxes = oriented_bbox
#        else:
#            object_oriented_bboxes = torch.vstack(( object_oriented_bboxes, oriented_bbox ))
#
#
#    return grasp_objs[goal_obj_index], obstacle_pose,  object_oriented_bboxes
#
# class DynamicObject(RigidPrim, GeometryPrim):
#     """Creates and adds a prim to stage from USD reference path, and wraps the prim with RigidPrim and GeometryPrim to
#        provide access to APIs for rigid body attributes, physics materials and collisions. Please note that this class
#        assumes the object has only a single mesh prim defining its geometry.

#     Args:
#         usd_path (str): USD reference path the Prim refers to.
#         prim_path (str): prim path of the Prim to encapsulate or create.
#         mesh_path (str): prim path of the underlying mesh Prim.
#         name (str, optional): shortname to be used as a key by Scene class. Note: needs to be unique if the object is
#                               added to the Scene. Defaults to "dynamic_object".
#         position (Optional[np.ndarray], optional): position in the world frame of the prim. Shape is (3, ). Defaults to
#                                                    None, which means left unchanged.
#         translation (Optional[np.ndarray], optional): translation in the local frame of the prim (with respect to its
#                                                       parent prim). Shape is (3, ). Defaults to None, which means left
#                                                       unchanged.
#         orientation (Optional[np.ndarray], optional): quaternion orientation in the world/local frame of the prim
#                                                       (depends if translation or position is specified). Quaternion is
#                                                       scalar-first (w, x, y, z). Shape is (4, ). Defaults to None, which
#                                                       means left unchanged.
#         scale (Optional[np.ndarray], optional): local scale to be applied to the prim's dimensions. Shape is (3, ).
#                                                 Defaults to None, which means left unchanged.
#         visible (bool, optional): set to false for an invisible prim in the stage while rendering. Defaults to True.
#         mass (Optional[float], optional): mass in kg. Defaults to None.
#         linear_velocity (Optional[np.ndarray], optional): linear velocity in the world frame. Defaults to None.
#         angular_velocity (Optional[np.ndarray], optional): angular velocity in the world frame. Defaults to None.
#     """

#     def __init__(
#         self,
#         usd_path: str,
#         prim_path: str,
#         mesh_path: str,
#         name: str = "dynamic_object",
#         position: Optional[np.ndarray] = None,
#         translation: Optional[np.ndarray] = None,
#         orientation: Optional[np.ndarray] = None,
#         scale: Optional[np.ndarray] = None,
#         visible: bool = True,
#         mass: Optional[float] = None,
#         linear_velocity: Optional[np.ndarray] = None,
#         angular_velocity: Optional[np.ndarray] = None,
#     ) -> None:

#         if is_prim_path_valid(mesh_path):
#             prim = get_prim_at_path(mesh_path)
#             if not prim.IsA(UsdGeom.Mesh):
#                 raise Exception("The prim at path {} cannot be parsed as a Mesh object".format(mesh_path))

#         self.usd_path = usd_path

#         add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

#         GeometryPrim.__init__(
#             self,
#             prim_path=mesh_path,
#             name=name,
#             translation=translation,
#             orientation=orientation,
#             visible=visible,
#             collision=True,
#         )

#         self.set_collision_approximation("convexHull")

#         RigidPrim.__init__(
#             self,
#             prim_path=prim_path,
#             name=name,
#             position=position,
#             translation=translation,
#             orientation=orientation,
#             scale=scale,
#             visible=visible,
#             mass=mass,
#             linear_velocity=linear_velocity,
#             angular_velocity=angular_velocity,
#         )
