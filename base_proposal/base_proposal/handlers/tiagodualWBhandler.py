import torch
import math

from base_proposal.handlers.base.tiagohandler import TiagoBaseHandler
from base_proposal.robots.articulations.tiago_dual_holo import TiagoDualHolo
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp
from omni.isaac.core.utils.stage import get_current_stage
import numpy as np


# Whole Body robot handler for the dual-armed Tiago robot
class TiagoDualWBHandler(TiagoBaseHandler):
    def __init__(self, move_group, use_torso, sim_config, num_envs, device):
        # self.task = task_class
        self._move_group = move_group
        self._use_torso = use_torso
        self._sim_config = sim_config
        self._num_envs = num_envs
        self._robot_positions = torch.tensor(
            [0, 0, 0]
        )  # placement of the robot in the world
        self._device = device

        # Custom default values of arm joints
        # middle of joint ranges
        # self.arm_left_start =  torch.tensor([0.19634954, 0.19634954, 1.5708,
        #                                     0.9817477, 0.0, 0.0, 0.0], device=self._device)
        # self.arm_right_start = torch.tensor([0.19634954, 0.19634954, 1.5708,
        #                                     0.9817477, 0.0, 0.0, 0.0], device=self._device)
        # Start at 'home' positions
        #        self.arm_left_start = torch.tensor([-1.2708, 1.3708, 0,
        #                                            -0.8, 1.5708, 1.5708, 0], device=self._device)
        #        self.arm_right_start = torch.tensor([-1.2708, 1.3708, 0,
        #                                            -0.8, 1.5708, 1.5708, 0], device=self._device)

        # self.joint_pos_min = np.array([-1.1780972451, -1.1780972451, -0.785398163397, -0.392699081699, -2.09439510239, -1.41371669412, -2.09439510239])
        # self.joint_pos_max = np.array([+1.57079632679, +1.57079632679, +3.92699081699, +2.35619449019, +2.09439510239, +1.41371669412, +2.09439510239])
        self.arm_left_start = torch.tensor(
            [
                1.1,  #  arm 向前
                -0.85,  # 抬高 shoulder
                0.3,
                1.0,  # elbow
                0.0,
                -1,
                0.0,  # No wrist roll
            ],
            device=self._device,
        )

        self.arm_right_start = torch.tensor(
            [0.25, 1.5707, 1.5707, 0.55, 1, -1.5707, 1.0], device=self._device
        )
        self.arm_left_start = torch.tensor(
            [0.25, 1.5707, 1.5707, 0.55, 1, -1.5707, 1.0], device=self._device
        )
        self.left_arm_pos = self.arm_left_start
        #  self.arm_left_start = torch.tensor([0.0, 1.5708, 1.5708,
        #                                      1.5708, 1.5708, -1.5708, 1.5708], device=self._device)
        #  self.arm_right_start = torch.tensor([0.0, 1.5708, 1.5708,
        #                                      1.5708, 1.5708, -1.5708, 1.5708], device=self._device)
        self.gripper_left_start = torch.tensor(
            [0.045, 0.045], device=self._device
        )  # Opened gripper by default
        self.gripper_left_pos = self.gripper_left_start
        self.gripper_right_start = torch.tensor(
            [0.045, 0.045], device=self._device
        )  # Opened gripper by default
        self.torso_fixed_state = torch.tensor([0.25], device=self._device)

        self.default_zero_env_path = "/World/envs/env_0"

        # self.max_arm_vel = torch.tensor(self._sim_config.task_config["env"]["max_rot_vel"], device=self._device)
        # self.max_base_rot_vel = torch.tensor(self._sim_config.task_config["env"]["max_rot_vel"], device=self._device)
        # self.max_base_xy_vel = torch.tensor(self._sim_config.task_config["env"]["max_base_xy_vel"], device=self._device)
        # Get dt for integrating velocity commands
        self.dt = torch.tensor(
            self._sim_config.task_config["sim"]["dt"]
            * self._sim_config.task_config["env"]["controlFrequencyInv"],
            device=self._device,
        )

        # articulation View will be created later
        self.robots = None

        # joint names
        self._base_joint_names = ["X", "Y", "R"]
        self._torso_joint_name = ["torso_lift_joint"]
        self._arm_left_names = []
        self._arm_right_names = []
        for i in range(7):
            self._arm_left_names.append(f"arm_left_{i + 1}_joint")
            self._arm_right_names.append(f"arm_right_{i + 1}_joint")

        # Future: Use end-effector link names and get their poses and velocities from Isaac
        self.ee_left_prim = ["gripper_left_grasping_frame"]
        self.ee_right_prim = ["gripper_right_grasping_frame"]
        # self.ee_left_tf =  torch.tensor([[ 0., 0.,  1.,  0.      ], # left_7_link to ee tf
        #                                  [ 0., 1.,  0.,  0.      ],
        #                                  [-1., 0.,  0., -0.196575],
        #                                  [ 0., 0.,  0.,  1.      ]])
        # self.ee_right_tf = torch.tensor([[ 0., 0., -1.,  0.      ], # right_7_link to ee tf
        #                                  [ 0., 1.,  0.,  0.      ],
        #                                  [ 1., 0.,  0.,  0.196575],
        #                                  [ 0., 0.,  0.,  1.      ]])
        self._gripper_left_names = [
            "gripper_left_left_finger_joint",
            "gripper_left_right_finger_joint",
        ]
        self._gripper_right_names = [
            "gripper_right_left_finger_joint",
            "gripper_right_right_finger_joint",
        ]

        # values are set in post_reset after model is loaded
        self.base_dof_idxs = []
        self.torso_dof_idx = []
        self.arm_left_dof_idxs = []
        self.arm_right_dof_idxs = []
        self.gripper_left_dof_idxs = []
        self.gripper_right_dof_idxs = []
        self.upper_body_dof_idxs = []
        self.combined_dof_idxs = []

        # dof joint position limits
        self.torso_dof_lower = []
        self.torso_dof_upper = []
        self.arm_left_dof_lower = []
        self.arm_left_dof_upper = []
        self.arm_right_dof_lower = []
        self.arm_right_dof_upper = []

    def get_robot(self):
        # make it in task and use handler as getter for path
        tiago = TiagoDualHolo(
            prim_path=self.default_zero_env_path + "/TiagoDualHolo",
            name="TiagoDualHolo",
            translation=self._robot_positions,
        )
        # Optional: Apply additional articulation settings
        self._sim_config.apply_articulation_settings(
            "TiagoDualHolo",
            get_prim_at_path(tiago.prim_path),
            self._sim_config.parse_actor_config("TiagoDualHolo"),
        )

    # call it in setup_up_scene in Task
    def create_articulation_view(self):
        self.robots = ArticulationView(
            prim_paths_expr="/World/envs/.*/TiagoDualHolo", name="tiago_dual_holo_view"
        )
        return self.robots

    def post_reset(self):
        # reset that takes place when the isaac world is reset (typically happens only once)
        # add dof indexes
        self._set_dof_idxs()
        # set dof limits
        self._set_dof_limits()
        # set new default state for reset
        self._set_default_state()
        # get stage
        self._stage = get_current_stage()

    def _set_dof_idxs(self):
        [
            self.base_dof_idxs.append(self.robots.get_dof_index(name))
            for name in self._base_joint_names
        ]
        [
            self.torso_dof_idx.append(self.robots.get_dof_index(name))
            for name in self._torso_joint_name
        ]
        [
            self.arm_left_dof_idxs.append(self.robots.get_dof_index(name))
            for name in self._arm_left_names
        ]
        [
            self.arm_right_dof_idxs.append(self.robots.get_dof_index(name))
            for name in self._arm_right_names
        ]
        [
            self.gripper_left_dof_idxs.append(self.robots.get_dof_index(name))
            for name in self._gripper_left_names
        ]
        [
            self.gripper_right_dof_idxs.append(self.robots.get_dof_index(name))
            for name in self._gripper_right_names
        ]
        self.upper_body_dof_idxs = []
        if self._use_torso:
            self.upper_body_dof_idxs += self.torso_dof_idx

        if self._move_group == "arm_left":
            self.upper_body_dof_idxs += self.arm_left_dof_idxs
        elif self._move_group == "arm_right":
            self.upper_body_dof_idxs += self.arm_right_dof_idxs
        elif self._move_group == "both_arms":
            self.upper_body_dof_idxs += self.arm_left_dof_idxs + self.arm_right_dof_idxs
        else:
            raise ValueError("move_group not defined")
        # Future: Add end-effector prim paths
        self.combined_dof_idxs = self.base_dof_idxs + self.upper_body_dof_idxs

    def _set_dof_limits(self):  # dof position limits
        # (num_envs, num_dofs, 2)
        dof_limits = self.robots.get_dof_limits()
        dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        dof_limits_upper = dof_limits[0, :, 1].to(self._device)

        # set relevant joint position limit values
        self.torso_dof_lower = dof_limits_upper[self.torso_dof_idx]
        self.torso_dof_upper = dof_limits_upper[self.torso_dof_idx]
        self.arm_left_dof_lower = dof_limits_lower[self.arm_left_dof_idxs]
        self.arm_left_dof_upper = dof_limits_upper[self.arm_left_dof_idxs]
        self.arm_right_dof_lower = dof_limits_lower[self.arm_right_dof_idxs]
        self.arm_right_dof_upper = dof_limits_upper[self.arm_right_dof_idxs]
        # self.gripper_dof_lower = dof_limits_lower[self.gripper_idxs]
        # self.gripper_dof_upper = dof_limits_upper[self.gripper_idxs]
        # Holo base has no limits

    def _set_default_state(self):
        joint_states = self.robots.get_joints_default_state()
        jt_pos = joint_states.positions
        jt_pos[:, self.torso_dof_idx] = self.torso_fixed_state
        jt_pos[:, self.arm_left_dof_idxs] = self.arm_left_start
        jt_pos[:, self.arm_right_dof_idxs] = self.arm_right_start
        jt_pos[:, self.gripper_left_dof_idxs] = self.gripper_left_start
        jt_pos[:, self.gripper_right_dof_idxs] = self.gripper_right_start

        self.robots.set_joints_default_state(positions=jt_pos)

    def apply_actions(self, actions):
        # Actions are velocity commands
        # The first three actions are the base velocities
        self.apply_base_actions(actions=actions[:, :3])
        self.apply_upper_body_actions(actions=actions[:, 3:])

    def open_gripper(self):
        # Open gripper
        # self.robots.set_joint_positions(
        #    positions=torch.tensor(
        #        [0.2, 0.2], device=self._device
        #    ),  # set joint positions to open gripper
        #    joint_indices=self.gripper_left_dof_idxs,
        # )
        self.robots.set_joint_efforts(  # set joint efforts to close gripper
            efforts=torch.tensor([15.0, 15.0], device=self._device),
            joint_indices=self.gripper_left_dof_idxs,
        )

        jt_pos = self.robots.get_joint_positions(
            joint_indices=self.gripper_left_dof_idxs, clone=True
        )
        actions = torch.tensor([0.1, 0.1], device=self._device)
        jt_pos = actions
        # jt_pos += actions * self.dt

        self.robots.set_joint_position_targets(
            positions=jt_pos, joint_indices=self.gripper_left_dof_idxs
        )

    def set_gripper_positions(self, positions):
        # Set gripper positions
        self.robots.set_joint_positions(
            positions=positions, joint_indices=self.gripper_left_dof_idxs
        )

    # self.robots.set_joint_position_targets(
    #     positions=positions, joint_indices=self.gripper_left_dof_idxs
    # )

    def close_gripper(self, action=torch.tensor([-0.1, -0.1])):
        # Close gripper
        self.robots.set_joint_efforts(  # set joint efforts to close gripper
            # efforts=torch.tensor([-1.0, -1.0], device=self._device),
            # efforts=torch.tensor([-15.0, -15.0], device=self._device),
            efforts=torch.tensor([-100.0, -100.0], device=self._device),
            joint_indices=self.gripper_left_dof_idxs,
        )

        jt_pos = self.robots.get_joint_positions(
            joint_indices=self.gripper_left_dof_idxs, clone=True
        )
        actions = action
        jt_pos = actions
        # jt_pos += actions * self.dt

        self.robots.set_joint_position_targets(
            positions=jt_pos, joint_indices=self.gripper_left_dof_idxs
        )

    def lift(self):
        # set arm
        arm_pos = self.robots.get_joint_positions(  # get current joint positions
            joint_indices=self.arm_left_dof_idxs, clone=True
        )
        self.robots.set_joint_positions(  # set joint position targets to lift arm
            positions=arm_pos, joint_indices=self.arm_left_dof_idxs
        )
        # lift torso
        jt_pos = self.robots.get_joint_positions(
            joint_indices=self.torso_dof_idx, clone=True
        )
        jt_pos += torch.tensor([0.1], device=self._device)
        # self.robots.set_joint_positions(  # set joint position targets to lift torso
        #     positions=jt_pos, joint_indices=self.torso_dof_idx
        # )
        self.robots.set_joint_position_targets(
            positions=jt_pos, joint_indices=self.torso_dof_idx
        )

    # self.robots.set_joint_position_targets(  # set joint position targets to lift torso
    #     positions=torch.tensor([3.5], device=self._device),
    #     joint_indices=self.torso_dof_idx,
    # )

    def apply_upper_body_actions(self, actions):
        # Apply actions as per the selected upper_body_dof_idxs (move_group)
        # Velocity commands (rad/s) will be converted to next-position (rad) targets
        jt_pos = self.robots.get_joint_positions(
            joint_indices=self.upper_body_dof_idxs, clone=True
        )
        jt_pos += actions * self.dt  # create new position targets
        # self.robots.set_joint_position_targets(positions=jt_pos, joint_indices=self.upper_body_dof_idxs)
        # TEMP: Use direct joint positions
        self.robots.set_joint_positions(
            positions=jt_pos, joint_indices=self.upper_body_dof_idxs
        )
        if not self._use_torso:
            # Hack to avoid torso falling when it isn't controlled
            pos = self.torso_fixed_state.unsqueeze(dim=0)
            self.robots.set_joint_positions(
                positions=pos, joint_indices=self.torso_dof_idx
            )
        if (
            self._move_group == "arm_left"
        ):  # Hack to avoid arm falling when it isn't controlled
            pos = self.arm_right_start.unsqueeze(dim=0)
            self.robots.set_joint_positions(
                positions=pos, joint_indices=self.arm_right_dof_idxs
            )
        elif (
            self._move_group == "arm_right"
        ):  # Hack to avoid arm falling when it isn't controlled
            pos = self.arm_left_start.unsqueeze(dim=0)
            self.robots.set_joint_positions(
                positions=pos, joint_indices=self.arm_left_dof_idxs
            )
        elif self._move_group == "both_arms":
            pass

    #    def apply_base_actions(self, actions):
    #        base_actions = actions.clone()
    #        # self.robots.set_joint_velocity_targets(velocities=base_actions, joint_indices=self.base_dof_idxs)
    #        # TEMP: Use direct joint positions
    #        jt_pos = self.robots.get_joint_positions(joint_indices=self.base_dof_idxs, clone=True)
    #        jt_pos += base_actions*self.dt # create new position targets
    #        self.robots.set_joint_positions(positions=jt_pos, joint_indices=self.base_dof_idxs)
    def base_rotate(self):
        self.set_joint_velocities(  # set joint velocity targets
            velocities=torch.tensor([0, 0, 1], device=self._device),
            joint_indices=self.base_dof_idxs,
        )

    def base_forward(self):
        self.set_joint_velocities(  # set joint velocity targets
            velocities=torch.tensor([1.0, 0.0, 0.0], device=self._device),
            joint_indices=self.base_dof_idxs,
        )

    def base_stop(self):
        self.robots.set_joint_velocities(  # set joint velocity targets
            velocities=torch.tensor([0.0, 0.0, 0.0], device=self._device),
            joint_indices=self.base_dof_idxs,
        )

    def base_backward(self):
        arm_pos = self.robots.get_joint_positions(  # get current joint positions
            joint_indices=self.arm_left_dof_idxs, clone=True
        )

        self.robots.set_joint_position_targets(  # set joint position targets to lift arm)
            positions=arm_pos, joint_indices=self.arm_left_dof_idxs
        )
        self.close_gripper()

        # gripper_pos = self.robots.get_joint_positions(
        #     joint_indices=self.gripper_left_dof_idxs, clone=True
        # )
        # self.robots.set_joint_position_targets(
        #     positions=gripper_pos, joint_indices=self.gripper_left_dof_idxs
        # )

        self.robots.set_joint_velocities(  # set joint velocity targets
            velocities=torch.tensor([0.0, 0.1, 0.0], device=self._device),
            joint_indices=self.base_dof_idxs,
        )

        self.robots.set_joint_efforts(
            efforts=torch.tensor([0, 1000.0, 0.0], device=self._device),
            joint_indices=self.base_dof_idxs,
        )

    def apply_base_actions1(self, actions):
        from pyquaternion import Quaternion

        # set arm
        arm_pos = self.robots.get_joint_positions(  # get current joint positions
            joint_indices=self.arm_left_dof_idxs, clone=True
        )
        self.robots.set_joint_positions(  # set joint position targets to lift arm
            positions=arm_pos, joint_indices=self.arm_left_dof_idxs
        )

        base_actions = actions.clone()

        # 取得當前 base 的位置
        jt_pos = self.robots.get_joint_positions(
            joint_indices=self.base_dof_idxs, clone=True
        )
        jt_pos = jt_pos[0]
        base_actions = base_actions[0]

        # **計算移動方向並旋轉**
        rotated_x = base_actions[0] * math.cos(jt_pos[2]) - base_actions[1] * math.sin(
            jt_pos[2]
        )
        rotated_y = base_actions[0] * math.sin(jt_pos[2]) + base_actions[1] * math.cos(
            jt_pos[2]
        )

        # **計算新的位置變化量**
        dx = rotated_x * self.dt
        dy = rotated_y * self.dt
        dtheta = base_actions[2] * self.dt

        # **計算速度**
        vx = dx / self.dt  # X 方向速度
        vy = dy / self.dt  # Y 方向速度
        omega = dtheta / self.dt  # 角速度

        # **應用速度控制**
        self.robots.set_joint_velocities(
            velocities=torch.tensor([vx, vy, omega], device=self._device),
            joint_indices=self.base_dof_idxs,
        )

        # **更新機器人的姿態（僅用於記錄）**
        jt_pos[0] += dx
        jt_pos[1] += dy
        jt_pos[2] += dtheta

        # 計算新的四元數姿態
        current_orientation = Quaternion(axis=[0, 0, 1], radians=jt_pos[2])
        rotation = Quaternion(axis=[0, 0, 1], radians=dtheta)
        new_orientation = current_orientation * rotation

        # 更新機器人的內部 pose
        quatpose = np.array(
            [
                jt_pos[0],
                jt_pos[1],
                0,
                new_orientation.real,
                new_orientation.imaginary[0],
                new_orientation.imaginary[1],
                new_orientation.imaginary[2],
            ]
        )
        self._robot_pose = quatpose

    def set_left_arm_dof_pos(self):
        dof_pos = self.robots.get_joint_positions(  # get joint positions
            joint_indices=self.arm_left_dof_idxs, clone=True
        )
        self.left_arm_pos = dof_pos
        dof_pos = self.robots.get_joint_positions(  # get joint positions
            joint_indices=self.gripper_left_dof_idxs, clone=True
        )
        self.gripper_left_pos = dof_pos

    def apply_base_actions(self, actions):
        from pyquaternion import Quaternion
        import math

        base_actions = actions.clone()

        #  arm_pos = self.robots.get_joint_positions(  # get current joint positions
        #      joint_indices=self.arm_left_dof_idxs, clone=True
        #  )

        self.robots.set_joint_positions(  # set joint position targets to lift arm)
            positions=self.left_arm_pos, joint_indices=self.arm_left_dof_idxs
        )
        # gripper_pos = self.robots.get_joint_positions(
        #     joint_indices=self.gripper_left_dof_idxs, clone=True
        # )
        gripper_pos = self.gripper_left_pos
        self.robots.set_joint_positions(
            positions=gripper_pos, joint_indices=self.gripper_left_dof_idxs
        )
        jt_pos = self.robots.get_joint_positions(
            joint_indices=self.base_dof_idxs, clone=True
        )
        jt_pos = jt_pos[0]
        base_actions = base_actions[0]

        rotated_x = base_actions[0] * math.cos(jt_pos[2]) - base_actions[1] * math.sin(
            jt_pos[2]
        )
        rotated_y = base_actions[0] * math.sin(jt_pos[2]) + base_actions[1] * math.cos(
            jt_pos[2]
        )

        jt_pos[0] += rotated_x * self.dt
        jt_pos[1] += rotated_y * self.dt
        jt_pos[2] += base_actions[2] * self.dt

        current_orientation = Quaternion(axis=[0, 0, 1], radians=jt_pos[2])
        rotation = Quaternion(axis=[0, 0, 1], radians=base_actions[2] * self.dt)
        new_orientation = current_orientation * rotation
        quatpose = [
            jt_pos[0],
            jt_pos[1],
            0,
            new_orientation.real,
            new_orientation.imaginary[0],
            new_orientation.imaginary[1],
            new_orientation.imaginary[2],
        ]

        self._robot_pose = quatpose

        self.robots.set_joint_positions(
            positions=jt_pos, joint_indices=self.base_dof_idxs
        )

    # self.robots.set_joint_velocities(
    #     velocities=base_actions, joint_indices=self.base_dof_idxs
    # )

    # self.robots.set_joint_position_targets(  # set joint position targets
    #    positions=jt_pos * 1000, joint_indices=self.base_dof_idxs
    # )

    def set_upper_body_positions(self, jnt_positions):
        # Set upper body joints to specific positions
        self.robots.set_joint_positions(
            positions=jnt_positions, joint_indices=self.upper_body_dof_idxs
        )
        # if not self._use_torso:
        #     # Hack to avoid torso falling when it isn't controlled
        #     pos = self.torso_fixed_state.unsqueeze(dim=0)
        #     self.robots.set_joint_positions(positions=pos, joint_indices=self.torso_dof_idx)
        if (
            self._move_group == "arm_left"
        ):  # Hack to avoid arm falling when it isn't controlled
            pos = self.arm_right_start.unsqueeze(dim=0)
            self.robots.set_joint_positions(
                positions=pos, joint_indices=self.arm_right_dof_idxs
            )
        # elif self._move_group == "arm_right": # Hack to avoid arm falling when it isn't controlled
        #     pos = self.arm_left_start.unsqueeze(dim=0)
        #     self.robots.set_joint_positions(positions=pos, joint_indices=self.arm_left_dof_idxs)
        # elif self._move_group == "both_arms":
        #     pass

    def get_upper_body_positions(self):
        # return positions and velocities of upper body joints
        combined_pos = self.robots.get_joint_positions(
            joint_indices=self.upper_body_dof_idxs, clone=True
        )
        return combined_pos

    def set_base_positions(self, jnt_positions):
        # Set base joints to specific positions
        self.robots.set_joint_positions(
            positions=jnt_positions, joint_indices=self.base_dof_idxs
        )

    def get_base_positions(self):
        # return positions and velocities of base joints
        combined_pos = self.robots.get_joint_positions(
            joint_indices=self.base_dof_idxs, clone=True
        )
        return combined_pos

    def get_robot_obs(self):
        # return positions and velocities of upper body and base joints
        combined_pos = self.robots.get_joint_positions(
            joint_indices=self.combined_dof_idxs, clone=True
        )
        # Base rotation continuous joint should be in range -pi to pi
        limits = combined_pos[:, 2] > torch.pi
        combined_pos[limits, 2] -= 2 * torch.pi
        limits = combined_pos[:, 2] < -torch.pi
        combined_pos[limits, 2] += 2 * torch.pi
        # NOTE: Velocities here will only be correct if the correct control mode is used!!
        combined_vel = self.robots.get_joint_velocities(
            joint_indices=self.combined_dof_idxs, clone=True
        )
        # Future: Add pose and velocity of end-effector from Isaac prims
        return torch.hstack((combined_pos, combined_vel))

    def set_arm_dof_init_pos(self):
        # Set arm joints to initial positions
        arm_left_pos = self.arm_left_start.unsqueeze(dim=0)
        arm_right_pos = self.arm_right_start.unsqueeze(dim=0)
        self.robots.set_joint_positions(
            positions=arm_left_pos, joint_indices=self.arm_left_dof_idxs
        )
        self.robots.set_joint_positions(
            positions=arm_right_pos, joint_indices=self.arm_right_dof_idxs
        )

    def get_arms_dof_pos(self):
        # (num_envs, num_dof)
        dof_pos = self.robots.get_joint_positions(clone=False)
        # left arm
        arm_left_pos = dof_pos[:, self.arm_left_dof_idxs]
        # right arm
        arm_right_pos = dof_pos[:, self.arm_right_dof_idxs]
        return arm_left_pos, arm_right_pos

    def get_arms_dof_vel(self):
        # (num_envs, num_dof)
        dof_vel = self.robots.get_joint_velocities(clone=False)
        # left arm
        arm_left_vel = dof_vel[:, self.arm_left_dof_idxs]
        # right arm
        arm_right_vel = dof_vel[:, self.arm_right_dof_idxs]
        return arm_left_vel, arm_right_vel

    def get_base_dof_values(self):
        base_pos = self.robots.get_joint_positions(
            joint_indices=self.base_dof_idxs, clone=False
        )
        base_vel = self.robots.get_joint_velocities(
            joint_indices=self.base_dof_idxs, clone=False
        )
        return base_pos, base_vel

    def get_torso_dof_values(self):
        torso_pos = self.robots.get_joint_positions(
            joint_indices=self.torso_dof_idx, clone=False
        )
        torso_vel = self.robots.get_joint_velocities(
            joint_indices=self.torso_dof_idx, clone=False
        )
        return torso_pos, torso_vel

    def reset(self, indices, randomize=False):
        num_resets = len(indices)
        joint_states = self.robots.get_joints_default_state()
        jt_pos = joint_states.positions.clone()
        jt_pos = jt_pos[0:num_resets]  # we need only num_resets rows
        if randomize:
            noise = torch_rand_float(
                -0.75,
                0.75,
                jt_pos[:, self.upper_body_dof_idxs].shape,
                device=self._device,
            )
            # Clip needed? dof_pos[:] = tensor_clamp(self.initial_dof_pos[env_ids] + dof_pos, self.dof_limits_lower, self.dof_limits_upper)
            # jt_pos[:, self.upper_body_dof_idxs] = noise
            jt_pos[:, self.upper_body_dof_idxs] += (
                noise  # Optional: Add to default instead
            )
        self.robots.set_joint_positions(jt_pos, indices=indices)
