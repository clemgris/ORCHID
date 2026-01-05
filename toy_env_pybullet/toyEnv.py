"""
Complete PyBullet Franka 3-Cube Multi-Task Environment
Full environment with reset, step, assets, simulation, and expert policies
Uses PyBullet's built-in inverse kinematics
"""

import random
from typing import Dict

import numpy as np
import pybullet as p
import pybullet_data
import torch

# Task definitions
TASK_NAMES = [
    "lift_A",  # 0: Lift cube A
    "lift_B",  # 1: Lift cube B
    "lift_C",  # 2: Lift cube C
    "stack",  # 3: Stack one cube on another (any valid pair)
    "push_C_left",  # 4: Push cube C to the left
    "push_C_right",  # 5: Push cube C to the right
    "push_B_left",  # 6: Push cube B to the left
    "push_B_right",  # 7: Push cube B to the right
    "push_A_left",  # 8: Push cube A to the left
    "push_A_right",  # 9: Push cube A to the right
]

TASK_INSTRUCTIONS = {
    "lift_A": "Lift the red block.",
    "lift_B": "Lift the green block.",
    "lift_C": "Lift the blue block.",
    "stack": "Stack one block on top of another block.",
    "push_A_left": "Push the red block to the left.",
    "push_A_right": "Push the red block to the right.",
    "push_B_left": "Push the green block to the left.",
    "push_B_right": "Push the green block to the right.",
    "push_C_left": "Push the blue block to the left.",
    "push_C_right": "Push the blue block to the right.",
}


class Franka3CubeEnvPyBullet:
    """
    Complete multi-task environment for Franka robot with 3 cubes using PyBullet
    Includes full PyBullet setup, reset, step, rewards, and expert policies
    """

    def __init__(
        self,
        cfg,
        headless=False,
        task_id=None,
    ):
        """
        Initialize the environment

        Args:
            cfg: Configuration dictionary with hyperparameters
            headless: Whether to run without visualization
            task_id: Fixed task ID (if None, will sample based on task_mode)
        """
        self.cfg = cfg
        self.headless = headless
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Environment parameters
        self.num_envs = cfg.get("num_envs", 1)  # PyBullet typically uses 1 env
        self.num_obs = cfg.get("num_obs", 27)
        self.num_actions = cfg.get("num_actions", 7)
        self.max_episode_length = cfg.get("max_episode_length", 500)

        # Control parameters
        self.dt = 1.0 / 240.0  # PyBullet default timestep
        self.control_freq_inv = cfg.get("control_freq_inv", 4)

        # EE Control parameters
        self.max_ee_pos_delta = 0.05
        self.max_ee_rot_delta = 0.1

        # Expert policy parameters
        self.grasp_height_offset = -0.001
        self.grasp_height_offset_x = 0.005
        self.grasp_threshold = 0.01
        self.lift_speed = 2.0
        self.approach_speed = 2.0

        # Cube parameters
        self.cube_size = 0.048
        self.cube_spacing = 0.10
        self.lift_height = 0.10
        self.stack_height = self.cube_size
        self.table_thickness = 0.25

        # Task-specific thresholds
        self.lift_height_threshold = 0.10
        self.lift_success_threshold = 0.02
        self.stack_height_threshold = self.cube_size
        self.stack_align_threshold = 0.02
        self.push_distance = 0.10
        self.push_success_threshold = 0.03
        self.phase_transition_threshold = 0.01
        self.grasp_close_threshold = self.cube_size
        self.push_height = 0.2 + self.cube_size / 2 + 0.02

        # Task management
        self.current_task_id = task_id if task_id is not None else 0
        self.task_mode = cfg.get("task_mode", "random")

        # State variables
        self.progress_buf = 0
        self.reset_buf = True
        self.success_buf = 0.0
        self.timeout_buf = False

        # Franka parameters
        self.franka_num_dofs = 9  # 7 arm + 2 gripper
        self.franka_ee_link_index = 11  # panda_hand link
        self.franka_gripper_indices = [9, 10]  # Gripper finger joints

        # Initialize simulation
        self._create_sim()
        self._load_assets()

        # Initial reset
        self.reset(task_id)

    def _create_sim(self):
        """Create PyBullet simulation"""
        if p.isConnected():
            p.disconnect()

        # Connect to PyBullet
        if self.headless:
            self.physics_client = p.connect(p.DIRECT)
        else:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.5, 0, 0.5],
            )

        # Set up simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

    def _load_assets(self):
        """Load Franka robot and create cubes"""
        # Load Franka Panda
        self.franka_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )

        # Set Franka joint damping for stability
        for i in range(p.getNumJoints(self.franka_id)):
            p.changeDynamics(self.franka_id, i, linearDamping=0, angularDamping=0)

        # Create table
        table_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.3, 0.8, self.table_thickness / 2]
        )
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.3, 0.8, self.table_thickness / 2],
            rgbaColor=[0.5, 0.5, 0.5, 1],
        )
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0.5, 0, self.table_thickness / 2],
        )

        # Create cubes with different colors
        cube_colors = [
            [1, 0, 0, 1],  # Red - Cube A
            [0, 1, 0, 1],  # Green - Cube B
            [0, 0, 1, 1],  # Blue - Cube C
        ]

        self.cube_ids = []
        for i, color in enumerate(cube_colors):
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[self.cube_size / 2] * 3
            )
            visual_shape = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[self.cube_size / 2] * 3, rgbaColor=color
            )
            cube_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[0.5, 0, self.table_thickness + self.cube_size / 2],
            )
            self.cube_ids.append(cube_id)

        # Table friction and restitution
        p.changeDynamics(
            self.table_id,
            -1,
            lateralFriction=1.0,
            restitution=0.1,  # Low restitution = less bouncy
        )

        # Cube dynamics - reduce bounciness
        for cube_id in self.cube_ids:
            p.changeDynamics(
                cube_id,
                -1,
                lateralFriction=1.0,
                restitution=0.1,  # Low restitution = less bouncy (0 = no bounce, 1 = perfectly elastic)
                linearDamping=0.5,  # Dampen linear motion
                angularDamping=0.5,  # Dampen rotation
                rollingFriction=0.01,  # Add rolling resistance
                spinningFriction=0.01,  # Add spinning resistance
            )

        self.cube_name_to_id = {
            "A": self.cube_ids[0],
            "B": self.cube_ids[1],
            "C": self.cube_ids[2],
        }

        # Get initial joint configuration
        self.franka_default_joints = [
            -7.7001e-05,
            -1.9163e-01,
            3.1069e-04,
            -2.0947e00,
            -1.1069e-04,
            1.9072e00,
            7.8494e-01,
            0.0000e00,
            0.0000e00,
        ]
        # Set up camera
        self.camera_setup()

    def camera_setup(self):
        """Setup camera for rendering"""
        self.camera_width = 256
        self.camera_height = 256

        # Camera parameters
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.9, -0.2, 0.6],
            cameraTargetPosition=[0.5, 0, 0.3],
            cameraUpVector=[0, 0, 1],
        )
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=3.0
        )

    def _get_camera_image(self):
        """Get camera image as tensor"""
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        # Convert to torch tensor [C, H, W]
        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel

        # Convert to torch and normalize
        img_tensor = torch.from_numpy(rgb_array).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # [C, H, W]

        return img_tensor.unsqueeze(0)  # [1, C, H, W]

    def reset(self, task_id=None, scene_obs=None):
        """Reset environment"""
        # Just reload the entire world
        p.resetSimulation()
        self._create_sim()
        self._load_assets()

        # Reset max cube heights atained by each cube during the episode
        self.max_cube_height = {
            "A": 0,
            "B": 0,
            "C": 0,
        }

        if scene_obs is not None:
            return self.reset_scene(scene_obs)
        else:
            # Reset Franka to default position
            for i, pos in enumerate(self.franka_default_joints):
                p.resetJointState(self.franka_id, i, pos, targetVelocity=0.0)
            # Reset cubes to random positions on table
            table_size = 0.2
            margin = 0.05 * table_size

            cube_positions = []
            for i, cube_id in enumerate(self.cube_ids):
                while True:
                    x = random.uniform(
                        0.5 - table_size / 2 + margin, 0.5 + table_size / 2 - margin
                    )
                    y = random.uniform(
                        -table_size / 2 + margin, table_size / 2 - margin
                    )
                    pos = [x, y, self.table_thickness + self.cube_size / 2]

                    # Check overlap with existing cubes
                    overlap = False
                    for prev_pos in cube_positions:
                        dist = np.linalg.norm(
                            np.array(pos[:2]) - np.array(prev_pos[:2])
                        )
                        if dist < self.cube_size + 0.005:
                            overlap = True
                            break

                    if not overlap:
                        cube_positions.append(pos)
                        break

                p.resetBasePositionAndOrientation(cube_id, pos, [0, 0, 0, 1])

        # Sample task
        if task_id is None:
            if self.task_mode == "random":
                self.current_task_id = random.randint(0, len(TASK_NAMES) - 1)
            elif self.task_mode == "sequential":
                self.current_task_id = (self.current_task_id + 1) % len(TASK_NAMES)
        else:
            self.current_task_id = task_id

        self.reward_fn = self.get_reward_function(self.current_task_id)

        # Reset tracking
        self.progress_buf = 0
        self.reset_buf = False
        self.success_buf = 0.0
        self.timeout_buf = False

        # Step simulation to settle
        for _ in range(100):
            p.stepSimulation()

        obs = self.compute_observations()
        self.init_obs = obs

        return obs

    def reset_scene(self, scene_obs):
        """Reset environment to specific scene observation"""
        if isinstance(scene_obs, Dict):
            state = self.parse_state(scene_obs["state"])
        else:
            state = self.parse_state(scene_obs)
        # Reset Franka joint positions
        # dof_pos = state["dof_pos"].squeeze(0).numpy()
        # for i in range(self.franka_num_dofs):
        #     p.resetJointState(self.franka_id, i, dof_pos[i], targetVelocity=0.0)
        # Reset cube positions
        cube_positions = [
            state["cube_a_pos"].squeeze(0).numpy(),
            state["cube_b_pos"].squeeze(0).numpy(),
            state["cube_c_pos"].squeeze(0).numpy(),
        ]
        for i, cube_id in enumerate(self.cube_ids):
            p.resetBasePositionAndOrientation(cube_id, cube_positions[i], [0, 0, 0, 1])

        # Step simulation to settle
        for _ in range(100):
            p.stepSimulation()

        obs = self.compute_observations()
        self.init_obs = obs

        return obs

    def compute_observations(self):
        """Compute current observations"""
        # Get end-effector state
        ee_state = p.getLinkState(self.franka_id, self.franka_ee_link_index)
        hand_pos = torch.tensor(ee_state[0], dtype=torch.float32)
        hand_rot = torch.tensor(
            ee_state[1], dtype=torch.float32
        )  # quaternion [x,y,z,w]

        # Get cube positions
        cube_positions = []
        for cube_id in self.cube_ids:
            pos, _ = p.getBasePositionAndOrientation(cube_id)
            cube_positions.append(torch.tensor(pos, dtype=torch.float32))

        # Get gripper position
        gripper_pos = p.getJointState(self.franka_id, self.franka_gripper_indices[0])[0]
        gripper_tensor = torch.tensor([gripper_pos], dtype=torch.float32)

        # Get all joint positions
        joint_positions = []
        for i in range(self.franka_num_dofs):
            joint_positions.append(p.getJointState(self.franka_id, i)[0])
        dof_pos = torch.tensor(joint_positions, dtype=torch.float32)

        # Construct state observation
        state_obs = torch.cat(
            [
                hand_pos,
                hand_rot,
                *cube_positions,
                gripper_tensor,
                dof_pos,
                torch.tensor([self.current_task_id], dtype=torch.float32),
            ]
        ).unsqueeze(0)  # Add batch dimension

        # Get camera observation
        camera_obs = self._get_camera_image()

        return {"state": state_obs, "pixels": camera_obs}

    def step(self, actions):
        """
        Step environment with 7-DOF end-effector control

        Args:
            actions: [7] or [1, 7] tensor
                    [0:3] - dx, dy, dz (position delta, normalized -1 to 1)
                    [3:6] - droll, dpitch, dyaw (orientation delta, normalized -1 to 1)
                    [6]   - dgripper (gripper command, -1=open, 1=close)
        """
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()

        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        actions = actions.to(torch.float32)

        # Get current EE state
        ee_state = p.getLinkState(self.franka_id, self.franka_ee_link_index)
        current_pos = np.array(ee_state[0])
        current_orn = np.array(ee_state[1])  # quaternion

        # Parse actions
        pos_delta = actions[0, :3].numpy() * self.max_ee_pos_delta
        rot_delta = actions[0, 3:6].numpy() * self.max_ee_rot_delta
        gripper_cmd = actions[0, 6].item()

        # Compute desired EE pose
        desired_pos = current_pos + pos_delta

        # Apply rotation delta (convert euler to quaternion and compose)
        rot_quat = p.getQuaternionFromEuler(rot_delta)
        desired_orn = p.multiplyTransforms([0, 0, 0], current_orn, [0, 0, 0], rot_quat)[
            1
        ]

        # Use PyBullet's built-in IK
        joint_poses = p.calculateInverseKinematics(
            self.franka_id,
            self.franka_ee_link_index,
            desired_pos,
            desired_orn,
            maxNumIterations=100,
            residualThreshold=1e-5,
        )

        # Set arm joint targets (first 7 joints)
        for i in range(7):
            p.setJointMotorControl2(
                self.franka_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500,
            )

        # Control gripper
        current_gripper = p.getJointState(
            self.franka_id, self.franka_gripper_indices[0]
        )[0]
        gripper_target = np.clip(current_gripper + gripper_cmd * 0.1, 0.0, 0.04)

        for gripper_idx in self.franka_gripper_indices:
            p.setJointMotorControl2(
                self.franka_id,
                gripper_idx,
                p.POSITION_CONTROL,
                targetPosition=gripper_target,
                force=200,
            )

        # Step simulation
        for _ in range(self.control_freq_inv):
            p.stepSimulation()

        # Update tracking
        self.progress_buf += 1

        # Compute observations and rewards
        obs = self.compute_observations()
        reward = self.reward_fn(self.init_obs["state"], obs["state"])

        # Check termination
        done = False
        if self.progress_buf >= self.max_episode_length:
            self.timeout_buf = True
            done = True

        if reward > 0.5:  # Success
            self.success_buf = 1.0
            done = True

        parsed_state = self.parse_state(obs["state"])

        self.max_cube_height["A"] = max(
            self.max_cube_height["A"],
            parsed_state["cube_a_pos"][0, 2].item(),
        )
        self.max_cube_height["B"] = max(
            self.max_cube_height["B"],
            parsed_state["cube_b_pos"][0, 2].item(),
        )
        self.max_cube_height["C"] = max(
            self.max_cube_height["C"],
            parsed_state["cube_c_pos"][0, 2].item(),
        )

        info = {
            "task_id": self.current_task_id,
            "success": self.success_buf,
            "timeout": self.timeout_buf,
        }

        return obs, reward, done, info

    def parse_state(self, state):
        """Parse state tensor into components"""
        if isinstance(state, Dict):
            state = state["state"]

        if state.dim() == 1:
            state = state.unsqueeze(0)

        return {
            "hand_pos": state[:, 0:3],
            "hand_rot": state[:, 3:7],
            "cube_a_pos": state[:, 7:10],
            "cube_b_pos": state[:, 10:13],
            "cube_c_pos": state[:, 13:16],
            "gripper_pos": state[:, 16:17],
            "dof_pos": state[:, 17:26],
            "task_id": state[:, 26:27],
        }

    def _go_to(
        self,
        current_obs,
        target_pos,
        target_quat=None,
        position_gain=0.65,
        orientation_gain=0.5,
        gripper_cmd=None,
        position_threshold=0.005,
    ):
        """
        Compute action to move end-effector toward target

        Args:
            current_obs: dict with state observations
            target_pos: [3] or [1, 3] target position
            target_quat: [4] or [1, 4] optional target orientation (x,y,z,w)
            position_gain: Scale factor for position control
            orientation_gain: Scale factor for orientation control
            gripper_cmd: Gripper command (-1 to 1)
            position_threshold: Distance threshold
        """
        current_state = self.parse_state(current_obs)
        hand_pos = current_state["hand_pos"]
        hand_rot = current_state["hand_rot"]

        if target_pos.dim() == 1:
            target_pos = target_pos.unsqueeze(0)

        # Initialize action
        actions = torch.zeros(1, 7)

        # Position control
        pos_error = target_pos - hand_pos
        pos_distance = torch.norm(pos_error, dim=-1, keepdim=True)

        pos_delta_normalized = pos_error / (self.max_ee_pos_delta + 1e-8)
        pos_delta_normalized = torch.clamp(pos_delta_normalized, -1.0, 1.0)

        distance_scale = torch.tanh(pos_distance / position_threshold)
        pos_delta_normalized = pos_delta_normalized * distance_scale * position_gain

        actions[:, 0:3] = pos_delta_normalized

        # Orientation control (if specified)
        if target_quat is not None:
            if target_quat.dim() == 1:
                target_quat = target_quat.unsqueeze(0)

            # Simple orientation error (could be improved)
            actions[:, 3:6] = 0.0  # Simplified for now
        else:
            actions[:, 3:6] = 0.0

        # Gripper control
        if gripper_cmd is not None:
            actions[:, 6] = gripper_cmd
        else:
            actions[:, 6] = 0.0

        return actions

    def get_gripper_contacts(self):
        """
        Returns a list of all contacts between the gripper and any other object.

        Each contact is a dictionary:
        {
            'gripper_link': int,
            'body_id': int,
            'contact_position': [x, y, z],
            'contact_normal': [x, y, z],
            'normal_force': float
        }
        """
        contacts_list = []

        for link_idx in self.franka_gripper_indices:
            # Get all contacts for this link
            contacts = p.getContactPoints(bodyA=self.franka_id, linkIndexA=link_idx)

            for c in contacts:
                # Ignore contacts with self (other links of the robot)
                if c[2] == self.franka_id:
                    continue

                contact_info = {
                    "gripper_link": link_idx,
                    "body_id": c[2],
                    "contact_position": c[6],  # contact point on body B
                    "contact_normal": c[7],
                    "normal_force": c[9],
                }
                contacts_list.append(contact_info)

        return contacts_list

    def has_cube(self, min_force=0.5):
        """
        Returns (True, cube_id) if BOTH gripper fingers
        are in contact with the SAME object.

        Args:
            min_force: minimum normal force to count as valid contact

        Returns:
            (bool, int or None)
        """
        contacts = self.get_gripper_contacts()

        # Map: body_id -> set of gripper links touching it
        body_to_links = {}

        for c in contacts:
            if c["normal_force"] < min_force:
                continue

            body_id = c["body_id"]
            link = c["gripper_link"]

            if body_id not in body_to_links:
                body_to_links[body_id] = set()

            body_to_links[body_id].add(link)

        # Check if any body is touched by BOTH fingers
        for body_id, links in body_to_links.items():
            if all(link in links for link in self.franka_gripper_indices):
                return True, body_id

        return False, None

    def is_touching_cube(self, cube_id, min_force=0.5):
        """
        Returns True if ANY gripper link is in contact with the specified cube.

        Args:
            cube_id: The body ID of the cube to check
            min_force: minimum normal force to count as valid contact

        Returns:
            bool: True if gripper is touching the cube, False otherwise
        """
        contacts = self.get_gripper_contacts()

        for c in contacts:
            if c["normal_force"] < min_force:
                continue

            if c["body_id"] == cube_id:
                return True

        return False

    def get_gripper_openness(self):
        """
        Returns gripper openness in meters [0.0 (closed) → ~0.04 (open)]
        """
        left = p.getJointState(self.franka_id, self.franka_gripper_indices[0])[0]
        right = p.getJointState(self.franka_id, self.franka_gripper_indices[1])[0]

        return 0.5 * (left + right)

    # ==================== LIFT TASKS ====================

    def compute_reward_lift(self, initial_state, current_state, cube_name="A"):
        """Binary reward for lifting a cube"""
        init = self.parse_state(initial_state)
        curr = self.parse_state(current_state)

        cube_map = {"A": "cube_a_pos", "B": "cube_b_pos", "C": "cube_c_pos"}
        cube_pos_key = cube_map[cube_name]

        init_cube_pos = init[cube_pos_key]
        curr_cube_pos = curr[cube_pos_key]

        target_height = init_cube_pos[:, 2] + self.lift_height_threshold
        at_target_height = curr_cube_pos[:, 2] >= target_height - 0.01

        horizontal_drift = torch.norm(
            curr_cube_pos[:, :2] - init_cube_pos[:, :2], dim=-1
        )
        is_stable = horizontal_drift < 0.08

        success = at_target_height & is_stable
        return success.float()

    def expert_lift(self, initial_state, current_state, cube_name="A"):
        """Expert policy for lifting"""
        init = self.parse_state(initial_state)
        curr = self.parse_state(current_state)

        cube_map = {"A": "cube_a_pos", "B": "cube_b_pos", "C": "cube_c_pos"}
        cube_pos = curr[cube_map[cube_name]]
        hand_pos = curr["hand_pos"]
        actions = torch.zeros(1, 7)

        grasp_pos = cube_pos.clone()
        grasp_pos[:, 2] = cube_pos[:, 2] + self.grasp_height_offset
        grasp_pos[:, 0] = cube_pos[:, 0] + self.grasp_height_offset_x

        dist_to_grasp = torch.norm(hand_pos - grasp_pos, dim=-1)
        cube_height = cube_pos[:, 2] - init["cube_" + cube_name.lower() + "_pos"][:, 2]

        if (cube_height[0] >= self.lift_height_threshold + 0.05) and self.has_cube()[0]:
            # print("done")
            return actions
        elif (cube_height[0] < self.lift_height_threshold + 0.05) and (
            self.has_cube()[0]
        ):
            # print("lift_up")
            actions[:, 2] = 0.4
            actions[:, 6] = -1.0
        elif dist_to_grasp[0] > self.phase_transition_threshold:
            # print('go_closer')
            return self._go_to(current_state, grasp_pos[0], gripper_cmd=1.0)
        elif (not self.has_cube()[0]) and (
            self.get_gripper_openness() < self.cube_size - 0.02
        ):
            # print('open_gripper')
            actions[:, 6] = 1.0
        elif not self.has_cube()[0]:
            # print('close_gripper')
            actions[:, 6] = -1.0

        return actions

    def compute_reward_lift_A(self, initial_state, current_state):
        return self.compute_reward_lift(initial_state, current_state, "A")

    def compute_reward_lift_B(self, initial_state, current_state):
        return self.compute_reward_lift(initial_state, current_state, "B")

    def compute_reward_lift_C(self, initial_state, current_state):
        return self.compute_reward_lift(initial_state, current_state, "C")

    def expert_lift_A(self, initial_state, current_state):
        return self.expert_lift(initial_state, current_state, "A")

    def expert_lift_B(self, initial_state, current_state):
        return self.expert_lift(initial_state, current_state, "B")

    def expert_lift_C(self, initial_state, current_state):
        return self.expert_lift(initial_state, current_state, "C")

    # ==================== STACK TASK ====================

    def _in_contact(self, body_a, body_b):
        return len(p.getContactPoints(bodyA=body_a, bodyB=body_b)) > 0

    def compute_reward_stack(
        self, initial_state, current_state, up_cube="A", bottom_cube="B"
    ):
        """
        Args:
            up_cube (str): 'A', 'B', or 'C'
            bottom_cube (str): 'A', 'B', or 'C'

        Returns:
            torch.Tensor([0.0 or 1.0])
        """
        assert up_cube != bottom_cube

        up_id = self.cube_name_to_id[up_cube]
        bottom_id = self.cube_name_to_id[bottom_cube]

        # --- 1. Cube-on-cube contact ---
        if not self._in_contact(up_id, bottom_id):
            return torch.tensor([0.0])

        # --- 2. Height condition (up cube above bottom cube) ---
        up_pos, _ = p.getBasePositionAndOrientation(up_id)
        bottom_pos, _ = p.getBasePositionAndOrientation(bottom_id)

        if up_pos[2] <= bottom_pos[2] + 0.8 * self.cube_size:
            return torch.tensor([0.0])

        # --- 4. Alignment condition (x,y within threshold) ---
        horizontal_dist = np.linalg.norm(
            np.array(up_pos[:2]) - np.array(bottom_pos[:2])
        )
        if horizontal_dist >= self.stack_align_threshold:
            return torch.tensor([0.0])

        return torch.tensor([1.0])

    def expert_stack(self, initial_state, current_state, up_cube="A", bottom_cube="B"):
        """Expert policy for stacking (stack A on B)"""
        init = self.parse_state(initial_state)
        curr = self.parse_state(current_state)

        assert up_cube != bottom_cube

        if up_cube == "A":
            cube_a_pos = curr["cube_a_pos"]
        elif up_cube == "B":
            cube_a_pos = curr["cube_b_pos"]
        else:
            cube_a_pos = curr["cube_c_pos"]

        if bottom_cube == "A":
            cube_b_pos = curr["cube_a_pos"]
        elif bottom_cube == "B":
            cube_b_pos = curr["cube_b_pos"]
        else:
            cube_b_pos = curr["cube_c_pos"]

        hand_pos = curr["hand_pos"]
        gripper_pos = curr["gripper_pos"].squeeze(-1)

        actions = torch.zeros(1, 7)

        grasp_a_pos = cube_a_pos.clone()
        grasp_a_pos[:, 2] += self.grasp_height_offset

        above_b_pos = cube_b_pos.clone()
        above_b_pos[:, 2] += self.cube_size + 0.03

        stack_pos = cube_b_pos.clone()
        stack_pos[:, 2] += self.cube_size + 0.1

        has_cube_a = self.has_cube()[0]
        dist_xy_to_b = torch.norm(hand_pos[:, :2] - above_b_pos[:, :2], dim=-1)
        is_above_b = (dist_xy_to_b < 0.02) & (
            hand_pos[:, 2] > cube_b_pos[:, 2] + self.cube_size
        )

        # if up_cub on top of bottom_cub do nothing
        if (
            self.compute_reward_stack(
                initial_state, current_state, up_cube, bottom_cube
            )[0]
            > 0.5
        ):
            # print("done")
            if hand_pos[0, 2] < self.cube_size * 4 + 0.2:
                actions[:, 2] = 0.4
            actions[:, 6] = 1.0
        else:
            if (not has_cube_a) or (
                self.max_cube_height[up_cube] < self.cube_size + 0.05
            ):
                # print("lift")
                return self.expert_lift(initial_state, current_state, up_cube)
            elif not is_above_b[0]:
                # print("go_above", above_b_pos, hand_pos)
                return self._go_to(current_state, above_b_pos, gripper_cmd=-1.0)
            elif hand_pos[0, 2] > self.cube_size:
                # print("go_down")
                actions[:, 2] = -0.4
                actions[:, 6] = 1.0
            else:
                actions[:, 6] = -1.0

        return actions

    # ==================== PUSH TASKS ====================

    def compute_reward_push(
        self, initial_state, current_state, cube_name="C", direction="left"
    ):
        """Binary reward for pushing"""
        init = self.parse_state(initial_state)
        curr = self.parse_state(current_state)

        cube_map = {"A": "cube_a_pos", "B": "cube_b_pos", "C": "cube_c_pos"}
        cube_pos_key = cube_map[cube_name]

        init_cube_pos = init[cube_pos_key]
        curr_cube_pos = curr[cube_pos_key]

        dir_map = {
            "left": torch.tensor([0.0, 1.0, 0.0]),
            "right": torch.tensor([0.0, -1.0, 0.0]),
        }
        push_dir = dir_map[direction]

        displacement = curr_cube_pos - init_cube_pos
        distance_pushed = torch.sum(displacement[:, :2] * push_dir[:2], dim=-1)

        pushed_enough = distance_pushed >= (
            self.push_distance - self.push_success_threshold
        )

        height_change = torch.abs(curr_cube_pos[:, 2] - init_cube_pos[:, 2])
        on_table = height_change < 0.03

        success = pushed_enough & on_table
        return success.float()

    def expert_push(
        self, initial_state, current_state, cube_name="C", direction="left"
    ):
        """Expert policy for pushing using _go_to"""

        init = self.parse_state(initial_state)
        curr = self.parse_state(current_state)

        cube_map = {
            "A": curr["cube_a_pos"],
            "B": curr["cube_b_pos"],
            "C": curr["cube_c_pos"],
        }
        cube_pos = cube_map[cube_name]
        init_cube_pos = init[f"cube_{cube_name.lower()}_pos"]

        hand_pos = curr["hand_pos"]
        dir_map = {
            "left": torch.tensor([0.0, 1.0, 0.0], device=hand_pos.device),
            "right": torch.tensor([0.0, -1.0, 0.0], device=hand_pos.device),
        }
        push_dir = dir_map[direction]

        displacement = cube_pos - init_cube_pos
        pushed_dist = torch.sum(displacement[:, :2] * push_dir[:2], dim=-1)

        if pushed_dist[0] > self.push_distance + 0.01:
            # print("done")
            actions = torch.zeros(1, 7, device=hand_pos.device)
            if hand_pos[0, 2] < self.cube_size * 4 + 0.2:
                actions[..., 2] = 0.4
                actions[..., 6] = 1
            return actions
        if not self.has_cube()[0]:
            # print("get_the_cube")
            return self.expert_lift(initial_state, current_state, cube_name)
        else:
            # print("push")
            actions = torch.zeros(1, 7, device=hand_pos.device)
            actions[..., 6] = -1
            actions[..., :3] = push_dir * 0.4
            return actions

    # Specific push task wrappers
    def compute_reward_push_C_left(self, initial_state, current_state):
        return self.compute_reward_push(initial_state, current_state, "C", "left")

    def compute_reward_push_C_right(self, initial_state, current_state):
        return self.compute_reward_push(initial_state, current_state, "C", "right")

    def compute_reward_push_B_left(self, initial_state, current_state):
        return self.compute_reward_push(initial_state, current_state, "B", "left")

    def compute_reward_push_B_right(self, initial_state, current_state):
        return self.compute_reward_push(initial_state, current_state, "B", "right")

    def compute_reward_push_A_left(self, initial_state, current_state):
        return self.compute_reward_push(initial_state, current_state, "A", "left")

    def compute_reward_push_A_right(self, initial_state, current_state):
        return self.compute_reward_push(initial_state, current_state, "A", "right")

    def expert_push_C_left(self, initial_state, current_state):
        return self.expert_push(initial_state, current_state, "C", "left")

    def expert_push_C_right(self, initial_state, current_state):
        return self.expert_push(initial_state, current_state, "C", "right")

    def expert_push_B_left(self, initial_state, current_state):
        return self.expert_push(initial_state, current_state, "B", "left")

    def expert_push_B_right(self, initial_state, current_state):
        return self.expert_push(initial_state, current_state, "B", "right")

    def expert_push_A_left(self, initial_state, current_state):
        return self.expert_push(initial_state, current_state, "A", "left")

    def expert_push_A_right(self, initial_state, current_state):
        return self.expert_push(initial_state, current_state, "A", "right")

    # ==================== REWARDS EXPERT POLICY ====================

    def get_reward_function(self, task_id):
        """Get reward function for a specific task ID"""
        reward_map = {
            0: self.compute_reward_lift_A,
            1: self.compute_reward_lift_B,
            2: self.compute_reward_lift_C,
            3: self.compute_reward_stack,
            4: self.compute_reward_push_C_left,
            5: self.compute_reward_push_C_right,
            6: self.compute_reward_push_B_left,
            7: self.compute_reward_push_B_right,
            8: self.compute_reward_push_A_left,
            9: self.compute_reward_push_A_right,
        }
        return reward_map[task_id]

    def get_expert_policy(self, task_id):
        """Get expert policy for a specific task ID"""
        expert_map = {
            0: self.expert_lift_A,
            1: self.expert_lift_B,
            2: self.expert_lift_C,
            3: self.expert_stack,
            4: self.expert_push_C_left,
            5: self.expert_push_C_right,
            6: self.expert_push_B_left,
            7: self.expert_push_B_right,
            8: self.expert_push_A_left,
            9: self.expert_push_A_right,
        }
        return expert_map[task_id]
