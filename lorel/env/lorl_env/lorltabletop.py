import os

# from gym.spaces import spaces.Box
import sys

import matplotlib
import mujoco
import numpy as np
from gymnasium import spaces

base_dir_path = os.path.abspath(os.path.dirname(__file__) + "/../../../")
sys.path.append(base_dir_path)
from metaworld.envs.mujoco.sawyer_xyz import SawyerXYZEnv

matplotlib.use("Agg")


class LorlTabletop(SawyerXYZEnv):
    def __init__(
        self,
        obj_low=None,
        obj_high=None,
        goal_low=None,
        goal_high=None,
        hand_init_pos=(0, 0.4, 0.2),
        liftThresh=0.04,
        rewMode="orig",
        rotMode="fixed",
        problem="rand",
        xml="updated_new",
        filepath="test",
        max_path_length=20,
        verbose=0,
        log_freq=100,  # in terms of episode num
        **kwargs,
    ):
        self.max_path_length = max_path_length
        self.cur_path_length = 0
        self.xml = xml

        # self.quick_init(locals())
        hand_low = (-0.3, 0.4, 0.0)
        hand_high = (0.3, 0.8, 0.15)
        obj_low = (-0.3, 0.4, 0.1)
        obj_high = (0.3, 0.8, 0.3)

        self.imsize = 64
        self.imsize_x = 64

        super().__init__(
            frame_skip=5,
            action_scale=1.0 / 5,
            hand_low=hand_low,
            hand_high=hand_high,
            # model_name=self.model_name, # Updated Mujoco version
            render_mode="rgb_array",
            camera_name="cam0",
            width=self.imsize_x,
            height=self.imsize,
            **kwargs,
        )

        self.observation_space = spaces.Box(
            0,
            1.0,
            (
                self.imsize_x,
                self.imsize,
                3,
            ),
            dtype=np.float64,
        )

        self.liftThresh = liftThresh
        self.rewMode = rewMode
        self.rotMode = rotMode
        self.hand_init_pos = np.array(hand_init_pos)
        self.action_rot_scale = 1.0 / 10
        self.action_space = spaces.Box(
            np.array([-1, -1, -1, -np.pi, -1]),
            np.array([1, 1, 1, np.pi, 1]),
        )
        self.hand_and_obj_space = spaces.Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
        )

    @property
    def model_name(self):
        dirname = os.path.dirname(__file__)
        file = "../assets_updated/sawyer_xyz/" + self.xml + ".xml"
        filename = os.path.join(dirname, file)
        return filename

    def step(self, action):
        action[3] = 0  ### Rotation always 0
        self.set_xyz_action_rotz(action[:4])
        self.do_simulation([action[-1], -action[-1]], self.frame_skip)

        ob = self._get_obs()
        if self.cur_path_length == self.max_path_length - 1:
            done = True
        else:
            done = False
        self.cur_path_length += 1
        return ob, 0, done, {}

    def _get_obs(self):
        # obs = self.sim.render(self.imsize_x, self.imsize, camera_name="cam0") / 255.0
        obs = self.render().copy().astype(np.float64) / 255.0
        # obs = np.flip(obs, 0).copy()
        return obs

    def reset_model(self):
        """For logging"""
        self.cur_path_length = 0

        ### Reset gripper
        self._reset_hand()
        for _ in range(100):
            try:
                self.do_simulation([0.0, 0.0], self.frame_skip)
            except:
                print("Got Mujoco Unstable Simulation Warning")
                continue
        self.cur_path_length = 0

        # Set inital pos for mugs
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        rd = np.random.uniform(-0.1, 0.1, (2,))
        qpos[9:11] = np.array([-0.2, 0.65]) + rd
        rd = np.random.uniform(-0.1, 0.1, (2,))
        qpos[11:13] = np.array([-0.2, 0.65]) + rd

        # Set initial pos for drawer, faucet, button
        qpos[13] = np.random.uniform(-np.pi / 4, np.pi / 4)
        qpos[14] = np.random.uniform(-0.09, 0.0)
        self.set_state(qpos, qvel)
        for _ in range(100):
            #     self.sim.step()
            mujoco.mj_step(self.model, self.data)  # Updated Mujoco version
        self._target_pos = self._get_object_position("goal")
        o = self._get_obs()
        return o

    def _reset_hand(self, pos=None):
        if pos is None:
            if np.random.uniform() < 0.5:
                pos = [-0.0, 0.5, 0.07]
            else:
                pos = [-0.2, 0.65, 0.07] + np.random.uniform(-0.05, 0.05, (3,))
                pos[2] = 0.07
        for _ in range(100):
            # self.data.set_mocap_pos("mocap", pos)
            # self.data.set_mocap_quat("mocap", np.array([0.707, 0.0, 0.707, 0.0]))
            self.data.mocap_pos[0] = pos  # Updated Mujoco version
            self.data.mocap_quat[0] = np.array(
                [0.707, 0.0, 0.707, 0.0]
            )  # Updated Mujoco version
            try:
                self.do_simulation([-1, 1], self.frame_skip)
            except:
                print("Got Mujoco Unstable Simulation Warning")
                continue
        rightFinger, leftFinger = (
            self.get_site_pos("rightEndEffector"),
            self.get_site_pos("leftEndEffector"),
        )
        self.init_fingerCOM = (rightFinger + leftFinger) / 2
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        # _id = self.model.site_names.index(siteName)
        _id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, siteName
        )  # Updated Mujoco version
        return self.data.site_xpos[_id].copy()

    def compute_reward(self):
        return 0.0

    def _get_pos_objects(self) -> np.ndarray:
        """Retrieves object positions."""
        # Retrieve positions from MuJoCo
        # goal_pos = self._get_object_position("goal")
        obj_pos = self._get_object_position("obj")
        obj2_pos = self._get_object_position("obj2")

        # Concatenate into a single flat array
        return np.concatenate([obj_pos, obj2_pos])

    def _get_object_position(self, name: str) -> np.ndarray:
        """Helper function to get the position of an object by its name."""
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[obj_id]

    def _get_object_quaternion(self, name: str) -> np.ndarray:
        """
        Helper function to get the quaternion of an object by its name.

        Args:
            name: The name of the object in the Mujoco model.

        Returns:
            The quaternion as a NumPy array [w, x, y, z].
        """
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xquat[obj_id]

    def _get_quat_objects(self) -> np.ndarray:
        """
        Retrieves object quaternions for all relevant objects in the environment.
        """
        object_names = ["obj", "obj2"]  # Quaternions for objects
        quaternions = [self._get_object_quaternion(name) for name in object_names]
        return np.concatenate(quaternions)
