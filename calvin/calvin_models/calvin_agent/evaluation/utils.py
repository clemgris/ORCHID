import contextlib
import json
import logging
import os
import random
import zlib
from collections import Counter
from pathlib import Path
from pydoc import locate

import cv2
import hydra
import numpy as np
import torch
from calvin_agent.utils.utils import add_text, format_sftp_path
from numpy import pi
from omegaconf import OmegaConf

# import pyhash
# hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)


def get_default_env(train_folder, dataset_path, checkpoint, env=None, device_id=0):
    train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    cfg = OmegaConf.load(train_cfg_path)
    lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize("../../conf/datamodule/datasets")
    # we don't want to use shm dataset for evaluation
    datasets_cfg = hydra.compose(
        "vision_lang.yaml", overrides=["lang_dataset.lang_folder=" + lang_folder]
    )
    # since we don't use the trainer during inference, manually set up data_module
    cfg.datamodule.datasets = datasets_cfg
    cfg.datamodule.root_data_dir = dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["lang"]
    device = torch.device(f"cuda:{device_id}")

    if env is None:
        rollout_cfg = OmegaConf.load(
            Path(__file__).parents[2] / "conf/callbacks/rollout/default.yaml"
        )
        env = hydra.utils.instantiate(
            rollout_cfg.env_cfg, dataset, device, show_gui=False
        )

    # checkpoint = format_sftp_path(checkpoint)
    # print(f"Loading model from {checkpoint}")
    # # import the model class that was used for the training
    # model_cls = locate(cfg.model._target_)
    # model = model_cls.load_from_checkpoint(checkpoint)
    # model.load_lang_embeddings(
    #     dataset.abs_datasets_dir / dataset.lang_folder / "embeddings.npy"
    # )
    # model.freeze()
    # if cfg.model.action_decoder.get("load_action_bounds", False):
    #     model.action_decoder._setup_action_bounds(
    #         cfg.datamodule.root_data_dir, None, None, True
    #     )
    # model = model.cuda(device)
    model = None
    print("Successfully loaded model.")

    return model, env, data_module


def get_default_model_and_env(
    train_folder, dataset_path, checkpoint, env=None, device_id=0
):
    train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    cfg = OmegaConf.load(train_cfg_path)
    lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize("../../conf/datamodule/datasets")
    # we don't want to use shm dataset for evaluation
    datasets_cfg = hydra.compose(
        "vision_lang.yaml", overrides=["lang_dataset.lang_folder=" + lang_folder]
    )
    # since we don't use the trainer during inference, manually set up data_module
    cfg.datamodule.datasets = datasets_cfg
    cfg.datamodule.root_data_dir = dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["lang"]
    device = torch.device(f"cuda:{device_id}")

    if env is None:
        rollout_cfg = OmegaConf.load(
            Path(__file__).parents[2] / "conf/callbacks/rollout/default.yaml"
        )
        env = hydra.utils.instantiate(
            rollout_cfg.env_cfg, dataset, device, show_gui=False
        )

    checkpoint = format_sftp_path(checkpoint)
    print(f"Loading model from {checkpoint}")
    # import the model class that was used for the training
    model_cls = locate(cfg.model._target_)
    model = model_cls.load_from_checkpoint(checkpoint)
    model.load_lang_embeddings(
        dataset.abs_datasets_dir / dataset.lang_folder / "embeddings.npy"
    )
    model.freeze()
    if cfg.model.action_decoder.get("load_action_bounds", False):
        model.action_decoder._setup_action_bounds(
            cfg.datamodule.root_data_dir, None, None, True
        )
    model = model.cuda(device)
    print("Successfully loaded model.")

    return model, env, data_module


def collect_plan(model, plans, subtask):
    try:
        plans[subtask].append((model.plan.cpu(), model.latent_goal.cpu()))
    except AttributeError:
        return


def join_vis_lang(img, lang_text):
    """Takes as input an image and a language instruction and visualizes them with cv2"""
    img = img[:, :, ::-1].copy()
    img = cv2.resize(img, (500, 500))
    add_text(img, lang_text)
    cv2.imshow("simulation cam", img)
    cv2.waitKey(1)


def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(results, sequences, log_dir, epoch=None):
    current_data = {}
    print(f"Results for Epoch {epoch}:")
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        print(
            f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%"
        )

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

    current_data[epoch] = data

    print()
    previous_data = {}
    try:
        with open(log_dir / "results.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file)
    print(
        f"Best model: epoch {max(json_data, key=lambda x: json_data[x]['avg_seq_len'])} "
        f"with average sequences length of {max(map(lambda x: x['avg_seq_len'], json_data.values()))}"
    )


def create_tsne(plan_dict, log_dir, epoch):
    ids, labels, plans, latent_goals = zip(
        *[
            (i, label, latent_goal, plan)
            for i, (label, plan_list) in enumerate(plan_dict.items())
            for latent_goal, plan in plan_list
        ]
    )
    latent_goals = torch.cat(latent_goals)
    plans = torch.cat(plans)
    np.savez(
        f"{log_dir / f'tsne_data_{epoch}.npz'}",
        ids=ids,
        labels=labels,
        plans=plans,
        latent_goals=latent_goals,
    )


def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")
            os.makedirs(log_dir, exist_ok=True)
    print(f"logging to {log_dir}")
    return log_dir


def imshow_tensor(window, img_tensor, wait=0, resize=True, keypoints=None, text=None):
    img_tensor = img_tensor.squeeze()
    img = np.transpose(img_tensor.cpu().numpy(), (1, 2, 0))
    img = np.clip(((img / 2) + 0.5) * 255, 0, 255).astype(np.uint8)

    if keypoints is not None:
        key_coords = np.clip(keypoints * 200 + 100, 0, 200)
        key_coords = key_coords.reshape(-1, 2)
        cv_kp1 = [cv2.KeyPoint(x=pt[1], y=pt[0], _size=1) for pt in key_coords]
        img = cv2.drawKeypoints(img, cv_kp1, None, color=(255, 0, 0))

    if text is not None:
        add_text(img, text)

    if resize:
        cv2.imshow(window, cv2.resize(img[:, :, ::-1], (500, 500)))
    else:
        cv2.imshow(window, img[:, :, ::-1])
    cv2.waitKey(wait)


def print_task_log(demo_task_counter, live_task_counter, mod):
    print()
    logger.info(f"Modality: {mod}")
    for task in demo_task_counter:
        logger.info(
            f"{task}: SR = {(live_task_counter[task] / demo_task_counter[task]) * 100:.0f}%"
            + f" |  {live_task_counter[task]} of {demo_task_counter[task]}"
        )
    logger.info(
        f"Average Success Rate {mod} = "
        + f"{(sum(live_task_counter.values()) / s if (s := sum(demo_task_counter.values())) > 0 else 0) * 100:.0f}% "
    )
    logger.info(
        f"Success Rates averaged throughout classes = {np.mean([live_task_counter[task] / demo_task_counter[task] for task in demo_task_counter]) * 100:.0f}%"
    )


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_env_state_for_initial_condition(initial_condition):
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    # we want to have a "deterministic" random seed for each initial condition
    seed = zlib.adler32(str(initial_condition.values()).encode("utf-8"))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        # red block
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs


def get_random_env_state_for_initial_condition(initial_condition):
    # lower_joint_limits = (
    #     (
    #         -1.8973,  # -2.8973,
    #         -1.0628,  # -1.7628,
    #         -1.8973,  # -2.8973,
    #         -2.0718,  # -3.0718,
    #         -1.8973,  # -2.8973,
    #         -0.0175,  # -0.0175,
    #         -1.8973,  # -2.8973,
    #     ),
    # )
    # upper_joint_limits = (
    #     1.8973,  # 2.8973,
    #     1.0628,  # 1.7628,
    #     1.8973,  # 2.8973,
    #     -0.0698,  # -0.0698,
    #     1.8973,  # 2.8973,
    #     2.7525,  # 3.7525,
    #     1.8973,  # 2.8973,
    # )

    robot_obs = np.array(
        [
            0.02586889,  # x derived from joint angles
            -0.2313129,  # y derived from joint angles
            0.5712808,  # z derived from joint angles
            3.09045411,  # theta_x derived from joint angles
            -0.02908596,  # theta_y derived from joint angles
            1.50013585,  # theta_z derived from joint angles
            0.07999963,  # gripper width
            -1.21779124,  # joint 0
            1.03987629,  # joint 1
            2.11978254,  # joint 2
            -2.34205014,  # joint 3
            -0.87015899,  # joint 4
            1.64119093,  # joint 5
            0.55344928,  # joint 6
            1.0,  # gripper actionn
        ]
    )

    # robot_joints = np.random.uniform(low=lower_joint_limits, high=upper_joint_limits)
    # robot_obs[7:14] = robot_joints

    robot_gripper_width = np.random.uniform(0.0, 0.08)
    robot_obs[6] = robot_gripper_width

    block_width = 0.04
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    block_drawer_open = [
        np.array([0.9170e-01, -2.7140e-01, 3.6213e-01]),
        np.array([1.7130e-01, -2.5120e-01, 3.6213e-01]),
    ]
    block_drawer_closed = [
        np.array([0.9090e-01, -1.0526e-01, 3.6213e-01]),
        np.array([1.7170e-01, -1.1120e-01, 3.6213e-01]),
    ]

    drawer_idx = [0, 1]
    random.shuffle(drawer_idx)

    table_idx = [0, 1]
    random.shuffle(table_idx)

    if initial_condition["drawer"] == "open":
        block_drawer = block_drawer_open
    elif initial_condition["drawer"] == "closed":
        block_drawer = block_drawer_closed

    # we want to have a "deterministic" random seed for each initial condition
    seed = zlib.adler32(str(initial_condition.values()).encode("utf-8"))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]

        # red block
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        elif initial_condition["red_block"] == "grasped":
            scene_obs[6:9] = robot_obs[0:3]
            robot_obs[6] = block_width
        elif initial_condition["red_block"] == "drawer":
            scene_obs[6:9] = block_drawer[drawer_idx.pop(0)]
        elif initial_condition["red_block"] == "table":
            scene_obs[6:9] = block_table[table_idx.pop(0)]
        else:
            raise ValueError("Unknown initial condition for red block")
        # Orientation
        if initial_condition["red_block"] == "grasped":
            scene_obs[11] = 0.0
        else:
            scene_obs[11] = np.random.uniform(*block_rot_z_range)

        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["blue_block"] == "grasped":
            scene_obs[12:15] = robot_obs[0:3]
            robot_obs[6] = block_width
        elif initial_condition["blue_block"] == "drawer":
            scene_obs[12:15] = block_drawer[drawer_idx.pop(0)]
        elif initial_condition["blue_block"] == "table":
            scene_obs[12:15] = block_table[table_idx.pop(0)]
        else:
            raise ValueError("Unknown initial condition for blue block")
        # Orientation
        if initial_condition["blue_block"] == "grasped":
            scene_obs[17] = 0.0
        else:
            scene_obs[17] = np.random.uniform(*block_rot_z_range)

        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        elif initial_condition["pink_block"] == "grasped":
            scene_obs[18:21] = robot_obs[0:3]
            robot_obs[6] = block_width
        elif initial_condition["pink_block"] == "drawer":
            scene_obs[18:21] = block_drawer[drawer_idx.pop(0)]
        elif initial_condition["pink_block"] == "table":
            scene_obs[18:21] = block_table[table_idx.pop(0)]
        else:
            raise ValueError("Unknown initial condition for pink block")
        # Orientation
        if initial_condition["pink_block"] == "grasped":
            scene_obs[23] = 0.0
        else:
            scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs


uids = {
    "table": 5,
    "block_blue": 3,
    "block_pink": 4,
    "block_red": 2,
}


def get_logic_state_from_state(info):
    # Light states
    logic_state = {
        "led": info["scene_info"]["lights"]["led"]["logical_state"],
        "lightbulb": info["scene_info"]["lights"]["lightbulb"]["logical_state"],
    }

    # Grasped blocks
    robot_contacts = [c[2] for c in info["robot_info"]["contacts"]]
    logic_state["grasped"] = 0
    if uids["block_red"] in robot_contacts:
        logic_state["red_block"] = "grasped"
        logic_state["grasped"] = 1
    elif uids["block_pink"] in robot_contacts:
        logic_state["pink_block"] = "grasped"
        logic_state["grasped"] = 1
    elif uids["block_blue"] in robot_contacts:
        logic_state["blue_block"] = "grasped"
        logic_state["grasped"] = 1

    # On table or slider
    drawer_condition = (5, 3)
    table_condition = (5, -1)
    slider_condition = (5, 6)

    block_blue_contacts = [
        (c[2], c[4])
        for c in info["scene_info"]["movable_objects"]["block_blue"]["contacts"]
    ]
    block_pink_contacts = [
        (c[2], c[4])
        for c in info["scene_info"]["movable_objects"]["block_pink"]["contacts"]
    ]
    block_red_contacts = [
        (c[2], c[4])
        for c in info["scene_info"]["movable_objects"]["block_red"]["contacts"]
    ]

    if "blue_block" not in logic_state:
        # In drawer
        if drawer_condition in block_blue_contacts:
            logic_state["blue_block"] = "drawer"
        # On table
        elif table_condition in block_blue_contacts:
            logic_state["blue_block"] = "table"
        # In slider
        elif slider_condition in block_blue_contacts:
            logic_state["blue_block"] = (
                "slider_left"
                if info["scene_info"]["movable_objects"]["block_blue"]["current_pos"][0]
                < -0.11
                else "slider_right"
            )
        # Default on table
        else:
            logic_state["blue_block"] = "table"

        # Stacked on another block
        if (uids["block_red"], -1) in block_blue_contacts:
            logic_state["blue_block"] = (
                "stacked_top"
                if info["scene_info"]["movable_objects"]["block_blue"]["current_pos"][2]
                > info["scene_info"]["movable_objects"]["block_red"]["current_pos"][2]
                else "stacked_bottom"
            )
        elif (uids["block_pink"], -1) in block_blue_contacts:
            logic_state["blue_block"] = (
                "stacked_top"
                if info["scene_info"]["movable_objects"]["block_blue"]["current_pos"][2]
                > info["scene_info"]["movable_objects"]["block_pink"]["current_pos"][2]
                else "stacked_bottom"
            )

    if "pink_block" not in logic_state:
        if drawer_condition in block_pink_contacts:
            logic_state["pink_block"] = "drawer"
        elif table_condition in block_pink_contacts:
            logic_state["pink_block"] = "table"
        elif slider_condition in block_pink_contacts:
            logic_state["pink_block"] = (
                "slider_left"
                if info["scene_info"]["movable_objects"]["block_pink"]["current_pos"][0]
                < -0.11
                else "slider_right"
            )
        else:
            logic_state["pink_block"] = "table"
        # Stacked on another block
        if (uids["block_red"], -1) in block_pink_contacts:
            logic_state["pink_block"] = (
                "stacked_top"
                if info["scene_info"]["movable_objects"]["block_pink"]["current_pos"][2]
                > info["scene_info"]["movable_objects"]["block_red"]["current_pos"][2]
                else "stacked_bottom"
            )
        elif (uids["block_blue"], -1) in block_pink_contacts:
            logic_state["pink_block"] = (
                "stacked_top"
                if info["scene_info"]["movable_objects"]["block_pink"]["current_pos"][2]
                > info["scene_info"]["movable_objects"]["block_blue"]["current_pos"][2]
                else "stacked_bottom"
            )

    if "red_block" not in logic_state:
        if drawer_condition in block_red_contacts:
            logic_state["red_block"] = "drawer"
        elif table_condition in block_red_contacts:
            logic_state["red_block"] = "table"
        elif slider_condition in block_red_contacts:
            logic_state["red_block"] = (
                "slider_left"
                if info["scene_info"]["movable_objects"]["block_red"]["current_pos"][0]
                < -0.11
                else "slider_right"
            )
        else:
            logic_state["red_block"] = "table"

        # Stacked on another block
        if (uids["block_blue"], -1) in block_red_contacts:
            logic_state["red_block"] = (
                "stacked_top"
                if info["scene_info"]["movable_objects"]["block_red"]["current_pos"][2]
                > info["scene_info"]["movable_objects"]["block_blue"]["current_pos"][2]
                else "stacked_bottom"
            )
        elif (uids["block_pink"], -1) in block_red_contacts:
            logic_state["red_block"] = (
                "stacked_top"
                if info["scene_info"]["movable_objects"]["block_red"]["current_pos"][2]
                > info["scene_info"]["movable_objects"]["block_pink"]["current_pos"][2]
                else "stacked_bottom"
            )

    # Open/closed doors
    logic_state["slider"] = (
        "left"
        if info["scene_info"]["doors"]["base__slide"]["current_state"] > 0.14
        else "right"
    )
    logic_state["drawer"] = (
        "closed"
        if info["scene_info"]["doors"]["base__drawer"]["current_state"] < 0.11
        else "open"
    )
    return logic_state
