import argparse
import multiprocessing as mp
import os
import sys
from functools import partial
from pathlib import Path

# import ddpo_pytorch.prompts
# import ddpo_pytorch.rewards
import hydra
import torch
import tqdm
from accelerate.logging import get_logger
from omegaconf import OmegaConf

# === Set Up Paths ===

CALVIN_ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CALVIN_ROOT_PATH))  # Top-level project root

sys.path.extend(
    [
        str(CALVIN_ROOT_PATH / "hd-expit"),
        str(CALVIN_ROOT_PATH / "calvin"),
        str(CALVIN_ROOT_PATH / "calvin/calvin_models"),
    ]
)

# === CALVIN Imports ===
from calvin.calvin_models.calvin_agent.datasets.calvin_data_module import (
    CalvinDataModule,
)
from calvin.calvin_models.calvin_agent.evaluation.multistep_sequences import (
    get_initial_states,
    tasks,
)
from calvin.calvin_models.calvin_agent.evaluation.utils import (
    get_env_state_for_initial_condition,
)
from calvin_agent.evaluation.utils import (
    get_logic_state_from_state,
)

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

from generate_data.state_buffer import StateBuffer

logger = get_logger(__name__)


mp.set_start_method("spawn", force=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Data path",
        default=None,
    )

    parser.add_argument(
        "--policy_checkpoint_num",
        type=int,
        help="Policy checkpoint num",
        default=9999,
    )

    parser.add_argument(
        "--policy_results_folder",
        type=str,
        help="Results folder",
        default="results_LL_calvin",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug images during training.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        help="Mode for state buffer saving.",
        default="start_end_all",
        choices=[
            "start",
            "start_others",
            "start_end_all",
            "start_all",
            "episodes",
            "start_end_others",
            "end_all",
            "reset",
        ],
    )

    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Balance states between tasks in the buffer",
    )

    args = parser.parse_args()

    ####### CALVIN config #######
    policy_data_config = OmegaConf.load(
        os.path.join(args.policy_results_folder, "data_config.yaml")
    )
    rollout_cfg_path = os.path.join(
        CALVIN_ROOT_PATH, "calvin/calvin_models/conf/callbacks/rollout/default.yaml"
    )
    policy_data_config.datamodule.lang_dataset.lang_folder = "lang_annotations"
    policy_data_config.root = args.data_path

    ##### Hierarchical model #####
    image_transforms_dict = OmegaConf.load(
        os.path.join(
            CALVIN_ROOT_PATH,
            "calvin/calvin_models/conf/datamodule/transforms/play_basic.yaml",
        )
    )

    # load low level config
    if "lang_dataset" not in policy_data_config.datamodule:
        assert "vis_dataset" in policy_data_config.datamodule, (
            "vis_dataset or lanfg_dataset must be present in policy_data_config.datamodule"
        )
        policy_data_config.datamodule.lang_dataset = (
            policy_data_config.datamodule.vis_dataset
        )
        policy_data_config.datamodule.lang_dataset.key = "lang"

    policy_data_config.datamodule.lang_dataset._target_ = (
        "calvin_agent.datasets.disk_dataset.DiskDiffusionOracleDataset"
    )
    del policy_data_config.datamodule.lang_dataset.prob_aug
    policy_data_config.root = args.data_path

    if "diffuse_on" in policy_data_config.datamodule.lang_dataset:
        # Old config
        policy_data_config.datamodule.lang_dataset.goal = (
            policy_data_config.datamodule.lang_dataset.diffuse_on
        )
        policy_data_config.datamodule.lang_dataset.obs = "pixel"
        del policy_data_config.datamodule.lang_dataset.diffuse_on

    ########## Calvin env #########
    data_module = CalvinDataModule(
        policy_data_config.datamodule,
        transforms=image_transforms_dict,
        root_data_dir=policy_data_config.root,
    )
    data_module.setup()

    dataloader = data_module.train_dataloader()
    policy_dataset = dataloader["lang"].dataset

    rollout_cfg = OmegaConf.load(rollout_cfg_path)

    # Initialize state buffer
    env = hydra.utils.instantiate(
        rollout_cfg.env_cfg,
        policy_dataset,
        device,
        show_gui=False,
        use_egl=True,
    )
    buffer = StateBuffer(tasks.keys(), max_size=1e6, balanced=args.balanced)
    if args.mode == "reset":
        for task in tasks.keys():
            initial_conditions = get_initial_states(2000, task)
            for init_cond, _ in initial_conditions:
                robot_obs, scene_obs = get_env_state_for_initial_condition(init_cond)
                buffer.add((init_cond, robot_obs, scene_obs), task)
    else:
        for episode in tqdm(policy_dataset):
            state_info = episode["state_info"]
            rgb_episode = episode["rgb_obs"]["rgb_static"][1:]
            if args.mode == "start_end_all" or args.mode == "start_end_others":
                used_idx = [0, -1]
            elif args.mode == "end_all":
                used_idx = [-1]
            elif (
                args.mode == "start_all"
                or args.mode == "episodes"
                or args.mode == "start"
                or args.mode == "start_others"
            ):
                used_idx = [0]
            else:
                raise NotImplementedError
            for idx in used_idx:
                robot_obs = state_info["robot_obs"][idx]
                scene_obs = state_info["scene_obs"][idx]

                _ = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
                info = env.get_info()
                logic_state = get_logic_state_from_state(info)
                if args.mode == "start":
                    buffer.add(
                        (logic_state, robot_obs, scene_obs),
                        task=episode["task"].replace(" ", "_"),
                    )
                elif args.mode == "episodes":
                    buffer.add(
                        (logic_state, robot_obs, scene_obs, rgb_episode),
                        task=episode["task"].replace(" ", "_"),
                    )
                elif args.mode == "start_end_others" or args.mode == "start_others":
                    if idx == 0:
                        buffer.add(
                            (logic_state, robot_obs, scene_obs),
                            exection=episode["task"].replace(" ", "_"),
                        )
                    else:
                        buffer.add(
                            (logic_state, robot_obs, scene_obs),
                        )
                elif args.mode in ["start_end_all", "start_all", "end_all"]:
                    buffer.add((logic_state, robot_obs, scene_obs))

    for task in tasks.keys():
        print(f"Task: {task}, Num valid states: {buffer.num_valid(task)}")

    # Save buffer to file
    buffer_name = (
        f"state_buffer_{args.mode}.pkl"
        if not args.balanced
        else f"state_buffer_{args.mode}_balanced.pkl"
    )
    buffer_save_path = os.path.join(policy_dataset.abs_datasets_dir, buffer_name)
    buffer.save(buffer_save_path)
    print(f"Saved state buffer to {buffer_save_path}")


if __name__ == "__main__":
    main()
