# === Standard Library ===
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

# === Third-party Libraries ===
import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm

# === Set Up Paths ===
ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))  # Top-level project root
sys.path.extend(
    [
        str(ROOT_PATH / "hd-expit"),
        str(ROOT_PATH / "calvin/calvin_models"),
        str(ROOT_PATH / "franka_3blocks_env_pybullet"),
    ]
)

# === Local Project Imports ===
# === CALVIN Imports ===
from calvin.calvin_env.calvin_env.envs.play_table_env import get_env
from calvin.calvin_models.calvin_agent.datasets.calvin_data_module import (
    CalvinDataModule,
)
from calvin.calvin_models.calvin_agent.evaluation.multistep_sequences import (
    tasks,
)
from hierarchical_policy.hierarchical_model_calvin import HierarchicalModel
from methods.generate_data import generate_new_data
from state_buffer import StateBuffer

# === Device Setup ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


def make_env(dataset_path):
    train_folder = Path(dataset_path) / "training"
    env = get_env(train_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


if __name__ == "__main__":
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(
        description="Generate new data using a pretrained hierarchical CALVIN model"
    )

    parser.add_argument(
        "--buffer_save_path",
        type=str,
        help="Patch to buffer checkpoint",
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
        default=None,
    )

    parser.add_argument(
        "--high_level_checkpoint_num",
        type=int,
        help="High level checkpoint number",
        default=199,
    )

    parser.add_argument(
        "--high_level_results_folder",
        type=str,
        help="Results folder",
        default=None,
    )

    parser.add_argument(
        "--debug_path",
        type=str,
        help="Path to save debug images.",
        default=None,
    )

    parser.add_argument(
        "--num_subgoals",
        type=int,
        help="Number of subgoals to generate.",
        default=8,
    )

    parser.add_argument(
        "--replan",
        action="store_true",
        help="Replan subgoals every 64 steps.",
    )

    parser.add_argument(
        "--saving_path",
        type=str,
        help="Path to save the generated data.",
        default=None,
    )

    parser.add_argument(
        "--num_data",
        type=int,
        help="Number of data points to generate.",
        default=100,
    )

    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        help="Task to generate data for.",
        default=None,
    )

    parser.add_argument(
        "--num_trials",
        type=int,
        help="Number of trials per data point.",
        default=1,
    )

    parser.add_argument(
        "--start_idx",
        type=int,
        help="Starting index for saving data.",
        default=0,
    )

    parser.add_argument(
        "--buffer_mode",
        type=str,
        help="Buffer of initial states",
        choices=[
            "start",
            "start_end_all",
            "start_all",
            "episodes",
            "start_end_others",
            "end_all",
            "reset",
        ],
    )

    parser.add_argument(
        "--policy_model",
        type=str,
        default="diffusion",
        choices=["diffusion", "act"],
        help="Policy model to use.",
    )

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    if args.debug_path:
        # Create debug folder
        debug_path = Path(args.debug_path)
        os.makedirs(debug_path, exist_ok=True)

    # Do not change
    args.ep_len = 240

    # Load data configs
    policy_data_config = OmegaConf.load(
        os.path.join(args.policy_results_folder, "data_config.yaml")
    )

    rollout_cfg_path = os.path.join(
        ROOT_PATH, "calvin/calvin_models/conf/callbacks/rollout/default.yaml"
    )
    conf_dir = Path(os.path.join(ROOT_PATH, "calvin/calvin_models/conf"))

    high_level_data_config = OmegaConf.load(
        os.path.join(args.high_level_results_folder, "data_config.yaml")
    )

    if isinstance(policy_data_config.root, ListConfig):
        policy_data_config.root = policy_data_config.root[0]

    config = DictConfig(
        {
            "policy": {
                "checkpoint_num": args.policy_checkpoint_num,
                "results_folder": args.policy_results_folder,
                "model": getattr(args, "policy_model", "diffusion"),
                **policy_data_config,
            },
            "high_level": {
                "checkpoint_num": args.high_level_checkpoint_num,
                "results_folder": args.high_level_results_folder,
                "use_oracle_subgoals": policy_data_config.datamodule.lang_dataset.get(
                    "without_guidance", False
                ),
                "sampling_timesteps": 100,
                **high_level_data_config,
            },
            "debug_path": args.debug_path,
            "num_subgoals": args.num_subgoals,
            "replan": False,
        }
    )

    policy_data_config.datamodule.lang_dataset._target_ = (
        "calvin_agent.datasets.disk_dataset.DiskDiffusionOracleDataset"
    )
    del policy_data_config.datamodule.lang_dataset.prob_aug

    image_transforms_dict = OmegaConf.load(
        os.path.join(
            ROOT_PATH,
            "calvin/calvin_models/conf/datamodule/transforms/play_basic.yaml",
        )
    )

    data_module = CalvinDataModule(
        policy_data_config.datamodule,
        transforms=image_transforms_dict,
        root_data_dir=policy_data_config.root,
    )
    data_module.setup()

    dataloader = data_module.train_dataloader()
    policy_dataset = dataloader["lang"].dataset

    device = torch.device("cuda:0")
    config.device = "cuda"

    print("Config:\n" + OmegaConf.to_yaml(config))

    saving_path = args.saving_path or os.path.join(args.debug_path, "generated_data")

    # Save config
    os.makedirs(saving_path, exist_ok=True)
    with open(
        os.path.join(saving_path, "config.yaml"),
        "w",
    ) as f:
        OmegaConf.save(config, f)

    os.makedirs(os.path.join(saving_path, f"training_{args.start_idx}"), exist_ok=True)
    os.makedirs(
        os.path.join(saving_path, f"training_{args.start_idx}", "lang_annotations"),
        exist_ok=True,
    )

    transforms_dict = {
        "pixel": image_transforms_dict,
    }

    rollout_cfg = OmegaConf.load(rollout_cfg_path)
    env = hydra.utils.instantiate(
        rollout_cfg.env_cfg, policy_dataset, device, show_gui=False
    )

    # Initialize state buffer with preloaded states from dataset
    buffer = StateBuffer(tasks.keys(), max_size=1e6)
    if args.buffer_save_path is None:
        buffer_save_path = os.path.join(
            policy_dataset.abs_datasets_dir, f"state_buffer_{args.buffer_mode}.pkl"
        )
    else:
        buffer_save_path = args.buffer_save_path
    print("Buffer load path:", buffer_save_path)
    buffer.load(buffer_save_path)
    for task in tasks.keys():
        print(f"# valid states {task}: {buffer.num_valid(task)}")

    model = HierarchicalModel(config, transforms_dict)

    if args.task is None:
        for task in tqdm(tasks.keys()):
            generate_new_data(
                model,
                env,
                debug_path=args.debug_path,
                conf_dir=conf_dir,
                num_data=args.num_data,
                task=task,
                saving_path=os.path.join(saving_path, f"training_{args.start_idx}"),
                num_trials=args.num_trials,
                state_buffer=buffer,
                start_idx=args.start_idx,
            )
    else:
        for task in args.task:
            generate_new_data(
                model,
                env,
                debug_path=args.debug_path,
                conf_dir=conf_dir,
                num_data=args.num_data,
                task=task,
                saving_path=os.path.join(saving_path, f"training_{args.start_idx}"),
                num_trials=args.num_trials,
                state_buffer=buffer,
                start_idx=args.start_idx,
            )

    shutil.copytree(
        os.path.join(policy_data_config.root, "training", ".hydra"),
        os.path.join(saving_path, "training", ".hydra"),
        dirs_exist_ok=True,
    )
    shutil.copy2(
        os.path.join(policy_data_config.root, "training", "statistics.yaml"),
        os.path.join(saving_path, "training", "statistics.yaml"),
    )

    os.makedirs(os.path.join(saving_path, "validation"), exist_ok=True)

    # Copy from training to validation as a placeholder
    shutil.copy2(
        os.path.join(
            saving_path,
            f"training_{args.start_idx}",
            f"episode_{(args.start_idx + 1):07d}.npz",
        ),
        os.path.join(
            saving_path, "validation", f"episode_{(args.start_idx + 1):07d}.npz"
        ),
    )
    shutil.copytree(
        os.path.join(saving_path, f"training_{args.start_idx}", "lang_annotations"),
        os.path.join(saving_path, "validation", "new_lang_annotations"),
        dirs_exist_ok=True,
    )
