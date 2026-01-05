# === Standard Library ===
import argparse
import logging
import os
import pickle
import shutil
import sys
from pathlib import Path

# === Third-party Libraries ===
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from tqdm import tqdm

# === Set Up Paths ===
ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))  # Top-level project root
sys.path.extend(
    [
        str(ROOT_PATH),
        str(ROOT_PATH / "flowdiffusion"),
        str(ROOT_PATH / "calvin/calvin_models"),
        str(ROOT_PATH / "toy_env_pybullet"),
    ]
)
from methods.generate_data import generate_new_data_toy
from model.hierarchical_model_toy import HierarchicalModel
from toy_env_pybullet.toyEnv import TASK_NAMES as tasks
from toy_env_pybullet.toyEnv import Franka3CubeEnvPyBullet

# === Device Setup ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate new data using a pretrained hierarchical CALVIN model"
    )

    parser.add_argument(
        "--policy_checkpoint_num",
        type=str,
        help="Policy checkpoint num",
        default=1406,
    )

    parser.add_argument(
        "--policy_results_folder",
        type=str,
        help="Results folder",
        default="/home/grislain/AVDC/debug_toy_LL",
    )

    parser.add_argument(
        "--high_level_checkpoint_num",
        type=str,
        help="High level checkpoint number",
        default=2,
    )

    parser.add_argument(
        "--high_level_results_folder",
        type=str,
        help="Results folder",
        default="/home/grislain/AVDC/debug_toy_HL",
    )

    parser.add_argument(
        "--high_level_sampling_timesteps",
        type=int,
        help="Number of sampling timesteps for high level model",
        default=100,
    )

    parser.add_argument(
        "--high_level_ddim_eta",
        type=float,
        help="DDIM eta for high level model",
        default=0.0,
    )

    parser.add_argument(
        "--server",
        "-s",
        type=str,
        help="Server",
        default="hacienda",
    )

    parser.add_argument(
        "--use_oracle_subgoals",
        action="store_true",
        help="Use oracle subgoals",
    )

    parser.add_argument(
        "--num_subgoals",
        type=int,
        default=8,
        help="Number of subgoals to generate.",
    )

    parser.add_argument(
        "--debug_path",
        type=str,
        default=None,
        help="Path to save debug images.",
    )

    parser.add_argument(
        "--eval_folder",
        type=str,
        default="eval",
        help="Folder to save evaluation results.",
    )

    parser.add_argument(
        "--replan",
        action="store_true",
        help="Replan subgoals every 64 steps.",
    )

    parser.add_argument(
        "--use_filtered_data",
        action="store_true",
        help="Use filtered data (expert sucesses) for evaluation.",
    )

    parser.add_argument(
        "--policy_model",
        type=str,
        default="diffusion",
        choices=["diffusion", "act"],
        help="Policy model to use.",
    )

    parser.add_argument(
        "--high_level_guidance",
        type=int,
        default=3,
        help="Guidance in high-level diffusion.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
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
        ],
    )

    parser.add_argument(
        "--buffer_save_path",
        type=str,
        help="Patch to buffer checkpoint",
        default=None,
    )

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)  # type:ignore

    saving_path = args.saving_path
    os.makedirs(saving_path, exist_ok=True)

    if args.debug_path:
        # Create debug folder
        debug_path = Path(args.debug_path)
        os.makedirs(debug_path, exist_ok=True)

        # Load data config
    policy_data_config = OmegaConf.load(
        os.path.join(args.policy_results_folder, "data_config.yaml")
    )
    if not args.use_oracle_subgoals:
        high_level_data_config = OmegaConf.load(
            os.path.join(args.high_level_results_folder, "data_config.yaml")
        )
    else:
        high_level_data_config = {}

    config = DictConfig(
        {
            "policy": {
                "model": args.policy_model,
                "checkpoint_num": args.policy_checkpoint_num,
                "results_folder": args.policy_results_folder,
                **policy_data_config,
            },
            "high_level": {
                "checkpoint_num": args.high_level_checkpoint_num,
                "results_folder": args.high_level_results_folder,
                "use_oracle_subgoals": args.use_oracle_subgoals,
                **high_level_data_config,
                "sampling_timesteps": args.high_level_sampling_timesteps,
                "ddim_sampling_eta": args.high_level_ddim_eta,
                "guidance": args.high_level_guidance,
            },
            "debug_path": args.debug_path,
            "server": args.server,
            "num_subgoals": args.num_subgoals,
            "replan": args.replan,
        }
    )
    device = torch.device("cuda:0")
    config.device = "cuda"

    # Initialize state buffer with preloaded states from dataset
    buffer_save_path = args.buffer_save_path
    print("Buffer load path:", buffer_save_path)
    buffer = pickle.load(open(buffer_save_path, "rb"))
    for task in tasks:
        print(f"# valid states {task}: {len(buffer)}")

    env_cfg = {"num_envs": 1, "max_episode_length": 500, "task_mode": "random"}
    env = Franka3CubeEnvPyBullet(env_cfg, headless=True)
    model = HierarchicalModel(config)

    if args.task is None:
        for task_id, task in tqdm(enumerate(tasks)):
            generate_new_data_toy(
                model,
                env,
                debug_path=args.debug_path,
                num_data=args.num_data,
                task=task,
                task_id=task_id,
                saving_path=os.path.join(saving_path, f"training_{args.start_idx}"),
                num_trials=args.num_trials,
                state_buffer=buffer,
                start_idx=args.start_idx,
            )
    else:
        for task in args.task:
            generate_new_data_toy(
                model,
                env,
                debug_path=args.debug_path,
                num_data=args.num_data,
                task=task,
                task_id=tasks.index(task),
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
        os.path.join(saving_path, "validation", "lang_annotations"),
        dirs_exist_ok=True,
    )
