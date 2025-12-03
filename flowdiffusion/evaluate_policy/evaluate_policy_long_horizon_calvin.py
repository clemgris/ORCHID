# === Standard Library ===
import argparse
import logging
import os
import sys
from pathlib import Path

# === Third-party Libraries ===
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

# === Set Up Paths ===
ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))  # Top-level project root
sys.path.extend(
    [
        str(ROOT_PATH / "flowdiffusion"),
        str(ROOT_PATH / "calvin/calvin_models"),
    ]
)

# === Local Project Imports ===
# === CALVIN Imports ===
from calvin.calvin_env.calvin_env.envs.play_table_env import get_env
from calvin.calvin_models.calvin_agent.datasets.calvin_data_module import (
    CalvinDataModule,
)
from methods.evaluate_policy import evaluate_policy
from model.hierarchical_model_calvin import HierarchicalModel
from utils.transform_feat import update_feat_transform

# === Device Setup ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


if __name__ == "__main__":
    seed_everything(0, workers=True)
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on multistep sequences with language goals."
    )

    parser.add_argument(
        "--policy_checkpoint_num",
        type=str,
        help="Policy checkpoint num",
        default=9999,
    )

    parser.add_argument(
        "--policy_results_folder",
        type=str,
        help="Results folder",
        default="/home/grislain/AVDC/calvin/models/LL_RGB",
    )

    parser.add_argument(
        "--high_level_checkpoint_num",
        type=str,
        help="High level checkpoint number",
        default=199,
    )

    parser.add_argument(
        "--high_level_results_folder",
        type=str,
        help="Results folder",
        default="/home/grislain/AVDC/calvin/models/HL_RGB",
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
        "--test_on",
        type=str,
        help="Train on train or val",
        default="train",
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
        default="/home/grislain/AVDC/debug",
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

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()
    args.save_failures = args.debug_path is not None

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

    if args.server == "jz":
        data_path = "/lustre/fsn1/projects/rech/fch/uxv44vw/CALVIN/task_D_D_jz"
        rollout_cfg_path = "/lustre/fswork/projects/rech/fch/uxv44vw/clemgris/avdc/calvin/calvin_models/conf/callbacks/rollout/default.yaml"
        conf_dir = Path(
            "/lustre/fswork/projects/rech/fch/uxv44vw/clemgris/avdc/calvin/calvin_models/conf"
        )

    elif args.server == "hacienda":
        data_path = "/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset"
        rollout_cfg_path = "/home/grislain/AVDC/calvin/calvin_models/conf/callbacks/rollout/default.yaml"
        conf_dir = Path("/home/grislain/AVDC/calvin/calvin_models/conf")

        policy_data_config.datamodule.lang_dataset.lang_folder = "lang_annotations"
    else:
        raise ValueError("Invalid server argument")

    # load low level config
    if "lang_dataset" not in policy_data_config.datamodule:
        assert "vis_dataset" in policy_data_config.datamodule, (
            "vis_dataset or lang_dataset must be present in policy_data_config.datamodule"
        )
        policy_data_config.datamodule.lang_dataset = (
            policy_data_config.datamodule.vis_dataset
        )
        policy_data_config.datamodule.lang_dataset.key = "lang"

    policy_data_config.datamodule.lang_dataset._target_ = (
        "calvin_agent.datasets.disk_dataset.DiskDiffusionOracleDataset"
    )
    del policy_data_config.datamodule.lang_dataset.prob_aug
    policy_data_config.root = data_path
    policy_data_config.datamodule.lang_dataset.auto_lang_name = (
        "filtered_auto_lang_ann" if args.use_filtered_data else "auto_lang_ann"
    )

    if "diffuse_on" in policy_data_config.datamodule.lang_dataset:
        # Old config
        policy_data_config.datamodule.lang_dataset.goal = (
            policy_data_config.datamodule.lang_dataset.diffuse_on
        )
        policy_data_config.datamodule.lang_dataset.obs = "pixel"
        del policy_data_config.datamodule.lang_dataset.diffuse_on

    if high_level_data_config != {} and (
        "diffuse_on" in high_level_data_config.datamodule.lang_dataset
    ):
        high_level_data_config.datamodule.lang_dataset.goal = (
            high_level_data_config.datamodule.lang_dataset.diffuse_on
        )
        del high_level_data_config.datamodule.lang_dataset.diffuse_on

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

    image_transforms_dict = OmegaConf.load(
        os.path.join(
            ROOT_PATH,
            "calvin/calvin_models/conf/datamodule/transforms/play_basic.yaml",
        )
    )
    feat_transforms_dict = OmegaConf.load(
        os.path.join(
            ROOT_PATH,
            "calvin/calvin_models/conf/datamodule/transforms/play_features_imagenet.yaml",
        )
    )
    feat_transforms_dict = update_feat_transform(
        policy_data_config, feat_transforms_dict
    )

    transforms_dict = {
        "pixel": image_transforms_dict,
        "feat": feat_transforms_dict,
    }

    data_module = CalvinDataModule(
        policy_data_config.datamodule,
        transforms=image_transforms_dict,
        root_data_dir=policy_data_config.root,
    )
    data_module.setup()

    dataloader = data_module.val_dataloader()
    policy_dataset = dataloader.dataset.datasets["lang"]

    device = torch.device("cuda:0")
    config.device = "cuda"

    print("Config:\n" + OmegaConf.to_yaml(config))

    # Save config
    os.makedirs(args.eval_folder, exist_ok=True)
    with open(
        os.path.join(args.eval_folder, "config.yaml"),
        "w",
    ) as f:
        OmegaConf.save(config, f)

    if args.debug_path:
        # Create debug folder
        debug_path = Path(config.debug_path)
        os.makedirs(debug_path, exist_ok=True)

    rollout_cfg = OmegaConf.load(rollout_cfg_path)
    env = hydra.utils.instantiate(
        rollout_cfg.env_cfg, policy_dataset, device, show_gui=False
    )
    model = HierarchicalModel(config, transforms_dict)
    evaluate_policy(
        model,
        env,
        eval_folder=args.eval_folder,
        debug_path=args.debug_path,
        conf_dir=conf_dir,
    )
