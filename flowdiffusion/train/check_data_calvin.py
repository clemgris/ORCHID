import argparse
import os
import sys

import torch
import torchvision

os.environ["TOKENIZERS_PARALLELISM"] = "false"

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)

from omegaconf import DictConfig, OmegaConf

sys.path.append(
    os.path.join(
        root_path,
        "calvin/calvin_models",
    )
)

from calvin.calvin_models.calvin_agent.datasets.calvin_data_module import (
    CalvinDataModule,  # noqa: E402
)

print(f"CUDA available: {torch.cuda.is_available()}")
num_gpus = torch.cuda.device_count()
print(f"Total GPUs available: {num_gpus}")
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def main(args):
    data_path = args.data_path

    cfg = DictConfig(
        {
            "root": data_path,
            "datamodule": {
                "lang_dataset": {
                    "_target_": "calvin_agent.datasets.disk_dataset.DiskDiffusionDataset",
                    "key": "lang",
                    "save_format": "npz",
                    "batch_size": 1,
                    "min_window_size": 16,
                    "max_window_size": 65,
                    "proprio_state": {
                        "n_state_obs": 8,
                        "keep_indices": [[0, 7], [14, 15]],
                        "robot_orientation_idx": [3, 6],
                        "normalize": True,
                        "normalize_robot_orientation": True,
                    },
                    "obs_space": {
                        "rgb_obs": ["rgb_static"],
                        "depth_obs": [],
                        "state_obs": ["robot_obs"],
                        "actions": ["actions"],
                        "language": ["language"],
                    },
                    "num_subgoals": 8,
                    "pad": True,
                    "lang_folder": "lang_annotations",
                    "num_workers": 2,
                    "goal": "pixel",
                    "norm_feat": None,
                    "feat_patch_size": 0,
                    "auto_lang_name": "auto_lang_ann",
                },
            },
            "train_num_steps": 1,
            "save_and_sample_every": 1,
            "diffusion_objective": "pred_v",
            "min_batch_size": 1,
            "global_batch_size": 1,
            "text_encoder": "CLIP",
        }
    )

    print("Config:\n" + OmegaConf.to_yaml(cfg))

    transforms = OmegaConf.load(
        os.path.join(
            root_path,
            "calvin/calvin_models/conf/datamodule/transforms/play_basic.yaml",
        )
    )

    data_module = CalvinDataModule(
        cfg.datamodule, transforms=transforms, root_data_dir=cfg.root
    )

    data_module.setup()

    train_set = data_module.train_datasets["lang"]
    valid_set = data_module.val_datasets["lang"]

    print("Train data:", len(train_set))
    print("Valid data:", len(valid_set))

    os.path.exists(args.saving_path) or os.makedirs(args.saving_path)

    for idx in range(len(train_set)):
        x, x_cond, task = train_set[idx]
        torchvision.utils.save_image(
            (x.reshape((8, 3, 96, 96)) + 1) / 2,
            args.saving_path + f"train_img_{idx}_{task}.png",
        )
        if idx > args.num_data:
            break

    for idx in range(len(valid_set)):
        x, x_cond, task = valid_set[idx]
        torchvision.utils.save_image(
            (x.reshape((8, 3, 96, 96)) + 1) / 2,
            args.saving_path + f"valid_img_{idx}_{task}.png",
        )
        if idx > args.num_data:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--server", type=str, default="hacienda", choices=["jz", "local"]
    )  # set to 'jz' to use jean zay server
    parser.add_argument(
        "-num", "--num_data", type=int, default=100
    )  # set to number of samples to generate
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/grislain/AVDC/calvin/dataset/calvin_debug_dataset",
    )  # set to data path
    parser.add_argument(
        "--saving_path",
        type=str,
        default="./check_data_calvin/",
        help="Path to save images",
    )
    args = parser.parse_args()

    print()
    print("Arguments:", args)
    print()
    main(args)
