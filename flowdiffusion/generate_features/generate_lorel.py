import os
import pickle
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from omegaconf import DictConfig
from torch.utils.data import random_split
from tqdm import tqdm

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)

from encoder import DinoV2Encoder, R3MEncoder, ViTEncoder  # noqa: E402

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import argparse

from lorel.expert_dataset import ExpertDataset, TrajDataset  # noqa: E402, F401


def main(args):
    if args.server == "hacienda":
        root_path = "/home/grislain/SkillDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_1k.pkl"
        num_data = 100
    elif args.server == "jz":
        root_path = "/lustre/fsn1/projects/rech/fch/uxv44vw/TrajectoryDiffuser/lorel/data/dec_24_sawyer_50k/dec_24_sawyer_50k/dec_24_sawyer_50k.pkl"
        num_data = 38225
    else:
        raise ValueError(f"Unknown server {args.server}")
    cfg = DictConfig(
        {
            "root": root_path,
            "num_data": num_data,
            "skip_frames": 1,
        },
    )

    data_filename = Path(cfg.root).stem
    folder_name = Path(cfg.root).parent

    # Load the full dataset
    full_dataset = TrajDataset(cfg.root, num_trajectories=cfg.num_data, use_state=False)

    # Calculate split sizes
    valid_size = int(0.1 * len(full_dataset))  # 10% for validation
    train_size = len(full_dataset) - valid_size

    # Split into train and validation
    train_set, valid_set = random_split(full_dataset, [train_size, valid_size])

    print("Train data (traj):", len(train_set))
    print("Valid data (traj):", len(valid_set))

    # Frozen encoder model
    if args.features == "dino":
        num_channels = 768
        if args.server == "hacienda":
            encoder_model = DinoV2Encoder(
                name="facebook/dinov2-base",
            )
        elif args.server == "jz":
            encoder_model = DinoV2Encoder(
                name="/lustre/fsn1/projects/rech/fch/uxv44vw/facebook/dinov2-base",
            )
        else:
            raise ValueError(f"Unknown server {args.server}")

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        )
    elif args.features == "dino_vit":
        num_channels = 768
        encoder_model = ViTEncoder()

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    elif args.features == "r3m":
        num_channels = 512
        encoder_model = R3MEncoder("resnet18")

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown feature type {args.features}")

    # Training set
    saving_path = folder_name / data_filename / "training"
    os.makedirs(
        os.path.join(saving_path, f"data_with_{args.features}_features"),
        exist_ok=True,
    )

    action_min = np.inf * np.ones((1, 5))
    action_max = -np.inf * np.ones((1, 5))

    for ii, data in tqdm(
        enumerate(train_set),
        desc=f"Generate {args.features} features of training data",
        total=len(train_set),
    ):
        traj = data.copy()
        images = traj["states"]
        # Appy transform
        images = [
            PIL.Image.fromarray((image.transpose(1, 2, 0) * 255).astype("uint8"))
            for image in images
        ]
        images = [transform(image) for image in images]
        images = torch.stack(images, dim=0).to("cuda")
        _, patch_emb = encoder_model(images)
        traj["dino_patch_emb"] = patch_emb.cpu().numpy()

        action_min = np.minimum(action_min, traj["actions"].min(axis=0))
        action_max = np.maximum(action_max, traj["actions"].max(axis=0))

        # Save as npz
        np.savez(
            os.path.join(
                saving_path, f"data_with_{args.features}_features/data_{ii}.npz"
            ),
            **traj,
        )

        # Save action min and max
        pickle.dump(
            {
                "action": {
                    "min": torch.tensor(action_min, dtype=torch.float32),
                    "max": torch.tensor(action_max, dtype=torch.float32),
                }
            },
            open(os.path.join(saving_path, "dataset_stats.pkl"), "wb"),
        )

    print(
        f"Training data features generated and saved in {saving_path}/training/data_with_{args.features}_features"
    )
    print(f"Training action stats saved in {saving_path}/dataset_stats.pkl")

    # Validation set
    saving_path = folder_name / data_filename / "validation"
    os.makedirs(
        os.path.join(saving_path, f"data_with_{args.features}_features"),
        exist_ok=True,
    )
    action_min = np.inf * np.ones((1, 5))
    action_max = -np.inf * np.ones((1, 5))

    for ii, data in tqdm(
        enumerate(valid_set),
        desc=f"Generate {args.features} features of training data",
        total=len(valid_set),
    ):
        traj = data.copy()
        images = traj["states"]
        # Appy transform
        images = [
            PIL.Image.fromarray((image.transpose(1, 2, 0) * 255).astype("uint8"))
            for image in images
        ]
        images = [transform(image) for image in images]
        images = torch.stack(images, dim=0).to("cuda")
        _, patch_emb = encoder_model(images)
        traj["dino_patch_emb"] = patch_emb.cpu().numpy()

        action_min = np.minimum(action_min, traj["actions"].min(axis=0))
        action_max = np.maximum(action_max, traj["actions"].max(axis=0))

        # Save as npz
        np.savez(
            os.path.join(
                saving_path, f"data_with_{args.features}_features/data_{ii}.npz"
            ),
            **traj,
        )

        # Save action min and max
        pickle.dump(
            {
                "action": {
                    "min": torch.tensor(action_min, dtype=torch.float32),
                    "max": torch.tensor(action_max, dtype=torch.float32),
                }
            },
            open(os.path.join(saving_path, "dataset_stats.pkl"), "wb"),
        )

    print(
        f"Validation data features generated and saved in {saving_path}/validation/data_with_{args.features}_features"
    )
    print(f"Validation action stats saved in {saving_path}/dataset_stats.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--override", type=bool, default=False
    )  # set to True to overwrite results folder
    parser.add_argument("-f", "--features", type=str, default="dino")
    parser.add_argument(
        "-s", "--server", type=str, default="hacienda"
    )  # hacienda or jz
    args = parser.parse_args()
    main(args)
