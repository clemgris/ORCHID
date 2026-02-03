import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
sys.path.append(
    os.path.join(
        root_path,
        "flowdiffusion",
    )
)

from goal_diffusion import GoalGaussianDiffusion, Trainer
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    SiglipTextModel,
    SiglipTokenizer,
    T5EncoderModel,
)
from unet import UnetMW as Unet

sys.path.append(
    os.path.join(
        root_path,
        "toy_env_pybullet/dataset",
    )
)
sys.path.append(
    os.path.join(
        root_path,
        "calvin/calvin_models",
    )
)

from toy_env_pybullet.dataset.dataset import ToyDataset

print(f"CUDA available: {torch.cuda.is_available()}")
num_gpus = torch.cuda.device_count()
print(f"Total GPUs available: {num_gpus}")
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def main(args):
    results_folder = args.results_folder
    data_paths = args.data_paths

    print("Training on ", len(data_paths), " datasets.")

    cfg = DictConfig(
        {
            "root": data_paths,
            "dataset": {
                "batch_size": args.batch_size // num_gpus,
                "min_window_size": 16,
                "max_window_size": 65,
                "obs_space": {
                    "rgb_static": ["rgb_static"],
                    "states": ["states"],
                    "actions": ["actions"],
                    "language": ["language"],
                },
                "num_subgoals": args.num_subgoals,
                "pad": True,
                "lang_folder": "lang_annotations",
                "auto_lang_name": "auto_lang_ann",
                "goal": "pixel",
                "norm_feat": args.norm,
            },
            "train_num_steps": args.train_num_steps,
            "save_and_sample_every": args.save_and_sample_every,
            "diffusion_objective": args.diff_objective,
            "min_batch_size": args.min_batch_size,
            "global_batch_size": args.batch_size,
            "text_encoder": args.text_encoder,
        }
    )

    print("Config:\n" + OmegaConf.to_yaml(cfg))

    sample_per_seq = cfg.dataset.num_subgoals + 1

    if "dino" in cfg.dataset.goal:
        assert args.feat_patch_size % 16 == 0, "Dino patch size must be multiple of 16"
        target_size = (args.feat_patch_size, args.feat_patch_size)
        channel = 768
    elif "r3m" in cfg.dataset.goal:
        assert args.feat_patch_size % 7 == 0, "R3M patch size must be multiple of 7"
        target_size = (args.feat_patch_size, args.feat_patch_size)
        channel = 512
    elif cfg.dataset.goal == "pixel":
        target_size = (96, 96)
        # RGB
        channel = 3
    else:
        raise ValueError(f"Diffusion type {cfg.dataset.goal} not supported.")

    transforms = OmegaConf.load(
        os.path.join(
            root_path,
            "toy_env_pybullet/conf/transforms.yaml",
        )
    )
    cfg.dataset.transforms = transforms

    results_folder = Path(results_folder)

    if args.mode == "train":
        if os.path.exists(results_folder):
            if not args.override and args.checkpoint_num is None:
                raise ValueError(
                    f"Results folder {results_folder} already exists. Use --override to overwrite."
                )
        results_folder.mkdir(exist_ok=True, parents=True)
        print("Results folder:", results_folder)

    if args.checkpoint_num is None:
        with open(os.path.join(results_folder, "data_config.yaml"), "w") as file:
            file.write(OmegaConf.to_yaml(cfg))
    else:
        # Load checkpoint config
        allowed_mismatch = ["train_num_steps"]
        mismatching_keys = []
        with open(os.path.join(results_folder, "data_config.yaml"), "r") as file:
            checkpoint_cfg = OmegaConf.load(file)
        for key in cfg.keys():
            if key not in checkpoint_cfg:
                print(f"Missing key {key} not in checkpoint config.")
                raise ValueError(f"Key {key} not in checkpoint config.")
            elif cfg[key] != checkpoint_cfg[key]:
                print(
                    f"Key {key} has different value in checkpoint config {checkpoint_cfg[key]} != {cfg[key]}"
                )
                mismatching_keys.append(key)
        assert all(key in allowed_mismatch for key in mismatching_keys), (
            f"Keys {mismatching_keys} are not in the allowed mismatch list {allowed_mismatch}"
        )

    # Datasets
    train_sets = []
    for path in data_paths:
        train_set = ToyDataset(path, cfg.dataset)
        train_sets.append(train_set)

    train_set = ConcatDataset(train_sets)
    valid_n = 1
    print("Train data:", len(train_set))
    print("Valid data:", 0)

    # ## DEBUG
    # import torchvision

    # for idx in range(len(train_set)):
    #     x, x_cond, task = train_set[idx]
    #     torchvision.utils.save_image(
    #         (x.reshape((8, 3, 96, 96)) + 1) / 2, f"train_img_{idx}_{task}.png"
    #     )
    #     if idx > 10:
    #         break
    # ## DEBUG

    # Text encoder
    if args.text_encoder == "CLIP":
        if args.server == "jz":
            text_pretrained_model = (
                "/lustre/fsmisc/dataset/HuggingFace_Models/openai/clip-vit-base-patch32"
            )
        else:
            text_pretrained_model = "openai/clip-vit-base-patch32"

        tokenizer = CLIPTokenizer.from_pretrained(text_pretrained_model)
        text_encoder = CLIPTextModel.from_pretrained(text_pretrained_model)
        text_embed_dim = 512
        amp = True
        precision = "bf16" if args.server == "jz" else "fp16"

    elif args.text_encoder == "Flan-t5":
        if args.server == "jz":
            text_pretrained_model = (
                "/lustre/fsmisc/dataset/HuggingFace_Models/google/flan-t5-base"
            )
        else:
            text_pretrained_model = "google/flan-t5-base"
        text_encoder = T5EncoderModel.from_pretrained(text_pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(text_pretrained_model)
        text_embed_dim = 768
        amp = False
        precision = "no"

    elif args.text_encoder == "Siglip":
        if args.server == "jz":
            text_pretrained_model = "/lustre/fsn1/projects/rech/fch/uxv44vw/models/google/siglip-base-patch16-224"
        else:
            text_pretrained_model = "google/siglip-base-patch16-224"
        tokenizer = SiglipTokenizer.from_pretrained(text_pretrained_model)
        text_encoder = SiglipTextModel.from_pretrained(text_pretrained_model)
        text_embed_dim = 768
        amp = True
        precision = "bf16" if args.server == "jz" else "fp16"

    text_encoder = text_encoder.to(device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    text_encoder_num_params = sum(p.numel() for p in text_encoder.parameters())
    print(
        f"Number of parameters in text encoder {text_pretrained_model}: {text_encoder_num_params / 1e6:.2f}M"
    )

    # Diffusion Unet

    if args.diffuse_on == "pixel":
        channel_mult = (1, 2, 3, 4, 5)
    elif "dino" in args.diffuse_on:
        if args.feat_patch_size == 16:
            channel_mult = (1, 2, 3)
        elif args.feat_patch_size == 32:
            channel_mult = (1, 2, 3, 4)
        elif args.feat_patch_size == 64:
            channel_mult = (1, 2, 3, 4, 5)
    elif args.diffuse_on == "r3m":
        if args.feat_patch_size == 7:
            channel_mult = (1, 2)
        if args.feat_patch_size == 14:
            channel_mult = (1, 2, 3)
        if args.feat_patch_size == 21:
            channel_mult = (1, 2, 3, 4)

    unet = Unet(channel, channel_mult=channel_mult, text_embed_dim=text_embed_dim)

    # Load decoder checkpoint
    decoder_weigth_path = args.feature_decoder_checkpoint_path
    if args.feature_decoder_checkpoint_path is not None:
        import torch
        from decoder import TransposedConvDecoder

        decoder_model = TransposedConvDecoder(
            emb_dim=768,
            observation_shape=(3, 96, 96),
            patch_size=args.feat_patch_size,
        )
        state_dict = torch.load(decoder_weigth_path)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        decoder_model.load_state_dict(state_dict)
        decoder_model = decoder_model.to(device)
        decoder_model.eval()

    else:
        decoder_model = None

    diffusion = GoalGaussianDiffusion(
        channels=channel * (sample_per_seq - 1),
        num_subgoals=(sample_per_seq - 1),
        model=unet,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=args.sample_steps,
        loss_type="l2",
        objective=args.diff_objective,
        beta_schedule="cosine",
        min_snr_loss_weight=True,
        auto_normalize=False,
        temporal_loss_weight=args.temporal_loss_weight,
        prob_temp_swaps=args.prob_temp_swaps,
        num_swaps=args.num_swaps,
    )

    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        feature_decoder=decoder_model,
        train_set=train_set,
        valid_set=train_set,
        train_lr=1e-4,
        train_num_steps=cfg.train_num_steps,
        save_and_sample_every=cfg.save_and_sample_every,
        ema_update_every=10,
        ema_decay=0.999,
        train_batch_size=cfg.dataset.batch_size,
        valid_batch_size=1,
        gradient_accumulate_every=max(
            1,
            args.min_batch_size // (cfg.dataset.batch_size * num_gpus),
        ),
        num_samples=valid_n,
        results_folder=results_folder,
        precision=precision,
        amp=amp,
        calculate_fid=False,
        feat_stats=train_sets[0].feat_stats,
        norm_feat=cfg.dataset.norm_feat,
    )

    if args.checkpoint_num is not None:
        print("Continuing training from checkpoint", args.checkpoint_num)
        trainer.load(args.results_folder, args.checkpoint_num)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--server", type=str, default="hacienda", choices=["jz", "local"]
    )  # set to 'jz' to use jean zay server
    parser.add_argument(
        "-o", "--override", type=bool, default=False
    )  # set to True to overwrite results folder
    parser.add_argument(
        "-num", "--num_samples", type=int, default=1
    )  # set to number of samples to generate
    parser.add_argument(
        "-m", "--mode", type=str, default="train", choices=["train", "inference"]
    )  # set to 'inference' to generate samples
    parser.add_argument(
        "-c", "--checkpoint_num", type=int, default=None
    )  # set to checkpoint number to resume training or generate samples
    parser.add_argument(
        "-p", "--inference_path", type=str, default=None
    )  # set to get initial image
    parser.add_argument(
        "-t", "--text", type=str, default=None
    )  # set to text to generate samples
    parser.add_argument(
        "-n", "--sample_steps", type=int, default=100
    )  # set to number of steps to sample
    parser.add_argument(
        "-g", "--guidance_weight", type=int, default=0
    )  # set to positive to use guidance
    parser.add_argument(
        "--diffuse_on",
        type=str,
        default="pixel",
        choices=["pixel", "dino", "dino_vit", "r3m"],
    )  # set to 'pixel' or 'dino_feat' to diffuse on pixel or dino features
    parser.add_argument(
        "--feat_patch_size", type=int, default=16
    )  # set to patch size for dino features
    parser.add_argument(
        "--num_subgoals", type=int, default=8
    )  # set to number of subgoals
    parser.add_argument(
        "-r", "--results_folder", type=str, default="../results_huit_ann/calvin"
    )  # set to results folder
    parser.add_argument(
        "--feature_decoder_checkpoint_path",
        type=str,
        default=None,  # "/home/grislain/AVDC/calvin/models/decoder/decoder_model_48.pth",
    )  # set to decoder checkpoint path
    parser.add_argument(
        "--data_paths",
        type=str,
        nargs="+",
        default=["/home/grislain/AVDC/data/toy_env_demos/training"],
    )  # set to data path
    parser.add_argument(
        "--train_num_steps", type=int, default=150000
    )  # set to number of training steps
    parser.add_argument(
        "--batch_size", type=int, default=16
    )  # set to batch size for training
    parser.add_argument(
        "--min_batch_size", type=int, default=8
    )  # set to batch size for training
    parser.add_argument(
        "--save_and_sample_every", type=int, default=2500
    )  # set to number of steps to save and sample
    parser.add_argument(
        "--diff_objective",
        type=str,
        default="pred_v",
        choices=["pred_x0", "pred_v", "pred_noise"],
    )  # set to diffusion objective
    parser.add_argument(
        "--norm", type=str, default=None, choices=[None, "l2", "z_score", "min_max"]
    )  # set to normalisation type for features
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="CLIP",
        choices=["CLIP", "Flan-t5", "Siglip"],
    )  # set to text encoder type
    parser.add_argument("--use_depth", action="store_true")  # set to use depth images
    parser.add_argument(
        "--use_gripper", action="store_true"
    )  # set to use gripper images
    parser.add_argument(
        "--temporal_loss_weight",
        type=float,
        default=0.0,
    )  # set to temporal loss weight
    parser.add_argument(
        "--prob_temp_swaps",
        type=float,
        default=0.0,
    )  # set to probability of temporal swaps
    parser.add_argument(
        "--num_swaps",
        type=int,
        default=1,
    )  # set to number of swaps for data aug in training
    parser.add_argument(
        "--use_filtered_data",
        action="store_true",
        help="Use filtered data (expert successes) for evaluation.",
    )
    args = parser.parse_args()

    print()
    print("Arguments:", args)
    print()

    if args.mode == "inference":
        assert args.checkpoint_num is not None
        assert args.inference_path is not None
        assert args.text is not None
        assert args.sample_steps <= 100
    if args.mode == "train":
        if args.use_depth:
            assert args.diffuse_on == "pixel", (
                "Depth images only supported for pixel diffusion"
            )
    main(args)
