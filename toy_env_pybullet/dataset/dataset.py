import random
from itertools import chain
from pathlib import Path
from typing import Dict, List, Tuple, Union

import hydra
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
from toy_env_pybullet.dataset.utils import (
    load_npz,
    lookup_naming_pattern,
    process_actions,
    process_language,
    process_rgb,
)


class ToyDataset(Dataset):
    """
    Dataset that loads episodes as individual files from disk.

    Args:
        num_subgoals: Number of subgoals per episodes.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        path,
        kwargs,
    ):
        self.abs_datasets_dir = Path(path)
        self.load_file = load_npz
        self.num_subgoals = kwargs.num_subgoals
        self.auto_lang_name = kwargs.auto_lang_name
        self.min_window_size = kwargs.min_window_size
        self.max_window_size = kwargs.max_window_size
        self.observation_space = kwargs.obs_space
        self.save_format = "npz"
        self.relative_actions = True
        self.pad = True
        self.with_lang = True
        self.feat_stats = None

        self.episode_lookup, self.lang_lookup, self.lang_ann, self.lang_task = (
            self._build_file_indices_lang(self.abs_datasets_dir)
        )
        self.naming_pattern, self.n_digits = lookup_naming_pattern(
            self.abs_datasets_dir, self.save_format
        )

        self.transforms = {
            cam: [
                hydra.utils.instantiate(transform)
                for transform in kwargs.transforms.train[cam]
            ]
            for cam in kwargs.transforms.train
        }

    def __len__(self) -> int:
        return len(self.episode_lookup)

    def _get_episode_name(self, file_idx: int) -> Path:
        return Path(
            f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def _load_episode(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Load num_goals frames of the episodes (plus the initial frame) evenly spaced.

        Args:
            idx: Index of first frame.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx, end_idx = self.episode_lookup[idx]

        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        frames_idx = np.linspace(
            start_idx,
            end_idx,
            min(self.max_window_size, self.num_subgoals + 1),
            dtype=int,
        )
        frames = [
            self.load_file(self._get_episode_name(file_idx)) for file_idx in frames_idx
        ]

        episode = {key: np.stack([ep[key] for ep in frames]) for key in keys}
        episode["language"] = self.lang_ann[self.lang_lookup[idx]]
        episode["task"] = self.lang_task[self.lang_lookup[idx]]

        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language tasks.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        # Load lang data from pickle
        try:
            print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / f"{self.auto_lang_name}.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / self.lang_folder / f"{self.auto_lang_name}.npy",
                allow_pickle=True,
            ).item()
        except Exception:
            print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / f"{self.auto_lang_name}.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / f"{self.auto_lang_name}.npy", allow_pickle=True
            ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are <=64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]  # length total number of annotations
        lang_lookup = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            assert end_idx >= self.min_window_size
            lang_lookup.append(i)
            episode_lookup.append((start_idx, end_idx))

        return np.array(episode_lookup), lang_lookup, lang_ann, lang_task

    def _get_sequences(self, idx: int) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx)

        seq_state_obs = {
            "states": torch.tensor(episode["states"])
        }  # No state processing needed
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        seq_lang = process_language(episode, self.transforms, self.with_lang)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_acts,
            **seq_lang,
        }  # type:ignore
        seq_dict["idx"] = idx  # type:ignore

        return seq_dict

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.num_subgoals + 1 - len(sequence["actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"states": self._pad_with_repetition(seq["states"], pad_size)})
        seq.update(
            {"rgb_static": self._pad_with_repetition(seq["rgb_static"], pad_size)}
        )

        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            # repeat action for world coordinates action space
            seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            seq_acts = torch.cat(
                [
                    self._pad_with_zeros(seq["actions"][..., :-1], pad_size),
                    self._pad_with_repetition(seq["actions"][..., -1:], pad_size),
                ],
                dim=-1,
            )
            seq.update({"actions": seq_acts})
        return seq

    def _pad_with_repetition(
        self, input_tensor: torch.Tensor, pad_size: int
    ) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        last_repeated = torch.repeat_interleave(
            torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    def _pad_with_zeros(
        self, input_tensor: torch.Tensor, pad_size: int
    ) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """

        sequence = self._get_sequences(idx)
        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size)

        images = sequence["rgb_static"]
        x_cond = images[0, ...]
        x = images[1:, ...]
        x_cond = x_cond.squeeze(0)
        x = rearrange(x, "f c h w -> (f c) h w")
        task = sequence["lang"]
        return x, x_cond, task


class ToyActionDataset(Dataset):
    """
    Dataset that loads episodes as individual files from disk.

    Args:
        num_subgoals: Number of subgoals per episodes.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        path,
        kwargs,
    ):
        self.abs_datasets_dir = Path(path)
        self.load_file = load_npz
        self.num_subgoals = kwargs.num_subgoals
        self.auto_lang_name = kwargs.auto_lang_name
        self.min_window_size = kwargs.min_window_size
        self.max_window_size = kwargs.max_window_size
        self.observation_space = kwargs.obs_space
        self.save_format = "npz"
        self.relative_actions = True
        self.pad = True
        self.with_lang = True
        self.feat_stats = None

        self.episode_lookup, self.lang_lookup, self.lang_ann, self.lang_task = (
            self._build_file_indices_lang(self.abs_datasets_dir)
        )
        self.naming_pattern, self.n_digits = lookup_naming_pattern(
            self.abs_datasets_dir, self.save_format
        )

        self.transforms = {
            cam: [
                hydra.utils.instantiate(transform)
                for transform in kwargs.transforms.train[cam]
            ]
            for cam in kwargs.transforms.train
        }

    def __len__(self) -> int:
        return len(self.episode_lookup)

    def _get_episode_name(self, file_idx: int) -> Path:
        return Path(
            f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}"
        )

    def _load_episode(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Load entire_episode.

        Args:
            idx: Index of first frame.
        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx, end_idx, j = self.episode_lookup[idx]

        num_frames = end_idx - start_idx + 1
        chunk_size = random.randint(
            int(np.ceil(min(self.min_window_size, num_frames) / self.num_subgoals)),
            int(np.ceil(num_frames / self.num_subgoals)),
        )

        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")

        episodes_idx = np.arange(start_idx, end_idx + 1 - chunk_size + 1)

        # Pick randm frame from the episode except the last one
        frame_idx = episodes_idx[j]

        # Action idx are from the frame_idx to the next frame
        assert frame_idx + chunk_size <= end_idx + 1
        actions_idx = np.arange(frame_idx, frame_idx + chunk_size)

        episodes = [
            self.load_file(self._get_episode_name(file_idx)) for file_idx in actions_idx
        ]

        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        episode["language"] = self.lang_ann[self.lang_lookup[idx]]
        episode["task"] = self.lang_task[self.lang_lookup[idx]]
        return episode

    def _build_file_indices_lang(
        self, abs_datasets_dir: Path
    ) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language tasks.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        # Load lang data from pickle
        try:
            print(
                "trying to load lang data from: ",
                abs_datasets_dir / self.lang_folder / f"{self.auto_lang_name}.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / self.lang_folder / f"{self.auto_lang_name}.npy",
                allow_pickle=True,
            ).item()
        except Exception:
            print(
                "Exception, trying to load lang data from: ",
                abs_datasets_dir / f"{self.auto_lang_name}.npy",
            )
            lang_data = np.load(
                abs_datasets_dir / f"{self.auto_lang_name}.npy", allow_pickle=True
            ).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are <=64
        lang_ann = lang_data["language"]["ann"]  # length total number of annotations
        lang_task = lang_data["language"]["task"]  # length total number of annotations
        lang_lookup = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            num_frames = end_idx - start_idx + 1
            chunk_size = int(np.ceil(num_frames / self.num_subgoals))
            for j in range(0, max(1, num_frames - chunk_size)):
                episode_lookup.append((start_idx, end_idx, j))
                lang_lookup.append(i)
        return np.array(episode_lookup), lang_lookup, lang_ann, lang_task

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")

        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            assert end_idx >= self.min_window_size
            num_frames = end_idx - start_idx + 1
            chunk_size = int(np.ceil(self.max_window_size / self.num_subgoals))
            for j in range(0, max(1, num_frames - chunk_size)):
                episode_lookup.append(
                    (
                        start_idx
                        + (j // (self.max_window_size - chunk_size + 1))
                        * (self.max_window_size - chunk_size + 1),
                        start_idx
                        + (j // (self.max_window_size - chunk_size + 1))
                        * (self.max_window_size - chunk_size + 1)
                        + self.max_window_size,
                        j % (self.max_window_size - chunk_size + 1),
                    )
                )
        return np.array(episode_lookup)

    def _get_sequences(self, idx: int) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx)

        seq_state_obs = {
            "states": torch.tensor(episode["states"])
        }  # No state processing needed
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        seq_lang = process_language(episode, self.transforms, self.with_lang)
        seq_dict = {
            **seq_state_obs,
            **seq_rgb_obs,
            **seq_acts,
            **seq_lang,
        }  # type:ignore
        seq_dict["idx"] = idx  # type:ignore

        return seq_dict

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.num_subgoals + 1 - len(sequence["actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"states": self._pad_with_repetition(seq["states"], pad_size)})
        seq.update(
            {"rgb_static": self._pad_with_repetition(seq["rgb_static"], pad_size)}
        )

        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            # repeat action for world coordinates action space
            seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            seq_acts = torch.cat(
                [
                    self._pad_with_zeros(seq["actions"][..., :-1], pad_size),
                    self._pad_with_repetition(seq["actions"][..., -1:], pad_size),
                ],
                dim=-1,
            )
            seq.update({"actions": seq_acts})
        return seq

    def _pad_with_repetition(
        self, input_tensor: torch.Tensor, pad_size: int
    ) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        last_repeated = torch.repeat_interleave(
            torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    def _pad_with_zeros(
        self, input_tensor: torch.Tensor, pad_size: int
    ) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0),
            repeats=pad_size,
            dim=0,
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        """
        Get a processed sequence from the dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Dictionary containing observations, actions, and optionally goals and features.
        """
        sequence = self._get_sequences(idx)

        if self.pad:
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size)

        init = sequence["rgb_static"][0][None, ...]
        goal = sequence["rgb_static"][-1][None, ...]
        state = np.zeros((init.shape[0], 0), dtype=np.float32)

        # Actions
        actions = sequence["actions"][:-1]
        action_is_pad = torch.zeros_like(actions).sum(dim=-1).bool()

        # Compose result
        res = {
            "observation.state": state,
            "action": actions,
            "action_is_pad": action_is_pad,
            "text": sequence["lang"],
            "observation.image_goal_static": goal,
            "observation.image_static": init,
        }

        return res
