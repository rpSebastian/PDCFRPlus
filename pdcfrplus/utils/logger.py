import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import wandb
from torch.utils.tensorboard import SummaryWriter

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50


class Logger:
    def __init__(
        self,
        writer_strings: Optional[List[str]] = ["stdout"],
        folder: Optional[Union[str, Path]] = None,
        log_level: Optional[Union[int, str]] = INFO,
        project: Optional[str] = None,
        config: Optional[Union[Dict, str, None]] = None,
        group: Optional[str] = None,
        tags: Optional[Sequence] = None,
        name: Optional[str] = None,
        sweep: Optional[bool] = False,
        csv_name: Optional[str] = "data",
    ):
        self.folder = folder
        self.name_to_value = defaultdict(float)
        self.name_to_count = defaultdict(int)
        self.name_to_excluded = defaultdict(str)
        self.log_level = self._to_int_log_level(log_level)
        self.project = project
        self.config = config
        self.group = group
        self.tags = tags
        self.name = name
        self.sweep = sweep
        self.csv_name = csv_name
        self._make_writers(folder, writer_strings)

    def _to_int_log_level(self, log_level: Union[int, str] = INFO) -> int:
        level_dict = {
            "debug": DEBUG,
            "info": INFO,
            "warn": WARN,
            "error": ERROR,
            "disabled": "DISABLED",
        }
        if isinstance(log_level, str):
            int_log_level = level_dict[log_level.lower()]
        else:
            int_log_level = log_level
        return int_log_level

    def _make_writers(
        self, folder: Union[str, Path], writer_strings: List[str]
    ) -> None:
        self.writers = []
        for writer_string in writer_strings:
            if writer_string == "tensorboard":
                if folder is None:
                    continue
                writer = TensorBoardWriter(folder)
            elif writer_string == "sacred":
                if folder is None:
                    continue
                writer = SacredWriter(folder)
            elif writer_string == "stdout":
                writer = StdoutWriter(self.log_level)
            elif writer_string == "wandb":
                writer = WandbWriter(
                    project=self.project,
                    config=self.config,
                    tags=self.tags,
                    group=self.group,
                    name=self.name,
                    sweep=self.sweep,
                )
            elif writer_string == "csv":
                if folder is None:
                    continue
                writer = CsvWriter(folder, self.csv_name)
            else:
                raise ValueError(f"Unknown writer: {writer_string}")
            self.writers.append(writer)

    def record(self, key: str, value: Any) -> None:
        self.name_to_value[key] = value

    def record_mean(self, key: str, value: Any) -> None:
        if value is None:
            self.name_to_value[key] = None
            return
        old_val, count = self.name_to_value[key], self.name_to_count[key]
        self.name_to_value[key] = old_val * count / (count + 1) + value / (count + 1)
        self.name_to_count[key] = count + 1

    def dump(self, step: int = 0) -> None:
        for writer in self.writers:
            writer.write(self.name_to_value, step)
        self.name_to_value.clear()
        self.name_to_count.clear()

    def close(self) -> None:
        for writer in self.writers:
            writer.close()

    def log(self, *args, level: int = INFO) -> None:
        self._do_log(args)

    def debug(self, *args) -> None:
        self.log(*args, level=DEBUG)

    def info(self, *args) -> None:
        self.log(*args, level=INFO)

    def warn(self, *args) -> None:
        self.log(*args, level=WARN)

    def error(self, *args) -> None:
        self.log(*args, level=ERROR)

    def _do_log(self, args) -> None:
        for writer in self.writers:
            if isinstance(writer, SeqWriter):
                writer.write_sequence(map(str, args))


class Writer:
    def write(self, key_values: Dict[str, Any], step: int = 0) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class SeqWriter:
    def write_sequence(self, sequence: List) -> None:
        raise NotImplementedError


class TensorBoardWriter(Writer):
    def __init__(self, folder: Union[str, Path]):
        self.writer = SummaryWriter(log_dir=folder)

    def write(self, key_values: Dict[str, Any], step: int = 0) -> None:
        for key, value in key_values.items():
            if isinstance(value, np.ScalarType):
                if isinstance(value, str):
                    self.writer.add_text(key, value, step)
                else:
                    self.writer.add_scalar(key, value, step)

            if isinstance(value, th.Tensor):
                self.writer.add_histogram(key, value, step)

            if isinstance(value, Video):
                self.writer.add_video(key, value.frames, step, value.fps)

            if isinstance(value, Figure):
                self.writer.add_figure(key, value.figure, step, close=value.close)

            if isinstance(value, Image):
                self.writer.add_image(
                    key, value.image, step, dataformats=value.dataformats
                )

        self.writer.flush()

    def close(self) -> None:
        if self.writer:
            self.writer.close()
            self.writer = None


class WandbWriter(Writer):
    def __init__(
        self,
        project: str = None,
        config: Union[Dict, str, None] = None,
        tags: Sequence = None,
        group: str = None,
        name: str = None,
        sweep: str = False,
    ):
        self.sweep = sweep
        if self.sweep:
            wandb.init(config=config)
        else:
            wandb.init(
                project=project,
                config=config,
                name=name,
                group=group,
                tags=tags,
            )

    def write(self, key_values: Dict[str, Any], step: int = 0) -> None:
        info = {}
        for key, value in key_values.items():
            if isinstance(value, np.ScalarType):
                info[key] = value

        if self.sweep:
            wandb.log(info, step)
        else:
            wandb.log(info, step, commit=True)

    def close(self) -> None:
        wandb.finish()


class SacredWriter(Writer):
    def __init__(self, folder: Union[str, Path]):
        from pdcfrplus.utils.exp import ex

        self.writer = ex
        self.folder = folder

    def write(self, key_values: Dict[str, Any], step: int = 0) -> None:
        for key, value in key_values.items():
            if isinstance(value, np.ScalarType):
                self.writer.log_scalar(key, value, step)

            if isinstance(value, nn.Module):
                model_folder = Path(self.folder) / "model"
                model_folder.mkdir(parents=True, exist_ok=True)
                file = model_folder / f"{key}_{step}.pkl"
                th.save(value.state_dict(), file)

            if isinstance(value, Dict):
                data_folder = Path(self.folder) / "data" / key
                data_folder.mkdir(parents=True, exist_ok=True)
                file = data_folder / f"{key}_{step}.pkl"
                with open(file, "wb") as f:
                    pickle.dump(value, f)

            if isinstance(value, th.Tensor):
                pass

            if isinstance(value, Video):
                pass

            if isinstance(value, Figure):
                pass

            if isinstance(value, Image):
                pass

    def close(self) -> None:
        pass


class CsvWriter(Writer):
    def __init__(self, folder: Union[str, Path], csv_name: str = "data"):
        self.folder = folder
        self.data = pd.DataFrame(columns=["step"])
        folder.mkdir(parents=True, exist_ok=True)
        self.csv_file = Path(self.folder) / f"{csv_name}.csv"

    def write(self, key_values: Dict[str, Any], step: int = 0) -> None:
        pd_key_values = {}
        new_key = False
        for key, value in key_values.items():
            if isinstance(value, np.ScalarType):
                pd_key_values[key] = value
                if key not in self.data.columns:
                    self.data[key] = None
                    new_key = True
        pd_key_values["step"] = step

        if step in self.data.index:
            data = self.data.loc[step].to_dict()
            data.update(pd_key_values)
            self.data.loc[step] = data
            new_key = True
        else:
            self.data.loc[step] = pd_key_values

        if new_key:
            self.data.to_csv(self.csv_file, index=False)
        else:
            new_data = pd.DataFrame(columns=self.data.columns)
            new_data.loc[step] = pd_key_values
            new_data.to_csv(self.csv_file, mode="a", index=False, header=False)

    def close(self) -> None:
        pass


class StdoutWriter(Writer, SeqWriter):
    def __init__(self, log_level: int = INFO):
        self.max_length = 36
        self.log_level = log_level
        self.writer = self.get_logger(log_level)

    def get_logger(self, log_level: int) -> logging.Logger:
        logger = logging.getLogger()
        logger.handlers = []
        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname).1s] %(filename)s:%(lineno)d - %(message)s ",
            "%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.setLevel(log_level)
        return logger

    def write(self, key_values: Dict[str, Any], step: int = 0) -> None:
        key2str = {}
        tag = None

        for key, value in sorted(key_values.items()):
            if not isinstance(value, np.ScalarType):
                continue
            if isinstance(value, float):
                value_str = f"{value:<10.5g}"
            else:
                value_str = str(value)

            if key.find("/") > 0:  # Find tag and add it to the dict
                tag = key[: key.find("/") + 1]
                key2str[self._truncate(tag)] = ""

            # Remove tag from key
            if tag is not None and tag in key:
                key = str("   " + key[len(tag) :])

            truncated_key = self._truncate(key)
            if truncated_key in key2str:
                raise ValueError(
                    f"Key '{key}' truncated to '{truncated_key}' that already exists. Consider increasing `max_length`."
                )
            key2str[truncated_key] = self._truncate(value_str)

        # Find max widths
        if len(key2str) == 0:
            import warnings

            warnings.warn("Tried to write empty key-value dict")
            return
        else:
            key_width = max(map(len, key2str.keys()))
            val_width = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (key_width + val_width + 7)
        lines = ["\n" + dashes]
        for key, value in key2str.items():
            key_space = " " * (key_width - len(key))
            val_space = " " * (val_width - len(value))
            lines.append(f"| {key}{key_space} | {value}{val_space} |")
        lines.append(dashes)
        self.writer.info("\n".join(lines))

    def _truncate(self, string: str) -> str:
        if len(string) > self.max_length:
            string = string[: self.max_length - 3] + "..."
        return string

    def close(self) -> None:
        pass

    def write_sequence(self, sequence: List) -> None:
        sequence = list(sequence)
        line = ""
        for i, elem in enumerate(sequence):
            line += elem
            if i < len(sequence) - 1:  # add space unless this is the last one
                line += " "
        self.writer.info(line)


class Video:
    """
    Video data class storing the video frames and the frame per seconds
    :param frames: frames to create the video from
    :param fps: frames per second
    """

    def __init__(self, frames: th.Tensor, fps: Union[float, int]):
        self.frames = frames
        self.fps = fps


class Figure:
    """
    Figure data class storing a matplotlib figure and whether to close the figure after logging it
    :param figure: figure to log
    :param close: if true, close the figure after logging it
    """

    def __init__(self, figure: plt.figure, close: bool):
        self.figure = figure
        self.close = close


class Image:
    """
    Image data class storing an image and data format
    :param image: image to log
    :param dataformats: Image data format specification of the form NCHW, NHWC, CHW, HWC, HW, WH, etc.
        More info in add_image method doc at https://pytorch.org/docs/stable/tensorboard.html
        Gym envs normally use 'HWC' (channel last)
    """

    def __init__(self, image: Union[th.Tensor, np.ndarray, str], dataformats: str):
        self.image = image
        self.dataformats = dataformats
