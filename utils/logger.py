"""
utils/logger.py
---------------
Dual-sink logger: writes to a plain-text file AND streams to stdout,
plus writes scalar summaries to TensorBoard.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Combines Python logging (file + console) and TensorBoard SummaryWriter.

    Args:
        log_file:         Path to the plain-text log file.
        tensorboard_dir:  Directory for TensorBoard event files.
        name:             Logger name (appears in log lines).
    """

    def __init__(
        self,
        log_file: str,
        tensorboard_dir: str,
        name: str = "gait_occlusion",
    ):
        # ---- File + console Python logger ----------------------------------
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        self._logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(logging.INFO)
        self._logger.addHandler(ch)

        # ---- TensorBoard ---------------------------------------------------
        self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)

        self.info(f"Logger initialised. Log file: {log_file}")
        self.info(f"TensorBoard dir: {tensorboard_dir}")

    # ------------------------------------------------------------------
    # Text logging proxies
    # ------------------------------------------------------------------
    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    # ------------------------------------------------------------------
    # TensorBoard helpers
    # ------------------------------------------------------------------

    def log_scalars(self, tag_value_dict: dict, step: int) -> None:
        """Log multiple scalars in one call."""
        for tag, value in tag_value_dict.items():
            self.tb_writer.add_scalar(tag, value, global_step=step)

    def log_epoch_metrics(
        self,
        metrics: dict,
        epoch: int,
        prefix: str = "train",
    ) -> None:
        """Log a full metrics dict with a given prefix."""
        for k, v in metrics.items():
            self.tb_writer.add_scalar(f"{prefix}/{k}", v, global_step=epoch)

    def log_lr(self, lr: float, step: int) -> None:
        self.tb_writer.add_scalar("train/lr", lr, global_step=step)

    def close(self) -> None:
        self.tb_writer.close()
