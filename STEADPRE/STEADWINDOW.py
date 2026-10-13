"""从 STEADSPLIT 的固定窗口清单懒加载波形；可直接交给 torch DataLoader。

不导入 torch、不把整个 HDF5 载入内存。输入仅三分量波形，标签/ID 不拼入模型特征。
train 可做非循环平移；val/test 禁止增强。persistent_workers 请设 False，以传播 epoch。
"""

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

if __package__:
    from .STEADSPLIT import read_waveform, sha256_file
    from .WAVEPRE import PreprocessConfig, preprocess_waveform, draw_shift, validate_manifest
else:
    from STEADSPLIT import read_waveform, sha256_file
    from WAVEPRE import PreprocessConfig, preprocess_waveform, draw_shift, validate_manifest


def prepare_window(raw, p_sample=np.nan, s_sample=np.nan, shift=0, sigma_samples=5.0):
    """兼容旧入口；唯一预处理实现现位于 WAVEPRE.py，避免训练与验收分叉。"""
    return preprocess_waveform(raw, p_sample, s_sample, shift,
                               PreprocessConfig(sigma_seconds=sigma_samples / 100))


class STEADWindowDataset:
    """Map-style Dataset；每进程独立只读 HDF5 句柄，支持 Windows spawn workers。

    Dataset(..., augment=True) 仅用于 train。每 epoch 调用 set_epoch(epoch)，再创建
    DataLoader 迭代器，且 persistent_workers=False。划分 seed 与训练 seed 相互独立。
    """

    def __init__(self, manifest, data_dir=None, augment=False, seed=42, max_shift_seconds=5.0, config=None):
        self.manifest = Path(manifest).resolve()
        summary_path = self.manifest.parent / "summary.json"
        if not summary_path.is_file():
            raise ValueError("Split is incomplete: missing summary.json. Run STEADSPLIT.py first.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = summary["output_sha256"].get(self.manifest.name)
        if expected is None or sha256_file(self.manifest) != expected:
            raise ValueError("Window manifest is not part of the frozen split or has changed.")
        if data_dir is None:
            self.hdf5_path = Path(summary["inputs"]["waveforms.hdf5"]["path"])
        else:
            self.hdf5_path = Path(data_dir) / "waveforms.hdf5"
        stat = self.hdf5_path.stat()
        identity = summary["inputs"]["waveforms.hdf5"]
        if stat.st_size != identity["size_bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
            raise ValueError("HDF5 changed after the split; audit before training.")
        self.rows = pd.read_csv(self.manifest, keep_default_na=False, low_memory=False,
                               dtype={"source_id": str, "station_key": str})
        self.config = config or PreprocessConfig(max_shift_seconds=max_shift_seconds)
        validate_manifest(self.rows, self.config)
        splits = set(self.rows["split"])
        if len(splits) != 1 or not splits <= {"train", "val", "test"}:
            raise ValueError("Dataset must contain exactly one valid split.")
        self.split = next(iter(splits))
        if augment and self.split != "train":
            raise ValueError("Validation/test windows must remain native and deterministic.")
        if not 0 <= max_shift_seconds <= 5 or seed < 0:
            raise ValueError("max_shift_seconds must be in [0,5]; seed must be nonnegative.")
        self.augment = augment
        self.seed, self.epoch = int(seed), 0
        self.max_shift = round(self.config.max_shift_seconds * 100)
        self._handle, self._pid = None, None

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __getitem__(self, index):
        if self._handle is None or self._pid != os.getpid():
            self.close()
            self._handle = h5py.File(self.hdf5_path, "r")
            self._pid = os.getpid()
        row = self.rows.iloc[index]
        raw = read_waveform(self._handle, row.trace_name)
        p = float(row.p_sample_in_window) if row.kind == "earthquake" else np.nan
        try:
            s = float(row.s_sample_in_window)
        except (TypeError, ValueError):
            s = np.nan
        shift = 0
        if self.augment:
            shift = draw_shift(p, self.seed, self.epoch, int(row.metadata_row), self.config)
        result = preprocess_waveform(raw, p, s, shift, self.config)
        result.update({"window_id": str(row.window_id), "event_group_key": str(row.event_group_key),
                       "station_key": str(row.station_key), "is_noise": row.kind == "noise"})
        return result

    def close(self):
        if self._handle is not None:
            self._handle.close()
        self._handle, self._pid = None, None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"], state["_pid"] = None, None
        return state

    def __del__(self):
        if hasattr(self, "_handle"):
            self.close()
