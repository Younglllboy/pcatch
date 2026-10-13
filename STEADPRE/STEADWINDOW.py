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
else:
    from STEADSPLIT import read_waveform, sha256_file


def prepare_window(raw, p_sample=np.nan, s_sample=np.nan, shift=0, sigma_samples=5.0):
    """返回 x[3,T], y[1,T], valid_mask[1,T]；不插值/截断 fractional pick。"""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (3, 6000) or not np.isfinite(raw).all():
        raise ValueError("Expected a finite native (3,6000) window.")
    n = raw.shape[1]
    shift = int(shift)
    if abs(shift) > 500 or sigma_samples <= 0:
        raise ValueError("Allowed shift is +/-500 samples; sigma must be positive.")
    p = float(p_sample) + shift if np.isfinite(p_sample) else np.nan
    s = float(s_sample) + shift if np.isfinite(s_sample) else np.nan
    if np.isfinite(p) and not 0 <= p < n:
        raise ValueError("Augmentation would remove P from window.")
    x = np.zeros_like(raw)
    start, stop = max(0, shift), min(n, n + shift)
    x[:, start:stop] = raw[:, start - shift:stop - shift]
    # 单窗口、有效样本内去均值/线性趋势，联合 RMS 保留分量相对振幅。
    valid_x = x[:, start:stop]
    t = np.arange(stop - start, dtype=np.float64)
    t -= t.mean()
    valid_x -= valid_x.mean(axis=1, keepdims=True)
    slopes = (valid_x @ t) / np.dot(t, t)
    valid_x -= slopes[:, None] * t[None, :]
    rms = float(np.sqrt(np.mean(valid_x ** 2)))
    valid_x /= max(rms, 1e-8)
    mask = np.zeros((1, n), dtype=np.float32)
    mask[:, start:stop] = 1
    target = np.zeros((1, n), dtype=np.float32)
    if np.isfinite(p):
        target[0] = np.exp(-0.5 * ((np.arange(n) - p) / sigma_samples) ** 2)
        target *= mask
    if not start <= s < stop:
        s = np.nan
    return {"waveform": x.astype(np.float32), "p_target": target, "valid_mask": mask,
            "p_sample": np.float32(p), "s_sample": np.float32(s), "shift_samples": shift}


class STEADWindowDataset:
    """Map-style Dataset；每进程独立只读 HDF5 句柄，支持 Windows spawn workers。

    Dataset(..., augment=True) 仅用于 train。每 epoch 调用 set_epoch(epoch)，再创建
    DataLoader 迭代器，且 persistent_workers=False。划分 seed 与训练 seed 相互独立。
    """

    def __init__(self, manifest, data_dir=None, augment=False, seed=42, max_shift_seconds=5.0):
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
        self.max_shift = round(max_shift_seconds * 100)
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
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(row.metadata_row)]))
            low, high = -self.max_shift, self.max_shift
            if np.isfinite(p):
                # 保留完整 +/-3 sigma P 标签，同时不使用循环卷绕。
                low = max(low, int(np.ceil(15 - p)))
                high = min(high, int(np.floor(5984 - p)))
            shift = int(rng.integers(low, high + 1))
        result = prepare_window(raw, p, s, shift, float(row.p_target_sigma_samples))
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
