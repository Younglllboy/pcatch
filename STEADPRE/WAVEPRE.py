"""实验 A / STEAD 第七步：冻结协议、全量预处理验收、训练接口与可视化。

python STEADPRE/WAVEPRE.py
python STEADPRE/WAVEPRE.py --verify

只读已有 stead_split/main，原生 ZNE / 100 Hz / 6000 点，无再次划分/重采样。
完整扫描原生窗口，并检查 train 的 epoch=0 平移版本。训练时通过 load_dataset
按需生成波形和标签，不复制整库或提前保存小波图。结果 artifacts/stead_wavepre/main。
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import platform
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

if __package__:
    from .STEADSPLIT import ROOT, SPLITS, log, rank_key, read_waveform, save_csv, sha256_file, verify_saved
else:
    from STEADSPLIT import ROOT, SPLITS, log, rank_key, read_waveform, save_csv, sha256_file, verify_saved


@dataclass(frozen=True)
class PreprocessConfig:
    sigma_seconds: float = 0.05
    rms_floor: float = 1e-8
    max_shift_seconds: float = 5.0
    min_valid_fraction: float = 0.9
    bandpass: bool = False
    sampling_rate: float = 100.0
    n_samples: int = 6000
    component_order: str = "ZNE"
    constant_run_seconds: float = 0.2
    rail_run_samples: int = 5

    def __post_init__(self):
        numeric = [self.sigma_seconds, self.rms_floor, self.max_shift_seconds,
                   self.min_valid_fraction, self.sampling_rate, self.constant_run_seconds]
        if not np.isfinite(numeric).all():
            raise ValueError("Preprocessing parameters must be finite.")
        if self.sigma_seconds <= 0 or self.rms_floor <= 0:
            raise ValueError("sigma and RMS floor must be positive.")
        if not 0 <= self.max_shift_seconds <= 5 or not 0.9 <= self.min_valid_fraction <= 1:
            raise ValueError("Require max shift <=5 s and valid fraction >=90%.")
        if (self.sampling_rate, self.n_samples, self.component_order) != (100, 6000, "ZNE"):
            raise ValueError("This experiment A protocol only accepts native ZNE / 100 Hz / 6000 samples.")
        if self.constant_run_seconds <= 0 or self.rail_run_samples < 2:
            raise ValueError("Invalid diagnostic run thresholds.")

    @property
    def sigma_samples(self):
        return self.sigma_seconds * self.sampling_rate


def _longest_true(values):
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return int(np.max(stops - starts, initial=0))


def waveform_qc(raw, config=None):
    """可证实异常硬失败；连续常值/零值/极值平台只是疑似诊断，不冒充缺口标志。"""
    config = config or PreprocessConfig()
    x = np.asarray(raw)
    result = {"hard_failures": [], "flags": [], "max_zero_run_samples": 0,
              "max_constant_run_samples": 0, "max_rail_run_samples": 0}
    if x.shape != (3, config.n_samples):
        result["hard_failures"].append("wrong_shape")
        return result
    if not np.isfinite(x).all():
        result["hard_failures"].append("nonfinite_samples")
        return result
    constant_threshold = int(np.ceil(config.constant_run_seconds * config.sampling_rate))
    for component, channel in zip("ZNE", x):
        zero_run = _longest_true(channel == 0)
        constant_run = 1 + _longest_true(np.diff(channel) == 0)
        rail_run = max(_longest_true(channel == channel.min()), _longest_true(channel == channel.max()))
        result["max_zero_run_samples"] = max(result["max_zero_run_samples"], zero_run)
        result["max_constant_run_samples"] = max(result["max_constant_run_samples"], constant_run)
        result["max_rail_run_samples"] = max(result["max_rail_run_samples"], rail_run)
        if constant_run == config.n_samples:
            result["hard_failures"].append(f"constant_component_{component}")
        if zero_run >= constant_threshold:
            result["flags"].append(f"possible_zero_gap_{component}")
        if constant_run >= constant_threshold:
            result["flags"].append(f"constant_run_{component}")
        if rail_run >= config.rail_run_samples:
            result["flags"].append(f"possible_clipping_plateau_{component}")
    return result


def _valid_limits(shift, config):
    if not np.isfinite(shift) or float(shift) != int(shift):
        raise ValueError("shift must be an integer number of samples.")
    shift = int(shift)
    if abs(shift) > round(config.max_shift_seconds * config.sampling_rate):
        raise ValueError("shift exceeds the frozen augmentation limit.")
    start, stop = max(0, shift), min(config.n_samples, config.n_samples + shift)
    if (stop - start) / config.n_samples < config.min_valid_fraction:
        raise ValueError("Insufficient valid samples after shift.")
    return start, stop


def _validate_p(p, shift, start, stop, config):
    if np.isnan(p):
        return
    if not np.isfinite(p) or not 0 <= p < config.n_samples:
        raise ValueError("Original P label is outside its parent waveform.")
    center, margin = p + shift, 3 * config.sigma_samples
    if center - margin < start or center + margin > stop - 1:
        raise ValueError("P +/-3 sigma must lie inside real, unpadded samples.")


def draw_shift(p_sample, seed, epoch, metadata_row, config=None):
    """与 worker 次序无关的可复现增强；原始/平移后都保留真实 +/-3sigma 上下文。"""
    config = config or PreprocessConfig()
    if any(int(v) != v or v < 0 for v in (seed, epoch, metadata_row)):
        raise ValueError("seed, epoch and metadata_row must be nonnegative integers.")
    _validate_p(float(p_sample), 0, 0, config.n_samples, config)
    maximum = min(round(config.max_shift_seconds * config.sampling_rate),
                  int(np.floor(config.n_samples * (1 - config.min_valid_fraction) + 1e-9)))
    low, high = -maximum, maximum
    if np.isfinite(p_sample):
        margin = 3 * config.sigma_samples
        low = max(low, int(np.ceil(margin - p_sample)))
        high = min(high, int(np.floor(config.n_samples - 1 - margin - p_sample)))
    if low > high:
        raise ValueError("No legal shift for this P label and sigma.")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), int(metadata_row)]))
    return int(rng.integers(low, high + 1))


def preprocess_waveform(raw, p_sample=np.nan, s_sample=np.nan, shift=0, config=None):
    """纯函数。p_sample=NaN 只表示调用者已确认的背景，不能据此推断记录类别。

    顺序：真实样本非循环平移 -> 有效区线性去趋势 -> 可选带通 -> 联合 RMS -> 标签。
    不修改输入。P/S 保留 float64 小数样本精度，波形/目标/mask 输出 float32。
    """
    config = config or PreprocessConfig()
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (3, config.n_samples) or not np.isfinite(raw).all():
        raise ValueError("Expected finite native ZNE waveform with shape (3,6000).")
    start, stop = _valid_limits(shift, config)
    shift = int(shift)
    _validate_p(float(p_sample), shift, start, stop, config)
    x = np.zeros_like(raw)
    x[:, start:stop] = raw[:, start - shift:stop - shift]
    valid = x[:, start:stop]
    removed_mean = valid.mean(axis=1, keepdims=True)
    valid -= removed_mean
    t = np.arange(stop - start, dtype=np.float64)
    t -= t.mean()
    removed_slope = (valid @ t) / np.dot(t, t)
    valid -= removed_slope[:, None] * t[None, :]
    if config.bandpass:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(4, [1, 40], btype="bandpass", fs=100, output="sos")
        valid[:] = sosfiltfilt(sos, valid, axis=1, padtype="odd", padlen=27)
    rms = float(np.sqrt(np.mean(valid ** 2)))
    if not np.isfinite(rms) or rms == 0:
        raise ValueError("No usable energy after detrending; do not normalize an empty record.")
    denominator = max(rms, config.rms_floor)
    valid /= denominator
    mask = np.zeros((1, config.n_samples), dtype=np.float32)
    mask[:, start:stop] = 1
    p = float(p_sample) + shift if np.isfinite(p_sample) else np.nan
    s = float(s_sample) + shift if np.isfinite(s_sample) else np.nan
    if not start <= s < stop:
        s = np.nan
    target = np.zeros_like(mask)
    if np.isfinite(p):
        target[0] = np.exp(-0.5 * ((np.arange(config.n_samples) - p) / config.sigma_samples) ** 2)
        target *= mask
    waveform = x.astype(np.float32)
    valid32 = waveform[:, start:stop].astype(np.float64)
    return {"waveform": waveform, "p_target": target, "valid_mask": mask,
            "p_sample": np.float64(p), "s_sample": np.float64(s), "shift_samples": shift,
            "stats": {"valid_start": start, "valid_stop": stop, "valid_fraction": (stop - start) / config.n_samples,
                      "rms_before_normalization": rms, "normalization_denominator": denominator,
                      "rms_floor_applied": rms < config.rms_floor,
                      "joint_rms_after": float(np.sqrt(np.mean(valid32 ** 2))),
                      "max_abs_component_mean_after": float(np.max(np.abs(valid32.mean(axis=1)))),
                      "max_abs_component_slope_after": float(np.max(np.abs((valid32 @ t) / np.dot(t, t)))),
                      "removed_mean": removed_mean.ravel().tolist(), "removed_slope_per_sample": removed_slope.tolist()}}


def masked_bce_with_logits(logits, target, valid_mask):
    """全 batch 有效采样点平均 BCE；只遮挡 loss，mask 不作为模型额外输入。"""
    import torch
    if logits.shape != target.shape or logits.shape != valid_mask.shape:
        raise ValueError("logits, target and valid_mask must have identical shapes.")
    if not torch.isfinite(target).all() or not ((target >= 0) & (target <= 1)).all():
        raise ValueError("Target must be finite and within [0,1].")
    if not torch.isfinite(valid_mask).all() or not ((valid_mask == 0) | (valid_mask == 1)).all():
        raise ValueError("valid_mask must be finite and binary.")
    count = valid_mask.sum(dtype=torch.float32)
    if count.item() <= 0:
        raise ValueError("No valid samples for loss.")
    # 先将 padding logits 置零，避免极端/非有限 padding 经 BCE 后出现 inf*0。
    safe_logits = torch.where(valid_mask.bool(), logits, torch.zeros_like(logits))
    safe_target = torch.where(valid_mask.bool(), target, torch.zeros_like(target))
    element = torch.nn.functional.binary_cross_entropy_with_logits(safe_logits, safe_target, reduction="none")
    return (element.to(torch.float32) * valid_mask.to(torch.float32)).sum() / count


def validate_manifest(frame, config):
    if frame.window_id.duplicated().any():
        raise ValueError("Duplicate window_id in frozen manifests.")
    for row in frame.itertuples():
        if (row.sampling_rate_hz, row.component_order, row.window_samples, row.window_start_sample) != (100, "ZNE", 6000, 0):
            raise ValueError(f"Unexpected native format: {row.window_id}")
        if row.kind == "earthquake":
            if row.p_status != "manual" or not row.event_group_key:
                raise ValueError(f"Unconfirmed manual P: {row.window_id}")
            try:
                p = float(row.p_sample_in_window)
            except (ValueError, TypeError):
                raise ValueError(f"Earthquake without valid P: {row.window_id}") from None
            if not np.isfinite(p):
                raise ValueError(f"Earthquake without valid P: {row.window_id}")
            _validate_p(p, 0, 0, 6000, config)
            if abs(p - float(row.p_sample)) > 1e-9:
                raise ValueError("Native labels were shifted before preprocessing.")
        elif row.kind == "noise":
            if (str(row.p_sample_in_window) not in ("", "nan")
                    or str(row.p_sample) not in ("", "nan") or row.trace_category != "noise"):
                raise ValueError("Noise must be an explicitly catalogued noise window without a P label.")
        else:
            raise ValueError("Unknown label class must not silently become background.")


def _numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _validate_output(item, config):
    x, target, mask = item["waveform"], item["p_target"], item["valid_mask"]
    if x.dtype != np.float32 or x.shape != (3, 6000):
        raise AssertionError("Invalid model input shape/dtype")
    if target.dtype != np.float32 or target.shape != (1, 6000) or mask.shape != target.shape:
        raise AssertionError("Invalid label/mask shape")
    if not (np.isfinite(x).all() and np.isfinite(target).all() and np.isfinite(mask).all()):
        raise AssertionError("Nonfinite preprocessing output")
    if not (np.isin(mask, [0, 1]).all() and 0 <= target.min() <= target.max() <= 1):
        raise AssertionError("Invalid target or mask range")
    invalid = mask[0] == 0
    if np.any(x[:, invalid] != 0) or np.any(target[0, invalid] != 0):
        raise AssertionError("Padding contributed to waveform or target")
    p = item["p_sample"]
    if np.isfinite(p):
        if abs(int(target.argmax()) - p) > 0.500001:
            raise AssertionError("Gaussian label peak drift")
    elif np.any(target != 0):
        raise AssertionError("Background target is not zero")
    stats = item["stats"]
    if stats["valid_fraction"] < config.min_valid_fraction:
        raise AssertionError("Insufficient valid duration")
    if not stats["rms_floor_applied"] and abs(stats["joint_rms_after"] - 1) > 2e-6:
        raise AssertionError("Joint RMS normalization failed")
    if not config.bandpass and (stats["max_abs_component_mean_after"] > 2e-6 or stats["max_abs_component_slope_after"] > 2e-8):
        raise AssertionError("Mean/trend removal failed")


def _sample_indices(frame, seed):
    chosen = []
    for split in SPLITS:
        for kind in ("earthquake", "noise"):
            pool = frame[(frame.split == split) & (frame.kind == kind)]
            ids = sorted(pool.index, key=lambda i: rank_key(frame.at[i, "window_id"], seed, "step7_qa"))[:10]
            chosen.extend(ids)
    # 小数标签若确实存在，额外加入一个真实示例，不制造虚假的实测标签。
    p = pd.to_numeric(frame.p_sample_in_window, errors="coerce")
    fractional = frame.index[frame.kind.eq("earthquake") & p.notna() & ((p - p.round()).abs() > 1e-8)]
    if len(fractional):
        chosen.append(int(fractional[0]))
    return set(chosen)


def _source_files():
    folder = Path(__file__).resolve().parent
    return {p.name: {"path": str(p), "sha256": sha256_file(p)} for p in
            [folder / "WAVEPRE.py", folder / "STEADWINDOW.py", folder / "WAVEPREVIEW.py"]}


def verify_prepared(output, full=True):
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete":
        raise ValueError("Preprocessing validation is incomplete.")
    split_dir = Path(summary["split_dir"])
    if sha256_file(split_dir / "summary.json") != summary["split_summary_sha256"]:
        raise ValueError("The parent split protocol changed.")
    for source in summary["implementation"].values():
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError("Preprocessing code changed; validate a new output protocol before training.")
    for relative, expected in summary["output_sha256"].items():
        if full or relative == "protocol.json":
            if sha256_file(output / relative) != expected:
                raise ValueError(f"Changed preprocessing artifact: {relative}")
    return summary


def load_dataset(output_dir=None, split="train", augment=None, seed=42):
    """推荐训练入口；固定第七步协议 + 原分组清单。数据与 label 按需生成。"""
    if __package__:
        from .STEADWINDOW import STEADWindowDataset
    else:
        from STEADWINDOW import STEADWindowDataset
    output = Path(output_dir) if output_dir is not None else ROOT / "artifacts" / "stead_wavepre" / "main"
    summary = verify_prepared(output, full=False)
    if split not in SPLITS:
        raise ValueError("split must be train/val/test")
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    config = PreprocessConfig(**protocol["config"])
    if augment is None:
        augment = split == "train"
    return STEADWindowDataset(Path(summary["split_dir"]) / f"{split}_windows.csv",
                              data_dir=summary["data_dir"], augment=augment, seed=seed,
                              max_shift_seconds=config.max_shift_seconds, config=config)


def _write_report(output, summary):
    totals = summary["totals"]
    table = "\n".join(f"| {s} | {c['records']:,} | {c['earthquake']:,} | {c['noise']:,} |" for s, c in summary["counts"].items())
    text = f"""# 实验 A：第七步验收报告

状态：全量自动验收完成；尚未训练科研模型。只处理已冻结的 STEAD 主实验，不涉及 Iquique/ETHZ。

| 集合 | 全部窗口 | 地震 | 背景 |
|---|---:|---:|---:|
{table}

## 已完成的操作

- 核对原划分 hash，重新读取每个原生波形并比对其内容 SHA256，原清单及事件归属不变。
- 显式核对 data_format 为 CW / ZNE / 100 Hz；6000 点已经是 60 s，不重复转置或重采样。
- 各窗口有效区去均值、去线性趋势，随后以三分量共同 RMS（下限 1e-8）归一化为 float32。保持既有仪器处理状态，原数据 unit=counts、instrument_response=not restituted；没有将 counts 冒称校正后的物理速度单位。
- Gaussian P 标签 sigma={summary['protocol']['config']['sigma_seconds']} s；中心保留小数样本精度。S 不生成第二个训练目标，状态保留作后续诊断。软标签宽度不解释为到时不确定性。
- 明确 kind=noise 的原数据背景才生成全零 P 目标。未标注 P 的地震不自动变成负样本。
- train 最多 ±5 s 非循环平移，移出部分丢弃、填充为零，P/S 同步平移；loss 仅统计 valid_mask 内采样点。P 的 ±3sigma 必须落在真实有效区。val/test 不增强、不按真值重新定位。
- 对全部 {totals['native_checked']:,} 个原生窗口和 {totals['train_augmented_checked']:,} 个训练 epoch=0 版本验证有限值、dtype、形状、归一化、标签峰位及有效掩码。

## 质控协议与结果

本轮硬失败 {totals['hard_failures']} 条；疑似异常提示 {totals['flagged_records']:,} 条。硬失败（非有限值、缺失/恒定整分量、无可用残差能量、错误形状或标签）会停止完成，不静默补样或重新划分。
连续零值或恒定段 >=0.2 s、连续极值平台 >=5 点仅作诊断提示。它们可能来自缺口、量化或削顶，也可能是合法背景；没有采集端状态证据，不能直接确诊。提示记录仍保留在固定集合，不根据测试波形外观调门槛或删困难样本。
详见 record_qc.csv.gz、qc_flagged_records.csv；本步骤不插值、不替换未知波形、不生成伪造缺口标签。

## 可视化验收

打开 index.html：选择样本与原生/训练平移版本；核对三分量原始和处理波形、P 初至、Gaussian 标签、mask；可切换全窗和 P 附近局部时间轴。
examples.npz 保存实际示例数组；sample_manifest.csv 保存来源。展示的是数据和参考标签，没有模型预测。overview.png 与 case_*.png 可直接查看。
随机示例每个 split 各 10 条地震、10 条背景；若存在真实小数 P 标签，另补一个示例。另补最多 12 条实际带质控提示记录以方便复核，selection_reason 区分固定随机选样与诊断补充。

## 本轮不做的内容

ETHZ 重采样及 Pg/Pn 合并、跨域统计、到时区间校准属于实验 B；SWT/多尺度/Mamba 特征属于模型步骤；自然 SNR 分层和加噪压力曲线属于后续鲁棒性步骤。
主流程不带通；可显式选择 --bandpass 或不同 --sigma-seconds 生成独立敏感性协议目录，仅可按 Train/Val 决策，不凭 Test 结果切换主协议。1–40 Hz 分支为四阶 Butterworth SOS、双向零相位、odd padding 27 点，不适合声称实时因果性。
本轮不拟合全局归一化统计、不移除仪器响应，也不把背景窗口的小时数直接当成已经人工审过的连续监测性能。

## 训练用法与边界

参见 STEADPRE/README_WAVEPRE.md。推荐 load_dataset(split='train') 与 masked_bce_with_logits；模型只输入 batch['waveform']。
清单、原波形及处理代码 hash 绑定在 summary.json。全部模型必须共享同一协议和增强配置；训练 seed/epoch 可变，split 不变。
原生 P 位置范围：{totals['p_min_seconds']:.3f}–{totals['p_max_seconds']:.3f} s。训练平移仅有限缓解位置先验，不等价于真实长记录中任意到时。归一化利用整个窗口，任务为离线拾取。
事件互斥仍允许台站跨集合。自动验收验证的是实现一致性和数值性质，不等价于地震学专家确认所有标签；背景也可能有漏标事件。
"""
    (output / "ACCEPTANCE.md").write_text(text, encoding="utf-8")


def run_preprocessing(split_dir, data_dir, output, config, seed):
    log("Verify frozen parent split and native STEAD format")
    parent = verify_saved(split_dir, data_dir)
    with h5py.File(data_dir / "waveforms.hdf5", "r") as handle:
        fmt = {k: v[()] for k, v in handle["data_format"].items()}
        fmt = {k: v.decode() if isinstance(v, bytes) else v.item() if isinstance(v, np.generic) else v for k, v in fmt.items()}
        if (fmt["dimension_order"], fmt["component_order"], fmt["sampling_rate"]) != ("CW", "ZNE", 100):
            raise ValueError("Native format disagrees with the frozen experiment A interface.")
    frame = pd.concat([pd.read_csv(split_dir / f"{s}_windows.csv", keep_default_na=False,
                                   low_memory=False, dtype={"source_id": str}) for s in SPLITS], ignore_index=True)
    validate_manifest(frame, config)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"experiment": "A", "config": asdict(config), "augmentation_seed_for_acceptance": seed,
                "acceptance_epoch": 0, "data_format": fmt, "resampling": "none; already 100 Hz",
                "normalization": "linear detrend of valid samples; joint ZNE RMS; no global fit",
                "gap_policy": "hard fail nonfinite/full constant; flag exact runs; no interpolation",
                "loss": "BCE with logits, sum over valid samples divided by valid sample count",
                "bandpass_details": {"enabled": config.bandpass, "type": "Butterworth", "order": 4,
                                     "frequency_hz": [1, 40], "implementation": "scipy.signal.sosfiltfilt", "padtype": "odd", "padlen": 27},
                "model_input": "waveform only; no reference P/S/SNR/IDs/mask channel",
                "test_policy": "native window, no augmentation, no test-guided parameter selection"}
    protocol_json = json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2)
    protocol_id = hashlib.sha256(protocol_json.encode()).hexdigest()
    (output / "protocol.json").write_text(protocol_json, encoding="utf-8")
    sample_ids = _sample_indices(frame, seed)
    samples, records, augment_records, hard_failures, flag_counts = [], [], [], [], Counter()
    extra_flag_samples = 0
    order = sorted(frame.index, key=lambda i: tuple(map(int, re.findall(r"\d+", frame.at[i, "trace_name"])[:2])))
    log(f"Full preprocessing validation: {len(frame):,} native windows, plus train epoch 0")
    with h5py.File(data_dir / "waveforms.hdf5", "r") as handle:
        for position, i in enumerate(order, 1):
            row = frame.loc[i]
            raw = read_waveform(handle, row.trace_name)
            actual_hash = hashlib.sha256(np.asarray(raw, dtype="<f4").tobytes()).hexdigest()
            if actual_hash != row.waveform_sha256:
                raise ValueError(f"Parent waveform changed: {row.window_id}")
            qc = waveform_qc(raw, config)
            record = {"window_id": row.window_id, "metadata_row": row.metadata_row, "split": row.split,
                      "kind": row.kind, "qc_flags": ";".join(qc["flags"]),
                      **{k: v for k, v in qc.items() if k not in ("flags", "hard_failures")}}
            if qc["hard_failures"]:
                hard_failures.append({**record, "reason": ";".join(qc["hard_failures"])})
                continue
            p = _numeric(row.p_sample_in_window) if row.kind == "earthquake" else np.nan
            s = _numeric(row.s_sample_in_window)
            try:
                native = preprocess_waveform(raw, p, s, config=config)
                _validate_output(native, config)
                augmented = None
                if row.split == "train":
                    shift = draw_shift(p, seed, 0, int(row.metadata_row), config)
                    augmented = preprocess_waveform(raw, p, s, shift, config)
                    _validate_output(augmented, config)
                    augment_records.append({"window_id": row.window_id, "metadata_row": row.metadata_row,
                                            "split": "train", "seed": seed, "epoch": 0,
                                            "shift_samples": shift, "p_sample": augmented["p_sample"],
                                            "s_sample": augmented["s_sample"], **{k: v for k, v in augmented["stats"].items() if not isinstance(v, list)}})
            except (ValueError, AssertionError) as exc:
                hard_failures.append({**record, "reason": str(exc)})
                continue
            stats = native["stats"]
            record.update({k: v for k, v in stats.items() if not isinstance(v, list)})
            record.update({"p_time_seconds": p / 100, "s_time_seconds": s / 100,
                           "p_peak_sample": int(native["p_target"].argmax()) if np.isfinite(p) else "",
                           "label_peak_error_samples": abs(int(native["p_target"].argmax()) - p) if np.isfinite(p) else 0,
                           "protocol_id": protocol_id})
            records.append(record)
            flag_counts.update(qc["flags"])
            diagnostic_sample = bool(qc["flags"]) and i not in sample_ids and extra_flag_samples < 12
            if i in sample_ids or diagnostic_sample:
                if diagnostic_sample:
                    extra_flag_samples += 1
                samples.append({"window_id": row.window_id, "metadata_row": int(row.metadata_row),
                                "split": row.split, "kind": row.kind, "station_key": row.station_key,
                                "p_sample": p if np.isfinite(p) else None, "s_sample": s if np.isfinite(s) else None,
                                "p_status": row.p_status, "s_status": row.s_status, "raw": raw,
                                "native": native, "augmented": augmented, "qc": qc,
                                "selection_reason": "qc_flag" if diagnostic_sample else "frozen_qa_selection"})
            if position % 10000 == 0:
                log(f"Validated {position:,}/{len(frame):,}")
    if not records:
        save_csv(pd.DataFrame(hard_failures), output / "hard_failures.csv")
        raise ValueError("No record passed hard QC; see hard_failures.csv.")
    qc_frame = pd.DataFrame(records).sort_values("metadata_row")
    save_csv(qc_frame, output / "record_qc.csv.gz")
    save_csv(qc_frame[qc_frame.qc_flags.ne("")], output / "qc_flagged_records.csv")
    if hard_failures:
        save_csv(pd.DataFrame(hard_failures), output / "hard_failures.csv")
        raise ValueError(f"{len(hard_failures)} hard failures; inspect hard_failures.csv. No complete protocol written.")
    aug_frame = pd.DataFrame(augment_records).sort_values("metadata_row")
    save_csv(aug_frame, output / "train_augmentation_epoch0.csv.gz")
    for split in SPLITS:
        prepared = frame[frame.split.eq(split)].copy()
        prepared["preprocess_protocol_id"] = protocol_id
        prepared["label_sigma_seconds"] = config.sigma_seconds
        prepared["preprocessing_mode"] = "online_per_window"
        prepared["p_time_seconds"] = pd.to_numeric(prepared.p_sample_in_window, errors="coerce") / 100
        prepared["source_sampling_rate_hz"] = 100
        prepared["output_sampling_rate_hz"] = 100
        prepared = prepared.merge(qc_frame[["window_id", "qc_flags"]], on="window_id", how="left", validate="one_to_one")
        save_csv(prepared, output / f"{split}_prepared.csv")
    samples.sort(key=lambda x: (SPLITS.index(x["split"]), x["kind"], x["metadata_row"]))
    sample_meta, arrays = [], {}
    for i, sample in enumerate(samples):
        prefix = f"sample_{i:03d}"
        sample_meta.append({"array_prefix": prefix, **{k: sample[k] for k in ("window_id", "metadata_row", "split", "kind", "station_key", "p_sample", "s_sample", "p_status", "s_status", "selection_reason")},
                            "qc_flags": ";".join(sample["qc"]["flags"]),
                            "augmented_shift_samples": sample["augmented"]["shift_samples"] if sample["augmented"] is not None else ""})
        arrays[prefix + "_raw"] = sample["raw"]
        for mode in ("native", "augmented"):
            if sample[mode] is not None:
                for key in ("waveform", "p_target", "valid_mask"):
                    arrays[f"{prefix}_{mode}_{key}"] = sample[mode][key]
    np.savez_compressed(output / "examples.npz", **arrays)
    save_csv(pd.DataFrame(sample_meta), output / "sample_manifest.csv")
    (output / "sample_manifest.json").write_text(json.dumps(sample_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {s: {"records": int((frame.split == s).sum()),
                  "earthquake": int(((frame.split == s) & (frame.kind == "earthquake")).sum()),
                  "noise": int(((frame.split == s) & (frame.kind == "noise")).sum())} for s in SPLITS}
    p_seconds = qc_frame.loc[qc_frame.kind.eq("earthquake"), "p_time_seconds"]
    summary = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Experiment A / STEAD / preprocessing and P-label generation only",
               "split_dir": str(split_dir.resolve()), "data_dir": str(data_dir.resolve()),
               "split_summary_sha256": sha256_file(split_dir / "summary.json"),
               "counts": counts, "protocol": protocol, "protocol_id": protocol_id,
               "checks": {"parent_manifest_hashes": "pass", "all_selected_waveform_hashes": "pass",
                          "all_native_numerical_checks": "pass", "train_epoch0_checks": "pass",
                          "parent_split_unchanged": "pass", "hard_failures": 0},
               "totals": {"native_checked": len(qc_frame), "train_augmented_checked": len(aug_frame),
                          "hard_failures": 0, "flagged_records": int(qc_frame.qc_flags.ne("").sum()),
                          "p_min_seconds": float(p_seconds.min()), "p_max_seconds": float(p_seconds.max()),
                          "max_label_peak_error_samples": float(qc_frame.label_peak_error_samples.max()),
                          "augmented_min_valid_fraction": float(aug_frame.valid_fraction.min()),
                          "native_max_mean_abs": float(qc_frame.max_abs_component_mean_after.max()),
                          "native_max_slope_abs": float(qc_frame.max_abs_component_slope_after.max()),
                          "rms_floor_count": int(qc_frame.rms_floor_applied.sum()), "qa_samples": len(samples),
                          "qa_diagnostic_extras": extra_flag_samples},
               "flag_counts": dict(flag_counts), "versions": {"python": platform.python_version(),
                          "numpy": np.__version__, "pandas": pd.__version__, "h5py": h5py.__version__}}
    if __package__:
        from .WAVEPREVIEW import build_visual_report
    else:
        from WAVEPREVIEW import build_visual_report
    build_visual_report(output, samples, summary)
    _write_report(output, summary)
    for split in SPLITS:
        name = f"{split}_windows.csv"
        if sha256_file(split_dir / name) != parent["output_sha256"][name]:
            raise ValueError("Parent manifest changed during preprocessing; no completion marker written.")
    summary["implementation"] = _source_files()
    summary["output_sha256"] = {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Completed: {output.resolve()}")
    log(json.dumps(summary["totals"], ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "artifacts" / "stead_split" / "main")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "STEAD")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "stead_wavepre" / "main")
    parser.add_argument("--sigma-seconds", type=float, choices=[0.02, 0.05, 0.1], default=0.05)
    parser.add_argument("--bandpass", action="store_true", help="Independent 1-40 Hz sensitivity protocol; use a separate output.")
    parser.add_argument("--seed", type=int, default=42, help="Training augmentation/QA seed; never changes split.")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    config = PreprocessConfig(sigma_seconds=args.sigma_seconds, bandpass=args.bandpass)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.verify or (args.output / "summary.json").exists():
        summary = verify_prepared(args.output)
        saved = summary["protocol"]
        if saved["config"] != asdict(config) or saved["augmentation_seed_for_acceptance"] != args.seed:
            raise ValueError("Existing preprocessing protocol differs; choose a new --output.")
        verify_saved(Path(summary["split_dir"]), Path(summary["data_dir"]))
        log("Frozen preprocessing artifacts verified; no records or parameters changed.")
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Incomplete/nonempty output directory; inspect before using a fresh --output.")
    run_preprocessing(args.split_dir, args.data_dir, args.output, config, args.seed)


if __name__ == "__main__":
    main()
