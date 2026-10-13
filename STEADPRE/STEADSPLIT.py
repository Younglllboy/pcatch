"""STEAD 主实验：审计候选 -> 关联分组 -> 事件抽样 -> 分组划分 -> 窗口清单。

直接运行默认生成 main（最多 50,000 地震 + 12,500 噪声），不复制 HDF5。
python STEADPRE/STEADSPLIT.py --profile debug
python STEADPRE/STEADSPLIT.py --profile main
python STEADPRE/STEADSPLIT.py --profile expanded
python STEADPRE/STEADSPLIT.py --verify

依赖已有 STEADAUDIT.py 的完整审计结果。旧 split 仅留作溯源，绝不用于划分。
输出已存在时只复核，避免训练过程中无意重抽测试集。改变协议请指定新 --output。
训练读取类见 STEADWINDOW.py；本脚本不训练模型、不从文件名推测事件。
"""

import argparse
import hashlib
import itertools
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {"debug": (2000, 500), "main": (50000, 12500), "expanded": (100000, 25000)}
SPLITS = ("train", "val", "test")
RATIOS = np.array([0.7, 0.1, 0.2])
REFERENCE_RE = re.compile(r"^(bucket\d+)\$(\d+),:3,:6000$")
COLUMNS = [
    "metadata_row", "trace_name", "trace_name_original", "trace_category", "source_id",
    "event_group_key", "noise_group_key", "station_key", "trace_start_time_utc",
    "source_origin_time_utc", "sampling_rate_hz", "component_order", "waveform_samples",
    "waveform_channels", "p_sample", "s_sample", "p_status", "s_status", "split",
    "main_p_candidate", "noise_candidate", "event_group_confirmed", "p_manual",
]
STRING_COLUMNS = {x: str for x in COLUMNS if x not in (
    "metadata_row", "sampling_rate_hz", "waveform_samples", "waveform_channels",
    "p_sample", "s_sample", "main_p_candidate", "noise_candidate",
    "event_group_confirmed", "p_manual",
)}


def log(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_key(value, seed, namespace):
    return hashlib.blake2b(f"{seed}|{namespace}|{value}".encode(), digest_size=16).hexdigest()


def read_csv(path, **kwargs):
    # 保留合法网络代码 NA，source_id 保持原始字符串（包括前导零）。
    return pd.read_csv(path, keep_default_na=False, low_memory=False, **kwargs)


def file_identity(path, digest=False):
    stat = path.stat()
    result = {"path": str(path.resolve()), "size_bytes": stat.st_size,
              "mtime_ns": stat.st_mtime_ns}
    if digest:
        result["sha256"] = sha256_file(path)
    return result


def validate_audit(data_dir, audit_dir):
    summary = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8"))
    identities = {}
    for name in ("metadata.csv", "waveforms.hdf5"):
        path = data_dir / name
        identity = file_identity(path, digest=name.endswith("csv"))
        expected = summary["files"][name]
        if identity["size_bytes"] != expected["size_bytes"]:
            raise ValueError(f"{name} changed since audit; rerun STEADAUDIT.py.")
        if name.endswith("csv"):
            if identity["sha256"] != expected["sha256"]:
                raise ValueError("metadata hash differs from audit; rerun STEADAUDIT.py.")
        else:
            recorded = datetime.fromisoformat(expected["mtime_utc"]).timestamp()
            if abs(path.stat().st_mtime - recorded) > 0.00001:
                raise ValueError("HDF5 modification time differs from audit; rerun audit.")
        identities[name] = identity
    with h5py.File(data_dir / "waveforms.hdf5", "r") as h:
        values = {k: v[()] for k, v in h["data_format"].items()}
        values = {k: v.decode() if isinstance(v, bytes) else v for k, v in values.items()}
        if (values["dimension_order"], values["component_order"], values["sampling_rate"]) != ("CW", "ZNE", 100):
            raise ValueError("This protocol requires native CW/ZNE/100 Hz data.")
    identities["audit_summary"] = file_identity(audit_dir / "summary.json", True)
    for name in ("main_p_candidates.csv.gz", "noise_candidates.csv.gz"):
        identities[name] = file_identity(audit_dir / name, True)
    return summary, identities


def load_candidates(audit_dir):
    frames = []
    for name, flag, kind in (("main_p_candidates", "main_p_candidate", "earthquake"),
                              ("noise_candidates", "noise_candidate", "noise")):
        d = read_csv(audit_dir / f"{name}.csv.gz", usecols=COLUMNS, dtype=STRING_COLUMNS)
        if not d[flag].eq(True).all():
            raise ValueError(f"Invalid eligibility flag in {name}.")
        d["kind"] = kind
        frames.append(d)
    d = pd.concat(frames, ignore_index=True).sort_values("metadata_row").reset_index(drop=True)
    if d.metadata_row.duplicated().any() or d.trace_name.duplicated().any():
        raise ValueError("Candidate manifest contains duplicate parent references.")
    if not (d.sampling_rate_hz.eq(100) & d.waveform_samples.eq(6000)
            & d.waveform_channels.eq(3) & d.component_order.eq("ZNE")).all():
        raise ValueError("Non-native shape/rate found; do not silently resample this protocol.")
    eq = d.kind.eq("earthquake")
    if not (d.loc[eq, "event_group_confirmed"].eq(True)
            & d.loc[eq, "p_manual"].eq(True) & d.loc[eq, "event_group_key"].ne("")).all():
        raise ValueError("Unconfirmed or nonmanual earthquake entered candidate pool.")
    if d.station_key.eq("").any() or d.loc[~eq, "noise_group_key"].eq("").any():
        raise ValueError("Missing station or noise date identity.")
    d.rename(columns={"split": "original_split"}, inplace=True)
    for column in ("p_sample", "s_sample"):
        d[column] = pd.to_numeric(d[column], errors="coerce")
    d["_start_ns"] = pd.to_datetime(d.trace_start_time_utc, utc=True, format="mixed").dt.as_unit("ns").astype("int64")
    d["_end_ns"] = d._start_ns + 60_000_000_000  # 半开区间 [start, start+60 s)。
    # Native P near an edge cannot support the full +/-3 sigma target (sigma=5 samples).
    eligible = ~eq | d.p_sample.between(15, 5984.999999)
    excluded = d.loc[~eligible, ["metadata_row", "trace_name", "event_group_key"]].copy()
    excluded["reason"] = "P_within_0.15s_of_native_boundary"
    d["sampling_eligible"] = eligible
    return d, excluded


class UnionFind:
    """关联组不是新事件；只表达这些原始记录必须同属一个 split。"""

    def __init__(self, n):
        self.parent = np.arange(n)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, a):
        a = int(a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = int(self.parent[a])
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return 0
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        return 1


def union_keys(uf, values, indices=None):
    first, links = {}, 0
    if indices is None:
        indices = range(len(values))
    for i, key in zip(indices, values):
        if key == "" or pd.isna(key):
            continue
        if key in first:
            links += uf.union(i, first[key])
        else:
            first[key] = i
    return links


def overlap_links(d):
    """线性扫描连通区间，覆盖包含、链式重叠、跨午夜；相接但无交集不连接。"""
    ordered = d.sort_values(["station_key", "_start_ns", "metadata_row"])
    station, leader, last_end = None, None, None
    for i, key, start, end in zip(ordered.index, ordered.station_key, ordered._start_ns, ordered._end_ns):
        if key != station or start >= last_end:
            station, leader, last_end = key, i, end
        else:
            yield leader, i
            if end > last_end:
                leader, last_end = i, end


def metadata_groups(d):
    uf = UnionFind(len(d))
    counts = {}
    for key in ("event_group_key", "noise_group_key", "trace_name", "trace_name_original"):
        counts[key] = union_keys(uf, d[key])
    # 一条噪声若跨午夜，则归属其触及的两个 UTC 日期，避免午夜拆组。
    day_nodes = {}
    midnight_links = 0
    for i in d.index[d.kind.eq("noise")]:
        for day in {int(d.at[i, "_start_ns"]) // 86_400_000_000_000,
                    (int(d.at[i, "_end_ns"]) - 1) // 86_400_000_000_000}:
            key = (d.at[i, "station_key"], day)
            if key in day_nodes:
                midnight_links += uf.union(i, day_nodes[key])
            else:
                day_nodes[key] = i
    counts["noise_dates_including_midnight"] = midnight_links
    counts["same_station_time_overlap"] = sum(uf.union(a, b) for a, b in overlap_links(d))
    return uf, counts


def prefix_budget(group_sizes, budget, seed, namespace):
    """随机排列组，保留不超过上限的完整前缀；不截断末组或为凑数优选小组。"""
    keys = sorted(group_sizes, key=lambda key: rank_key(key, seed, namespace))
    selected, total = [], 0
    for key in keys:
        size = int(group_sizes[key])
        if total + size > budget:
            break
        selected.append(key)
        total += size
    if not selected:
        raise ValueError(f"First {namespace} group exceeds budget {budget}; increase budget.")
    return selected


def sample_records(d, eq_budget, noise_budget, station_cap, seed):
    eq = d.loc[d.kind.eq("earthquake") & d.sampling_eligible].copy()
    eq["_rank"] = [rank_key(x, seed, "station") for x in eq.trace_name]
    eq = eq.sort_values("_rank").drop_duplicates(["event_group_key", "station_key"])
    eq = eq.groupby("event_group_key", sort=False).head(station_cap)
    groups = prefix_budget(eq.groupby("event_group_key").size().to_dict(), eq_budget, seed, "events")
    eq = eq[eq.event_group_key.isin(groups)]
    noise = d[d.kind.eq("noise")]
    groups = prefix_budget(noise.groupby("noise_group_key").size().to_dict(), noise_budget, seed, "noise_days")
    # 抽样保留完整 station-day；更大的重叠连通组稍后仍会共同分配。
    return d.loc[sorted(list(eq.index) + list(noise[noise.noise_group_key.isin(groups)].index))].copy()


def read_waveform(handle, reference):
    match = REFERENCE_RE.fullmatch(reference)
    if not match:
        raise ValueError(f"Unsupported SeisBench reference: {reference!r}")
    bucket, row = match.groups()
    x = np.asarray(handle[f"data/{bucket}"][int(row), :3, :6000], dtype=np.float32)
    if x.shape != (3, 6000):
        raise ValueError(f"Unexpected waveform shape {x.shape}: {reference}")
    return x


def waveform_qc(selected, hdf5_path, uf):
    """逐条读选中波形；精确内容 SHA256 + 同台站近重复候选筛查。"""
    hashes, features, bad = {}, {}, []
    ordered = sorted(selected.index, key=lambda i: tuple(map(int, re.findall(r"\d+", selected.at[i, "trace_name"])[:2])))
    with h5py.File(hdf5_path, "r") as handle:
        for k, i in enumerate(ordered, 1):
            x = read_waveform(handle, selected.at[i, "trace_name"])
            if not np.isfinite(x).all():
                bad.append((i, "nonfinite_waveform"))
                continue
            centered = x.astype(np.float64) - x.mean(axis=1, keepdims=True, dtype=np.float64)
            if np.any(np.mean(centered ** 2, axis=1) <= 1e-20):
                bad.append((i, "constant_or_empty_component"))
                continue
            # 规范为 little-endian float32；不更改原始波形文件。
            hashes[i] = hashlib.sha256(np.asarray(x, dtype="<f4").tobytes()).hexdigest()
            feature = centered.reshape(3, 60, 100).mean(axis=2).ravel()
            norm = np.linalg.norm(feature)
            features[i] = (feature / norm if norm else feature).astype(np.float32)
            if k % 10000 == 0:
                log(f"Waveform QC {k:,}/{len(ordered):,}")
        selected["waveform_sha256"] = pd.Series(hashes)
        valid = selected.drop(index=[i for i, _ in bad]).copy()
        duplicate_groups = valid[valid.waveform_sha256.duplicated(False)].groupby("waveform_sha256")
        duplicate_rows = []
        for digest, group in duplicate_groups:
            indices = list(group.index)
            for i in indices[1:]:
                uf.union(indices[0], i)
            # 相同波形对应不同事件/类别时身份存在矛盾，全部隔离，而非伪造事件。
            identities = set(zip(group.kind, group.event_group_key))
            conflicting = len(identities) > 1
            remove = indices if conflicting else indices[1:]
            for i in remove:
                bad.append((i, "identical_waveform_conflicting_identity" if conflicting else "duplicate_waveform"))
            duplicate_rows.append({"waveform_sha256": digest, "records": len(indices),
                                   "identity_conflict": conflicting, "removed": len(remove)})
        valid = selected.drop(index=[i for i, _ in bad]).copy()
        pairs, candidates = [], 0
        # 平均降采样仅用于候选检索；最终阈值使用完整 3x6000 波形的去均值余弦。
        # 此检查不是穷尽的滞后互相关搜索，不能据此声称排除一切物理近重复。
        for _, group in valid.groupby("station_key", sort=True):
            ids = group.index.to_numpy()
            if len(ids) < 2:
                continue
            matrix = np.stack([features[i] for i in ids])
            for start in range(0, len(ids), 256):
                corr = np.abs(matrix[start:start + 256] @ matrix.T)
                a_rows, b_rows = np.where(corr >= 0.98)
                for a, b in zip(a_rows + start, b_rows):
                    if b <= a:
                        continue
                    i, j = int(ids[a]), int(ids[b])
                    # 已被事件/时间关系连接的对不会造成跨 split 泄漏，无需反复读取。
                    if uf.find(i) == uf.find(j):
                        continue
                    candidates += 1
                    x = read_waveform(handle, valid.at[i, "trace_name"]).astype(np.float64)
                    y = read_waveform(handle, valid.at[j, "trace_name"]).astype(np.float64)
                    x -= x.mean(axis=1, keepdims=True)
                    y -= y.mean(axis=1, keepdims=True)
                    similarity = abs(float(np.sum(x * y) / (np.linalg.norm(x) * np.linalg.norm(y))))
                    if similarity >= 0.995:
                        uf.union(i, j)
                        pairs.append({"metadata_row_a": int(valid.at[i, "metadata_row"]),
                                      "metadata_row_b": int(valid.at[j, "metadata_row"]),
                                      "station_key": valid.at[i, "station_key"], "abs_cosine": similarity})
    exclusions = selected.loc[[i for i, _ in bad], ["metadata_row", "trace_name", "event_group_key"]].copy()
    exclusions["reason"] = [reason for _, reason in bad]
    return valid, exclusions, duplicate_rows, pairs, candidates


def component_keys(d, uf):
    roots = np.array([uf.find(i) for i in range(len(d))])
    minimum = pd.Series(d.metadata_row.to_numpy()).groupby(roots).min().to_dict()
    return pd.Series([f"G:{minimum[r]}" for r in roots], index=d.index)


def assign_splits(selected, seed):
    """按不可拆组的事件数平衡 70/10/20，另平衡噪声记录数；不按地震 trace 数。"""
    groups = selected.groupby("leakage_group")
    weights = pd.DataFrame({
        "events": groups.event_group_key.agg(lambda s: len(set(s) - {""})),
        "noise_records": groups.kind.agg(lambda s: int(s.eq("noise").sum())),
    })
    total = weights.sum().to_numpy(dtype=float)
    targets = RATIOS[:, None] * total[None, :]
    counts = np.zeros_like(targets)
    # 大关联组优先，种子仅控制同权重时次序；最小化归一化配额偏差。
    order = sorted(weights.index, key=lambda g: (-weights.at[g, "events"],
                   -weights.at[g, "noise_records"], rank_key(g, seed, "assignment")))
    assignment = {}
    for g in order:
        w = weights.loc[g].to_numpy(dtype=float)
        costs = (((counts + w - targets) ** 2 - (counts - targets) ** 2) / np.maximum(targets, 1)).sum(axis=1)
        chosen = int(np.argmin(costs))
        assignment[g] = SPLITS[chosen]
        counts[chosen] += w
    selected = selected.copy()
    selected["split"] = selected.leakage_group.map(assignment)
    return selected, assignment


def leakage_checks(d):
    checks = []
    keys = ("event_group_key", "noise_group_key", "metadata_row", "trace_name",
            "trace_name_original", "leakage_group", "waveform_sha256")
    for a, b in itertools.combinations(SPLITS, 2):
        for key in keys:
            left = set(d.loc[d.split.eq(a), key]) - {""}
            right = set(d.loc[d.split.eq(b), key]) - {""}
            checks.append({"check": key, "split_a": a, "split_b": b, "violations": len(left & right)})
    overlap_count = sum(d.at[a, "split"] != d.at[b, "split"] for a, b in overlap_links(d))
    checks.append({"check": "same_station_overlapping_intervals", "split_a": "all", "split_b": "all", "violations": overlap_count})
    touched_days = {}
    day_conflicts = set()
    for row in d[d.kind.eq("noise")].itertuples():
        start = pd.Timestamp(row.trace_start_time_utc).value
        for day in {start // 86_400_000_000_000, (start + 60_000_000_000 - 1) // 86_400_000_000_000}:
            key = (row.station_key, day)
            if key in touched_days and touched_days[key] != row.split:
                day_conflicts.add(key)
            touched_days[key] = row.split
    checks.append({"check": "noise_touched_UTC_dates", "split_a": "all", "split_b": "all", "violations": len(day_conflicts)})
    if any(x["violations"] for x in checks):
        raise AssertionError(f"Cross-split leakage: {[x for x in checks if x['violations']]}")
    if not set(d.split) == set(SPLITS):
        raise ValueError("Empty split; increase sample budget.")
    return checks


def make_windows(d):
    """只在 split 已确定之后生成索引；原始 60 s 不按真值重新居中。"""
    windows = d.copy()
    windows["parent_record_id"] = "STEAD:row:" + windows.metadata_row.astype(str)
    windows["window_id"] = windows.parent_record_id + ":0:6000"
    windows["window_start_sample"] = 0
    windows["window_samples"] = 6000
    windows["valid_start_sample"] = 0
    windows["valid_stop_sample"] = 6000
    windows["p_sample_in_window"] = windows.p_sample
    windows["s_sample_in_window"] = windows.s_sample
    windows["p_target_sigma_samples"] = 5.0
    return windows


def save_csv(d, path):
    # mtime=0 让 gzip 文件本身也可复现；不用时间戳污染清单 hash。
    compression = {"method": "gzip", "mtime": 0} if str(path).endswith(".gz") else None
    d.to_csv(path, index=False, encoding="utf-8", compression=compression, lineterminator="\n")


def qa_figures(windows, hdf5_path, output, seed, per_class):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen = []
    for kind in ("earthquake", "noise"):
        pool = windows[windows.kind.eq(kind)]
        ids = sorted(pool.index, key=lambda i: rank_key(pool.at[i, "window_id"], seed, "qa"))[:per_class]
        chosen.extend(ids)
    qa = windows.loc[chosen].copy()
    save_csv(qa, output / "qa_sample.csv")
    (output / "qa").mkdir()
    with h5py.File(hdf5_path, "r") as handle:
        for page, start in enumerate(range(0, len(qa), 6), 1):
            rows = qa.iloc[start:start + 6]
            fig, axes = plt.subplots(len(rows), 3, figsize=(15, 2 * len(rows)), squeeze=False)
            for r, row in enumerate(rows.itertuples()):
                x = read_waveform(handle, row.trace_name)
                for c, component in enumerate("ZNE"):
                    ax = axes[r, c]
                    ax.plot(np.arange(6000) / 100, x[c], lw=0.45, color="#334155")
                    if row.kind == "earthquake":
                        ax.axvline(row.p_sample / 100, color="#dc2626", ls="--", lw=1, label="Manual P")
                    ax.set_title(f"row {row.metadata_row} | {row.split} | {row.station_key} | {component}", fontsize=8)
                    ax.set_xlim(0, 60)
                    ax.set_xlabel("Time from native start (s)", fontsize=8)
            fig.tight_layout()
            fig.savefig(output / "qa" / f"waveforms_{page:02d}.png", dpi=130)
            plt.close(fig)


def verify_saved(output, data_dir):
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    for name, identity in summary["inputs"].items():
        path = data_dir / name if name in ("metadata.csv", "waveforms.hdf5") else Path(identity["path"])
        stat = path.stat()
        if stat.st_size != identity["size_bytes"]:
            raise ValueError(f"Changed input size: {path}")
        if "sha256" in identity and sha256_file(path) != identity["sha256"]:
            raise ValueError(f"Changed input hash: {path}")
        if name == "waveforms.hdf5" and stat.st_mtime_ns != identity["mtime_ns"]:
            raise ValueError("HDF5 changed after split.")
    for relative, expected in summary["output_sha256"].items():
        if sha256_file(output / relative) != expected:
            raise ValueError(f"Frozen output changed: {relative}")
    frames = [read_csv(output / f"{s}_windows.csv", dtype=STRING_COLUMNS) for s in SPLITS]
    d = pd.concat(frames, ignore_index=True)
    d["_start_ns"] = pd.to_datetime(d.trace_start_time_utc, utc=True, format="mixed").dt.as_unit("ns").astype("int64")
    d["_end_ns"] = d._start_ns + 60_000_000_000
    leakage_checks(d)
    if not (d.window_start_sample.eq(0) & d.window_samples.eq(6000)).all():
        raise AssertionError("Frozen evaluation windows changed.")
    log(f"Verified frozen split: {len(d):,} windows; all pairwise leakage checks passed.")
    return summary


def write_report(output, summary):
    rows = summary["split_counts"]
    table = "\n".join(f"| {s} | {rows[s]['events']:,} | {rows[s]['earthquake']:,} | {rows[s]['noise']:,} | {rows[s]['total']:,} |" for s in SPLITS)
    report = f"""# STEAD 主实验固定划分

生成时间（UTC）：{summary['created_utc']}；协议：{summary['profile']}；划分种子：{summary['seed']}。

## 结果与使用

| 集合 | 地震事件 | 地震窗口 | 噪声窗口 | 总窗口 |
|---|---:|---:|---:|---:|
{table}

训练使用 train_windows.csv，选阈值/检查点使用 val_windows.csv，最终一次性评估使用 test_windows.csv。
同一套清单供全部模型和训练随机种子使用。训练种子变化不能重新划分数据。
三个 CSV 索引原 HDF5，没有另存波形。每条原始记录只生成一个固定 60 s 窗口，ZNE / 100 Hz / (3,6000)。
PyTorch 数据读取与带掩码增强见 STEADPRE/STEADWINDOW.py 和 STEADPRE/README_SPLIT.md。

## 协议和防泄漏

1. 只接受审计确认的原始 source_id、台站、时间、格式与人工 P 标签。未确认事件、自动 P 标签不纳入本主分析。
2. 对完整候选池先连接同事件、同噪声台站-UTC 日期（含跨午夜）、相同原始记录、同台站重叠时间段。
   时间是 [start,start+60 s)；相接但不重叠的地震记录不因边界相等而合并。
3. 以固定种子随机排列事件；每事件最多 {summary['station_cap']} 个不同台站，每站随机保留一条。完整事件前缀满足记录预算，不为凑整数拆分末事件。
   噪声随机抽取完整台站-日期组。抽样不是按 SNR 或测试结果挑样；仍需报告抽样分布。
4. 对所有入选记录读取完整波形，排除非有限值、恒定分量；计算原始 float32 内容 SHA256。
   同波形跨事件/类别身份冲突全部隔离，其余完全重复只留一条。
5. 同台站用 180 维均值特征余弦 >=0.98 检索候选，完整波形去分量均值后的绝对余弦 >=0.995 时连接为同一划分组。
   这是保守关联，不改写 source_id，也不声称这些记录属于同一物理事件。
6. 最后按不可拆的关联组分配 train/val/test，以事件数 70/10/20 为主，兼顾噪声记录数；不是强制窗口比例。
   关联约束可能导致比例略有偏差。实际事件比例：{summary['event_fractions']}。
7. 先保存 earthquake_manifest.csv、noise_manifest.csv，再生成窗口清单。事件衍生的预 P 噪声、裁剪和平移必须继承父记录 split 和 leakage_group。
   本版不生成预 P 噪声、不重新居中验证/测试窗口，不沿用 metadata 的原 split。
8. leakage_checks.csv 对 train-val、train-test、val-test 分别检查事件、父记录、噪声日期、波形 hash、关联组，以及同台站重叠。全部通过才写完成标记 summary.json。

## 已知边界

- 这是新事件泛化划分；同一台站可同时出现在多个集合，不可称为未见台站测试或严格时间外推。
- 主分析只含人工 P；S 状态单独保留，不要求 S 人工标签，也不将自动 S 当人工参考。
- UTC 来自现有 metadata 的解释，未回溯原始采集档案；缺少更细连续原文件标识时，以台站-日期和时间重叠代理约束噪声。
- 波形完整扫描覆盖本次入选记录；HDF5 整文件仅核对大小和修改时间，未对 91 GB 文件计算整文件 hash。
- 近重复筛查为同台站、零时移的候选方法，不是全库穷尽的滞后互相关或跨数据集去重，不能承诺所有未知重复均已排除。
- 该划分不包含 Iquique/ETHZ，不能作为其跨区域互斥证明；该迁移主线仍须独立初始化训练。
- 自动 P 与不确定事件若另做敏感性分析，应单独声明并重新审计其与主清单的事件/时间/重复关系；不能直接随机追加。
- 预处理只对单窗口去趋势并按三分量联合 RMS 归一化，不拟合全数据统计。Gaussian sigma=0.05 s；噪声标签为全零只是数据集背景定义，不保证不存在漏标事件。

## QC 与溯源

完整候选池记录：{summary['candidate_rows']:,}；本轮波形 QC 排除：{summary['waveform_excluded_rows']:,}。
边界 P 排除数量：{summary['edge_excluded_rows']:,}；精确重复组：{summary['exact_duplicate_groups']}；近重复连接：{summary['near_duplicate_links']}。
原始审计中未确认事件记录 {summary['audit_uncertain_event_records']:,} 未进入主候选池。
qa_sample.csv 与 qa/ 提供随机波形检查页（各类最多 {summary['qa_per_class']} 条）；这是待人工复核清单，不等于已完成人工质量认证。
eligible_record_groups.csv.gz 保存完整候选池关联分组及所继承 split；unused 表示未被本实验分配，绝不能直接当作可自由加入的新训练数据。
exclusions.csv 只记录本轮新增排除，前置审计的其他排除详见 artifacts/stead_audit。
输入路径、metadata/audit SHA256、各 CSV SHA256、代码版本、软件版本和阈值见 summary.json。
"""
    (output / "SPLIT_REPORT.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "STEAD")
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "artifacts" / "stead_audit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="main")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--station-cap", type=int, default=6)
    parser.add_argument("--qa-per-class", type=int, default=30)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.seed < 0 or args.station_cap < 1 or args.qa_per_class < 0:
        parser.error("seed/qa-per-class must be nonnegative; station-cap must be positive")
    output = args.output or ROOT / "artifacts" / "stead_split" / args.profile
    if args.verify or (output / "summary.json").exists():
        saved = verify_saved(output, args.data_dir)
        if saved["seed"] != args.seed or saved["profile"] != args.profile or saved["station_cap"] != args.station_cap:
            raise ValueError("Existing protocol differs; use another --output, do not overwrite frozen split.")
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Incomplete/nonempty output: {output}. Inspect it and use a new --output.")
    log("Verify audit provenance and load eligible metadata")
    audit, identities = validate_audit(args.data_dir, args.audit_dir)
    d, edge_excluded = load_candidates(args.audit_dir)
    log(f"Build event/noise/time groups over all {len(d):,} eligible candidate records")
    uf, grouping_counts = metadata_groups(d)
    selected = sample_records(d, *PROFILES[args.profile], args.station_cap, args.seed)
    log(f"Read complete waveforms for {len(selected):,} sampled records")
    selected, wave_excluded, duplicates, near_pairs, near_candidates = waveform_qc(selected, args.data_dir / "waveforms.hdf5", uf)
    d["leakage_group"] = component_keys(d, uf)
    selected["leakage_group"] = d.loc[selected.index, "leakage_group"]
    selected, assignment = assign_splits(selected, args.seed)
    checks = leakage_checks(selected)
    log("All pairwise leakage checks passed; write grouped manifests, then native windows")
    output.mkdir(parents=True, exist_ok=True)
    public = selected.drop(columns=[c for c in selected if c.startswith("_") or c == "sampling_eligible"])
    public = public.sort_values("metadata_row")
    for kind in ("earthquake", "noise"):
        save_csv(public[public.kind.eq(kind)], output / f"{kind}_manifest.csv")
    d["assigned_split"] = d.leakage_group.map(assignment).fillna("unused")
    save_csv(d[["metadata_row", "event_group_key", "noise_group_key", "leakage_group", "assigned_split"]], output / "eligible_record_groups.csv.gz")
    windows = make_windows(public)
    for split in SPLITS:
        save_csv(windows[windows.split.eq(split)], output / f"{split}_windows.csv")
    save_csv(pd.concat([edge_excluded, wave_excluded], ignore_index=True), output / "exclusions.csv")
    save_csv(pd.DataFrame(duplicates, columns=["waveform_sha256", "records", "identity_conflict", "removed"]), output / "exact_duplicates.csv")
    save_csv(pd.DataFrame(near_pairs, columns=["metadata_row_a", "metadata_row_b", "station_key", "abs_cosine"]), output / "near_duplicates.csv")
    save_csv(pd.DataFrame(checks), output / "leakage_checks.csv")
    events = public[public.kind.eq("earthquake")].groupby("event_group_key").agg(
        source_id=("source_id", "first"), split=("split", "first"), leakage_group=("leakage_group", "first"),
        traces=("metadata_row", "size"), stations=("station_key", "nunique"), origin_time=("source_origin_time_utc", "first"))
    save_csv(events.reset_index(), output / "event_assignments.csv")
    station_presence = public.groupby(["station_key", "split"]).size().unstack(fill_value=0).reset_index()
    save_csv(station_presence, output / "station_split_presence.csv")
    p_positions = public[public.kind.eq("earthquake")].copy()
    p_positions["p_seconds_bin"] = pd.cut(p_positions.p_sample / 100, bins=[0, 5, 10, 20, 30, 45, 60], right=False).astype(str)
    save_csv(p_positions.groupby(["split", "p_seconds_bin"]).size().reset_index(name="records"), output / "p_position_distribution.csv")
    qa_figures(windows, args.data_dir / "waveforms.hdf5", output, args.seed, args.qa_per_class)
    counts = {}
    for split in SPLITS:
        part = public[public.split.eq(split)]
        counts[split] = {"events": int(part.loc[part.kind.eq("earthquake"), "event_group_key"].nunique()),
                         "earthquake": int(part.kind.eq("earthquake").sum()), "noise": int(part.kind.eq("noise").sum()), "total": len(part)}
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "profile": args.profile,
               "seed": args.seed, "station_cap": args.station_cap, "budgets": PROFILES[args.profile],
               "qa_per_class": args.qa_per_class, "ratios": RATIOS.tolist(), "split_counts": counts,
               "event_fractions": {s: round(counts[s]["events"] / len(events), 6) for s in SPLITS},
               "candidate_rows": len(d), "edge_excluded_rows": len(edge_excluded),
               "waveform_excluded_rows": len(wave_excluded), "exact_duplicate_groups": len(duplicates),
               "near_duplicate_links": len(near_pairs), "near_candidates_full_comparison": near_candidates,
               "grouping_union_counts": grouping_counts, "audit_uncertain_event_records": audit["uncertain_event_records"],
               "inputs": identities, "script_sha256": sha256_file(Path(__file__)),
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "h5py": h5py.__version__},
               "near_duplicate_parameters": {"sketch_abs_cosine": 0.98, "waveform_abs_cosine": 0.995, "lag_samples": 0, "same_station_only": True}}
    write_report(output, summary)
    summary["output_sha256"] = {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}
    # 最后才写完成标记；中途失败目录不会被视为可训练的完整数据集。
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(counts, ensure_ascii=False))
    log(f"Completed: {output.resolve()}")


if __name__ == "__main__":
    main()
