"""STEAD 元数据审计：缺失率、唯一值、事件/台站一致性与主分析候选清单。

运行：python STEADPRE/STEADAUDIT.py
默认只读 datasets/STEAD，结果写入 artifacts/stead_audit。
不修改数据、不下载文件、不按文件名推断事件、不自动修复/拆分/合并 source_id。
主分析候选并非已经完成事件互斥划分；不能直接沿用原始 split。
依赖：pandas、numpy、h5py、matplotlib（使用项目 dizhenczsq 环境）。
"""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")  # 批量审计保存图片，不弹窗或依赖交互式后端。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "STEAD"
OUTPUT_DIR = ROOT / "artifacts" / "stead_audit"
REQUIRED = [
    "source_id", "source_origin_time", "source_latitude_deg", "source_longitude_deg",
    "source_depth_km", "station_network_code", "station_code", "trace_channel",
    "trace_start_time", "trace_category", "trace_name", "trace_p_arrival_sample",
    "trace_s_arrival_sample", "trace_p_status", "trace_s_status",
]
COORDINATES = ["source_latitude_deg", "source_longitude_deg", "source_depth_km"]


def read_metadata(path):
    # 禁用 pandas 的默认 NA 词表，避免把合法网络/台站标识 NA 当成空值。
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")
    for column in df:
        values = df[column].astype("string").str.strip()
        df[column] = values.mask(values.str.lower().isin(["", "nan", "null", "none"]))
    return df


def read_hdf5_layout(path):
    """检查全部数据块的结构，不将约 91 GB 波形加载到内存。"""
    with h5py.File(path, "r") as handle:
        if "data" not in handle or "data_format" not in handle:
            raise ValueError("Expected SeisBench data and data_format groups.")
        data_format = {}
        for key, dataset in handle["data_format"].items():
            value = dataset[()]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, np.generic):
                value = value.item()
            data_format[key] = value
        buckets = {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                   for name, value in handle["data"].items()}
    if data_format.get("dimension_order") != "CW":
        raise ValueError("This audit expects stored dimension_order=CW; inspect layout before adapting.")
    return data_format, buckets


def numeric(series):
    return pd.to_numeric(series, errors="coerce").astype(float)


def timestamps(series):
    # 无时区字符串按 UTC 解释；报告保留这一假设，不从文件名恢复时间。
    return pd.to_datetime(series, format="mixed", errors="coerce", utc=True)


def audit_metadata(df, data_format, buckets):
    """纯元数据审计。confirmed 仅表示目录键通过本次一致性检查。"""
    a = pd.DataFrame(index=df.index)
    a["metadata_row"] = np.arange(len(df))
    eq = df.trace_category.eq("earthquake_local").fillna(False)
    noise = df.trace_category.eq("noise").fillna(False)
    a["is_earthquake"] = eq
    a["is_noise"] = noise
    a["source_id"] = df.source_id
    a["station_key"] = (df.station_network_code + "." + df.station_code).where(
        df.station_network_code.notna() & df.station_code.notna()
    )
    a["trace_start_time_utc"] = timestamps(df.trace_start_time)
    a["source_origin_time_utc"] = timestamps(df.source_origin_time)
    a["sampling_rate_hz"] = numeric(df.get("trace_sampling_rate_hz", pd.Series(np.nan, index=df.index)))
    global_rate = pd.to_numeric(data_format.get("sampling_rate"), errors="coerce")
    # 仅填补实际缺失值；格式错误的显式值不应被全局默认值掩盖。
    row_rate_missing = df.get("trace_sampling_rate_hz", pd.Series(pd.NA, index=df.index)).isna()
    a.loc[row_rate_missing, "sampling_rate_hz"] = global_rate
    row_order = df.get("trace_component_order", pd.Series(pd.NA, index=df.index, dtype="string"))
    a["component_order"] = row_order.fillna(data_format.get("component_order", pd.NA))
    a["sampling_rate_source"] = np.where(row_rate_missing, "HDF5:data_format/sampling_rate", "metadata:trace_sampling_rate_hz")
    a["component_order_source"] = np.where(row_order.isna(), "HDF5:data_format/component_order", "metadata:trace_component_order")

    parts = df.trace_name.str.extract(r"^(?P<bucket>[^$]+)\$(?P<row>\d+),:(?P<channels>\d+),:(?P<samples>\d+)$")
    shapes = {k: v["shape"] for k, v in buckets.items() if len(v["shape"]) == 3}
    a["waveform_samples"] = numeric(parts.samples)
    a["waveform_channels"] = numeric(parts.channels)
    bucket_row = numeric(parts.row)
    capacity = parts.bucket.map({k: v[0] for k, v in shapes.items()})
    channels = parts.bucket.map({k: v[1] for k, v in shapes.items()})
    samples = parts.bucket.map({k: v[2] for k, v in shapes.items()})
    a["waveform_reference_valid"] = (
        parts.notna().all(axis=1) & bucket_row.ge(0) & bucket_row.lt(capacity)
        & a.waveform_channels.eq(3) & a.waveform_channels.le(channels)
        & a.waveform_samples.gt(0) & a.waveform_samples.le(samples)
    )
    a["duplicate_trace_reference"] = df.trace_name.notna() & df.trace_name.duplicated(keep=False)
    original = df.get("trace_name_original", pd.Series(pd.NA, index=df.index, dtype="string"))
    a["duplicate_original_name"] = original.notna() & original.duplicated(keep=False)
    a["sampling_rate_valid"] = np.isfinite(a.sampling_rate_hz) & a.sampling_rate_hz.gt(0)
    a["component_order_valid"] = a.component_order.isin(["ZNE", "ZEN", "NZE", "NEZ", "EZN", "ENZ"])
    a["station_valid"] = a.station_key.notna()
    a["trace_start_time_valid"] = a.trace_start_time_utc.notna()

    for phase in ("p", "s"):
        values = numeric(df[f"trace_{phase}_arrival_sample"])
        a[f"{phase}_sample"] = values
        a[f"{phase}_label_valid"] = np.isfinite(values) & values.ge(0) & values.lt(a.waveform_samples)
        a[f"{phase}_label_fractional"] = np.isfinite(values) & values.mod(1).abs().gt(1e-8)
        a[f"{phase}_status"] = df[f"trace_{phase}_status"].str.lower()
        a[f"{phase}_manual"] = a[f"{phase}_status"].eq("manual").fillna(False)
    a["s_not_after_p"] = a.p_label_valid & a.s_label_valid & a.s_sample.le(a.p_sample)
    relative_p = (a.trace_start_time_utc - a.source_origin_time_utc).dt.total_seconds() + a.p_sample / a.sampling_rate_hz
    a["p_before_origin"] = eq & a.p_label_valid & relative_p.lt(0)

    event_data = pd.DataFrame({"source_id": df.source_id, "origin": a.source_origin_time_utc})
    for col in COORDINATES:
        event_data[col] = numeric(df[col])
    event_data["station_key"] = a.station_key
    event_data["split"] = df.get("split", pd.Series(pd.NA, index=df.index, dtype="string"))
    catalog = event_data.loc[eq & df.source_id.notna()].copy()
    group = catalog.groupby("source_id", sort=True)
    events = group.size().rename("trace_count").to_frame()
    events["station_count"] = group.station_key.nunique()
    events["missing_station_rows"] = catalog.station_key.isna().groupby(catalog.source_id).sum()
    fields = ["origin"] + COORDINATES
    for col in fields:
        counts = group[col].nunique()
        events[f"{col}_nunique"] = counts
        lo, hi = group[col].min(), group[col].max()
        events[f"{col}_min"] = lo
        events[f"{col}_max"] = hi
        events[f"{col}_span"] = (hi-lo).dt.total_seconds() if col == "origin" else hi-lo
    events["catalog_conflict"] = events[[f"{col}_nunique" for col in fields]].gt(1).any(axis=1)

    location_valid = (
        event_data.source_latitude_deg.between(-90, 90)
        & event_data.source_longitude_deg.between(-180, 180)
        & np.isfinite(event_data.source_depth_km)
    )  # 负深度不直接判错：部分目录以海平面为参考。
    corroboration_valid = a.source_origin_time_utc.notna() & location_valid
    bad_ids = set(df.loc[eq & df.source_id.notna() & ~corroboration_valid, "source_id"])
    events["incomplete_or_invalid_catalog"] = events.index.isin(bad_ids)
    # 完全相同的发震时刻/位置/深度却有多个 ID：只列候选冲突，不自动合并。
    fingerprint_ok = catalog[fields].notna().all(axis=1)
    candidate_catalog = catalog.loc[fingerprint_ok]
    id_counts = candidate_catalog.groupby(fields, dropna=False).source_id.transform("nunique")
    aliases = candidate_catalog.loc[id_counts.gt(1), ["source_id"] + fields].drop_duplicates()
    events["cross_id_fingerprint_candidate"] = events.index.isin(aliases.source_id.unique())
    events["event_group_confirmed"] = ~(
        events.catalog_conflict | events.incomplete_or_invalid_catalog | events.cross_id_fingerprint_candidate
    )
    events["original_split_count"] = group["split"].nunique()
    events["original_splits"] = ""
    for split, members in catalog.dropna(subset=["split"]).groupby("split", sort=True):
        ids = members.source_id.unique()
        events.loc[ids, "original_splits"] += str(split) + "|"
    events["original_splits"] = events.original_splits.str.rstrip("|")
    a["event_group_confirmed"] = eq & df.source_id.map(events.event_group_confirmed).fillna(False).astype(bool)
    a["catalog_conflict"] = eq & df.source_id.isin(events.index[events.catalog_conflict])
    a["cross_id_fingerprint_candidate"] = eq & df.source_id.isin(events.index[events.cross_id_fingerprint_candidate])

    parent = pd.DataFrame({"station": a.station_key, "channel": df.trace_channel, "start": a.trace_start_time_utc})
    a["duplicate_station_channel_start"] = parent.notna().all(axis=1) & parent.duplicated(keep=False)
    common_valid = (
        a.station_valid & a.trace_start_time_valid & a.sampling_rate_valid
        & a.component_order_valid & a.waveform_reference_valid
        & ~a.duplicate_trace_reference & ~a.duplicate_original_name & ~a.duplicate_station_channel_start
    )
    p_usable = common_valid & a.p_label_valid & ~a.s_not_after_p & ~a.p_before_origin
    a["main_p_candidate"] = a.event_group_confirmed & p_usable & a.p_manual
    a["automatic_p_sensitivity_candidate"] = (
        a.event_group_confirmed & p_usable & a.p_status.isin(["automatic", "autopicker"])
    )
    a["noise_candidate"] = noise & common_valid & df.trace_p_arrival_sample.isna() & df.trace_s_arrival_sample.isna()
    a["noise_group_key"] = (a.station_key + "|" + a.trace_start_time_utc.dt.strftime("%Y-%m-%d")).where(a.noise_candidate)

    reasons = pd.Series("", index=df.index, dtype="string")
    rules = {
        "event_id_missing": eq & df.source_id.isna(),
        "catalog_conflict": a.catalog_conflict,
        "cross_id_fingerprint_candidate": a.cross_id_fingerprint_candidate,
        "catalog_incomplete_or_invalid": eq & df.source_id.isin(bad_ids),
        "station_unconfirmed": ~a.station_valid,
        "start_time_invalid": ~a.trace_start_time_valid,
        "sampling_rate_invalid": ~a.sampling_rate_valid,
        "component_order_invalid": ~a.component_order_valid,
        "waveform_reference_invalid": ~a.waveform_reference_valid,
        "duplicate_trace_reference": a.duplicate_trace_reference,
        "duplicate_original_name": a.duplicate_original_name,
        "duplicate_station_channel_start": a.duplicate_station_channel_start,
        "p_label_invalid": eq & ~a.p_label_valid,
        "p_not_manual": eq & ~a.p_manual,
        "s_not_after_p": a.s_not_after_p,
        "p_before_origin": a.p_before_origin,
        "noise_with_phase_label": noise & (df.trace_p_arrival_sample.notna() | df.trace_s_arrival_sample.notna()),
        "unknown_category": ~(eq | noise),
    }
    for reason, mask in rules.items():
        reasons.loc[mask] = reasons.loc[mask] + reason + ";"
    a["review_reasons"] = reasons.str.rstrip(";")
    a["event_group_key"] = ("STEAD:" + df.source_id).where(a.event_group_confirmed)
    for label, mask in [("main_p", a.main_p_candidate), ("automatic_p", a.automatic_p_sensitivity_candidate)]:
        grouped = a.loc[mask].groupby("source_id")
        events[f"{label}_trace_count"] = grouped.size().reindex(events.index, fill_value=0)
        events[f"{label}_station_count"] = grouped.station_key.nunique().reindex(events.index, fill_value=0)

    overlaps = []
    for cohort, mask in [("catalog_ids", eq), ("confirmed_event_ids", a.event_group_confirmed)]:
        pairs = event_data.loc[mask & df.source_id.notna(), ["source_id", "split"]].dropna().drop_duplicates()
        split_sets = {key: set(rows.source_id) for key, rows in pairs.groupby("split")}
        labels = sorted(split_sets)
        for i, left in enumerate(labels):
            for right in labels[i+1:]:
                overlaps.append({"cohort": cohort, "split_a": left, "split_b": right,
                                 "shared_source_ids": len(split_sets[left] & split_sets[right])})
    issues = [{"rule": name, "all_rows": int(mask.sum()), "earthquake_rows": int((mask & eq).sum()),
               "noise_rows": int((mask & noise).sum())} for name, mask in rules.items()]
    return a, events, aliases, pd.DataFrame(overlaps), pd.DataFrame(issues)


def field_profiles(df, a, output):
    """输出每字段分层缺失率/基数，并保存全量唯一值及其频数。"""
    unique_dir = output / "unique_values"
    unique_dir.mkdir(exist_ok=True)
    cohorts = {"all": np.ones(len(df), dtype=bool), "earthquake": a.is_earthquake, "noise": a.is_noise}
    fields = {col: (df[col], "metadata.csv") for col in df}
    fields.update({
        "audit_station_key": (a.station_key, "station_network_code + station_code"),
        "audit_sampling_rate_hz": (a.sampling_rate_hz, "row metadata or HDF5 data_format/sampling_rate"),
        "audit_component_order": (a.component_order, "row metadata or HDF5 data_format/component_order"),
        "audit_event_group_key": (a.event_group_key, "validated source_id only"),
    })
    rows, top = [], []
    for field, (series, origin) in fields.items():
        counts = series.dropna().value_counts()
        counts.rename_axis("value").rename("count").reset_index().to_csv(
            unique_dir / f"{field}.csv.gz", index=False, compression="gzip", encoding="utf-8-sig"
        )
        for value, count in counts.head(15).items():
            top.append({"field": field, "value": value, "count": int(count), "is_missing": False})
        if series.isna().any():
            top.append({"field": field, "value": "<MISSING>", "count": int(series.isna().sum()), "is_missing": True})
        for cohort, mask in cohorts.items():
            s = series.loc[mask]
            missing = int(s.isna().sum())
            rows.append({"field": field, "cohort": cohort, "rows": len(s), "missing": missing,
                         "missing_pct": 100*missing/len(s) if len(s) else np.nan,
                         "unique_nonmissing": int(s.nunique(dropna=True)), "source": origin})
    profile = pd.DataFrame(rows)
    profile.to_csv(output / "field_profile.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(top).to_csv(output / "top_values.csv", index=False, encoding="utf-8-sig")
    return profile


def save_plots(profile, a, events, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fields = profile.loc[profile.cohort.eq("all"), "field"].tolist()
    matrix = profile.pivot(index="field", columns="cohort", values="missing_pct").reindex(fields)[["all", "earthquake", "noise"]]
    fig, ax = plt.subplots(figsize=(10, max(9, len(fields)*0.27)))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=100, cmap="Blues")
    ax.set_xticks(range(3), ["All traces", "Earthquake traces", "Noise traces"])
    ax.set_yticks(range(len(fields)), fields)
    for i in range(len(fields)):
        for j in range(3):
            value = matrix.iloc[i, j]
            label = "NA" if pd.isna(value) else (f"{value:.3f}%" if 0 < value < 0.1 else f"{value:.1f}%")
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white" if value > 55 else "black")
    ax.set_title("STEAD field missingness by trace category\nRaw fields plus explicitly derived audit fields", pad=14)
    fig.colorbar(im, ax=ax, label="Missing rows (%)", fraction=0.03)
    fig.tight_layout(); fig.savefig(output / "field_missingness.png", dpi=170); plt.close(fig)

    counts = profile.loc[profile.cohort.eq("all")].set_index("field").loc[fields, "unique_nonmissing"]
    fig, ax = plt.subplots(figsize=(12, max(9, len(fields)*0.27)))
    ax.barh(fields, counts, color="#356A89")
    ax.set_xscale("symlog", linthresh=1)
    for i, value in enumerate(counts):
        ax.text(max(value, 1)*1.13, i, f"{value:,}", va="center", fontsize=8)
    ax.set_xlim(0, max(counts.max(), 1)*12)
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.15)
    ax.set_xlabel("Distinct non-missing values (symmetric-log axis; zero retained)")
    ax.set_title("STEAD field cardinality | all traces")
    fig.tight_layout(); fig.savefig(output / "field_unique_counts.png", dpi=170); plt.close(fig)

    distribution = []
    populations = {
        "All catalog IDs": events.station_count,
        "Confirmed catalog IDs": events.loc[events.event_group_confirmed, "station_count"],
        "Manual-P candidate events": events.loc[events.main_p_trace_count.gt(0), "main_p_station_count"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#7B848B", "#356A89", "#BB792F"]
    for (label, values), color in zip(populations.items(), colors):
        freq = values.value_counts().sort_index()
        for station_count, event_count in freq.items():
            distribution.append({"cohort": label, "stations": int(station_count), "events": int(event_count)})
        if len(values):
            axes[0].plot(freq.index, freq.values, marker=".", linewidth=1.1, color=color, label=f"{label} (n={len(values):,})")
            axes[1].step(freq.index, 100*freq.cumsum()/len(values), where="post", color=color, label=label)
    axes[0].set_yscale("log"); axes[0].set_ylabel("Events (log scale)")
    axes[1].set_ylabel("Cumulative events (%)"); axes[1].set_ylim(0, 102)
    for ax in axes:
        ax.set_xlabel("Distinct network.station keys per event"); ax.grid(alpha=0.2); ax.legend(fontsize=8)
    fig.suptitle("STEAD event station-count distribution\nMissing station identities are not counted; a trace is not a station")
    fig.tight_layout(); fig.savefig(output / "stations_per_event.png", dpi=170); plt.close(fig)
    pd.DataFrame(distribution).to_csv(output / "station_count_distribution.csv", index=False, encoding="utf-8-sig")

    records = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for phase, ax in zip(("p", "s"), axes):
        counts = a.loc[a.is_earthquake, f"{phase}_status"].fillna("<MISSING>").value_counts()
        bars = ax.bar(counts.index, counts.values, color="#356A89")
        for bar, (status, count) in zip(bars, counts.items()):
            ax.text(bar.get_x()+bar.get_width()/2, count, f"{count:,}\n{count/a.is_earthquake.sum():.1%}", ha="center", va="bottom", fontsize=8)
            records.append({"phase": phase.upper(), "status": status, "earthquake_rows": int(count)})
        ax.set_ylim(0, max(counts.max(), 1)*1.25); ax.set_title(f"{phase.upper()} pick status | earthquake traces")
        ax.set_ylabel("Traces"); ax.grid(axis="y", alpha=0.15)
    fig.tight_layout(); fig.savefig(output / "pick_status.png", dpi=170); plt.close(fig)
    pd.DataFrame(records).to_csv(output / "pick_status_counts.csv", index=False, encoding="utf-8-sig")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_outputs(df, a, events, aliases, overlaps, issues, data_format, buckets, output, data_dir):
    events.to_csv(output / "event_inventory.csv.gz", index_label="source_id", compression="gzip", encoding="utf-8-sig")
    unresolved = ~events.event_group_confirmed
    events.loc[unresolved].to_csv(output / "event_groups_to_review.csv", index_label="source_id", encoding="utf-8-sig")
    aliases.to_csv(output / "cross_id_fingerprint_candidates.csv", index=False, encoding="utf-8-sig")
    overlaps.to_csv(output / "original_split_event_overlap.csv", index=False, encoding="utf-8-sig")
    issues.to_csv(output / "quality_checks.csv", index=False, encoding="utf-8-sig")
    raw_columns = [c for c in ["trace_name", "trace_name_original", "trace_category", "station_network_code", "station_code", "trace_channel", "trace_start_time", "source_origin_time", "split"] if c in df]
    manifest = pd.concat([df[raw_columns], a], axis=1)
    paths = {
        "main_p_candidates": a.main_p_candidate,
        "uncertain_event_records": a.is_earthquake & ~a.event_group_confirmed,
        "automatic_p_sensitivity": a.automatic_p_sensitivity_candidate,
        "excluded_earthquake_records": a.is_earthquake & ~a.main_p_candidate,
        "noise_candidates": a.noise_candidate,
    }
    a.to_csv(output / "record_audit.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    for name, mask in paths.items():
        manifest.loc[mask].to_csv(output / f"{name}.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    existing_noise_split_crossing = None
    if "split" in df:
        noise_splits = pd.DataFrame({"key": a.noise_group_key, "split": df["split"]}).dropna()
        existing_noise_split_crossing = int(noise_splits.groupby("key")["split"].nunique().gt(1).sum())
    station_stats = {}
    for label, values in [("catalog_ids", events.station_count), ("confirmed_ids", events.loc[events.event_group_confirmed, "station_count"]), ("manual_p_events", events.loc[events.main_p_trace_count.gt(0), "main_p_station_count"])]:
        station_stats[label] = {"events": len(values), "min": int(values.min()) if len(values) else None,
                                "median": float(values.median()) if len(values) else None,
                                "p95": float(values.quantile(.95)) if len(values) else None,
                                "max": int(values.max()) if len(values) else None,
                                "events_ge_3_stations": int(values.ge(3).sum())}
    provenance = {}
    for name in ("metadata.csv", "waveforms.hdf5"):
        file = data_dir / name
        stat = file.stat()
        provenance[name] = {"path": str(file.resolve()), "size_bytes": stat.st_size,
                            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}
    provenance["metadata.csv"]["sha256"] = sha256(data_dir / "metadata.csv")
    summary = {
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script_sha256": sha256(Path(__file__)),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "h5py": h5py.__version__},
        "files": provenance, "data_format": data_format,
        "rows": len(df), "raw_fields": len(df.columns), "earthquake_rows": int(a.is_earthquake.sum()),
        "noise_rows": int(a.is_noise.sum()), "source_ids": len(events),
        "catalog_conflict_ids": int(events.catalog_conflict.sum()),
        "origin_time_conflict_ids": int(events.origin_nunique.gt(1).sum()),
        "cross_id_fingerprint_candidate_ids": int(events.cross_id_fingerprint_candidate.sum()),
        "unresolved_event_ids": int(unresolved.sum()), "confirmed_event_ids": int(events.event_group_confirmed.sum()),
        "uncertain_event_records": int((a.is_earthquake & ~a.event_group_confirmed).sum()),
        "main_p_candidate_rows": int(a.main_p_candidate.sum()),
        "main_p_candidate_events": int(events.main_p_trace_count.gt(0).sum()),
        "automatic_p_sensitivity_rows": int(a.automatic_p_sensitivity_candidate.sum()),
        "noise_candidate_rows": int(a.noise_candidate.sum()),
        "original_split_crossing_ids": int(events.original_split_count.gt(1).sum()),
        "confirmed_original_split_crossing_ids": int((events.event_group_confirmed & events.original_split_count.gt(1)).sum()),
        "noise_station_date_groups_crossing_original_split": existing_noise_split_crossing,
        "hdf5_buckets": len(buckets), "hdf5_stored_rows": sum(v["shape"][0] for v in buckets.values() if len(v["shape"])==3),
        "valid_waveform_references": int(a.waveform_reference_valid.sum()),
        "fractional_p_labels": int((a.is_earthquake & a.p_label_fractional).sum()),
        "fractional_s_labels": int((a.is_earthquake & a.s_label_fractional).sum()),
        "station_count_statistics": station_stats,
        "rules": {
            "event_identity": "Use source_id verbatim after whitespace normalization; no filename inference, no automatic splitting/merging. Exact within-ID conflicts after numeric/time normalization and cross-ID exact fingerprint candidates are quarantined as whole ID groups.",
            "station_identity": "network.station; channel is not a separate station. Missing network/code is unconfirmed; no location-code inference.",
            "main_p": "Confirmed event group + valid station/time/format/HDF reference + unique reference/original name/station-channel-start + in-range manual P + no P-before-origin or S<=P contradiction. Manual S is not required for a P-only task.",
            "missingness": "Trim whitespace; empty/nan/null/none are missing. NA is retained as a possible valid identifier. Distinct counts exclude missing. No numeric sentinel is silently treated as missing.",
            "time": "Parse metadata timestamps; timezone-naive values interpreted as UTC, not independently verified from original recordings.",
            "scope": "Metadata and HDF5 structure audit; no full waveform/checksum scan, no authoritative catalog reconciliation, no fuzzy-event matching, no interval-overlap or cross-dataset deduplication. Candidate manifests still require new grouped splits.",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def write_report(summary, issues, overlaps, output):
    s = summary
    lines = [
        "# STEAD 元数据审计报告", "",
        f"审计时间（UTC）：{s['audit_time_utc']}。本报告由 STEADPRE/STEADAUDIT.py 生成，原始数据只读。", "",
        "## 主要结论", "",
        f"共 {s['rows']:,} 条 trace，其中地震 {s['earthquake_rows']:,} 条、噪声 {s['noise_rows']:,} 条；地震目录含 {s['source_ids']:,} 个不同 source_id。",
        f"HDF5 声明采样率 {s['data_format'].get('sampling_rate')} Hz、分量顺序 {s['data_format'].get('component_order')}、单条维度 {s['data_format'].get('dimension_order')}；{s['valid_waveform_references']:,} 条 metadata 波形引用通过结构/范围检查。",
        f"{s['catalog_conflict_ids']:,} 个 ID 存在组内发震时间或经纬度、深度不一致，其中 {s['origin_time_conflict_ids']:,} 个 ID 的发震时间不一致。{s['cross_id_fingerprint_candidate_ids']:,} 个 ID 涉及跨 ID 相同时间/位置/深度指纹。差异可能来自 ID 冲突、目录修订或精度差异，本审计不擅自判断为同一或不同地震。",
        f"共隔离 {s['unresolved_event_ids']:,} 个待复核 ID，涉及 {s['uncertain_event_records']:,} 条地震记录（还包括无法确认事件身份的记录）；通过严格目录一致性规则的 ID 为 {s['confirmed_event_ids']:,} 个。",
        f"人工 P 标签的主分析候选为 {s['main_p_candidate_rows']:,} 条、{s['main_p_candidate_events']:,} 个事件；自动 P 标签敏感性候选 {s['automatic_p_sensitivity_rows']:,} 条；独立噪声候选 {s['noise_candidate_rows']:,} 条。候选数并非最终训练/测试数。", "",
        "## 事件互斥风险", "",
        f"原始 split 中有 {s['original_split_crossing_ids']:,} 个 source_id 跨集合；通过本次事件身份审核后仍有 {s['confirmed_original_split_crossing_ids']:,} 个 ID 跨集合。不能将现有 train/dev/test 直接称为事件互斥划分。",
        f"噪声候选中的 network.station+UTC 日期分组，有 {s['noise_station_date_groups_crossing_original_split']} 组跨原始 split。",
        "应从候选清单重新按事件分组；同事件所有台站、裁剪、增强和预 P 噪声继承同一分组。噪声按台站日期分组只是起点，还需检查跨午夜和时间重叠。此脚本不自动创建训练/验证/测试划分。", "",
        "|范围|集合 A|集合 B|共享 source_id 数|", "|---|---|---|---:|",
    ]
    lines += [f"|{r.cohort}|{r.split_a}|{r.split_b}|{r.shared_source_ids:,}|" for r in overlaps.itertuples()]
    lines += ["", "## 字段口径与判定", "",
              "- source_id：只使用实际目录标识。对时间、经纬度及深度按解析后的值严格检查一致性；差异阈值为零，这是保守隔离规则，不代表每个差异都是物理事件冲突。",
              "- 等价事件标识：本次没有可用的已核实映射表。缺失/冲突 ID 不用文件名前缀、台站名或相似字符串替代；也不自动拼接时间形成新事件号。",
              "- 跨 ID 指纹：完全相同的发震时间、经纬度和深度对应多个 ID 时，只输出候选供回查目录，并隔离所涉完整 ID。相近但不完全相同的重复事件尚未排除。",
              "- 台站：network.station 的不同组合才算不同台站；同台站多通道或多个 trace 不重复计数。网络号缺失时不凭文件名补全。位置码未提供，无法区分台站代码复用或仪器位置历史。",
              "- 原始起始时间：保留 metadata 中 trace_start_time 原字符串并解析；无时区字符串按 UTC 解释。不能由本地副本证明它与原始采集文件完全一致。",
              "- 采样率和分量：优先逐行 metadata，缺失时读取 HDF5 data_format。audit_sampling_rate_hz 与 audit_component_order 是有效值；CSV 没有这些列不等于数据未知。",
              "- P/S：使用采样点坐标，检验有限值及 0≤sample<实际窗口长度；秒数为 sample/fs。非整数样本位置保留为亚采样标签，不直接判错。",
              f"- 地震记录的非整数 P/S 标签分别有 {s['fractional_p_labels']:,}/{s['fractional_s_labels']:,} 条。",
              "- 人工状态：manual 与 autopicker/automatic 严格区分。manual 仅表示数据集的标签声明，并非本次人工重拾取质量保证；P-only 主候选要求 manual P，不强制 manual S。",
              "- 噪声通常没有 source_id、发震时间和 P/S 标签，这是预期缺失，单独报告，不作为坏地震记录。",
              "- 缺失率分母分别为全部、地震、噪声记录数；唯一值数排除缺失。原字段按字符串统计，数值/时间合法性另查，避免混淆完整性与正确性。", "",
              "## 质量检查", "", "各行原因可以重叠，不能相加当作排除总数。", "",
              "|规则|全部记录|地震记录|噪声记录|", "|---|---:|---:|---:|"]
    lines += [f"|{r.rule}|{r.all_rows:,}|{r.earthquake_rows:,}|{r.noise_rows:,}|" for r in issues.itertuples()]
    lines += ["", "## 每事件台站数", "", "|范围|事件数|中位数|95% 分位数|最大值|至少 3 台站事件数|", "|---|---:|---:|---:|---:|---:|"]
    for label, row in s["station_count_statistics"].items():
        lines.append(f"|{label}|{row['events']:,}|{row['median']}|{row['p95']}|{row['max']}|{row['events_ge_3_stations']:,}|")
    lines += ["", "![字段缺失率](field_missingness.png)", "", "![字段唯一值数](field_unique_counts.png)", "", "![每事件台站数](stations_per_event.png)", "", "![P/S 拾取状态](pick_status.png)", "",
              "## 输出与后续使用", "",
              "- field_profile.csv：所有原字段及明确派生字段，按 all/earthquake/noise 分层统计。unique_values/*.csv.gz 保存各字段完整非缺失唯一值及频数，top_values.csv 保存常见值及缺失值数量。",
              "- event_inventory.csv.gz：所有目录 ID 的 trace/台站数、时间和坐标取值范围、原始 split、确认状态。event_groups_to_review.csv 与 cross_id_fingerprint_candidates.csv 供回查原始目录。",
              "- record_audit.csv.gz：每条记录的布尔审计标志与排除原因；metadata_row 是 CSV 从零开始的数据行位置，可关联原始 metadata。",
              "- main_p_candidates.csv.gz：只含满足本次严格规则的地震记录，可作为重新事件分组的输入。既有 split 仅保留作审计证据，不代表可复用。",
              "- uncertain_event_records.csv.gz：无法确认事件身份的地震记录；excluded_earthquake_records.csv.gz：全部未进入主候选的地震记录及原因。前者禁止混入声称事件互斥的主分析；只能另报敏感性研究且不得声称已排除事件泄漏。",
              "- automatic_p_sensitivity.csv.gz：事件身份/技术检查通过但 P 来自 autopicker/automatic 的记录；与主候选共享事件的记录必须沿用同一事件划分，不能单独随机切分。",
              "- noise_candidates.csv.gz：可用于后续背景分组的候选，不赋予地震 event_group_key。",
              "- summary.json：输入路径、大小、修改时间、metadata 和脚本 SHA256、包版本、核心数量与审核规则。HDF5 未做全文件哈希。", "",
              "本报告仅完成元数据与 HDF5 结构审计；尚未回查权威事件目录、扫描全量波形有效值/哈希、检测相似事件或跨数据集重复、完成全部时间区间重叠检查。主候选的事件确认是基于现有目录的一致性判定，不是物理事件唯一性的无条件证明。"]
    (output / "AUDIT_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    data_dir, output = args.data_dir.resolve(), args.output_dir.resolve()
    if output == data_dir or data_dir in output.parents:
        parser.error("Keep audit outputs outside the input dataset directory.")
    output.mkdir(parents=True, exist_ok=True)
    print("[1/5] Reading metadata and HDF5 structure...", flush=True)
    df = read_metadata(data_dir / "metadata.csv")
    data_format, buckets = read_hdf5_layout(data_dir / "waveforms.hdf5")
    print(f"[2/5] Auditing {len(df):,} records and event identity...", flush=True)
    a, events, aliases, overlaps, issues = audit_metadata(df, data_format, buckets)
    print("[3/5] Profiling fields and exporting complete unique-value counts...", flush=True)
    profile = field_profiles(df, a, output)
    print("[4/5] Saving review/candidate manifests and provenance...", flush=True)
    summary = write_outputs(df, a, events, aliases, overlaps, issues, data_format, buckets, output, data_dir)
    print("[5/5] Plotting and writing audit report...", flush=True)
    save_plots(profile, a, events, output)
    write_report(summary, issues, overlaps, output)
    print(json.dumps({key: summary[key] for key in ["rows", "source_ids", "unresolved_event_ids", "main_p_candidate_rows", "original_split_crossing_ids"]}, indent=2), flush=True)
    print(f"Audit complete: {output / 'AUDIT_REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
