"""查看本地 STEAD 波形（SeisBench 格式）。

前五条地震：python STEADPRE/STEADLOOK.py --mode first5
按 metadata 原顺序选取前五条 earthquake_local 记录，跳过噪声并标记 P 到时。
直接运行时使用下方 VIEW_MODE 指定的模式。
随机事件：python STEADPRE/STEADLOOK.py --mode event
固定随机种子：python STEADPRE/STEADLOOK.py --mode event --seed 42
同时查看两种模式：python STEADPRE/STEADLOOK.py --mode both

事件中每个 trace 显示 Z/N/E 三分量；使用 Prev/Next 或左右方向键翻页。
红色虚线表示 metadata 中已有的 P 初至标签，不是模型预测。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pandas as pd
import seisbench.data as sbd


DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "STEAD"
VIEW_MODE = "first5"  # PyCharm 直接运行时，可选 "first5"、"event" 或 "both"。
COMPONENTS = ("Z", "N", "E")


def load_stead(data_dir=DATA_DIR):
    """只加载 metadata；波形按页读取，不预加载约 91 GB 的 HDF5。"""
    data_dir = Path(data_dir)
    for filename in ("metadata.csv", "waveforms.hdf5"):
        if not (data_dir / filename).is_file():
            raise FileNotFoundError(f"Missing STEAD file: {data_dir / filename}")
    return sbd.WaveformDataset(
        path=data_dir,
        name="STEAD",
        cache=None,
        component_order="ZNE",
        dimension_order="NCW",
    )


def p_arrival_seconds(metadata, n_samples, sampling_rate):
    """只接受波形范围内的有限 P 标签，返回相对本条记录起点的秒数。"""
    try:
        sample = float(metadata.get("trace_p_arrival_sample", np.nan))
    except (TypeError, ValueError):
        return None, "No valid P label"
    if not np.isfinite(sample):
        return None, "No P label"
    if not 0 <= sample < n_samples:
        return None, "P label outside trace"
    seconds = sample / sampling_rate
    return seconds, f"P = {seconds:.3f} s (sample {sample:g})"


def random_event_indices(metadata, seed=None):
    """等概率抽取一个地震事件，返回该 source_id 对应的全部行位置。"""
    event_ids = metadata["source_id"].astype("string").str.strip()
    valid_id = event_ids.notna() & ~event_ids.str.lower().isin(
        ["", "nan", "none", "null"]
    )
    earthquake = metadata["trace_category"].eq("earthquake_local")
    candidates = event_ids[valid_id & earthquake].unique()
    if len(candidates) == 0:
        raise ValueError("No earthquake event with a valid source_id was found.")

    rng = np.random.default_rng(seed)
    event_id = str(candidates[rng.integers(len(candidates))])
    # 先抽唯一事件，再选择其全部 trace，避免多台站事件更易被抽中的偏差。
    # 使用行位置以匹配 get_sample(idx)，不依赖 DataFrame 的索引标签。
    positions = np.flatnonzero(event_ids.eq(event_id).fillna(False).to_numpy(dtype=bool))
    return event_id, positions


class WaveformViewer:
    """每行一个 trace、每列一个分量；事件的全部记录可通过翻页查看。"""

    def __init__(self, dataset, indices, title, traces_per_page=5):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=int)
        if len(self.indices) == 0:
            raise ValueError("No traces to display.")
        if traces_per_page < 1:
            raise ValueError("traces_per_page must be at least 1.")
        self.rows = min(traces_per_page, len(self.indices))
        self.page_count = (len(self.indices) + self.rows - 1) // self.rows
        self.page = 0
        self.title = title

        self.figure, self.axes = plt.subplots(
            self.rows, 3, figsize=(16, 2.1 * self.rows + 1.5), squeeze=False,
            sharex="row",
        )
        self.figure.subplots_adjust(
            left=0.065, right=0.98, bottom=0.12, top=0.89, hspace=0.85, wspace=0.25,
        )
        self.previous_button = Button(self.figure.add_axes([0.73, 0.025, 0.10, 0.035]), "Prev")
        self.next_button = Button(self.figure.add_axes([0.85, 0.025, 0.10, 0.035]), "Next")
        self.previous_button.on_clicked(self.previous_page)
        self.next_button.on_clicked(self.next_page)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.footer = self.figure.text(0.065, 0.04, "", fontsize=9)
        self.draw_page()

    def draw_page(self):
        start = self.page * self.rows
        stop = min(start + self.rows, len(self.indices))
        for row, axes in enumerate(self.axes):
            for axis in axes:
                axis.clear()
                axis.set_visible(False)
            if start + row >= stop:
                continue

            index = int(self.indices[start + row])
            # get_sample 同时返回波形和对应 metadata，避免标签与波形错位。
            waveform, metadata = self.dataset.get_sample(index)
            if waveform.ndim != 2 or waveform.shape[0] != 3:
                raise ValueError(f"Trace {index}: expected (3, samples), got {waveform.shape}")
            sampling_rate = float(metadata["trace_sampling_rate_hz"])
            if not np.isfinite(sampling_rate) or sampling_rate <= 0:
                raise ValueError(f"Trace {index}: invalid sampling rate {sampling_rate}")
            n_samples = waveform.shape[1]
            times = np.arange(n_samples) / sampling_rate
            p_time, p_description = p_arrival_seconds(metadata, n_samples, sampling_rate)
            trace_name = metadata.get("trace_name_original", metadata["trace_name"])
            category = metadata.get("trace_category", "unknown")

            for component, axis in enumerate(axes):
                axis.set_visible(True)
                axis.plot(times, waveform[component], color="#245575", linewidth=0.55)
                if p_time is not None:
                    axis.axvline(p_time, color="#D62728", linestyle="--", linewidth=1.1)
                    axis.text(
                        p_time, 0.97, " P", transform=axis.get_xaxis_transform(),
                        color="#D62728", ha="left", va="top", fontsize=9,
                    )
                axis.set_xlim(0, times[-1])
                axis.grid(alpha=0.2)
                axis.set_xlabel("Time since trace start (s)", fontsize=8)
                axis.tick_params(labelsize=8, labelbottom=True)
                axis.set_title(COMPONENTS[component], fontsize=10, loc="right")
            axes[0].set_ylabel("Amplitude (counts)", fontsize=8)
            axes[0].set_title(f"metadata row {index}", fontsize=9, loc="left")
            axes[1].set_title(str(trace_name), fontsize=8, loc="left")
            axes[2].set_title(f"{category} | {p_description}", fontsize=8, loc="left")

        self.figure.suptitle(
            f"{self.title}\nTraces {start + 1}-{stop} / {len(self.indices)}"
            f"  |  Page {self.page + 1}/{self.page_count}  |  Components: Z / N / E",
            fontsize=12,
        )
        self.footer.set_text(
            "Red dashed line: metadata P arrival. No marker if label missing/invalid. "
            "Use Prev/Next or left/right keys."
        )
        self.previous_button.set_active(self.page > 0)
        self.next_button.set_active(self.page + 1 < self.page_count)
        self.figure.canvas.draw_idle()

    def previous_page(self, event=None):
        if self.page > 0:
            self.page -= 1
            self.draw_page()

    def next_page(self, event=None):
        if self.page + 1 < self.page_count:
            self.page += 1
            self.draw_page()

    def on_key(self, event):
        if event.key == "left":
            self.previous_page()
        elif event.key == "right":
            self.next_page()


def show_first_five(dataset):
    """按 metadata 原顺序显示前五条真实地震 trace，并标记已有 P 到时。"""
    earthquake = dataset.metadata["trace_category"].eq("earthquake_local").fillna(False)
    # 使用原始行位置读取波形，不重置筛选后的索引，以免波形与标签错位。
    indices = np.flatnonzero(earthquake.to_numpy(dtype=bool))[:5]
    if len(indices) == 0:
        raise ValueError("No earthquake_local traces were found in metadata.")
    print("\nFirst five earthquake_local metadata rows:")
    print(dataset.metadata.iloc[indices])
    return WaveformViewer(dataset, indices, "First five earthquake traces")


def show_random_event(dataset, seed=None, traces_per_page=5):
    event_id, indices = random_event_indices(dataset.metadata, seed)
    print(f"\nRandom event source_id: {event_id}")
    print(f"All traces in this event: {len(indices)}")
    columns = [
        "source_id", "station_network_code", "station_code", "trace_name_original",
        "trace_p_arrival_sample", "trace_start_time",
    ]
    columns = [column for column in columns if column in dataset.metadata.columns]
    print(dataset.metadata.iloc[indices][columns].to_string())
    return WaveformViewer(dataset, indices, f"Event: {event_id}", traces_per_page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("first5", "event", "both"), default=VIEW_MODE)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=None, help="Random event selection seed")
    parser.add_argument("--traces-per-page", type=int, default=5)
    args = parser.parse_args()
    if args.traces_per_page < 1:
        parser.error("--traces-per-page must be at least 1")

    dataset = load_stead(args.data_dir)
    print(f"STEAD: {args.data_dir}\nTotal traces: {len(dataset):,}")
    # 持有 viewer 引用，保证 Matplotlib 按钮和键盘回调持续有效。
    viewers = []
    if args.mode in ("first5", "both"):
        viewers.append(show_first_five(dataset))
    if args.mode in ("event", "both"):
        viewers.append(show_random_event(dataset, args.seed, args.traces_per_page))
    plt.show()


if __name__ == "__main__":
    main()
