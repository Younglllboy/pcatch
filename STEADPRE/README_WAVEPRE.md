# 实验 A：第七步波形预处理与 P 标签生成

入口为 `WAVEPRE.py`。原方案第 10 页与“分析模型与实验重复性”中的第七步讨论已落实为本地 STEAD 专用协议。
基于现有 `artifacts/stead_split/main`，不重新分组、不重抽集合。实验 A 为 STEAD 上的离线 P 波拾取。

## 运行与查看

项目 `dizhenczsq` Python 环境、项目根目录下运行：

```powershell
python STEADPRE/WAVEPRE.py
python STEADPRE/WAVEPRE.py --verify
python -m unittest STEADPRE.test_wavepre -v
python STEADPRE/check_wavepre_training.py
```

默认正式输出为 `artifacts/stead_wavepre/main/`。已有完整结果时，再次运行只校验，不改动清单和参数。
双击 `index.html` 可在普通浏览器离线验收；在页面选择样本、原生/训练增强、完整 60 s/参考 P 附近。
P 附近按钮仅缩放绘图，不改变模型输入或测试窗口。val/test 禁止选择增强。

| 输出 | 用途 |
|---|---|
| index.html | 原始/处理 ZNE、P/S 参考线、Gaussian 标签、mask 的交互对照，含需复核记录 |
| case_*.png、overview.png | 关键验收图，可直接查看或插入实验记录 |
| qa_page_*.png | 全部随机验收样本及诊断补充样本的波形/标签图 |
| record_qc.csv.gz | 全部窗口的质控、去趋势、归一化、标签对齐数值 |
| qc_flagged_records.csv | 连续零值/常值/极值平台等待复核提示，不等于已确诊缺口或饱和 |
| train_augmentation_epoch0.csv.gz | seed=42、epoch=0 的每条训练记录的偏移、P/S 位置和有效区 |
| train/val/test_prepared.csv | 继承原窗口/事件/台站/划分并附加协议、物理相对到时和 QC 信息的审计清单 |
| examples.npz、sample_manifest.* | 验收样本实际数组及来源，可重新检查，不是全库缓存 |
| protocol.json、summary.json、ACCEPTANCE.md | 固定规则、来源/产物/代码 hash、统计及完成标记 |

## 已筛选并实现的 A 线工作

1. 重新验证被冻结清单，核对所有入选波形的内容 SHA256。读取真实 HDF5 data_format。
2. 本地数据已是 ZNE、100 Hz、(3,6000)；只检查，不重复转置、重排或重采样。P/S 相对物理时间为原小数 sample / 100，原生标签不改。
3. 硬失败：非有限波形、错误形状、恒定整分量、无可用残差能量、无效人工 P/类别、P±3σ缺少真实上下文。遇到硬失败输出诊断并停止，不自动补样或伪造标签。
4. 对连续零值/常值 >=0.2 s 或在最小/最大值连续停留 >=5 点作固定规则提示。提示记录不自动删除、插值或修改类别；数值平台并不单独证明采集缺口。无响应档案时不猜测物理单位转换。
5. 每个窗口有效区去均值、线性趋势；三分量共用一个 RMS（下限 1e-8），输出 float32。没有全数据/验证/测试集拟合，没有默认带通。
6. P 使用 σ=0.05 s（5 samples）的 Gaussian 软标签，保留小数中心；明确 STEAD 背景才用零目标。S 不生成正 P 标签，原 S 状态留作诊断。P 标签平滑不是不确定性概率密度。
7. train 可做最多 ±5 s 非循环平移，P/S 同步移动，零填充、mask同步，有效区不少于90%。P±3σ必须处于真实样本内，边界阈值随σ变化。
8. val/test 固定原窗口和确定性处理。所有架构、基线与训练种子共享同一协议。Gaussian、归一化或滤波的选择只能依据 Train/Val；不得依据 Test 表现切换协议。

记录中的 `valid_mask` 只表明本实现已知的有效/填充区，不是传感器原生缺口标志。疑似自然缺口另通过 QC 提示记录，不能因数值等于零就擅自确定缺测。
数据仍保留 STEAD 的 counts 和既有仪器处理状态；窗口尺度归一化不等价于仪器响应校正。原生窗口 P 为 4–10 s 的位置先验仍然存在。

## 训练接入：推荐使用固定第七步协议

```python
import torch
from torch.utils.data import DataLoader
from STEADPRE.WAVEPRE import load_dataset, masked_bce_with_logits

def main():
    train = load_dataset(split="train", augment=True, seed=42)
    val = load_dataset(split="val")  # 默认无增强
    loader = DataLoader(train, batch_size=8, shuffle=True, num_workers=2,
                        persistent_workers=False,
                        generator=torch.Generator().manual_seed(42))
    # model / optimizer 由模型步骤提供。
    for epoch in range(30):
        train.set_epoch(epoch)  # 在本epoch创建迭代器之前设置。
        for batch in loader:
            waveform = batch["waveform"]       # B x 3 x 6000，唯一模型输入
            target = batch["p_target"]         # B x 1 x 6000
            valid_mask = batch["valid_mask"]   # B x 1 x 6000
            # logits = model(waveform)         # 使用logits，不先做sigmoid
            # loss = masked_bce_with_logits(logits, target, valid_mask)
            # optimizer.zero_grad(); loss.backward(); optimizer.step()
    train.close(); val.close()

if __name__ == "__main__":
    main()
```

mask/标签和 logits 应移到同一 device。损失按全 batch 有效采样点数归一，FP32 累加保护半精度 mask 的计数；padding 不贡献损失或梯度。
不把参考 P、S、SNR、事件 ID 或 mask 作为额外通道传入模型。masked loss 本身也不能防止网络看到填充值，因而模型结构对边界的处理和评估适用范围仍需单独说明。
`persistent_workers=False` 让每个 epoch 新建 worker 时获得最新 epoch；否则主进程 set_epoch 不会自动更新旧 worker。

波形/标签在线按需生成：本次已经对全体窗口运行预处理并完成数值检查，磁盘保留诊断与验收样本，不额外复制数 GB 全库。
`*_prepared.csv` 是衍生审计清单，不直接交给旧的 `STEADWindowDataset` 构造器；`load_dataset` 自动核对第七步协议后使用原冻结窗口清单和同一预处理核心。
旧 `STEADWINDOW.prepare_window` 兼容入口已转发到 `WAVEPRE.preprocess_waveform`；推荐新训练代码使用 `load_dataset`，以同时校验预处理协议。

## 明确未混入本环节的工作

- ETHZ 变采样率抗混叠、P-family合并、目标域适配与校准：实验 B，不执行。
- SWT/时序增强/多尺度卷积/BiMamba2：模型步骤，不提前生成全部小波系数。
- 自然 SNR 分层、白噪声/有色噪声/真实背景混合、分量丢失压力测试：后续鲁棒性步骤，不作为这次默认训练增强。
- 全局训练集 robust scaling：可另做敏感性对照，当前不拟合，也不视为必需步骤。

可选预先定义的敏感性协议（未作为本次主结果运行）：

```powershell
python STEADPRE/WAVEPRE.py --sigma-seconds 0.10 --output artifacts/stead_wavepre/sigma010
python STEADPRE/WAVEPRE.py --bandpass --output artifacts/stead_wavepre/bandpass_1_40
```

带通固定为4阶 Butterworth、SOS、1–40 Hz、sosfiltfilt、odd padding 27点，在有效区做离线零相位处理。它会削弱40–50 Hz，不能用于证明完整高频小波尺度的主贡献。
任何敏感性协议都保留原事件与集合；不能混用不同协议的训练/验证/测试预处理。编辑核心后须用新输出目录重新验收，避免旧完成标记掩盖实现变更。
