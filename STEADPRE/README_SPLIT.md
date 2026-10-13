# STEAD 主实验数据准备

在项目根目录、已有 `dizhenczsq` Python 环境运行：

```powershell
python STEADPRE/STEADSPLIT.py
python STEADPRE/STEADSPLIT.py --verify
python -m unittest STEADPRE.test_stead_split -v
python STEADPRE/check_stead_training.py
```

默认输出 `artifacts/stead_split/main`，上限为 50,000 地震记录 + 12,500 噪声记录，每事件最多 6 个不同台站。
`--profile debug` 使用 2,000+500 上限；`--profile expanded` 使用 100,000+25,000 上限。
预算不是精确配额：末组不截断，波形质量排除后不补抽。文件中的 `split` 是重新分组得到的；`original_split` 仅用于溯源。
不同 profile 是各自冻结的协议，不能把不同 profile 的训练集与测试集混用，也不能将较大 profile 的测试成绩与较小 profile 直接当作同一固定测试集上的学习曲线。
所有模型与所有训练随机种子必须复用同一份 main 清单。直接重跑已有输出只校验、不覆盖；另做协议试验请指定新 `--output`。

## 生成顺序

1. 验证已有审计与 metadata 哈希、HDF5 大小/修改时间一致；以全部人工 P 主候选和合格噪声建立关系图。
2. 按事件 ID、噪声网络-台站-UTC 日期、父记录引用及同台站重叠区间分组；跨午夜和传递重叠不能拆开。
3. 随机抽事件并限制不同台站数量，抽取噪声日期组；完整读取入选波形做有限值、恒定分量、精确重复及近重复筛查。
4. 将不可拆关系的连通组整体分给 train/val/test，以事件比例 70/10/20 为主约束；保留原 source_id，不把关系图分组冒充新事件。
5. 保存地震/噪声 manifest，再保存固定原始 60 s 的三份 windows.csv。只有最后存在 summary.json 的目录才算完成。

`SPLIT_REPORT.md` 给出实际计数和限制。`leakage_checks.csv` 给出两两集合交叉检查。`eligible_record_groups.csv.gz` 给出完整候选的继承分组；其中 unused 不能直接随机追加。
`qa_sample.csv` 与 `qa/` 提供各类 30 条波形的人工复核页面，红虚线为人工 P 标签。

## PyTorch 训练接入

```python
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from STEADPRE.STEADWINDOW import STEADWindowDataset

def main():
    root = Path("artifacts/stead_split/main")
    train = STEADWindowDataset(root / "train_windows.csv", augment=True, seed=42)
    val = STEADWindowDataset(root / "val_windows.csv")
    loader = DataLoader(train, batch_size=8, shuffle=True, num_workers=2,
                        persistent_workers=False,
                        generator=torch.Generator().manual_seed(42))
    for epoch in range(30):
        train.set_epoch(epoch)  # 必须在创建本 epoch 的 loader 迭代器之前。
        for batch in loader:
            x = batch["waveform"]       # [B,3,6000] float32，Z/N/E
            y = batch["p_target"]       # [B,1,6000]，Gaussian sigma=0.05 s
            mask = batch["valid_mask"]  # [B,1,6000]，padding 不计损失
            # logits = model(x)         # [B,1,6000]；只给模型 x
            # element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            #     logits, y, reduction="none")
            # loss = (element_loss * mask).sum() / mask.sum().clamp_min(1)
            # loss.backward(); optimizer.step(); optimizer.zero_grad()
    train.close()
    val.close()

if __name__ == "__main__":  # Windows 多进程 DataLoader 必须保护入口。
    main()
```

训练中可做最大 ±5 s 非循环平移，P/S 标签同步平移，填充为零并输出有效掩码；P 不得被移出窗口或落入 padding。保留至少 91.7% 原始样本。
各窗口只在有效样本内去均值、线性趋势、三分量联合 RMS 归一化；无全数据拟合，无默认带通滤波。验证/测试不允许增强、不按 P 真值移动窗口。
`p_sample`、`s_sample` 可含小数；噪声的 P 为 NaN、目标全零。S 标签状态保存在清单，P-only 训练不使用它作为人工 S 参考。
epoch 状态不会自动同步给持久 worker，故示例固定 `persistent_workers=False`。多种子实验只改变训练 seed，不改划分 seed。

## 评估边界

验证集选择阈值/检查点；冻结后再测测试集。预先固定主容差 ±0.10 s，另报 ±0.05/0.20 s。
不要只对成功拾取计算误差而忽略漏检；保持相同噪声比例，另报噪声每小时误触发。
这是事件互斥、允许台站交叉的主实验，不是台站互斥或严格按时间外推。重复检查覆盖已知元数据关系、入选波形精确内容与限定的近重复候选，不等于穷尽所有未知重复。
本次 main 实际保留 49,975 条地震和 12,283 条噪声；地震 P 标签均在原始窗口的 4–10 s。这个位置先验可能让模型学习到数据集裁剪习惯，事件互斥本身不能消除它。训练平移只能有限缓解，不能凭空取得原始窗口外的真实背景。论文需报告 `p_position_distribution.csv`，并考虑独立的长连续记录/位置扰动敏感性实验；不要通过真值重切主测试集来掩盖这一问题。
若增加预 P 噪声、裁剪、特征缓存或增强版本，全部继承原事件和父记录的 split，先分组再生成，不能再次随机拆分。
Iquique→ETHZ 是独立实验，不继承 STEAD 权重；STEAD 本清单不能证明跨数据集无重复。
