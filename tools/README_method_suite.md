# TCT_NGC 方法实验套件使用说明

`tools/run_method_suite.sh` 是 WeDetect TCT_NGC @1024 方法实验的**一键入口**。协作方只需跑
**一条命令**,脚本会自动:**下权重 → 下数据 → 训练 → 评测(base + novel)→ 出分析图**。
结果落在一个**不含权重的结果包**里,打 tar 回传即可。

---

## 0. 前提(只配一次)

1. **conda 环境**:名为 `wedetect` 的环境(pytorch 2.x / mmcv 2.1 / mmdet 3.3 / mmengine /
   open_clip / huggingface_hub / modelscope)。脚本默认用
   `$HOME/anaconda3/envs/wedetect/bin/python`;若你的路径不同,跑命令时加 `PYBIN=/your/python`。
2. **数据集 token**(唯一的 secret):数据集是 ModelScope **私有库** `doilion/TCT_NGC_1024_TAR`,
   需要作者给的临时 token。**权重是公开的,不用 token。**
   ```bash
   export MODELSCOPE_API_TOKEN='<作者给你的 token>'
   ```
3. GPU:推荐 8×H800(或任意 N 卡)。脚本会按卡数自动缩放学习率(见 §4)。

> 不需要手动下任何东西、不需要 scp 权重 —— 脚本全自动。

---

## 1. 快速开始

```bash
# 先预演:只打印每个实验 -> 配置/工作目录/有效batch/学习率/输出路径,不真跑
bash tools/run_method_suite.sh --list

# 正式跑(8 卡,每卡 batch 2 = 有效 batch 16 = 调好的配方 lr 3e-4)
export MODELSCOPE_API_TOKEN='<token>'
bash tools/run_method_suite.sh --gpus 0,1,2,3,4,5,6,7 --batch 2
```

跑完,结果 + 图 + 结论都在 `results/<时间戳>/`。**回传给作者:**
```bash
tar czf results_<时间戳>.tgz results/<时间戳>      # 不含权重,只有指标 JSON + 图 + 日志
```

---

## 2. 自动流程(一条命令背后做了什么)

| 步 | 脚本 | 做什么 |
|---|---|---|
| 1 | `tools/fetch_checkpoints.sh` | 下 `checkpoints/wedetect_tiny.pth`(公开 HF `fushh7/WeDetect`,走 hf-mirror,免 token)。已存在就跳过。 |
| 2 | `tools/fetch_dataset.sh` | 下 `data/TCT_NGC_1024/`(ModelScope `doilion/TCT_NGC_1024_TAR`,需 token)。数据集是 tar 包,**脚本自动 `sha256sum -c` 校验 + `tar -xf` 解压**到 `images/` 并核验布局 —— **无需手动 untar**。已存在就跳过。 |
| 3 | `train.py` | 逐个实验训练(fp32,DDP 多卡,LR 按卡数缩放)。已有最终 ckpt 就跳过训练。 |
| 4 | `test_exclude_negative.py --metric organ` | base 评测:器官-macro headline + **每类 AP** + **排除 6 个负类**(含 NS)。 |
| 5 | `tools/eval_novel_split.py` | novel 评测:9 类纯零样本。 |
| 6 | `tools/analyze_suite.py` | 自动出图 + `analysis.md`(判生死结论),写进结果包。 |

---

## 3. 实验清单(arm)

默认跑前 6 个(用 `--configs` 改)。每个 arm 都 train + base 评测 + novel 评测。

| arm | = 什么 | 作用 |
|---|---|---|
| `baseline` | 冻结 PSC 文本,无模块 | 总基线 |
| `attr_mean` | 6 属性**固定平均**分类 | 模块①的对照 |
| `attr` | 6 属性**区域自适应**加权 | **★模块①**;判生死=`attr − attr_mean` |
| `decone` | 训练免统计去锥 | 模块②的"光统计"对照 |
| `reldistill` | 视觉→文本关系蒸馏(无白化) | 模块②的"光蒸馏"对照 |
| `decone_reldistill` | 去锥 + 关系蒸馏 | **★模块②**;判生死=`− baseline (novel)` |
| `stitch`(选) | 模块①+② 晚融合 | 缝合 A |
| `stitchb`(选) | 模块①+② 共享文本 | 缝合 B |
| `attr_b5` / `p2` / `p2_nwd`(选) | 去锥+区域自适应 / 召回探索 | 备用 |

> 归因纪律:`attr`/`attr_mean` 用 morph6 文本,其余用 fullnames 文本 —— **只在同家族内比**
> (`attr − attr_mean`、`decone_reldistill − baseline`),别跨家族比。

---

## 4. 参数(CLI 优先,环境变量兜底)

```bash
bash tools/run_method_suite.sh [选项]
```

| CLI 选项 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| `--gpus 0,1,..` | `GPUS` | `0,1` | GPU 列表;卡数 = 逗号个数 |
| `--batch N` | `BATCH` | 配置值(@1024 为 8) | 每卡 batch |
| `--configs "a b"` | `CONFIGS` | 默认 6 个 arm | 选实验 |
| `--amp 0\|1` | `AMP` | `0`(fp32) | **别开 1**:@1024 混精会 nan |
| `--eval 0\|1` | `EVAL` | `1` | 训练后是否评测 |
| `--analyze 0\|1` | `ANALYZE` | `1` | 评测后是否自动出图 |
| `--port N` | `PORT` | `29610` | DDP 端口(并行多实例时**每个要不同**) |
| `--auto-lr 0\|1` | `AUTO_LR` | `1` | 是否按卡数线性缩放 lr |
| `--skip-fetch` | `SKIP_FETCH` | `0` | 跳过下权重+数据(本地已有) |
| `--list` | — | — | 干跑预演,不训练 |
| — | `MODELSCOPE_API_TOKEN` | — | **数据集私有库 token** |
| — | `TCT_NGC_1024_ROOT` | `data/TCT_NGC_1024` | 已有数据集就指过去 |
| — | `CKPT_MS_REPO` | (空) | 权重改走 ModelScope(默认走公开 HF) |
| — | `PYBIN` | `$HOME/anaconda3/envs/wedetect/bin/python` | wedetect 环境的 python |

**学习率自动缩放**(= mmengine auto_scale_lr):`lr = 3e-4 × 卡数 × batch ÷ 16`(调好的配方是
有效 batch 16 → lr 3e-4)。例:`8卡×batch2=eff16→3e-4`、`4卡×batch4=eff16→3e-4`、`8卡×batch8=eff64→1.2e-3`。

---

## 5. 8×H800 怎么用(两种)

```bash
# A) 推荐:保持调好的配方(有效 batch 16),用 8 卡换吞吐
bash tools/run_method_suite.sh --gpus 0,1,2,3,4,5,6,7 --batch 2    # eff16, lr3e-4

# B) 更快:开 4 个实例 ×2 卡,切分实验并行(每个 native eff16/lr3e-4,端口要不同)
export MODELSCOPE_API_TOKEN='<token>'
CONFIGS="baseline attr_mean"      GPUS=0,1 PORT=29610 bash tools/run_method_suite.sh &
CONFIGS="attr decone"             GPUS=2,3 PORT=29611 bash tools/run_method_suite.sh &
CONFIGS="reldistill"              GPUS=4,5 PORT=29612 bash tools/run_method_suite.sh &
CONFIGS="decone_reldistill"       GPUS=6,7 PORT=29613 bash tools/run_method_suite.sh &
wait
```

> ⚠️ 有效 batch 越大,AdamW 线性缩放越不稳;建议有效 batch ≤ ~32(否则给更长 warmup)。
> @1024 显存大概是 @640 的 ~2.6 倍,batch 8 若 OOM 就降 `--batch`。

---

## 6. 结果在哪 / 回传什么

```
results/<时间戳>/
├── summary.tsv                  # 每个 arm 的 base 器官-macro + novel mAP
├── <arm>/
│   ├── base_metrics.json        # base 器官-macro + 每类 AP(机器可读)
│   ├── novel_metrics.json       # novel 每类 AP + mAP
│   ├── train_curve.log          # loss 曲线点 + RelDistill 自检行
│   └── meta.txt                 # 配置/卡数/batch/lr/ckpt/git/机器
└── analysis/
    ├── headline_mAP.png         # 各 arm base+novel 柱状图
    ├── per_class_base.png       # 逐类 AP 热图
    ├── training_loss.png        # loss 曲线
    └── analysis.md              # 自动判生死结论 + caveat
```
**回传**:`tar czf results_<时间戳>.tgz results/<时间戳>`(**不含权重**,可放心传)。

---

## 7. 单独用法(一般不用,自动流程已包含)

```bash
# --- 单独下数据集(自动流程已含;这里是手动 / 调试 / 想先把数据放好用)---
export MODELSCOPE_API_TOKEN='<token>'        # 或先: ms login --token <token>
bash tools/fetch_dataset.sh                  # 推荐:幂等,自动校验 sha256 + 解压到 data/TCT_NGC_1024
#   └ 已有数据集就指过去:TCT_NGC_1024_ROOT=/your/TCT_NGC_1024 bash tools/fetch_dataset.sh(有就直接跳过)
#
# 等价的原始三步(仓库已自带下载脚本,所以平时只需上面那一行;留作参考/无 clone 时用):
#   ms login --token <token>
#   ms download doilion/TCT_NGC_1024_TAR scripts/download_tct_ngc_1024_modelscope_tar.sh --repo-type dataset --local-dir .
#   bash scripts/download_tct_ngc_1024_modelscope_tar.sh /path/to/TCT_NGC_1024
#
# 数据集内容:annotations/(train_dev / val_dev / test_base_clean_dev30 / test_novel_merged_9)
#            + 5 个器官图像 tar(Serous_effusion / TCT_CCD / Thyroid_gland / Urine / respiratory_tract)
#            + SHA256SUMS。base 训练/评测 + 9 类 novel 评测全都在这一个包里。

# --- 单独下权重 ---
bash tools/fetch_checkpoints.sh              # -> checkpoints/wedetect_tiny.pth(公开 HF fushh7/WeDetect,免 token)

# 拿到回传的结果包后,在本地重新出图
PYTHONPATH=. python tools/analyze_suite.py --results-dir results/<时间戳>

# 单独评测某个已训练 ckpt
PYTHONPATH=. python test_exclude_negative.py --config <cfg> --checkpoint <ckpt> --metric organ --metrics-out base.json
PYTHONPATH=. python tools/eval_novel_split.py --config <cfg> --checkpoint <ckpt> \
  --data-root "$TCT_NGC_1024_ROOT" --ann-file annotations/instances_test_novel_merged_9.json \
  --text-json data/texts/tct_ngc_novel_merged_9.json \
  --text-emb data/texts/tct_ngc_novel_merged_9_emb_biomedclip.pth \
  --work-dir wd/novel_eval --metrics-out novel.json \
  [--attr-text data/texts/tct_ngc_morph6_novel9_per_attr_biomedclip.pth]   # 属性 arm 才加
```

---

## 8. 排错 / 跑的时候盯什么

| 现象 | 处理 |
|---|---|
| 训练 OOM | 降 `--batch`(@1024 在 24G 卡上常用 batch 2~4) |
| loss 变 `nan` | 确认 `AMP=0`(默认就是);@1024 不要开 `--amp` |
| 数据集下载失败 | 确认 `MODELSCOPE_API_TOKEN` 已 export;或 `ms login`;或把已有数据集指给 `TCT_NGC_1024_ROOT` |
| 权重下载失败 | HF mirror 不通时设 `CKPT_MS_REPO=<你的 ModelScope 权重库>`,或手动放 `checkpoints/wedetect_tiny.pth` |
| **RelDistill 自检** | 训练 log 里 `[RelDistill diag @step100] ... gap(text-image)=` **必须为负**(图像比文本更判别才对);为正说明 teacher 在帮倒忙,告诉作者 |
| 判生死都 ≈0(±0.005) | 模块没起作用(wash);把 summary.tsv + analysis/ 回传给作者判 |

---

*入口脚本顶部 `bash tools/run_method_suite.sh --help` 有同样的速查。*
