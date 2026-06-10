# `data/texts/` — 文本嵌入与视觉原型缓存

存放所有 prompt JSON、文本 embedding `.pth`、视觉 prototype `.pth`、以及实验中间产物。
按文件名前缀和 suffix 一眼能分清用途。

## 1. 跨数据集（不动）

| 文件 | 用途 |
|---|---|
| `coco_class_texts.json` / `coco_zh_class_texts.json` | COCO 80 类 prompt（中英）|
| `lvis_v1_class_texts.json` / `lvis_v1_zh_class_texts.json` | LVIS 1203 类 prompt |

## 2. dev30 baseline（**configs 引用，别动**）

| 文件 | 用途 |
|---|---|
| `tct_ngc_fullnames_30.json` | dev30 的 30 个 base 类 prompts（**config inputs**）|
| `tct_ngc_fullnames_30_embeddings_wedetect_tiny.pth` | 上述的 XLM-Roberta embedding 缓存（**config inputs**）|

## 3. dev32 baseline（**configs 引用，别动**）

| 文件 | 用途 |
|---|---|
| `tct_ngc_fullnames_32.json` | dev32 的 32 个 base 类 prompts（dev32 chain configs 引用）|
| `tct_ngc_fullnames_32_embeddings_wedetect_tiny.pth` | dev32 文本 embedding |
| `tct_ngc_fullnames_32_random_embeddings_seed20260506.pth` | random text 消融实验（dev32 randomemb config 引用）|

## 4. Phase 2 输入（structured attribute training）

| 文件 | 用途 |
|---|---|
| `tct_ngc_fullnames_39_attr.json` | 30 base + 9 novel 类的 5 属性 JSON（病理教科书 grounding；item 19 hierarchical training 输入）|

## 5. Novel zero-shot 评估资产（4 splits × 多种来源）

splits：`main_3` (3 类) / `pseudo_2` (2 类) / `hard_4` (4 类) / `full_5` (5 类)

### 5.1 prompts + 文本 embedding（v2 baseline）

| Pattern | 用途 |
|---|---|
| `tct_ngc_novel_<split>.json` | novel 类 v2 国际标准 prompts (PSC/MAL-S/Bethesda)|
| `tct_ngc_novel_<split>_emb.pth` | 上述 XLM-Roberta 编码后的缓存（v2 baseline）|

### 5.2 视觉 prototype（visproto，5-shot 版）

| Pattern | 用途 |
|---|---|
| `tct_ngc_novel_<split>_visproto_emb.pth` | **直接** visproto（从 test set 5 张 GT crop 提，⚠ 有 leakage）|
| `tct_ngc_novel_<split>_visproto_emb.holdout_anns.json` | 上述用过的 ann_id 列表（strict ZS 评估时排除）|
| `tct_ngc_novel_<split>_visproto_calibrated_emb.pth` | **Procrustes 对齐**后的 visproto（落在文本几何空间，可与 text 共存）|

### 5.3 dual-anchor fusion（实验中间产物）

| Pattern | 用途 |
|---|---|
| `tct_ngc_novel_<split>_calfused_emb.pth` | calibrated 二元路由融合（Serous→text，Resp/Thyroid→calibrated visproto）—— Phase 1.4 实验产物 |

### 5.4 Procrustes 基础设施

| 文件 | 用途 |
|---|---|
| `tct_ngc_base30_visproto_train.pth` | 30 base 类的 visproto（**从训练集**提，无 leakage）—— Procrustes anchor pairs 之一 |
| `tct_ngc_base30_visproto_train.holdout_anns.json` | 上述用过的 ann_id（base 类 train set，可忽略）|
| `procrustes_R.pth` | 768×768 正交旋转矩阵 R（解：R · vis_base ≈ text_base）+ common_keys 列表 |

## 6. `_archive/`（实验失败 / 已废弃，不要再用）

| Pattern | 死亡时间 | 死因 |
|---|---|---|
| `tct_ngc_novel_<split>_fused_emb.pth` | 2026-05-09 | **DEAD-3** 几何不匹配 binary fusion，每个 split 把 text-routed 类 mAP 拉到 0.000。计划文档 `docs/tct_ngc_novel_zero_shot_review_20260509.md` §2.3 记录失败原因。kept for ablation history only. |

## 命名约定速查表

| Suffix | 含义 |
|---|---|
| `.json` | prompt 文本 JSON（list-of-list 格式：每个 inner list = 一个类的 variants）|
| `_emb.pth` | XLM-Roberta 文本 embedding 缓存（dict: prompt str → 768-dim tensor）|
| `_visproto_emb.pth` | 图像 encoder 提取的视觉 prototype（dict: prompt str → 768-dim tensor）|
| `_visproto_calibrated_emb.pth` | Procrustes 旋转到文本空间的 visproto |
| `_fused_emb.pth` | binary 类路由融合（DEAD-3，归档）|
| `_calfused_emb.pth` | calibrated 类路由融合（current best dual-anchor）|
| `_holdout_anns.json` | strict zero-shot 评估排除的 ann_id 列表 |
| `_random_embeddings_seed*.pth` | 随机 text embedding 消融（控制实验）|

## 重要文件相互依赖

```
fullnames_30.json + fullnames_30_emb.pth   ←→  PseudoLanguageBackbone (dev30 config)
fullnames_30_emb.pth + base30_visproto_train.pth  →  procrustes_R.pth
procrustes_R.pth × novel_<split>_visproto_emb  →  novel_<split>_visproto_calibrated_emb
novel_<split>_emb (text) + novel_<split>_visproto_calibrated_emb  →  novel_<split>_calfused_emb
```

## 重生成命令（如果某个文件丢了）

```bash
# Procrustes anchor (base 30 类 visproto from training set)
python tools/build_visual_prototype.py \
  --config config/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_2gpu.py \
  --checkpoint work_dirs/.../best_coco_bbox_mAP_epoch_9.pth \
  --ann-file annotations/instances_train_dev_disjoint_dev30.json \
  --data-root "${TCT_NGC_640_ROOT:-data/TCT_NGC_640}/" --img-prefix images/ \
  --text-json data/texts/tct_ngc_fullnames_30.json \
  --out data/texts/tct_ngc_base30_visproto_train.pth --n-per-class 5

# Procrustes 矩阵 + 4 个 calibrated novel
python tools/procrustes_text_visual.py \
  --vis-proto-novel data/texts/tct_ngc_novel_{main_3,pseudo_2,hard_4,full_5}_visproto_emb.pth
```
