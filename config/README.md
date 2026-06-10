# config/ 索引

本目录 **31 个** `.py` 配置。**文件未做物理移动**——原因见底部「维护须知」。

**图例**:`◆`=intermediate(被其它 config 继承,改名/移动会断子配置);`·`=leaf(改动安全)。
状态:**RUNNING**=正在训练 / ACTIVE=当前线可用 / LEGACY=旧线,仅复现保留。

> 2026-06-09 两轮清理:删了 dev32 旧线、DINOv3 / BiomedCLIP-ViT 骨干实验、res1280、独立 m1_nwd、
> organdisc_regiongap 合并配置、@640 均值对照(morph6mean/5attrmean)、@640 organdisc/regiongap、
> XLM-R ICF、reldistill 变体、6 个 ICF ablation 变体。**变体结果都在
> `docs/results/clean_retrain_summary.xlsx`,配置 git 可恢复**——只去 sprawl,不丢排除法证据。

## `_base_` 继承树(→ = 「被继承出」)
```
default_runtime.py
├─ wedetect_tiny / base / large.py                          (原版 WeDetect)
└─ dev30_2gpu → fullnames_1gpu → cache640_fullnames_disjoint_2gpu → ..._disjoint_clean_2gpu
     ├─ ochmta_m1_xlmr_2gpu → xlmr_res1024 → xlmr_p2_res1024     (§3 LEGACY)
     └─ biomedclip_noTHAF_2gpu → ochmta_m1_biomedclip_2gpu  ★Row1 baseline★
          ├─ res1024_biomedclip_2gpu  ★@1024 baseline★
          │    ├─ attr_b4_morph6 (B2) → {attr_b4_5attr, attr_b4_morph6_novel,
          │    │                         attr_b5_morph6 (B3) → {attr_whitenmean (B1),
          │    │                                                attr_b5_dv2 (B3+D), attr_b5_morph6_novel},
          │    │                         attr_mean_morph6 (B0)}
          │    └─ dv2 ; whiten ; p2 → p2_nwd ; organdisc_v1_res1024 ; regiongap_v1_res1024
          ├─ icf_biomedclip → icf_morph6                          (§3 诊断)
          └─ reldistill                                           (§3 诊断)
```

## ① Base Chain(dev30/@1024 继承脊梁,必保留)
| 文件 | 用途 | _base_ | 状态 |
|---|---|---|---|
| `default_runtime.py` | MMEngine 运行时默认 | — | ◆ ACTIVE |
| `wedetect_{tiny,base,large}.py` | 原版 WeDetect(80训/1203测) | default_runtime | · ACTIVE |
| `..._dev30_2gpu.py` | dev30 根(30 类) | default_runtime | ◆ ACTIVE |
| `..._dev30_fullnames_1gpu.py` | 缓存全名文本基线 | dev30_2gpu | ◆ ACTIVE |
| `..._dev30_cache640_fullnames_disjoint_2gpu.py` | 640 缓存 + 患者 disjoint | fullnames_1gpu | ◆ ACTIVE |
| `..._dev30_cache640_fullnames_disjoint_clean_2gpu.py` | 干净重训 | disjoint_2gpu | ◆ ACTIVE |
| `..._dev30_biomedclip_noTHAF_2gpu.py` | 1-PSC BiomedCLIP 文本基线 | disjoint_clean_2gpu | ◆ ACTIVE |
| `..._dev30_ochmta_m1_biomedclip_2gpu.py` | **★Row1**(M1 器官 mask + 冻结 BiomedCLIP) | biomedclip_noTHAF | ◆ ACTIVE |

## ② Current Method Configs(论文主线 @1024)
**De-collapse 矩阵**(Deconed Attribute-Adaptive Classifier;判据见
`docs/method_design/deconed_attr_adaptive_experiment_plan_20260609.md`):
| 文件 | arm | adaptive / whiten / dv2 | _base_ | 状态 |
|---|---|---|---|---|
| `..._res1024_biomedclip_2gpu.py` | **@1024 baseline**(PSC 通用) | – | ochmta_m1_biomedclip | ◆ RUNNING(GPU0,1) |
| `..._attr_mean_morph6_..._2gpu.py` | **B0** 原始均值 | ✗ / ✗ / ✗ | attr_b4_morph6 | · ACTIVE |
| `..._attr_whitenmean_morph6_..._2gpu.py` | **B1** 白化均值 | ✗ / ✓ / ✗ | attr_b5_morph6 | · ACTIVE |
| `..._attr_b4_morph6_..._2gpu.py` | **B2** 自适应 | ✓ / ✗ / ✗ | res1024 | ◆ RUNNING(GPU2,3) |
| `..._attr_b5_morph6_..._2gpu.py` | **B3** 白化+自适应 | ✓ / ✓ / ✗ | attr_b4_morph6 | ◆ ACTIVE |
| `..._attr_b5_dv2_morph6_..._2gpu.py` | **B3+D** 全模型 | ✓ / ✓ / ✓ | attr_b5_morph6 | · ACTIVE |
| `..._attr_b4_morph6_novel_..._2gpu.py` | B2 novel(39类) | ✓ / ✗ / ✗ | attr_b4_morph6 | · ACTIVE |
| `..._attr_b5_morph6_novel_..._2gpu.py` | B3 novel(39类,transductive 对照) | ✓ / ✓ / ✗ | attr_b4_morph6_novel | · ACTIVE |
| `..._attr_b4_5attr_..._2gpu.py` | B2 的 5 属性变体(次要) | ✓ / ✗ / ✗ | attr_b4_morph6 | · ACTIVE |
| `..._dv2_biomedclip_2gpu.py` | **D** 视觉原型接地(VisualPrototypeAnchor) | – | res1024 | · ACTIVE |
| `..._whiten_biomedclip_2gpu.py` | 文本去各向异性(standalone) | – | res1024 | · ACTIVE |

**正交召回**(进表不进标题):
| `..._p2_biomedclip_2gpu.py` | stride-4 P2 小细胞召回 | res1024 | ◆ ACTIVE |
| `..._p2_nwd_biomedclip_2gpu.py` | P2 + NWD tiny-object 分配 | p2 | · ACTIVE |

**@1024 器官方法 + XLM-R baseline**:
| `..._organdisc_v1_res1024_..._2gpu.py` | 器官内 hard-neg 判别 loss | res1024 | · ACTIVE |
| `..._regiongap_v1_res1024_..._2gpu.py` | region-gap 蒸馏(冻结 BiomedCLIP teacher) | res1024 | · ACTIVE |
| `..._ochmta_m1_xlmr_2gpu.py` | XLM-R 文本线根 | disjoint_clean_2gpu | ◆ LEGACY |
| `..._xlmr_res1024_2gpu.py` / `..._xlmr_p2_res1024_2gpu.py` | XLM-R @1024 (+P2) | xlmr_2gpu / xlmr_res1024 | LEGACY |

## ③ Diagnostic / Retained(消融·排除法证据;变体已删,结果见 xlsx)
| 文件 | 保留原因 | _base_ | 状态 |
|---|---|---|---|
| `..._icf_biomedclip_2gpu.py` | ICF 基线(Image-Conditional Fusion 树根) | ochmta_m1_biomedclip | ◆ ACTIVE |
| `..._icf_morph6_biomedclip_2gpu.py` | **唯一真 work 的 ICF**(6 形态属性,fused_cos 0.877) | icf_biomedclip | · ACTIVE |
| `..._ochmta_m1_reldistill_biomedclip_2gpu.py` | 关系蒸馏 D(gauge-freedom motivation) | ochmta_m1_biomedclip | · ACTIVE |

## ⚠️ 维护须知(为什么不分子目录)
1. **`_base_` 是相对路径**(`./X.py`)。移动/改名任一文件断继承链——尤其 `◆`
   (`ochmta_m1_biomedclip`、`res1024_biomedclip`、`attr_b4_morph6`、`attr_b5_morph6`、`icf_biomedclip` 有子)。
2. **被多处硬编码引用**:`train.py`/`test*.py`/`dist_*.sh`、`tools/run_method_suite.sh`、
   `test_novel.py`/`test_exclude_negative.py` 默认。改名/移动前必须同步全部 + 子配置 `_base_`,再
   `Config.fromfile` 逐个验证。
