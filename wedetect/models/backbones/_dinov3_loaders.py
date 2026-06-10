"""HF safetensors -> wedetect / timm key mappers for DINOv3 weights.

Used by:
  - ConvNextVisionBackbone(pretrained_weights=...)  (Stage A)
      via hf_to_wedetect_convnext: DINOv3-ConvNeXt-Tiny HF -> wedetect ConvNeXt key namespace.
      Drop-in load DINOv3 SSL weights instead of supervised ImageNet ones; zero arch changes.
  - DINOv3VisionBackbone  (Stage B)
      via hf_to_timm_vit_dinov3: facebook/dinov3-vit* HF -> timm vit_*_patch16_dinov3 namespace.
      Identical logic to tools/dinov3_smoke.py (verified missing=0/unexpected=0 on ViT-S/B/L).

Both return (state_dict, list_of_warnings). Mapping rules verified against actual key dumps:
  - HF DINOv3 ConvNeXt-Tiny: 180 keys, shape 100% matches wedetect ConvNeXt(depths=[3,3,9,3],
    dims=[96,192,384,768]); only key naming differs.
  - HF DINOv3 ViT-S/B/L: 211/something/something keys; differs from timm by q/k/v split
    (HF separate -> timm concat into qkv).
"""
import re
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch


def hf_to_wedetect_convnext(
    hf_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """HF DINOv3 ConvNeXt-Tiny -> wedetect ConvNeXt key namespace.

    Mapping table (HF -> wedetect):
      stages.{i}.downsample_layers.{j}.* -> model.downsample_layers.{i}.{j}.*
        (stem i=0:  j=0 Conv(3->96,4x4), j=1 LayerNorm(96)
         downs i>0: j=0 LayerNorm,        j=1 Conv(stride 2))
      stages.{i}.layers.{j}.depthwise_conv.*   -> model.stages.{i}.{j}.dwconv.*
      stages.{i}.layers.{j}.pointwise_conv1.*  -> model.stages.{i}.{j}.pwconv1.*
      stages.{i}.layers.{j}.pointwise_conv2.*  -> model.stages.{i}.{j}.pwconv2.*
      stages.{i}.layers.{j}.layer_norm.*       -> model.stages.{i}.{j}.norm.*
      stages.{i}.layers.{j}.gamma              -> model.stages.{i}.{j}.gamma  (unchanged)
      layer_norm.*                              -> model.norm.*  (final norm)

    Not covered by HF (left at init):
      model.head.{weight,bias}   wedetect classifier head; forward() doesn't use it.
    """
    out: Dict[str, torch.Tensor] = OrderedDict()
    warns: List[str] = []

    for k, v in hf_state.items():
        # final norm (top-level layer_norm.*)
        if k.startswith("layer_norm."):
            out["model.norm." + k.split(".", 1)[1]] = v
            continue

        # downsample layers (stem + 3 inter-stage downsamples)
        m = re.match(r"stages\.(\d+)\.downsample_layers\.(\d+)\.(.+)", k)
        if m:
            i, j, sub = m.group(1), m.group(2), m.group(3)
            out[f"model.downsample_layers.{i}.{j}.{sub}"] = v
            continue

        # block layers
        m = re.match(r"stages\.(\d+)\.layers\.(\d+)\.(.+)", k)
        if m:
            i, j, sub = m.group(1), m.group(2), m.group(3)
            sub_remapped = (
                sub.replace("depthwise_conv", "dwconv")
                .replace("pointwise_conv1", "pwconv1")
                .replace("pointwise_conv2", "pwconv2")
                .replace("layer_norm", "norm")
            )
            out[f"model.stages.{i}.{j}.{sub_remapped}"] = v
            continue

        warns.append(f"unmapped HF key: {k}")
    return out, warns


def hf_to_timm_vit_dinov3(
    hf_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """facebook/dinov3-vit* HF safetensors -> timm vit_*_patch16_dinov3 namespace.

    Mapping table (HF -> timm):
      embeddings.cls_token                  -> cls_token
      embeddings.register_tokens            -> reg_token
      embeddings.mask_token                 -> (dropped; MIM pretraining only)
      embeddings.patch_embeddings.{w,b}     -> patch_embed.proj.{w,b}
      layer.{i}.norm{1,2}.{w,b}             -> blocks.{i}.norm{1,2}.{w,b}
      layer.{i}.attention.{q,k,v}_proj.{w,b} (3 separate) -> blocks.{i}.attn.qkv.{w,b} (concat dim 0)
      layer.{i}.attention.o_proj.{w,b}      -> blocks.{i}.attn.proj.{w,b}
      layer.{i}.layer_scale{1,2}.lambda1    -> blocks.{i}.gamma_{1,2}
      layer.{i}.mlp.up_proj.{w,b}           -> blocks.{i}.mlp.fc1.{w,b}
      layer.{i}.mlp.down_proj.{w,b}         -> blocks.{i}.mlp.fc2.{w,b}
      norm.{w,b}                             -> norm.{w,b}  (unchanged)
    """
    out: Dict[str, torch.Tensor] = OrderedDict()
    qkv_parts: Dict[str, Dict[str, torch.Tensor]] = {}
    warns: List[str] = []

    for k, v in hf_state.items():
        if k == "embeddings.cls_token":
            out["cls_token"] = v
        elif k == "embeddings.register_tokens":
            out["reg_token"] = v
        elif k == "embeddings.mask_token":
            continue  # MIM only
        elif k == "embeddings.patch_embeddings.weight":
            out["patch_embed.proj.weight"] = v
        elif k == "embeddings.patch_embeddings.bias":
            out["patch_embed.proj.bias"] = v
        elif k in ("norm.weight", "norm.bias"):
            out[k] = v
        else:
            m = re.match(r"layer\.(\d+)\.(.+)", k)
            if not m:
                warns.append(f"unmapped HF key: {k}")
                continue
            i, sub = m.group(1), m.group(2)
            prefix = f"blocks.{i}."
            if sub.startswith("norm1.") or sub.startswith("norm2."):
                out[prefix + sub] = v
            elif sub == "layer_scale1.lambda1":
                out[prefix + "gamma_1"] = v
            elif sub == "layer_scale2.lambda1":
                out[prefix + "gamma_2"] = v
            elif sub == "mlp.up_proj.weight":
                out[prefix + "mlp.fc1.weight"] = v
            elif sub == "mlp.up_proj.bias":
                out[prefix + "mlp.fc1.bias"] = v
            elif sub == "mlp.down_proj.weight":
                out[prefix + "mlp.fc2.weight"] = v
            elif sub == "mlp.down_proj.bias":
                out[prefix + "mlp.fc2.bias"] = v
            elif sub == "attention.o_proj.weight":
                out[prefix + "attn.proj.weight"] = v
            elif sub == "attention.o_proj.bias":
                out[prefix + "attn.proj.bias"] = v
            elif sub == "attention.q_proj.weight":
                qkv_parts.setdefault(i, {})["qw"] = v
            elif sub == "attention.q_proj.bias":
                qkv_parts.setdefault(i, {})["qb"] = v
            elif sub == "attention.k_proj.weight":
                qkv_parts.setdefault(i, {})["kw"] = v
            elif sub == "attention.k_proj.bias":
                qkv_parts.setdefault(i, {})["kb"] = v
            elif sub == "attention.v_proj.weight":
                qkv_parts.setdefault(i, {})["vw"] = v
            elif sub == "attention.v_proj.bias":
                qkv_parts.setdefault(i, {})["vb"] = v
            else:
                warns.append(f"unmapped HF sub-key: layer.{i}.{sub}")

    for i, parts in qkv_parts.items():
        prefix = f"blocks.{i}."
        # QKV WEIGHTS: require all three (DINOv3 ViT always has q/k/v weights);
        # missing any is a real bug worth surfacing.
        if {"qw", "kw", "vw"} <= parts.keys():
            out[prefix + "attn.qkv.weight"] = torch.cat(
                [parts["qw"], parts["kw"], parts["vw"]], dim=0
            )
        else:
            warns.append(
                f"layer {i}: incomplete q/k/v weights "
                f"(have {sorted(parts.keys() & {'qw','kw','vw'})}); "
                f"qkv.weight NOT emitted"
            )
        # QKV BIASES: DINOv3 ViT-S/B/L published on HF have q_proj.bias and
        # v_proj.bias but NO k_proj.bias (key bias is fixed at 0 in the upstream
        # arch). The original gate `{qb,kb,vb} <= parts.keys()` required all
        # three and therefore silently dropped the q/v biases. timm's default
        # vit_*_patch16_dinov3 has qkv_bias=False (no qkv.bias slot, so even an
        # emitted qkv.bias would land as unexpected) -- but the `_qkvb` timm
        # variants DO expect a qkv.bias and would silently keep zero biases
        # without this fix. Pad the missing component with zeros so the SSL
        # q/v biases are actually transferred when the timm side accepts them.
        present_b = parts.keys() & {"qb", "kb", "vb"}
        if present_b:
            sample = parts[next(iter(present_b))]
            zero = torch.zeros_like(sample)
            qb = parts.get("qb", zero)
            kb = parts.get("kb", zero)
            vb = parts.get("vb", zero)
            out[prefix + "attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)
            missing = {"qb", "kb", "vb"} - present_b
            if missing:
                warns.append(
                    f"layer {i}: padded missing {sorted(missing)} with zeros "
                    f"when emitting qkv.bias"
                )
    return out, warns
