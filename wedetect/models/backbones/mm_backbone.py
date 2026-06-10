# Copyright (c) Tencent Inc. All rights reserved.
import itertools
import logging
from typing import List, Optional, Sequence, Tuple
import torch
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmdet.utils import OptMultiConfig, ConfigType
from ._freeze_utils import freeze_submodules
from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPTextConfig,
    CLIPVisionModelWithProjection,
)
from transformers import CLIPTextModelWithProjection as CLIPTP
from transformers import AutoConfig, XLMRobertaModel
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import os.path as osp
import collections
import math
from collections import OrderedDict



@MODELS.register_module()
class HuggingCLIPVisionBackbone(BaseModule):

    def __init__(
        self,
        model_name: str,
        frozen_modules: Sequence[str] = (),
        dropout: float = 0.0,
        training_use_cache: bool = False,
        init_cfg: OptMultiConfig = None,
    ) -> None:

        super().__init__(init_cfg=init_cfg)

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        self._freeze_modules()

    def forward(self, image: Tensor) -> Tuple[Tensor]:

        img_feats = self.model(image, output_hidden_states=True)
        img_feats = img_feats["last_hidden_state"][:, 0, :]

        return img_feats

    def _freeze_modules(self):
        freeze_submodules(self.model, self.frozen_modules)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()





class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
    ):
        super().__init__()

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

        self.avgpool = nn.AvgPool2d(7, stride=1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, model_name):

        outputs = []
        c1 = self.downsample_layers[0](x)
        c1 = self.stages[0](c1)
        outputs.append(c1)

        c2 = self.downsample_layers[1](c1)
        c2 = self.stages[1](c2)
        outputs.append(c2)

        c3 = self.downsample_layers[2](c2)
        c3 = self.stages[2](c3)
        outputs.append(c3)

        c4 = self.downsample_layers[3](c3)
        c4 = self.stages[3](c4)
        outputs.append(c4)

        if model_name == 'xlarge':
            return c2, c3, c4
        else:
            return c1, c2, c3, c4
        


@MODELS.register_module()
class ConvNextVisionBackbone(BaseModule):

    def __init__(
        self,
        model_name: str,
        out_indices: Sequence[int] = (0, 1, 2),
        norm_eval: bool = True,
        frozen_modules: Sequence[str] = (),
        pretrained_weights: Optional[str] = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:

        super().__init__(init_cfg=init_cfg)

        self.norm_eval = norm_eval
        self.frozen_modules = frozen_modules
        self.model_name = model_name

        # 新增定义降维
        if self.model_name == "xlarge":
            self.down_mlp = nn.Conv2d(2048, 1024, kernel_size=1)

        if self.model_name == "xlarge":
            self.model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048])
        if self.model_name == "base":
            self.model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
        if self.model_name == "large":
            self.model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
        if self.model_name == "tiny":
            self.model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])

        if pretrained_weights is not None:
            self._load_dinov3_convnext_weights(pretrained_weights)

        self._freeze_modules()

    def _load_dinov3_convnext_weights(self, pretrained_path: str) -> None:
        """Load DINOv3-style HF safetensors into the underlying ConvNeXt.

        Currently only the 'tiny' variant matches a published HF DINOv3 ConvNeXt
        release (facebook/dinov3-convnext-tiny-pretrain-lvd1689m).

        Caveat: pairing pretrained_weights with a non-'tiny' model_name will trigger
        a shape-mismatch RuntimeError inside load_state_dict (strict=False suppresses
        only missing/unexpected, not shape mismatches). That loud failure is the safe
        outcome; silent partial copy cannot happen here.
        """
        from safetensors.torch import load_file
        from wedetect.models.backbones._dinov3_loaders import hf_to_wedetect_convnext

        if not pretrained_path.endswith(".safetensors"):
            raise ValueError(
                f"pretrained_weights must be a .safetensors path, got {pretrained_path}"
            )
        hf_state = load_file(pretrained_path)
        mapped, warns = hf_to_wedetect_convnext(hf_state)
        # mapper emits wedetect-side keys (prefixed with 'model.'), so load at
        # the backbone level, NOT self.model.
        result = self.load_state_dict(mapped, strict=False)

        n_loaded = len(mapped)
        n_missing = len(result.missing_keys)
        n_unexpected = len(result.unexpected_keys)
        head_hint = " (expected model.head.weight/bias)" if self.model_name == "tiny" else ""
        print_log(
            f"[ConvNextVisionBackbone:{self.model_name}] loaded {n_loaded} keys "
            f"from {pretrained_path}; missing={n_missing}{head_hint}, "
            f"unexpected={n_unexpected}; mapper_warns={len(warns)}",
            logger="current",
        )
        # Sanity gate: tiny should have exactly 2 missing (head.{w,b}) and 0 unexpected.
        if n_unexpected > 0 or (self.model_name == "tiny" and n_missing > 2):
            print_log(
                f"  missing[:5]={result.missing_keys[:5]}  "
                f"unexpected[:5]={result.unexpected_keys[:5]}",
                logger="current",
                level=logging.WARNING,
            )

    def forward(self, image: Tensor) -> Tuple[Tensor]:
        if self.model_name == "xlarge":
            img_feats_1, img_feats_2, img_feats_3 = self.model(image, self.model_name)
            img_feats_3 = self.down_mlp(img_feats_3)
            img_feats = [img_feats_1, img_feats_2, img_feats_3]
            return tuple(img_feats)
        else:
            img_feats_0, img_feats_1, img_feats_2, img_feats_3 = self.model(image, self.model_name)
            img_feats = [img_feats_0, img_feats_1, img_feats_2, img_feats_3]
            return tuple(img_feats)

    # def forward(self, image: Tensor) -> Tuple[Tensor]:
    #     img_feats_1, img_feats_2, img_feats_3 = self.model(image)
    #     img_feats_3 = self.down_mlp(img_feats_3)
    #     img_feats = [img_feats_1, img_feats_2, img_feats_3]
    #     return tuple(img_feats)


   
    def _freeze_modules(self):
        freeze_submodules(self.model, self.frozen_modules)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@MODELS.register_module()
class XLMRobertaLanguageBackbone(BaseModule):

    def __init__(
        self,
        model_name: str ,
        model_size: str ,
        frozen_modules: Sequence[str] = (),
        dropout: float = 0.0,
        training_use_cache: bool = False,
        init_cfg: OptMultiConfig = None,
    ) -> None:

        super().__init__(init_cfg=init_cfg)

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = XLMRobertaModel(AutoConfig.from_pretrained(model_name))
        if model_size == 'base' or model_size == 'tiny':
            self.head = nn.Linear(768, 768, bias=True) # Tiny Base
        elif model_size == 'large':  
            self.head = nn.Linear(1024, 768, bias=True) # Large
        elif model_size == 'xlarge':
            self.head = nn.Linear(1024, 1024, bias=True) # XLarge

        self._freeze_modules()

    def forward_tokenizer(self, texts):
        if not hasattr(self, "text"):
            text = list(itertools.chain(*texts))
            text = self.tokenizer(text=text, return_tensors="pt", padding=True)
            self.text = text.to(device=self.model.device)
        return self.text

    def forward(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(
            num_per_batch
        ), "number of sequences not equal in batch"
        text = list(itertools.chain(*text))
        text = self.tokenizer(text=text, return_tensors="pt", padding=True)
        text = text.to(device=self.model.device)

        txt_feats = self.model(**text)["last_hidden_state"][:, 0]
        txt_feats = self.head(txt_feats)
        txt_feats = F.normalize(txt_feats, dim=-1)
        txt_feats = txt_feats.reshape(-1, num_per_batch[0], txt_feats.shape[-1])
    
        return txt_feats


    def _freeze_modules(self):

        if len(self.frozen_modules) == 0:
            # not freeze
            return
        if self.frozen_modules[0] == "all":
            self.model.eval()
            for _, module in self.model.named_modules():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            # head 也需要freeze
            self.head.eval()
            for _, module in self.head.named_modules():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            return
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()


@MODELS.register_module()
class HuggingVisionBackbone(BaseModule):

    def __init__(
        self,
        model_name: str,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        norm_eval: bool = True,
        frozen_modules: Sequence[str] = (),
        init_cfg: OptMultiConfig = None,
    ) -> None:

        super().__init__(init_cfg=init_cfg)

        self.norm_eval = norm_eval
        self.frozen_modules = frozen_modules
        self.model = AutoModel.from_pretrained(model_name)

        self._freeze_modules()

    def forward(self, image: Tensor) -> Tuple[Tensor]:
        encoded_dict = self.image_model(pixel_values=image, output_hidden_states=True)
        hidden_states = encoded_dict.hidden_states
        img_feats = encoded_dict.get("reshaped_hidden_states", hidden_states)
        img_feats = [img_feats[i] for i in self.image_out_indices]
        return tuple(img_feats)

    def _freeze_modules(self):
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()



@MODELS.register_module()
class HuggingCLIPLanguageBackbone(BaseModule):

    def __init__(
        self,
        model_name: str,
        frozen_modules: Sequence[str] = (),
        dropout: float = 0.0,
        training_use_cache: bool = False,
        init_cfg: OptMultiConfig = None,
    ) -> None:

        super().__init__(init_cfg=init_cfg)

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_config = CLIPTextConfig.from_pretrained(
            model_name, attention_dropout=dropout
        )
        self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
        self._freeze_modules()

    def forward_tokenizer(self, texts):
        if not hasattr(self, "text"):
            text = list(itertools.chain(*texts))
            text = self.tokenizer(text=text, return_tensors="pt", padding=True)
            self.text = text.to(device=self.model.device)
        return self.text

    def forward(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(
            num_per_batch
        ), "number of sequences not equal in batch"
        text = list(itertools.chain(*text))
        text = self.tokenizer(text=text, return_tensors="pt", padding=True)
        text = text.to(device=self.model.device)
        txt_outputs = self.model(**text)
        txt_feats = txt_outputs.text_embeds
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        txt_feats = txt_feats.reshape(-1, num_per_batch[0], txt_feats.shape[-1])
        return txt_feats

    def _freeze_modules(self):
        freeze_submodules(self.model, self.frozen_modules)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()


@MODELS.register_module()
class PseudoLanguageBackbone(BaseModule):
    """Pseudo Language Backbone
    Args:
        text_embed_path (str): path to the text embedding file
    """

    def __init__(
        self,
        text_embed_path: str = "",
        test_embed_path: str = None,
        text_adapter: Optional[ConfigType] = None,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(init_cfg)
        # {text:embed}
        self.text_embed = torch.load(text_embed_path, map_location="cpu")
        if test_embed_path is None:
            self.test_embed = self.text_embed
        else:
            self.test_embed = torch.load(test_embed_path)
        # Innovation D (2026-06-05): optional learnable residual adapter g(.) on the
        # frozen cached text. None (default) -> behaves exactly as before. The cached
        # vectors stay frozen (requires_grad_(False) in forward_text); only g trains,
        # supervised by RelationalDistillationLoss + the main cls loss.
        if text_adapter is not None:
            self.text_adapter = MODELS.build(text_adapter)
        else:
            self.text_adapter = None
        self.register_buffer(
            "buff",
            torch.zeros(
                [
                    1,
                ]
            ),
        )

    def forward_cache(self, text: List[List[str]]) -> Tensor:
        # Innovation D: a learnable text_adapter g(.) keeps training across
        # epochs, so any text vector cached at an earlier eval is STALE — val
        # metrics / save_best would score an out-of-date adapter. The adapter is
        # a cheap residual MLP, so just recompute each eval (no cache at all).
        if self.text_adapter is not None:
            return self.forward_text(text)

        cache_key = tuple(text[0])
        if any(tuple(t) != cache_key for t in text):
            return self.forward_text(text)

        if not hasattr(self, "cache_bank"):
            self.cache_bank = {}
        if cache_key not in self.cache_bank:
            # Cache one copy and expand to the current eval batch size. The
            # final validation batch can be smaller than the first one.
            self.cache_bank[cache_key] = self.forward_text([list(cache_key)]).detach()

        # expand returns a non-contiguous broadcast view; safe because the
        # contrastive head only reads (cosine matmul). Do not call .view()
        # on the result without .contiguous() first.
        return self.cache_bank[cache_key].expand(len(text), -1, -1)

    def forward(self, text: List[List[str]]) -> Tensor:
        if self.training:
            return self.forward_text(text)
        else:
            return self.forward_cache(text)

    def forward_text(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(
            num_per_batch
        ), "number of sequences not equal in batch"
        text = list(itertools.chain(*text))
        if self.training:
            text_embed_dict = self.text_embed
        else:
            text_embed_dict = self.test_embed
        text_embeds = torch.stack([text_embed_dict[x.split("/")[0]] for x in text])
        # requires no grad and force to float
        text_embeds = text_embeds.to(self.buff.device).requires_grad_(False).float()
        # Innovation D: learnable residual adapter g(.) on the frozen vectors.
        # forward_cache() routes through here too, so train + eval both adapt.
        if self.text_adapter is not None:
            text_embeds = self.text_adapter(text_embeds)
        text_embeds = text_embeds.reshape(-1, num_per_batch[0], text_embeds.shape[-1])
        return text_embeds


@MODELS.register_module()
class PseudoMultiAttrLanguageBackbone(BaseModule):
    """OC-HMTA Module 2 text backbone: 5-attribute cache + hierarchical adapter.

    Replaces the 1-PSC PseudoLanguageBackbone for Module 2 training/eval.
    Looks up 5 attribute embeddings per class from a prompt-keyed cache,
    builds [B, C, A, D], and passes through HierarchicalTextAdapter (Stage
    1 attention pool + Stage 2 organ MoE + Stage 3 rank embedding) to produce
    final [B, C, D] for the head.

    Class metadata (organ_id, axis_id, rank_along_axis) is loaded from a
    .pt file built by tools/build_class_metadata_tensor.py.

    Args:
        class_metadata_path: .pt with {'class_names', 'class_ids',
            'organ_ids', 'axis_ids', 'rank_along_axis', 'system_ids',
            'attr_emb' [C, A, D]} all in cat_id order.
        adapter: HierarchicalTextAdapter config dict (or None for mean pool).
    """

    def __init__(
        self,
        class_metadata_path: str,
        adapter: OptMultiConfig = None,
        pool_mode: Optional[str] = None,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(init_cfg)
        # Auto-detect pool_mode when caller doesn't set it explicitly, so M2
        # configs that predate this flag continue to work unchanged:
        #   adapter is None    -> 'mean'    (M1-5attr static mean-pool)
        #   adapter is not None -> 'adapter' (M2 HierarchicalTextAdapter path)
        # Set pool_mode='none' explicitly to feed [B, C, A, D] raw attribute
        # embeddings to a downstream consumer (Design A ImageConditionalFusion).
        if pool_mode is None:
            pool_mode = 'adapter' if adapter is not None else 'mean'
        if pool_mode not in ('mean', 'adapter', 'none'):
            raise ValueError(
                f"pool_mode must be 'mean' / 'adapter' / 'none', got "
                f"{pool_mode!r}. 'mean' = static mean over 5 attrs (M1-5attr "
                f"baseline). 'adapter' = run trainable HTA (M2). 'none' = "
                f"return raw [B, C, A, D] without pooling (Design A "
                f"ImageConditionalFusion path; consumer must downstream pool)."
            )
        if pool_mode == 'adapter' and adapter is None:
            raise ValueError(
                "pool_mode='adapter' requires adapter config to be set.")
        if pool_mode != 'adapter' and adapter is not None:
            raise ValueError(
                f"pool_mode={pool_mode!r} expects adapter=None; got adapter "
                f"config — adapter would never be called.")
        self.pool_mode = pool_mode
        meta = torch.load(class_metadata_path, weights_only=False, map_location='cpu')
        self.class_names = list(meta['class_names'])
        self.num_classes = len(self.class_names)
        self.num_attrs = meta['attr_emb'].shape[1]
        self.embed_dim = meta['attr_emb'].shape[-1]

        # Register as non-persistent buffers so they move with .cuda() / DDP
        # but don't bloat state_dict (class metadata is reproducible from cfg).
        self.register_buffer('attr_emb', meta['attr_emb'].float(),
                             persistent=False)              # [C, A, D]
        self.register_buffer('class_organ_ids',
                             meta['organ_ids'].long(), persistent=False)
        self.register_buffer('class_axis_ids',
                             meta['axis_ids'].long(), persistent=False)
        self.register_buffer('class_ranks',
                             meta['rank_along_axis'].long(), persistent=False)
        self.register_buffer('class_system_ids',
                             meta['system_ids'].long(), persistent=False)

        if adapter is not None:
            self.adapter = MODELS.build(adapter)
        else:
            self.adapter = None

        # Dummy buff to anchor device (compat with PseudoLanguageBackbone)
        self.register_buffer('buff', torch.zeros(1), persistent=False)

    def forward(self, text):
        """Return per-class embeddings, shape determined by pool_mode.

        Ignores `text` content — backbone broadcasts the stored class metadata
        to every batch item (consistent with PseudoLanguageBackbone behavior
        under LoadText, which feeds the same class set to all samples).

        Returns:
            pool_mode='mean':    [B, C, D]     mean over 5 attrs (M1-5attr平均)
            pool_mode='adapter': [B, C, D]     trainable HTA adapter (M2)
            pool_mode='none':    [B, C, A, D]  raw 5-attr (Design A / ICF feed)

        Note: callers using sample-level class subset sampling (RandomLoadText)
        would need a different lookup mechanism; not supported here.
        """
        B = len(text) if isinstance(text, (list, tuple)) else 1
        attr_emb = self.attr_emb.unsqueeze(0).expand(B, -1, -1, -1)  # [B, C, A, D]

        if self.pool_mode == 'mean':
            return attr_emb.mean(dim=2)                              # [B, C, D]
        if self.pool_mode == 'none':
            return attr_emb                                          # [B, C, A, D]
        # pool_mode == 'adapter' (validated in __init__ to require non-None)
        emb_final = self.adapter(
            attr_emb,
            self.class_organ_ids,
            self.class_axis_ids,
            self.class_ranks,
        )                                                            # [B, C, D]
        return emb_final

    def forward_text(self, text):
        return self.forward(text)


@MODELS.register_module()
class MultiModalYOLOBackbone(BaseModule):

    def __init__(
        self,
        image_model: ConfigType,
        text_model: ConfigType,
        #  visual_prompt_model : ConfigType,
        frozen_stages: int = -1,
        with_text_model: bool = True,
        cross_modal_fusion: OptMultiConfig = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg)
        self.with_text_model = with_text_model
        self.image_model = MODELS.build(image_model)
        # self.visual_prompt_model = MODELS.build(visual_prompt_model)
        if self.with_text_model:
            self.text_model = MODELS.build(text_model)
        else:
            self.text_model = None
        # Optional image-text fusion module (Design A / ICF). When provided,
        # it consumes (img_feats[-1], txt_feats) inside forward() after both
        # image_model and text_model have run. The text_model must emit a
        # 4-D tensor [B, C, A, D] (pool_mode='none') for the fusion module
        # to be applied; otherwise the fusion is skipped (legacy path).
        if cross_modal_fusion is not None:
            self.cross_modal_fusion = MODELS.build(cross_modal_fusion)
        else:
            self.cross_modal_fusion = None
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self.image_model, self.image_model.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super().train(mode)
        self._freeze_stages()

    def forward(
        self, image: Tensor, text: List[List[str]]
    ) -> Tuple[Tuple[Tensor], Tensor]:
        img_feats = self.image_model(image)

        if not self.with_text_model:
            return img_feats, None

        txt_feats = self.text_model(text)
        # Image-conditional fusion (Design A / ICF). Only fires when the
        # text_model emits a 4-D tensor [B, C, A, D] (pool_mode='none' in
        # PseudoMultiAttrLanguageBackbone) AND a fusion module is configured.
        # The fusion module consumes the deepest backbone feature
        # (img_feats[-1], typically ConvNext stage 4 = 768d, stride 32) plus
        # the raw 5-attr text emb, and returns [B, C, D] for the head.
        if self.cross_modal_fusion is not None and txt_feats.dim() == 4:
            txt_feats = self.cross_modal_fusion(img_feats[-1], txt_feats)
        return img_feats, txt_feats

    def forward_text(self, text: List[List[str]]) -> Tensor:
        assert self.with_text_model, "forward_text() requires a text model"
        txt_feats = self.text_model(text)
        return txt_feats

    def forward_image(self, image: Tensor) -> Tuple[Tensor]:
        return self.image_model(image)

    def forward_visual_prompt(self, image: Tensor) -> Tuple[Tensor]:
        return self.visual_prompt_model(image)
