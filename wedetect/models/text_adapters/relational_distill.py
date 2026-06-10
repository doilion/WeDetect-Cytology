"""Innovation D (2026-06-05): learnable residual adapter on frozen cytology text.

See ``wedetect/models/losses/relational_distill_loss.py`` for the relational
distillation loss that trains this adapter. The adapter is an EXACT identity at
init (zero-init residual MLP) so the detector is regression-safe at training
start; the relational loss + the main cls loss then jointly reshape the
(otherwise collapsed) frozen-text geometry.

Why an adapter and not new prompts: the 3-prompt collapse study
(docs/results/figures/text_collapse_3sets_20260605.png) shows fullnames / 5attr
/ morph6 text all collapse within-organ (cos 0.80-0.88, max ~0.96) — prompt
engineering is a dead end. A learnable adapter is the only principled lever left
on the text side, and it is supervised by the discriminative IMAGE structure.
"""
import torch
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class TextRelationalAdapter(nn.Module):
    """g(t) = t + alpha * MLP(t), zero-init so g == identity at training start.

    ``in_dim == out_dim`` keeps the frozen-vector contract (downstream head sees
    no shape change). Returns the UN-normalized residual: the contrastive head
    L2-normalizes text internally, and the relational loss normalizes its own
    copy, so normalizing here would be redundant.

    Args:
        dim: text embedding dim (512 for BiomedCLIP).
        hidden_dim: bottleneck width of the residual MLP.
        alpha: fixed scale on the MLP branch (caps how fast text can move early).
    """

    def __init__(self, dim: int = 512, hidden_dim: int = 256, alpha: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.alpha = float(alpha)
        # zero-init the last layer -> MLP(t)=0 at init -> g(t)=t exactly (identity).
        # Survives mmengine BaseModule.init_weights cascade because this is a plain
        # nn.Module child (no init_weights to override the zero-init).
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return t + self.alpha * self.fc2(self.act(self.fc1(t)))
