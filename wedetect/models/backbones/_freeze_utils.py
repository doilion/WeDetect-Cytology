"""Shared submodule-freezing helper for wedetect backbones.

Replicates the long-standing per-backbone ``_freeze_modules`` logic *exactly*,
extracted to dedupe 3 byte-identical copies in ``mm_backbone.py``:

  - ``frozen_modules == ()``           -> no-op (nothing frozen)
  - ``frozen_modules == ('all',)``     -> ``eval()`` + ``requires_grad=False`` on
                                          every submodule
  - otherwise                          -> freeze every submodule whose qualified
                                          name starts with any given prefix

Behaviour is identical to the previous inline copies (verified by md5); the
``train()`` overrides in each backbone still call ``_freeze_modules`` every
``.train()`` so frozen modules stay in eval mode.
"""


def freeze_submodules(model, frozen_modules):
    if len(frozen_modules) == 0:
        # not freeze
        return
    if frozen_modules[0] == "all":
        model.eval()
        for _, module in model.named_modules():
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        return
    for name, module in model.named_modules():
        for frozen_name in frozen_modules:
            if name.startswith(frozen_name):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
                break
