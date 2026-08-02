from types import SimpleNamespace

import torch.nn as nn

from src.train.utils import set_model


class _TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Linear(4, 4)
        self.merger = nn.Linear(4, 4)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = _TinyVision()
        self.model.language_model = nn.Linear(4, 4)
        self.panovggt_mlp = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 4)


def _config(**trainable_modules):
    return SimpleNamespace(
        model=SimpleNamespace(trainable_modules=trainable_modules),
    )


def _all_require_grad(module: nn.Module, expected: bool) -> None:
    assert all(parameter.requires_grad is expected for parameter in module.parameters())


def test_visual_and_merger_trainability_are_independent():
    model = _TinyModel()
    set_model(
        _config(
            visual=True,
            visual_merger=False,
            language_model=False,
            panovggt_mlp=False,
        ),
        model,
    )
    _all_require_grad(model.model.visual.patch, True)
    _all_require_grad(model.model.visual.merger, False)

    model = _TinyModel()
    set_model(
        _config(
            visual=False,
            visual_merger=True,
            language_model=False,
            panovggt_mlp=False,
        ),
        model,
    )
    _all_require_grad(model.model.visual.patch, False)
    _all_require_grad(model.model.visual.merger, True)
