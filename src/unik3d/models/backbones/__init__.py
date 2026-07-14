from .dinov2 import _make_dinov2_model

__all__ = [
    "SwinTransformerV2",
    "ConvNeXtV2",
    "_make_dinov2_model",
    "ConvNeXt",
]


def __getattr__(name):
    if name == "ConvNeXt":
        from .convnext import ConvNeXt

        return ConvNeXt
    if name == "ConvNeXtV2":
        from .convnext2 import ConvNeXtV2

        return ConvNeXtV2
    if name == "SwinTransformerV2":
        from .swinv2 import SwinTransformerV2

        return SwinTransformerV2
    raise AttributeError(name)
