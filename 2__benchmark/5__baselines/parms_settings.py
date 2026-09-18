"""benchmark 用最小配置：官方 CBAM.py/DenseNet.py import 依赖，仅返回默认值。"""


class _Args:
    reduction = 16
    spatial_kernel = 9


def settings():
    return _Args()
