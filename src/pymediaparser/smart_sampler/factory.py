"""智能采样器工厂模块 - 提供采样器创建和注册功能"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Union

from .base import BaseSamplerConfig, SmartSampler
from .configs import MLSamplerConfig, SimpleSamplerConfig

logger = logging.getLogger(__name__)


def create_sampler(
    name: str,
    config: Union[BaseSamplerConfig, Dict[str, Any], None] = None,
) -> SmartSampler:
    """创建采样器实例

    Args:
        name: 采样器名称（'simple' 或 'ml'）
        config: 配置对象或配置字典，为 None 时使用默认配置

    Returns:
        SmartSampler 子类实例

    Raises:
        ValueError: 未知的采样器类型
    """
    if name == 'simple':
        from .simple_smart_sampler import SimpleSmartSampler

        if config is None:
            config = SimpleSamplerConfig()
        elif isinstance(config, dict):
            config = SimpleSamplerConfig(**config)
        return SimpleSmartSampler(**vars(config))

    elif name == 'ml':
        from .ml_smart_sampler import MLSmartSampler

        if config is None:
            config = MLSamplerConfig()
        elif isinstance(config, dict):
            config = MLSamplerConfig(**config)
        return MLSmartSampler(**vars(config))

    else:
        available = ', '.join(list_samplers())
        raise ValueError(f"未知的采样器类型: {name}。可用: {available}")


def list_samplers() -> List[str]:
    """列出所有可用的采样器名称

    Returns:
        采样器名称列表
    """
    return ['simple', 'ml']
