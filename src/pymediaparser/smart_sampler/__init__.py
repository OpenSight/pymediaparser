"""智能采样模块 - 基于计算机视觉的智能帧筛选"""

from .base import BaseSamplerConfig, SmartSampler
from .configs import MLSamplerConfig, SimpleSamplerConfig
from .factory import create_sampler, list_samplers
from .ml_smart_sampler import MLSmartSampler
from .simple_smart_sampler import SimpleSmartSampler
from .change_analyzer import ChangeAnalyzer
from .motion_detector import MotionDetector

__all__ = [
    # 基类与配置
    'SmartSampler',
    'BaseSamplerConfig',
    'SimpleSamplerConfig',
    'MLSamplerConfig',
    # 具体实现
    'SimpleSmartSampler',
    'MLSmartSampler',
    # 工厂函数
    'create_sampler',
    'list_samplers',
    # 辅助组件
    'MotionDetector',
    'ChangeAnalyzer',
]
