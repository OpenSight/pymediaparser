"""智能采样器配置类定义"""

from dataclasses import dataclass
from typing import Literal

from .base import BaseSamplerConfig


@dataclass
class SimpleSamplerConfig(BaseSamplerConfig):
    """SimpleSmartSampler 配置 - 基础二层架构

    基于运动检测 + SSIM变化分析的简单智能采样器。
    """

    motion_method: Literal['MOG2', 'KNN'] = 'MOG2'
    """运动检测方法：MOG2 或 KNN 背景减除"""

    ssim_threshold: float = 0.80
    """SSIM 相似度阈值，低于此值认为有显著变化"""

    motion_threshold: float = 0.1
    """运动检测阈值（运动像素比例）"""


@dataclass
class MLSamplerConfig(BaseSamplerConfig):
    """MLSmartSampler 配置 - 三层漏斗架构

    Layer 0 (硬过滤): 快速排除无价值帧，90%+ 拒绝率
    Layer 1 (快速触发): 多路 OR 并行检测，高召回率
    Layer 2 (精细验证): 多特征融合打分 + 峰值检测

    注：内部技术参数采用最优默认值，用户无需关心。
    """

    motion_method: Literal['MOG2', 'KNN'] = 'MOG2'
    """运动检测方法：MOG2 或 KNN 背景减除"""

    motion_threshold: float = 0.1
    """运动检测阈值（运动像素比例）"""

    scene_switch_threshold: float = 0.5
    """场景切换阈值，HSV直方图最小相关系数 < 此值视为场景切换

    - 值越高：越敏感，更容易触发场景切换，采样更多帧
    - 值越低：越不敏感，可能漏掉场景切换，采样更少帧
    """
