"""智能采样器配置类定义"""

from dataclasses import dataclass
from typing import Literal

from .base import BaseSamplerConfig


@dataclass
class SimpleSamplerConfig(BaseSamplerConfig):
    """SimpleSmartSampler 配置 - 活动状态机架构（优化版）

    采用活动状态机 + 多策略活动检测 + 多维度静态过滤。
    检测人、宠物、飞鸟、牲畜等所有有意义的动物活动。
    """

    sensitivity: float = 0.5
    """活动检测灵敏度 (0.0-1.0)
    
    - 值越低：越敏感，更容易检测到活动，采样更多帧
    - 值越高：越严格，仅明显活动被检测到，采样更少帧
    
    该参数会自动映射到内部各检测器的阈值。
    """

    activity_duration: float = 3.0
    """活动期持续时间（秒）
    
    检测到活动后，此期间内所有帧都会被采样。
    - 值越大：活动期越长，采样越多
    - 值越小：活动期越短，采样越少
    """

    quiet_frames_threshold: int = 10
    """静默帧阈值（连续无活动帧数）
    
    连续多少帧无活动后退出活动期。
    - 值越大：活动期越长，更保守
    - 值越小：活动期越短，更激进
    """

    # 以下为功能参数
    motion_method: Literal['MOG2', 'KNN'] = 'MOG2'
    """运动检测方法：MOG2 或 KNN 背景减除"""


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
