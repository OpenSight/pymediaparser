"""SmartSampler 抽象基类 - 定义智能采样器的统一接口"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class BaseSamplerConfig:
    """采样器配置基类 - 所有采样器配置的公共字段"""

    enable_smart_sampling: bool = True
    """是否启用智能采样"""

    backup_interval: float = 30.0
    """保底间隔（秒），画面无变化时强制采样"""

    min_frame_interval: float = 1.0
    """最小帧间隔（秒），防止过于频繁采样"""


class SmartSampler(ABC):
    """智能采样器抽象基类 - 定义所有采样器必须实现的接口。

    子类必须实现:
        - sample(): 核心采样方法
        - reset(): 重置状态

    子类可选重写:
        - get_statistics(): 获取统计信息
    """

    def __init__(
        self,
        enable_smart_sampling: bool = True,
        backup_interval: float = 30.0,
        min_frame_interval: float = 1.0,
    ) -> None:
        """初始化基类公共属性。

        Args:
            enable_smart_sampling: 是否启用智能采样。
            backup_interval: 保底间隔（秒），画面无变化时强制采样。
            min_frame_interval: 最小帧间隔（秒），防止过于频繁采样。
        """
        self.enable_smart = enable_smart_sampling
        self._backup_interval = backup_interval
        self._min_frame_interval = min_frame_interval
        self._last_emit_ts: float = -float('inf')
        self._input_frame_count: int = 0

    # ── 公共属性 ──────────────────────────────────────────

    @property
    def frame_count(self) -> int:
        """已送入的输入帧总数。"""
        return self._input_frame_count

    # ── 抽象方法 ──────────────────────────────────────────

    @abstractmethod
    def sample(
        self, frames: Iterator[tuple[Image.Image, float, int]],
    ) -> Iterator[Dict[str, Any]]:
        """核心采样方法。

        Args:
            frames: 帧迭代器，每个元素是 (PIL图像, 时间戳, 帧序号) 元组。

        Yields:
            采样结果字典，包含以下字段:
                - image: PIL.Image 对象（处理后的图像）
                - timestamp: 时间戳
                - frame_index: 帧序号（来自输入，非内部统计）
                - significant: 是否为显著帧
                - source: 触发来源 ('smart', 'time', 等)
                - change_metrics: 变化指标字典（可选）
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """重置所有内部状态。"""
        pass

    # ── 可选重写方法 ──────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息。

        子类可重写以添加更多统计字段。
        """
        return {
            'total_frames_processed': self.frame_count,
            'smart_sampling_enabled': self.enable_smart,
            'backup_interval': self._backup_interval,
            'min_frame_interval': self._min_frame_interval,
        }

    # ── 公共工具方法 ──────────────────────────────────────

    def _should_emit_by_time(self, ts: float) -> bool:
        """检查是否满足保底时间间隔。

        当画面长时间无变化时，强制采样一帧避免遗漏。
        """
        if ts < self._last_emit_ts:
            logger.info(
                "检测到时间戳回跳 (%.3f -> %.3f)，重置采样状态",
                self._last_emit_ts, ts,
            )
            self._last_emit_ts = -float('inf')
            return True
        return (ts - self._last_emit_ts) >= self._backup_interval

    @staticmethod
    def _numpy_to_pil(frame_np: np.ndarray) -> Image.Image:
        """BGR numpy → RGB PIL。"""
        rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)
