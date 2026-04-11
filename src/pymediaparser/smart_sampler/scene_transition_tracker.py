"""SceneTransitionTracker - 场景连续变化追踪器

检测场景连续变化（云台旋转、镜头推移等），
通过 HSV 分块直方图相关性和滑动窗口趋势分析实现。
"""

from __future__ import annotations
import logging
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SceneTransitionTracker:
    """场景连续变化追踪器 - 检测云台旋转等连续场景变换。"""

    def __init__(
        self,
        sensitivity: float = 0.5,
    ) -> None:
        """初始化追踪器。

        Args:
            sensitivity: 活动检测灵敏度 (0.0-1.0)，用于调整场景变化检测阈值。
        """
        self.sensitivity = sensitivity
        self._apply_sensitivity(sensitivity)

        # 状态
        self._prev_hsv_hists: Optional[List[np.ndarray]] = None
        self._prev_gray_small: Optional[np.ndarray] = None
        self._corr_window: Deque[float] = deque(maxlen=self._window_size)
        self._diff_window: Deque[float] = deque(maxlen=self._window_size)

        # 场景变化状态
        self._is_in_transition: bool = False
        self._transition_remaining_frames: int = 0

        logger.info(
            "SceneTransitionTracker 初始化完成 - 灵敏度: %.2f",
            sensitivity,
        )

    def _apply_sensitivity(self, sensitivity: float) -> None:
        """将用户灵敏度映射到内部参数。"""
        # 场景变化相关性阈值（值越低越容易触发）
        self._corr_threshold = 0.7 - sensitivity * 0.2  # 0.5-0.7

        # 内部固定参数
        self._window_size: int = 5
        self._min_consecutive: int = 3
        self._corr_low: float = 0.6
        self._corr_high: float = 0.95
        self._diff_low: float = 5.0
        self._diff_high: float = 50.0
        self._grace_period_frames: int = 3  # 检测到变化后的持续采样帧数

    def detect(
        self,
        frame_np: np.ndarray,
    ) -> Tuple[bool, float]:
        """检测当前帧是否处于场景连续变化中。

        Args:
            frame_np: BGR 格式的帧。

        Returns:
            (is_in_transition, transition_score) 元组。
        """
        if frame_np is None or frame_np.size == 0:
            return False, 0.0

        # 如果在 grace period 中，直接返回 True
        if self._is_in_transition:
            self._transition_remaining_frames -= 1
            if self._transition_remaining_frames <= 0:
                self._is_in_transition = False
            return True, 1.0

        # 下采样
        gray_small = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
        small_bgr = cv2.resize(frame_np, (160, 90), interpolation=cv2.INTER_AREA)

        # 计算 HSV 分块直方图
        hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
        curr_hists = self._compute_block_histograms(hsv)

        # 计算与上一帧的相关性
        corr_value = 1.0
        if self._prev_hsv_hists is not None:
            min_correl = 1.0
            for h_curr, h_prev in zip(curr_hists, self._prev_hsv_hists):
                correl = cv2.compareHist(h_curr, h_prev, cv2.HISTCMP_CORREL)
                min_correl = min(min_correl, correl)
            corr_value = min_correl

        # 计算帧间差分
        diff_value = 0.0
        if self._prev_gray_small is not None:
            diff = cv2.absdiff(gray_small, self._prev_gray_small)
            diff_value = float(np.mean(diff))

        # 更新上一帧状态
        self._prev_hsv_hists = curr_hists
        self._prev_gray_small = gray_small

        # 检测场景突变（在添加到窗口之前，使用当前相关性与窗口中前一帧比较）
        is_sudden_transition = self._check_sudden_transition(corr_value)

        # 添加到滑动窗口
        self._corr_window.append(corr_value)
        self._diff_window.append(diff_value)

        # 检测场景连续变化
        is_transition, transition_score = self._check_scene_transition()

        # 如果检测到场景变化（渐变或突变），设置 grace period
        if (is_transition or is_sudden_transition) and not self._is_in_transition:
            self._is_in_transition = True
            self._transition_remaining_frames = self._grace_period_frames
            logger.info(
                "检测到场景变化 - 相关性=%.3f, 差分=%.1f, 类型=%s",
                corr_value, diff_value,
                '突变' if is_sudden_transition else '渐变',
            )
            return True, max(transition_score, 0.8)  # 突变给高分

        logger.debug(
            "场景变化检测 - 相关性=%.3f, 差分=%.1f, 场景变化=%s",
            corr_value, diff_value,
            '是' if is_transition else '否',
        )

        return is_transition, transition_score

    def _compute_block_histograms(
        self, hsv: np.ndarray
    ) -> List[np.ndarray]:
        """计算 3x3 分块 HSV 直方图。

        Args:
            hsv: HSV 格式的图像。

        Returns:
            9 个分块的直方图列表。
        """
        h, w = hsv.shape[:2]
        bh, bw = h // 3, w // 3
        hists = []

        for row in range(3):
            for col in range(3):
                block = hsv[row * bh:(row + 1) * bh, col * bw:(col + 1) * bw]
                hist = cv2.calcHist([block], [0, 1], None, [30, 32], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                hists.append(hist)

        return hists

    def _check_scene_transition(self) -> Tuple[bool, float]:
        """检查是否处于场景连续变化中。

        判定条件:
        - 连续 N 帧相关性在 [corr_low, corr_high] 区间
        - 帧间差分在 [diff_low, diff_high] 区间
        - 相关性呈单调趋势（持续下降或上升）

        Returns:
            (is_transition, transition_score) 元组。
        """
        if len(self._corr_window) < self._min_consecutive:
            return False, 0.0

        corr_values = list(self._corr_window)
        diff_values = list(self._diff_window)

        # 检查连续帧的相关性和差分是否在目标区间
        consecutive_count = 0
        for corr, diff in zip(corr_values, diff_values):
            if (self._corr_low < corr < self._corr_high and
                    self._diff_low < diff < self._diff_high):
                consecutive_count += 1

        if consecutive_count < self._min_consecutive:
            return False, 0.0

        # 检查相关性趋势是否单调（持续变化）
        is_monotonic = self._check_monotonic_trend(corr_values)

        if is_monotonic:
            # 计算过渡分数（基于相关性变化幅度）
            corr_range = max(corr_values) - min(corr_values)
            transition_score = min(1.0, corr_range * 5.0)  # 归一化
            return True, transition_score

        return False, 0.0

    def _check_monotonic_trend(self, values: List[float]) -> bool:
        """检查序列是否呈单调趋势（允许最多 1 个异常点）。

        Args:
            values: 数值序列。

        Returns:
            是否单调。
        """
        if len(values) < 3:
            return False

        # 计算相邻差值
        diffs = [values[i+1] - values[i] for i in range(len(values) - 1)]

        # 统计正负差值数量
        positive = sum(1 for d in diffs if d > 0)
        negative = sum(1 for d in diffs if d < 0)

        # 允许最多 1 个异常点
        max_allowed = max(positive, negative)
        min_allowed = min(positive, negative)

        return min_allowed <= 1 and max_allowed >= len(diffs) - 1

    def _check_sudden_transition(self, current_corr: float) -> bool:
        """检测场景突变（相关性骤降）。

        当摄像机位置突然改变时，相关性会从接近 1.0 骤降到很低甚至负值。
        如果当前相关性比前一帧下降超过阈值，判定为场景突变。

        Args:
            current_corr: 当前帧的相关性值（尚未添加到窗口）。

        Returns:
            是否检测到场景突变。
        """
        if len(self._corr_window) < 1:
            return False

        # 取前一帧的相关性（窗口中的最后一帧）
        prev_corr = self._corr_window[-1]

        # 计算相关性下降幅度
        corr_drop = prev_corr - current_corr

        # 如果下降超过 0.3（例如从 1.0 降到 0.7 以下，或从 0.9 降到 0.6 以下）
        # 且当前相关性低于 0.7，判定为突变
        if corr_drop > 0.3 and current_corr < 0.7:
            logger.debug(
                "检测到场景突变 - 前一帧相关性=%.3f, 当前相关性=%.3f, 下降幅度=%.3f",
                prev_corr, current_corr, corr_drop,
            )
            return True

        return False

    def reset(self) -> None:
        """重置所有状态。"""
        self._prev_hsv_hists = None
        self._prev_gray_small = None
        self._corr_window.clear()
        self._diff_window.clear()
        self._is_in_transition = False
        self._transition_remaining_frames = 0
        logger.info("SceneTransitionTracker 状态已重置")
