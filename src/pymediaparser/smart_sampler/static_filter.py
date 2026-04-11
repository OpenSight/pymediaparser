"""StaticFilter - 多维度 Z-Score 静态过滤器

通过多维度特征分析（帧间差分、颜色方差、边缘密度），
精准过滤静态画面、微光变化、风吹草动等准静态画面。
"""

from __future__ import annotations
import logging
from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StaticFilter:
    """多维度 Z-Score 静态过滤器 - 排除静态/准静态帧。"""

    def __init__(
        self,
        sensitivity: float = 0.5,
    ) -> None:
        """初始化过滤器。

        Args:
            sensitivity: 活动检测灵敏度 (0.0-1.0)，用于调整静态判定阈值。
        """
        self.sensitivity = sensitivity
        self._apply_sensitivity(sensitivity)

        # 状态
        self._prev_gray_small: Optional[np.ndarray] = None
        self._feature_window: Deque[np.ndarray] = deque(maxlen=self._window_size)

        logger.info(
            "StaticFilter 初始化完成 - 灵敏度: %.2f",
            sensitivity,
        )

    def _apply_sensitivity(self, sensitivity: float) -> None:
        """将用户灵敏度映射到内部参数。"""
        # Z-Score 阈值（值越高越容易判定为静态）
        self._zscore_threshold = 0.5 + sensitivity * 1.0  # 0.5-1.5

        # 内部固定参数
        self._window_size: int = 10
        self._still_threshold: float = 3.0  # 帧间差分阈值
        self._lighting_only_factor: float = 1.5  # 光线微变判定因子

    def check(
        self,
        frame_np: np.ndarray,
        motion_score: float = 0.0,
        is_human_like: bool = False,
    ) -> Tuple[bool, str]:
        """检查当前帧是否为静态/准静态帧。

        Args:
            frame_np: BGR 格式的帧。
            motion_score: 运动像素比例（来自 MOG2）。
            is_human_like: 是否符合人体/动物活动模式。

        Returns:
            (passed, reason) 元组。
            passed: True 表示通过检查（非静态），False 表示应排除。
            reason: 排除原因（仅当 passed=False 时有意义）。
        """
        if frame_np is None or frame_np.size == 0:
            return False, '空帧'

        # 下采样
        gray = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_NEAREST)

        # 检查 1: 帧间差分
        mean_diff = 0.0
        if self._prev_gray_small is not None:
            diff = cv2.absdiff(gray_small, self._prev_gray_small)
            mean_diff = float(np.mean(diff))

            if mean_diff < self._still_threshold:
                # 即使判定为静止帧，也更新参考帧以保持基准一致
                self._prev_gray_small = gray_small
                return False, '静止帧'

        # 更新上一帧
        self._prev_gray_small = gray_small

        # 提取多维度特征
        features = self._extract_features(gray_small, mean_diff)

        # 添加到滑动窗口
        self._feature_window.append(features)

        # 如果窗口未满，保守通过
        if len(self._feature_window) < 3:
            return True, ''

        # 检查 2: 多维度 Z-Score 分析
        is_static, reason = self._check_zscore(features, motion_score, is_human_like)

        if not is_static:
            return False, reason

        return True, ''

    def _extract_features(
        self,
        gray_small: np.ndarray,
        mean_diff: float,
    ) -> np.ndarray:
        """提取多维度特征向量。

        Args:
            gray_small: 下采样后的灰度图。
            mean_diff: 帧间差分均值。

        Returns:
            特征向量 [帧间差分, 颜色方差, 边缘密度]。
        """
        # 特征 1: 帧间差分均值
        f1 = mean_diff

        # 特征 2: 颜色方差（像素强度方差）
        f2 = float(np.var(gray_small))

        # 特征 3: 边缘密度（Sobel 梯度均值）
        sobel_x = cv2.Sobel(gray_small, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_small, cv2.CV_64F, 0, 1, ksize=3)
        f3 = float(np.mean(np.sqrt(sobel_x ** 2 + sobel_y ** 2)))

        return np.array([f1, f2, f3])

    def _check_zscore(
        self,
        features: np.ndarray,
        motion_score: float,
        is_human_like: bool,
    ) -> Tuple[bool, str]:
        """多维度 Z-Score 分析。

        Returns:
            (passed, reason) 元组。
            passed: True 表示非静态，False 表示应排除。
        """
        # 计算窗口统计量
        window_arr = np.array(self._feature_window)  # shape: (N, 3)
        mean = np.mean(window_arr, axis=0)
        std = np.std(window_arr, axis=0)

        # 计算 Z-Score
        z_scores = np.abs((features - mean) / (std + 1e-8))

        # 判定逻辑
        # 1. 所有维度 |z| < threshold → 静态帧
        if np.all(z_scores < self._zscore_threshold):
            logger.debug(
                "静态过滤 - Z-Score 全低 [%.2f, %.2f, %.2f]，判定为静态",
                z_scores[0], z_scores[1], z_scores[2],
            )
            return False, '静态帧 (Z-Score 全低)'

        # 2. 仅颜色方差 |z| > threshold 但其他维度正常 → 光线微变
        color_only_anomaly = (
            z_scores[1] > self._zscore_threshold * self._lighting_only_factor and
            z_scores[0] < self._zscore_threshold and
            z_scores[2] < self._zscore_threshold
        )
        if color_only_anomaly:
            logger.debug(
                "静态过滤 - 仅颜色异常 [%.2f, %.2f, %.2f]，判定为光线微变",
                z_scores[0], z_scores[1], z_scores[2],
            )
            return False, '光线微变'

        # 3. 运动像素占比低且不符合动物活动模式 → 风吹植物
        low_motion = motion_score < 0.02
        if low_motion and not is_human_like:
            logger.debug(
                "静态过滤 - 运动占比低 (%.3f) 且不符合动物模式",
                motion_score,
            )
            return False, '风吹植物/干扰'

        return True, ''

    def reset(self) -> None:
        """重置所有状态。"""
        self._prev_gray_small = None
        self._feature_window.clear()
        logger.info("StaticFilter 状态已重置")
