"""MeaningfulActivityDetector - 有意义的活动检测器

检测画面中所有有意义的动物活动（人、宠物、飞鸟、牲畜等），
采用多策略融合方案：形态学分析 + 稀疏光流 + 运动持续性追踪。
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MeaningfulActivityDetector:
    """有意义的活动检测器 - 检测动物活动并区分风吹植物等干扰。"""

    def __init__(
        self,
        sensitivity: float = 0.5,
    ) -> None:
        """初始化检测器。

        Args:
            sensitivity: 活动检测灵敏度 (0.0-1.0)，值越低越敏感。
        """
        self.sensitivity = sensitivity
        self._apply_sensitivity(sensitivity)

        # 光流分析状态
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None

        # 运动持续性追踪
        self._motion_history: List[float] = []
        self._consecutive_motion_count: int = 0

        logger.info(
            "MeaningfulActivityDetector 初始化完成 - 灵敏度: %.2f",
            sensitivity,
        )

    def _apply_sensitivity(self, sensitivity: float) -> None:
        """将用户灵敏度映射到内部参数。"""
        # 形态学分析阈值
        self._morph_threshold = 0.3 + (1.0 - sensitivity) * 0.4  # 0.3-0.7

        # 光流分析阈值
        self._flow_threshold = 0.02 + (1.0 - sensitivity) * 0.03  # 0.02-0.05

        # 内部固定参数（用户不可见）
        self._min_components: int = 1
        self._max_components: int = 15
        self._min_area_ratio: float = 0.01
        self._max_area_ratio: float = 0.40
        self._min_aspect_ratio: float = 0.2
        self._max_aspect_ratio: float = 2.0
        self._min_solidity: float = 0.3
        self._min_centroid_y_ratio: float = 0.35

        # 光流参数
        self._max_corners: int = 100
        self._quality_level: float = 0.01
        self._min_distance: float = 5.0
        self._displacement_threshold: float = 1.0
        self._direction_consistency_threshold: float = 0.5

        # 运动持续性参数
        self._motion_history_window: int = 5
        self._min_consecutive_motion_frames: int = 2

    def detect(
        self,
        frame_np: np.ndarray,
        fg_mask: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float]:
        """检测当前帧是否有有意义的动物活动。

        Args:
            frame_np: BGR 格式的帧。
            fg_mask: MOG2 前景掩码（可选，如未提供则内部计算）。

        Returns:
            (has_activity, activity_score) 元组。
        """
        if frame_np is None or frame_np.size == 0:
            return False, 0.0

        gray = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
        # 下采样以加速光流计算
        gray_small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)

        # 策略 A: 形态学分析
        morph_score = 0.0
        if fg_mask is not None:
            morph_score = self._analyze_morphology(fg_mask)

        # 策略 B: 光流分析（使用下采样图）
        flow_score, flow_autonomous = self._analyze_optical_flow(gray_small)

        # 策略 C: 运动持续性
        is_motion_continuous = self._check_motion_continuity(
            morph_score > self._morph_threshold or flow_score > self._flow_threshold
        )

        # 综合判定
        has_activity = self._combine_strategies(
            morph_score, flow_score, flow_autonomous, is_motion_continuous
        )

        # 计算综合得分
        activity_score = (morph_score * 0.4 + flow_score * 0.4 +
                         (1.0 if is_motion_continuous else 0.0) * 0.2)

        logger.debug(
            "活动检测 - 形态学=%.3f, 光流=%.3f, 自主运动=%s, 持续性=%s, 结果=%s",
            morph_score, flow_score,
            '是' if flow_autonomous else '否',
            '是' if is_motion_continuous else '否',
            '有活动' if has_activity else '无活动',
        )

        return has_activity, activity_score

    def _analyze_morphology(self, fg_mask: np.ndarray) -> float:
        """策略 A: 分析前景掩码的形态学特征。

        Returns:
            形态学评分 (0.0-1.0)。
        """
        # 下采样以加速
        h, w = fg_mask.shape[:2]
        small_mask = cv2.resize(fg_mask, (160, 90), interpolation=cv2.INTER_NEAREST)

        # 查找连通域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            small_mask, connectivity=8
        )

        # 跳过背景（label 0）
        num_components = num_labels - 1
        if num_components == 0:
            return 0.0

        # 分析最大连通域
        max_area = 0
        max_aspect_ratio = 0.0
        max_solidity = 0.0
        max_centroid_y = 0.0
        total_pixels = small_mask.shape[0] * small_mask.shape[1]

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]

                # 高宽比
                max_aspect_ratio = min(bh, bw) / max(bh, bw) if max(bh, bw) > 0 else 0.0

                # 实心度
                max_solidity = area / (bw * bh) if (bw * bh) > 0 else 0.0

                # 重心位置（归一化）
                max_centroid_y = centroids[i][1] / small_mask.shape[0]

        # 计算形态学评分
        area_ratio = max_area / total_pixels if total_pixels > 0 else 0.0

        score = 0.0

        # 连通域数量评分
        if self._min_components <= num_components <= self._max_components:
            score += 0.25

        # 面积占比评分
        if self._min_area_ratio <= area_ratio <= self._max_area_ratio:
            score += 0.25

        # 高宽比评分
        if self._min_aspect_ratio <= max_aspect_ratio <= self._max_aspect_ratio:
            score += 0.2

        # 重心位置评分
        if max_centroid_y >= self._min_centroid_y_ratio:
            score += 0.15

        # 实心度评分
        if max_solidity >= self._min_solidity:
            score += 0.15

        return min(1.0, score)

    def _analyze_optical_flow(
        self, gray: np.ndarray
    ) -> Tuple[float, bool]:
        """策略 B: 稀疏光流分析运动模式。

        Returns:
            (flow_score, is_autonomous) 元组。
            flow_score: 光流强度评分 (0.0-1.0)
            is_autonomous: 是否为自主运动（非风吹等一致运动）
        """
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0, False

        # 检测特征点
        prev_pts = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=self._max_corners,
            qualityLevel=self._quality_level,
            minDistance=self._min_distance,
        )

        if prev_pts is None or len(prev_pts) == 0:
            self._prev_gray = gray
            return 0.0, False

        # 计算光流
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            gray,
            prev_pts,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self._prev_gray = gray

        if next_pts is None:
            return 0.0, False

        # 筛选有效点并展平
        valid_mask = status.ravel() == 1
        if not np.any(valid_mask):
            return 0.0, False

        # goodFeaturesToTrack 返回 (N, 1, 2)，需要 squeeze
        prev_good = prev_pts.reshape(-1, 2)[valid_mask]
        next_good = next_pts.reshape(-1, 2)[valid_mask]

        # 检查点数
        n_points = len(prev_good)
        if n_points == 0:
            return 0.0, False

        # 计算位移
        displacements = np.sqrt(
            np.sum((next_good - prev_good) ** 2, axis=-1)
        )

        # 计算平均位移
        avg_displacement = float(np.mean(displacements))
        flow_score = min(1.0, avg_displacement / 20.0)  # 归一化到 0-1

        # 分析运动方向一致性（区分自主运动 vs 被风吹动）
        if n_points > 2:
            # 计算运动方向
            dx = next_good[:, 0] - prev_good[:, 0]
            dy = next_good[:, 1] - prev_good[:, 1]
            angles = np.arctan2(dy, dx)

            # 计算方向一致性（角度方差）
            angle_variance = float(np.var(angles))
            is_autonomous = angle_variance > self._direction_consistency_threshold
        else:
            # 点数不足，保守判定为非自主运动
            is_autonomous = False

        return flow_score, is_autonomous

    def _check_motion_continuity(self, has_motion: bool) -> bool:
        """策略 C: 检查运动持续性。

        Returns:
            是否满足连续性要求。
        """
        # 更新运动历史
        self._motion_history.append(1.0 if has_motion else 0.0)

        # 保持窗口大小
        if len(self._motion_history) > self._motion_history_window:
            self._motion_history = self._motion_history[-self._motion_history_window:]

        # 计算连续运动帧数
        if has_motion:
            self._consecutive_motion_count += 1
        else:
            self._consecutive_motion_count = 0

        # 检查是否满足最小连续帧数
        return self._consecutive_motion_count >= self._min_consecutive_motion_frames

    def _combine_strategies(
        self,
        morph_score: float,
        flow_score: float,
        flow_autonomous: bool,
        is_continuous: bool,
    ) -> bool:
        """综合判定是否有有意义的活动。

        判定规则:
        - (形态学评分 > 阈值 AND 光流显示自主运动) OR
        - (形态学评分 > 高阈值 AND 运动持续3帧+) OR
        - (光流显示强自主运动 AND 运动持续2帧+)
        """
        high_morph_threshold = self._morph_threshold * 1.3

        rule1 = (morph_score > self._morph_threshold and
                 flow_autonomous and
                 flow_score > self._flow_threshold)

        rule2 = (morph_score > high_morph_threshold and
                 self._consecutive_motion_count >= 3)

        rule3 = (flow_autonomous and
                 flow_score > self._flow_threshold * 1.5 and
                 self._consecutive_motion_count >= 2)

        return rule1 or rule2 or rule3

    def reset(self) -> None:
        """重置所有状态。"""
        self._prev_gray = None
        self._prev_pts = None
        self._motion_history.clear()
        self._consecutive_motion_count = 0
        logger.info("MeaningfulActivityDetector 状态已重置")
