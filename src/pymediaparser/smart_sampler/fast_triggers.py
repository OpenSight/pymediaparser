"""快速触发器组 (Layer 1) - 高召回率识别潜在关键帧，OR 逻辑多路并行检测"""

from __future__ import annotations
import collections
import logging
from typing import Deque, List, Optional

import cv2
import numpy as np
from PIL import Image

from .motion_detector import MotionDetector

logger = logging.getLogger(__name__)

# 下采样尺寸
_TRIGGER_WIDTH = 320
_TRIGGER_HEIGHT = 180


class FastTriggers:
    """Layer 1 快速触发器 - 轻量级 CV 算法，纯 CPU，目标耗时 1-3ms/帧。

    四路触发器 (OR 逻辑，任一触发即通过)：
    1. 剧烈运动检测 - 帧差 + 背景减除
    2. 场景切换检测 - HSV 直方图比对 + 感知哈希
    3. 异常事件检测 - 时序 Z-Score
    4. 周期强制采样 - 定时器
    """

    def __init__(
        self,
        motion_method: str = 'MOG2',
        motion_threshold: float = 0.1,
        scene_switch_threshold: float = 0.5,
        anomaly_zscore_threshold: float = 2.0,
        periodic_interval: float = 30.0,
    ) -> None:
        self._motion_threshold = motion_threshold
        self._scene_switch_threshold = scene_switch_threshold
        self._anomaly_zscore_threshold = anomaly_zscore_threshold
        self._periodic_interval = periodic_interval
        self._motion_method_name = motion_method

        # 运动检测器（复用现有组件）
        self._motion_detector = MotionDetector(
            method=motion_method, threshold=motion_threshold,
        )

        # 场景切换状态
        self._prev_hsv_hists: Optional[List[np.ndarray]] = None  # 3×3 分块直方图
        self._prev_phash: Optional[object] = None  # imagehash 对象

        # 异常检测状态（滑动窗口）
        self._anomaly_window_size = 30
        self._feature_window: Deque[np.ndarray] = collections.deque(maxlen=self._anomaly_window_size)
        # 用于异常检测运动强度计算的上一帧（相邻帧差异）
        self._last_gray_for_anomaly: Optional[np.ndarray] = None

        # 周期采样状态
        self._last_periodic_ts: float = -float('inf')

        # 缓存（供主控类读取）
        self._last_motion_score: float = 0.0
        self._last_fg_mask: Optional[np.ndarray] = None
        self._last_diff_mask: Optional[np.ndarray] = None
        self._prev_gray_small: Optional[np.ndarray] = None

        # 统计
        self._trigger_count: int = 0

        logger.debug(
            "FastTriggers 初始化完成 - 运动方法: %s, 运动阈值: %.2f, "
            "场景切换阈值: %.2f, 异常Z阈值: %.1f, 周期间隔: %.1fs",
            motion_method, motion_threshold,
            scene_switch_threshold, anomaly_zscore_threshold,
            periodic_interval,
        )

    # ── 公共接口 ──────────────────────────────────────

    def detect(self, frame: np.ndarray, timestamp: float) -> List[str]:
        """对帧运行所有触发器（OR 逻辑）。

        Returns:
            触发器名称列表，空列表表示无触发。
        """
        triggers: List[str] = []

        # 下采样
        small = cv2.resize(frame, (_TRIGGER_WIDTH, _TRIGGER_HEIGHT), interpolation=cv2.INTER_NEAREST)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # ── 首帧：初始化参考帧并以周期触发方式返回 ──
        if self._prev_gray_small is None:
            # 更新所有参考帧
            self._prev_gray_small = gray_small
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            self._prev_hsv_hists = self._compute_block_histograms(hsv)
            try:
                import imagehash
                pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                self._prev_phash = imagehash.phash(pil_img)
            except ImportError:
                pass
            # 更新周期采样的起始时间
            self._last_periodic_ts = timestamp
            # 更新异常检测的相邻帧参考
            self._last_gray_for_anomaly = gray_small.copy()
            self._trigger_count += 1
            logger.debug("[L1首帧] 自动触发 periodic")
            return ['periodic']

        # 生成帧差掩码（供 scene_switch / anomaly / 裁剪使用）
        # 注意：此时使用的是上一次触发时的参考帧，而非上一帧
        if self._prev_gray_small is not None:
            diff = cv2.absdiff(gray_small, self._prev_gray_small)
            _, diff_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            # 上采样到原始尺寸供裁剪使用
            h, w = frame.shape[:2]
            self._last_diff_mask = cv2.resize(diff_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Trigger 1: 剧烈运动
        if self._check_motion(frame):
            triggers.append('motion')

        # Trigger 2: 场景切换
        if self._check_scene_switch(small):
            triggers.append('scene_switch')

        # Trigger 3: 异常事件
        if self._check_anomaly(gray_small):
            triggers.append('anomaly')

        # Trigger 4: 周期强制
        if self._check_periodic(timestamp):
            triggers.append('periodic')

        # ⭐ 关键逻辑：只有触发时才更新参考帧，且根据触发类型选择性更新
        if triggers:
            self._trigger_count += 1
            
            # 所有触发类型都需要更新灰度参考帧（用于帧差计算）
            # 原因：触发后，该帧应成为后续帧差计算的基准
            self._prev_gray_small = gray_small
            
            # 场景切换或异常触发 → 额外更新场景参考帧
            # 原因：这两种情况说明场景本身发生了变化
            if 'scene_switch' in triggers or 'anomaly' in triggers:
                # 更新 HSV 直方图
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                self._prev_hsv_hists = self._compute_block_histograms(hsv)
                # 更新感知哈希
                try:
                    import imagehash
                    pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                    self._prev_phash = imagehash.phash(pil_img)
                except ImportError:
                    pass
            elif 'periodic' in triggers:
                # 周期强制触发 → 额外更新场景参考帧
                # 原因：长时间未采样，需要重置参考基准
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
                self._prev_hsv_hists = self._compute_block_histograms(hsv)
                try:
                    import imagehash
                    pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                    self._prev_phash = imagehash.phash(pil_img)
                except ImportError:
                    pass
            # 'motion' only → 只更新 _prev_gray_small，不更新场景参考帧
            # 原因：运动不代表场景变化，但帧差基准需要更新
        
        # 更新用于异常检测的上一帧（始终更新，用于计算相邻帧差异）
        self._last_gray_for_anomaly = gray_small.copy()
        
        # 注意：last_diff_mask 始终保留，供 Layer 2 裁剪使用
        # 只有在 reset() 时才清除

        return triggers

    def reset(self) -> None:
        """重置所有触发器状态。"""
        self._motion_detector.reset()
        self._prev_hsv_hists = None
        self._prev_phash = None
        self._feature_window.clear()
        self._last_gray_for_anomaly = None
        self._last_periodic_ts = -float('inf')
        self._last_motion_score = 0.0
        self._last_fg_mask = None
        self._last_diff_mask = None
        self._prev_gray_small = None
        self._trigger_count = 0
        logger.info("FastTriggers 状态已重置")

    # ── 属性 ──────────────────────────────────────────

    @property
    def trigger_count(self) -> int:
        """累计有触发的帧数。"""
        return self._trigger_count

    @property
    def motion_method(self) -> str:
        return self._motion_method_name

    @property
    def last_motion_score(self) -> float:
        return self._last_motion_score

    @property
    def last_fg_mask(self) -> Optional[np.ndarray]:
        """最近一帧的背景减除前景掩码。"""
        return self._last_fg_mask

    @property
    def last_diff_mask(self) -> Optional[np.ndarray]:
        """最近一帧的帧差掩码（原始分辨率），供 scene_switch/anomaly 裁剪使用。"""
        return self._last_diff_mask

    # ── Trigger 1: 剧烈运动检测 ──────────────────────

    def _check_motion(self, frame: np.ndarray) -> bool:
        """使用背景减除法检测运动。"""
        has_motion, motion_score, fg_mask = self._motion_detector.detect_motion(frame)
        self._last_motion_score = motion_score
        self._last_fg_mask = fg_mask
        return has_motion

    # ── Trigger 2: 场景切换检测 ────────────────────────

    def _check_scene_switch(self, small_bgr: np.ndarray) -> bool:
        """HSV 分块直方图比对 + 感知哈希预筛。
        
        注意：参考帧仅在成功触发时更新，因此这里对比的是
        当前帧与上一次成功触发的帧之间的差异，而非与上一帧对比。
        这样可以检测到累积的场景变化。
        """
        hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
        curr_hists = self._compute_block_histograms(hsv)

        triggered = False

        if self._prev_hsv_hists is not None:
            # 分块直方图比对，取最小相关系数
            min_correl = 1.0
            for h_curr, h_prev in zip(curr_hists, self._prev_hsv_hists):
                correl = cv2.compareHist(h_curr, h_prev, cv2.HISTCMP_CORREL)
                min_correl = min(min_correl, correl)

            if min_correl < self._scene_switch_threshold:
                triggered = True
            else:
                # 感知哈希辅助预筛
                triggered = self._check_phash(small_bgr)

        # ⭐ 不在这里更新参考帧！参考帧由 detect() 方法在触发时统一更新
        return triggered

    def _compute_block_histograms(self, hsv: np.ndarray) -> List[np.ndarray]:
        """计算 3×3 九宫格分块 HSV 直方图。"""
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

    def _check_phash(self, small_bgr: np.ndarray) -> bool:
        """感知哈希快速比对。
        
        注意：参考帧由 detect() 方法统一管理，不在这里更新。
        """
        try:
            import imagehash
            pil_img = Image.fromarray(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB))
            curr_hash = imagehash.phash(pil_img)
            if self._prev_phash is not None:
                distance = curr_hash - self._prev_phash
                # ⭐ 不在这里更新！由 detect() 在触发时统一更新
                return distance > 15
            # _prev_phash 为 None 说明是首帧，但首帧已在 detect() 中初始化
            # 如果到这里仍未初始化，说明 imagehash 在首帧时不可用，跳过检测
            return False
        except ImportError:
            # imagehash 不可用时跳过
            return False

    # ── Trigger 3: 异常事件检测 ────────────────────────

    def _check_anomaly(self, gray_small: np.ndarray) -> bool:
        """时序异常检测 - 多维 Z-Score。"""
        # 提取三维特征
        features = self._extract_anomaly_features(gray_small)
        self._feature_window.append(features)

        # 窗口未满时不检测
        if len(self._feature_window) < 10:
            return False

        # 计算 Z-Score
        window_arr = np.array(self._feature_window)  # shape: (N, 3)
        mean = np.mean(window_arr, axis=0)
        std = np.std(window_arr, axis=0)

        z_scores = np.abs((features - mean) / (std + 1e-8))
        return bool(np.any(z_scores > self._anomaly_zscore_threshold))

    def _extract_anomaly_features(self, gray_small: np.ndarray) -> np.ndarray:
        """提取三维特征向量: [运动强度, 颜色方差, 边缘密度]。"""
        # 运动强度：与上一帧的差分均值（使用相邻帧差异，而非参考帧差异）
        motion_intensity = 0.0
        if self._last_gray_for_anomaly is not None:
            diff = cv2.absdiff(gray_small, self._last_gray_for_anomaly)
            motion_intensity = float(np.mean(diff))

        # 颜色方差
        color_variance = float(np.var(gray_small))

        # 边缘密度（Sobel 梯度均值）
        sobel_x = cv2.Sobel(gray_small, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_small, cv2.CV_64F, 0, 1, ksize=3)
        edge_density = float(np.mean(np.sqrt(sobel_x ** 2 + sobel_y ** 2)))

        return np.array([motion_intensity, color_variance, edge_density])

    # ── Trigger 4: 周期强制采样 ────────────────────────

    def _check_periodic(self, timestamp: float) -> bool:
        """定时器机制强制触发。"""
        if timestamp - self._last_periodic_ts >= self._periodic_interval:
            self._last_periodic_ts = timestamp
            return True
        return False
