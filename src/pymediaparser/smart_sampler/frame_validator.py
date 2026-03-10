"""精细验证器 (Layer 2) - 多特征融合评分，过滤误触发，输出高置信度关键帧"""

from __future__ import annotations
import collections
import logging
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 下采样尺寸（光流计算用）
_FLOW_WIDTH = 320
_FLOW_HEIGHT = 180


class FrameValidator:
    """Layer 2 精细验证器 - 多特征融合打分 + 时序平滑 + 峰值检测 + NMS。

    五维特征：
    - motion_persistence: 稀疏光流运动持续性
    - structural_change: 边缘 IoU 结构变化
    - content_semantics: 感知哈希汉明距离
    - temporal_coherence: 与历史关键帧时序连贯性
    - anomaly_significance: 触发器推断的异常显著性
    """

    def __init__(
        self,
        score_window_size: int = 10,
        peak_ratio: float = 1.10,
        nms_interval: float = 0.5,
    ) -> None:
        self._window_size = score_window_size
        self._peak_ratio = peak_ratio
        self._nms_interval = nms_interval

        # 基础权重
        self._base_weights = {
            'motion_persistence': 0.30,
            'structural_change': 0.25,
            'content_semantics': 0.20,
            'temporal_coherence': 0.15,
            'anomaly_significance': 0.10,
        }

        # 状态
        self._prev_gray_small: Optional[np.ndarray] = None
        self._prev_edges: Optional[np.ndarray] = None
        self._prev_phash: Optional[object] = None
        self._reference_grays: Deque[np.ndarray] = collections.deque(maxlen=5)  # 历史关键帧灰度
        self._score_window: Deque[float] = collections.deque(maxlen=score_window_size)
        self._last_pass_ts: float = -float('inf')
        self._last_pass_score: float = 0.0
        self._pass_count: int = 0

        logger.debug(
            "FrameValidator 初始化完成 - 窗口: %d, 峰值比: %.2f, NMS间隔: %.1fs",
            score_window_size, peak_ratio, nms_interval,
        )

    # ── 公共接口 ──────────────────────────────────────

    def validate(
        self,
        frame: np.ndarray,
        timestamp: float,
        triggers: List[str],
    ) -> Dict[str, Any]:
        """对候选帧进行多特征融合验证。

        验证通过时自动更新参考帧，无需外部调用 update_reference()。

        Args:
            frame: BGR 格式图像帧。
            timestamp: 帧时间戳。
            triggers: Layer 1 触发器列表。

        Returns:
            验证结果字典:
            - passed: 是否通过
            - score: 融合分数
            - features: 各特征分值
            - is_peak: 是否为峰值
            - window_mean: 窗口均值
        """
        # 提取灰度图（所有路径都需要）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (_FLOW_WIDTH, _FLOW_HEIGHT), interpolation=cv2.INTER_AREA)

        # 首帧或周期触发：直接通过并更新参考帧
        is_first_frame = self._prev_gray_small is None
        is_periodic = triggers == ['periodic']
        if is_first_frame or is_periodic:
            self._pass_count += 1
            self._last_pass_ts = timestamp
            self._update_prev_states(gray_small)
            self._reference_grays.append(gray_small.copy())
            reason = "首帧" if is_first_frame else "周期触发"
            logger.debug("[L2%s] 自动通过 | 历史帧数=%d", reason, len(self._reference_grays))
            return {
                'passed': True,
                'score': 0.5,
                'features': {k: 0.5 for k in self._base_weights},
                'is_peak': False,
                'window_mean': 0.5,
            }

        # 提取特征
        features = self._extract_features(gray_small, triggers)

        # 动态权重调整
        weights = self._adjust_weights(triggers)

        # 融合打分
        score = sum(
            features.get(k, 0.0) * w
            for k, w in weights.items()
        )

        # 时序平滑
        self._score_window.append(score)
        window_mean = float(np.mean(self._score_window)) if self._score_window else score

        # 峰值检测
        is_peak = False
        if len(self._score_window) >= 3:
            # 方式1: 分数超过均值的 peak_ratio 倍
            exceeds_mean = score > self._peak_ratio * window_mean

            # 方式2: 局部极大值（当前分数是最近5帧中的最大值）
            recent = list(self._score_window)[-5:]
            is_local_max = (score >= max(recent))

            # 峰值条件：满足以下任一
            # 1. 同时满足超过均值比例 + 局部极大值
            # 2. 或者分数显著高于均值（超过1.5倍峰值比，即比均值高出65%以上）
            significant_exceed = score > (self._peak_ratio * 1.5) * window_mean
            is_peak = (exceeds_mean and is_local_max) or significant_exceed

            if is_peak and not is_local_max:
                logger.debug(
                    "[L2峰值] 分数=%.3f 显著高于均值=%.3f（比=%.2f），直接通过",
                    score, window_mean, score / (window_mean + 1e-8),
                )
        else:
            # 窗口数据不足时：使用绝对分数阈值判断
            # 如果分数足够高（> 0.15），直接标记为峰值
            if score > 0.15:
                is_peak = True
                logger.debug(
                    "[L2窗口不足] 分数=%.3f 超过阈值0.15，标记为峰值",
                    score,
                )

        # 多触发并存时降低阈值
        effective_peak_ratio = self._peak_ratio
        if len([t for t in triggers if t != 'periodic']) > 1:
            effective_peak_ratio *= 0.9  # 降低 10%
            is_peak = is_peak or (score > effective_peak_ratio * window_mean)

        # NMS 抑制
        passed = False
        if is_peak:
            if timestamp - self._last_pass_ts >= self._nms_interval:
                passed = True
            elif score > self._last_pass_score:
                passed = True

        if passed:
            self._last_pass_ts = timestamp
            self._last_pass_score = score
            self._pass_count += 1

        # 每次验证后都更新相邻帧状态（用于下一帧的特征计算）
        self._update_prev_states(gray_small)

        # 验证通过时自动更新参考帧
        if passed:
            self._reference_grays.append(gray_small.copy())
            logger.debug("[L2验证通过] 已更新参考帧 | 历史帧数=%d", len(self._reference_grays))

        # 过滤内部字段
        clean_features = {k: v for k, v in features.items() if not k.startswith('_')}

        return {
            'passed': passed,
            'score': score,
            'features': clean_features,
            'is_peak': is_peak,
            'window_mean': window_mean,
        }

    def update_reference(self, frame: np.ndarray) -> None:
        """手动设置参考帧（可选）。

        首帧会自动检测并通过，无需预先调用此方法。

        适用场景：
        - 场景切换后强制重置参考帧
        - 外部需要强制设置特定帧作为参考
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (_FLOW_WIDTH, _FLOW_HEIGHT), interpolation=cv2.INTER_AREA)

        # 更新参考帧列表
        self._reference_grays.append(gray_small.copy())

        # 更新相邻帧状态
        self._update_prev_states(gray_small)

        logger.debug(
            "[L2] 手动设置参考帧 | 历史帧数=%d",
            len(self._reference_grays),
        )

    def _update_prev_states(self, gray_small: np.ndarray) -> None:
        """更新前一帧状态（每次验证后调用，用于下一帧的特征计算）。

        更新内容：
        - _prev_gray_small: 用于光流计算
        - _prev_edges: 用于边缘IoU计算
        - _prev_phash: 用于感知哈希距离计算
        """
        # 更新前一帧灰度图
        self._prev_gray_small = gray_small.copy()

        # 更新前一帧边缘图
        sobel_x = cv2.Sobel(gray_small, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_small, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        self._prev_edges = (edge_mag > 50).astype(np.uint8)

        # 更新感知哈希
        try:
            import imagehash
            from PIL import Image
            pil_img = Image.fromarray(gray_small)
            self._prev_phash = imagehash.phash(pil_img)
        except ImportError:
            pass

    def reset(self) -> None:
        """重置验证器状态。"""
        self._prev_gray_small = None
        self._prev_edges = None
        self._prev_phash = None
        self._reference_grays.clear()
        self._score_window.clear()
        self._last_pass_ts = -float('inf')
        self._last_pass_score = 0.0
        self._pass_count = 0
        logger.info("FrameValidator 状态已重置")

    @property
    def pass_count(self) -> int:
        """累计通过帧数。"""
        return self._pass_count

    # ── 特征提取 ──────────────────────────────────────

    def _extract_features(
        self, gray_small: np.ndarray, triggers: List[str],
    ) -> Dict[str, float]:
        """提取五维特征，各特征归一化到 [0, 1]。"""
        features: Dict[str, float] = {}

        # 1. 运动持续性（稀疏光流）
        features['motion_persistence'] = self._calc_motion_persistence(gray_small)

        # 2. 结构变化（边缘 IoU）
        features['structural_change'] = self._calc_structural_change(gray_small)

        # 3. 内容语义（感知哈希）
        features['content_semantics'] = self._calc_content_semantics(gray_small)

        # 4. 时序连贯性（与历史关键帧 SSIM）
        ssim_val = self._calc_temporal_coherence(gray_small)
        features['temporal_coherence'] = ssim_val
        features['_ssim_raw'] = 1.0 - ssim_val  # 原始 SSIM 相似度（用于日志）

        # 5. 异常显著性（基于触发器推断）
        features['anomaly_significance'] = self._calc_anomaly_significance(triggers)

        return features

    def _calc_motion_persistence(self, gray_small: np.ndarray) -> float:
        """稀疏光流：统计特征点位移 > 阈值的比例。"""
        if self._prev_gray_small is None:
            return 0.0

        # 在前一帧检测特征点
        prev_pts = cv2.goodFeaturesToTrack(
            self._prev_gray_small,
            maxCorners=50,
            qualityLevel=0.3,
            minDistance=7.0,
            blockSize=7,
        )
        if prev_pts is None or len(prev_pts) == 0:
            return 0.0

        # 计算光流
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray_small,
            gray_small,
            prev_pts,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        if next_pts is None:
            return 0.0

        # 统计大位移点比例
        good_mask = status.ravel() == 1
        if not np.any(good_mask):
            return 0.0

        prev_good = prev_pts[good_mask]
        next_good = next_pts[good_mask]
        displacements = np.sqrt(np.sum((next_good - prev_good) ** 2, axis=-1)).ravel()
        moving_ratio = float(np.mean(displacements > 3.0))  # 位移 > 3 像素

        return min(1.0, moving_ratio)

    def _calc_structural_change(self, gray_small: np.ndarray) -> float:
        """边缘检测 + Edge IoU。"""
        # 当前帧边缘
        sobel_x = cv2.Sobel(gray_small, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_small, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        curr_edges = (edge_mag > 50).astype(np.uint8)

        if self._prev_edges is None:
            return 0.0

        # Edge IoU
        intersection = float(np.sum(cv2.bitwise_and(curr_edges, self._prev_edges)))
        union = float(np.sum(cv2.bitwise_or(curr_edges, self._prev_edges)))
        iou = intersection / (union + 1e-8)

        return min(1.0, 1.0 - iou)

    def _calc_content_semantics(self, gray_small: np.ndarray) -> float:
        """感知哈希汉明距离。"""
        try:
            import imagehash
            from PIL import Image
            pil_img = Image.fromarray(gray_small)
            curr_hash = imagehash.phash(pil_img)
            if self._prev_phash is not None:
                # imagehash 对象支持减法运算，返回汉明距离
                distance = int(curr_hash - self._prev_phash)  # type: ignore[operator]
                self._prev_phash = curr_hash
                return min(1.0, distance / 64.0)
            self._prev_phash = curr_hash
            return 0.0
        except ImportError:
            # 回退到简单像素差异
            if self._prev_gray_small is not None:
                diff = cv2.absdiff(gray_small, self._prev_gray_small)
                return min(1.0, float(np.mean(diff)) / 128.0)
            return 0.0

    def _calc_temporal_coherence(self, gray_small: np.ndarray) -> float:
        """与最近关键帧的 SSIM 距离（1 - avg_ssim）。"""
        if not self._reference_grays:
            return 0.0

        similarities = []
        for ref_gray in self._reference_grays:
            sim = self._fast_ssim(ref_gray, gray_small)
            similarities.append(sim)

        avg_ssim = float(np.mean(similarities))
        return min(1.0, max(0.0, 1.0 - avg_ssim))

    def _calc_anomaly_significance(self, triggers: List[str]) -> float:
        """基于触发器推断异常显著性。"""
        if 'anomaly' in triggers:
            return 0.8
        non_periodic = [t for t in triggers if t != 'periodic']
        if len(non_periodic) > 1:
            return 0.5
        return 0.2

    # ── 权重调整 ──────────────────────────────────────

    def _adjust_weights(self, triggers: List[str]) -> Dict[str, float]:
        """根据触发类型动态调整权重。"""
        weights = self._base_weights.copy()
        non_periodic = [t for t in triggers if t != 'periodic']

        if non_periodic == ['motion']:
            # 纯运动触发
            weights['motion_persistence'] *= 1.5
            weights['structural_change'] *= 0.5
        elif non_periodic == ['scene_switch']:
            # 纯场景触发
            weights['structural_change'] *= 1.3
            weights['content_semantics'] *= 1.3
            weights['motion_persistence'] *= 0.7
        elif 'periodic' in triggers and len(non_periodic) == 0:
            # 纯周期触发（均等权重）
            weights = {k: 0.2 for k in weights}

        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    # ── 工具方法 ──────────────────────────────────────

    @staticmethod
    def _fast_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """快速 SSIM 近似计算（基于均值和方差）。"""
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        img1_f = img1.astype(np.float64)
        img2_f = img2.astype(np.float64)

        mu1 = np.mean(img1_f)
        mu2 = np.mean(img2_f)
        sigma1_sq = np.var(img1_f)
        sigma2_sq = np.var(img2_f)
        sigma12 = np.mean((img1_f - mu1) * (img2_f - mu2))

        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)

        return float(numerator / denominator)
