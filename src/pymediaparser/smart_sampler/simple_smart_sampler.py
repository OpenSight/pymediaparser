"""SimpleSmartSampler - 简单智能采样器实现（优化版）

采用活动状态机 + 多策略活动检测 + 多维度静态过滤。
检测人、宠物、飞鸟、牲畜等所有有意义的动物活动。
"""

from __future__ import annotations
import logging
from typing import Iterator, Dict, Any, Optional, List
from enum import Enum
import numpy as np
from PIL import Image
import cv2

from .base import SmartSampler
from .motion_detector import MotionDetector
from .meaningful_activity_detector import MeaningfulActivityDetector
from .scene_transition_tracker import SceneTransitionTracker
from .static_filter import StaticFilter

logger = logging.getLogger(__name__)


class ActivityState(Enum):
    """活动状态枚举。"""
    IDLE = "idle"          # 空闲状态，需触发条件才采样
    ACTIVE = "active"      # 活动状态，所有帧都采样


class SimpleSmartSampler(SmartSampler):
    """简单智能采样器 - 基于活动状态机的自适应帧采样（优化版）。

    继承自 SmartSampler 基类，实现活动状态机架构的智能采样。
    
    新架构：
    - 有意义的活动检测（人/宠物/飞鸟/牲畜等）
    - 场景连续变化追踪（云台旋转/镜头推移）
    - 多维度静态过滤（排除静态/微光变/风吹植物）
    """

    def __init__(
        self,
        # 核心效果参数（用户可配置）
        sensitivity: float = 0.5,              # 活动检测灵敏度 (0.0-1.0)
        activity_duration: float = 3.0,        # 活动期持续时间（秒）
        quiet_frames_threshold: int = 10,      # 静默帧阈值（连续无活动帧数）
        backup_interval: float = 30.0,         # 保底采样间隔（秒）
        min_frame_interval: float = 1.0,       # 最小帧间隔（秒）
        
        # 基类兼容参数
        enable_smart_sampling: bool = True,
        
        # 保留的功能参数
        motion_method: str = 'MOG2',
    ) -> None:
        """初始化优化版 SimpleSmartSampler。

        Args:
            sensitivity: 活动检测灵敏度 (0.0-1.0)，值越低越敏感。
            activity_duration: 活动期持续时间（秒），检测到活动后此期间内所有帧都采样。
            quiet_frames_threshold: 连续多少帧无活动后退出活动期。
            backup_interval: 保底采样间隔（秒），长时间无活动时的强制采样频率。
            min_frame_interval: 最小帧间隔（秒），防止过于频繁采样。
            enable_smart_sampling: 是否启用智能采样。
            motion_method: 运动检测方法（MOG2 或 KNN）。
        """
        super().__init__(
            enable_smart_sampling=enable_smart_sampling,
            backup_interval=backup_interval,
            min_frame_interval=min_frame_interval,
        )

        # 核心参数
        self._sensitivity = sensitivity
        self._activity_duration = activity_duration
        self._quiet_frames_threshold = quiet_frames_threshold

        # 活动状态
        self._activity_state: ActivityState = ActivityState.IDLE
        self._quiet_counter: int = 0  # 连续无活动帧计数
        self._last_activity_ts: float = -float('inf')  # 上次检测到活动的时间戳

        # 检测器组件
        self.motion_detector = MotionDetector(
            method=motion_method,
            threshold=0.01,  # 降低阈值，让 MOG2 更敏感
        )
        self.activity_detector = MeaningfulActivityDetector(
            sensitivity=sensitivity,
        )
        self.scene_tracker = SceneTransitionTracker(
            sensitivity=sensitivity,
        )
        self.static_filter = StaticFilter(
            sensitivity=sensitivity,
        )

        logger.info(
            "SimpleSmartSampler 初始化完成 - 智能采样: %s, "
            "灵敏度: %.2f, 活动持续时间: %.1fs, 静默阈值: %d帧, 保底间隔: %.1fs",
            "启用" if enable_smart_sampling else "禁用",
            sensitivity, activity_duration, quiet_frames_threshold, backup_interval,
        )

    # ── 属性 ──────────────────────────────────────────────

    @property
    def frame_count(self) -> int:
        """已送入的输入帧总数。"""
        return self._input_frame_count

    # ── 核心采样接口 ──────────────────────────────────────

    def sample(
        self, frames: Iterator[tuple[Image.Image, float, int]],
    ) -> Iterator[Dict[str, Any]]:
        """智能采样主入口。

        Args:
            frames: 帧迭代器，每个元素是 (PIL图像, 时间戳, 帧序号) 元组。

        Yields:
            采样结果字典。
        """
        # 只在首次调用时打印一次日志
        if not hasattr(self, '_sample_started'):
            logger.info("开始智能采样 - 模式: %s", "智能采样" if self.enable_smart else "时间采样")
            self._sample_started = True
        
        for pil_image, ts, idx in frames:
            # 跳过 None 和空帧
            if pil_image is None:
                continue
            # 如果是 numpy 数组（向后兼容），检查是否为空
            if isinstance(pil_image, np.ndarray):
                if pil_image.size == 0:
                    continue
                # 转换为 PIL
                pil_image = cv2.cvtColor(pil_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(pil_image)
            
            # 递增输入帧计数器
            self._input_frame_count += 1
            current_frame_idx = idx
            
            # 检查最小帧间隔（在保底时间检查之前）
            if not self._check_min_frame_interval(ts):
                continue
                
            time_based_emit = self._should_emit_by_time(ts)
            
            if self.enable_smart:
                # 转换为 numpy 供内部处理
                frame_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                sample_result = self._smart_sample_frame(frame_np, ts, current_frame_idx, time_based_emit)
                if sample_result:
                    yield sample_result
            else:
                if time_based_emit:
                    yield {
                        'image': pil_image,
                        'timestamp': ts,
                        'frame_index': current_frame_idx,
                        'significant': False,
                        'source': ['traditional'],
                    }
                    self._last_emit_ts = ts

    # ── 内部方法 ──────────────────────────────────────────

    def _check_min_frame_interval(self, ts: float) -> bool:
        """检查是否满足最小帧间隔。
        
        防止过于频繁的采样，即使检测到活动也需遵守最小间隔。
        
        Args:
            ts: 当前帧时间戳。
            
        Returns:
            True 表示可以处理此帧，False 表示应跳过。
        """
        if ts - self._last_emit_ts < self._min_frame_interval:
            return False
        return True

    def _smart_sample_frame(
        self, frame_np: np.ndarray, ts: float, frame_idx: int,
        time_based_emit: bool,
    ) -> Optional[Dict[str, Any]]:
        """智能采样单帧处理。

        Args:
            frame_np: BGR 格式的帧。
            ts: 时间戳。
            frame_idx: 帧序号。
            time_based_emit: 是否满足保底时间触发。

        Returns:
            采样结果字典，或 None（跳过）。
        """
        # 1. 保底时间触发（强制采样，但不进入 ACTIVE 状态）
        if time_based_emit:
            return self._emit_frame(
                frame_np, ts, frame_idx, ['periodic'],
                activity_score=0.0, motion_score=0.0, ssim_score=1.0,
            )

        # 2. 运行 MOG2 获取前景掩码（用于后续分析）
        has_motion, motion_score, fg_mask = self.motion_detector.detect_motion(frame_np)

        # 3. 根据当前状态处理
        if self._activity_state == ActivityState.ACTIVE:
            return self._process_active_state(
                frame_np, ts, frame_idx, motion_score, fg_mask,
            )
        else:  # IDLE 状态
            return self._process_idle_state(
                frame_np, ts, frame_idx, motion_score, fg_mask, has_motion,
            )

    def _process_active_state(
        self,
        frame_np: np.ndarray,
        ts: float,
        frame_idx: int,
        motion_score: float,
        fg_mask: Optional[np.ndarray],
    ) -> Optional[Dict[str, Any]]:
        """处理 ACTIVE 状态：所有帧都 emit，同时检测活动是否结束。

        Returns:
            采样结果字典。
        """
        # 检测是否有活动
        has_activity, activity_score = self.activity_detector.detect(frame_np, fg_mask)
        is_scene_transition, _ = self.scene_tracker.detect(frame_np)

        # 更新静默计数器
        if has_activity or is_scene_transition:
            self._quiet_counter = 0
            self._last_activity_ts = ts
        else:
            self._quiet_counter += 1

        # 检查是否应该退出 ACTIVE
        should_exit = False

        # 条件 1: 连续 N 帧无活动
        if self._quiet_counter >= self._quiet_frames_threshold:
            should_exit = True

        # 条件 2: 超过 activity_duration 秒无活动
        if ts - self._last_activity_ts > self._activity_duration:
            should_exit = True

        if should_exit:
            self._enter_idle()

        # ACTIVE 状态下所有帧都 emit
        triggers = []
        if has_activity or motion_score > 0.01:
            triggers.append('motion')
        if is_scene_transition:
            triggers.append('scene_switch')
        if not triggers:
            triggers.append('motion')  # 默认标记为 motion

        return self._emit_frame(
            frame_np, ts, frame_idx, triggers,
            activity_score=activity_score,
            motion_score=motion_score,
            ssim_score=1.0,
        )

    def _process_idle_state(
        self,
        frame_np: np.ndarray,
        ts: float,
        frame_idx: int,
        motion_score: float,
        fg_mask: Optional[np.ndarray],
        has_motion: bool,
    ) -> Optional[Dict[str, Any]]:
        """处理 IDLE 状态：需触发条件才进入 ACTIVE。

        Returns:
            采样结果字典，或 None（跳过）。
        """
        # 无运动且无前景 → 快速跳过
        fg_nonzero = cv2.countNonZero(fg_mask) if fg_mask is not None else 0
        total_pixels = frame_np.shape[0] * frame_np.shape[1]
        fg_ratio = fg_nonzero / total_pixels if total_pixels > 0 else 0.0

        if not has_motion and fg_nonzero == 0:
            logger.debug(
                "[IDLE-快速跳过] 帧#%d | 无运动前景", frame_idx,
            )
            return None

        # 运动像素比例过高（>50%）通常是光线变化而非真实活动
        # 但云台旋转也会导致高 fg_ratio，需先检查场景变化
        is_scene_transition, scene_score = self.scene_tracker.detect(frame_np)

        # 场景追踪器需要至少窗口满才能检测场景变化
        # 在此之前，允许高 fg_ratio 帧通过，避免误杀
        tracker_warmup = len(self.scene_tracker._corr_window) < self.scene_tracker._min_consecutive

        # 检查是否有相关性下降趋势（云台旋转早期信号）
        has_corr_downtrend = self._check_correlation_downtrend()

        if fg_ratio > 0.50 and not is_scene_transition and not tracker_warmup and not has_corr_downtrend:
            logger.debug(
                "[IDLE-光线变化] 帧#%d | 前景占比=%.1f%% (>50%%) 且非场景变化",
                frame_idx, fg_ratio * 100,
            )
            return None

        # 活动检测（较重，仅在有运动前景且非光线变化时执行）
        has_activity, activity_score = self.activity_detector.detect(frame_np, fg_mask)

        # 检测触发条件
        should_trigger = has_activity or is_scene_transition

        if should_trigger:
            # 检测到有意义活动或场景变化 → 直接 emit，不受静态过滤约束
            self._enter_active(ts)

            # 组装触发源
            triggers = []
            if has_activity or has_motion:
                triggers.append('motion')
            if is_scene_transition:
                triggers.append('scene_switch')
            if not triggers:
                triggers.append('motion')

            return self._emit_frame(
                frame_np, ts, frame_idx, triggers,
                activity_score=activity_score,
                motion_score=motion_score,
                ssim_score=1.0,
            )

        # 有运动前景但未检测到有意义活动 → 静态过滤（排除风吹植物等）
        passed, reason = self.static_filter.check(
            frame_np, motion_score, has_activity,
        )
        if not passed:
            logger.debug(
                "[IDLE-静态过滤] 帧#%d | 原因=%s",
                frame_idx, reason,
            )
            return None

        # 通过静态过滤但无触发 → 跳过
        return None

    def _enter_active(self, ts: float) -> None:
        """进入 ACTIVE 状态。"""
        if self._activity_state != ActivityState.ACTIVE:
            logger.info(
                "[状态转换] IDLE → ACTIVE (ts=%.3f)", ts,
            )
        self._activity_state = ActivityState.ACTIVE
        self._quiet_counter = 0
        self._last_activity_ts = ts

    def _enter_idle(self) -> None:
        """进入 IDLE 状态。"""
        if self._activity_state != ActivityState.IDLE:
            logger.info(
                "[状态转换] ACTIVE → IDLE (静默帧数=%d)",
                self._quiet_counter,
            )
        self._activity_state = ActivityState.IDLE
        self._quiet_counter = 0

    def _emit_frame(
        self,
        frame_np: np.ndarray,
        ts: float,
        frame_idx: int,
        triggers: List[str],
        activity_score: float,
        motion_score: float,
        ssim_score: float,
    ) -> Dict[str, Any]:
        """组装并返回采样结果。"""
        pil_image = self._numpy_to_pil(frame_np)

        # 判断是否为显著帧：非周期触发即为显著
        is_significant = 'periodic' not in triggers or len(triggers) > 1

        result = {
            'image': pil_image,
            'timestamp': ts,
            'frame_index': frame_idx,
            'significant': is_significant,
            'source': triggers,
            'change_metrics': {
                'ssim_score': ssim_score,
                'combined_score': activity_score,
                'motion_score': motion_score,
            }
        }

        self._last_emit_ts = ts

        # 打印日志
        source_label = '、'.join(triggers)
        logger.info(
            "[送VLM] 帧#%d | ts=%.3fs | "
            "活动得分=%.3f | 运动=%.3f | "
            "状态=%s | 静默计数=%d | "
            "来源=%s",
            frame_idx, ts,
            activity_score, motion_score,
            self._activity_state.value,
            self._quiet_counter,
            source_label,
        )

        return result

    # ── 状态管理 ──────────────────────────────────────────

    def _check_correlation_downtrend(self) -> bool:
        """检查 SceneTransitionTracker 的相关性是否有下降趋势。

        用于早期检测云台旋转，即使还没正式触发场景变化。
        如果窗口中有至少 2 帧，且相关性持续下降（允许 1 个异常点），
        则返回 True，表示可能是云台旋转而非光线突变。

        Returns:
            是否有相关性下降趋势。
        """
        corr_window = self.scene_tracker._corr_window
        if len(corr_window) < 2:
            return False

        # 取最近 3 帧（或更少）
        recent = list(corr_window)[-3:]
        if len(recent) < 2:
            return False

        # 检查是否整体呈下降趋势
        # 简单方法：最后一帧 < 第一帧，且中间没有大幅上升
        if recent[-1] < recent[0] - 0.005:  # 至少下降 0.005
            # 检查是否有大幅上升（超过 0.01 的上升可能是噪声）
            for i in range(1, len(recent)):
                if recent[i] > recent[i-1] + 0.01:
                    return False
            return True

        return False

    def reset(self) -> None:
        """重置所有状态。"""
        self._last_emit_ts = -float('inf')
        self._input_frame_count = 0
        self._activity_state = ActivityState.IDLE
        self._quiet_counter = 0
        self._last_activity_ts = -float('inf')
        self.motion_detector.reset()
        self.activity_detector.reset()
        self.scene_tracker.reset()
        self.static_filter.reset()
        logger.info("SimpleSmartSampler 状态已重置")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息。"""
        return {
            'total_frames_processed': self.frame_count,
            'smart_sampling_enabled': self.enable_smart,
            'backup_interval': self._backup_interval,
            'sensitivity': self._sensitivity,
            'activity_duration': self._activity_duration,
            'quiet_frames_threshold': self._quiet_frames_threshold,
            'current_state': self._activity_state.value,
            'quiet_counter': self._quiet_counter,
            'motion_detector_method': self.motion_detector.method,
        }
