"""MLSmartSampler - 分层漏斗型智能帧过滤器

三层架构:
  Layer 0 (硬过滤): 快速排除无价值帧，90%+ 拒绝率，<0.1ms
  Layer 1 (快速触发): 多路 OR 并行检测，1-3ms
  Layer 2 (精细验证): 多特征融合打分 + 峰值检测，最终通过 1-2%

接口与 SmartSampler 完全一致，可在 LivePipeline 中无缝替换。
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Iterator, List, Optional

import cv2
import numpy as np
from PIL import Image

from .base import SmartSampler
from .hard_filter import HardFilter
from .fast_triggers import FastTriggers
from .frame_validator import FrameValidator

logger = logging.getLogger(__name__)


class MLSmartSampler(SmartSampler):
    """分层漏斗型智能帧过滤器。

    继承自 SmartSampler 基类，实现三层漏斗架构的智能采样。
    """

    def __init__(
        self,
        enable_smart_sampling: bool = True,
        motion_method: str = 'MOG2',
        motion_threshold: float = 0.1,
        backup_interval: float = 30.0,
        min_frame_interval: float = 1.0,
        scene_switch_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            enable_smart_sampling=enable_smart_sampling,
            backup_interval=backup_interval,
            min_frame_interval=min_frame_interval,
        )
        self._yield_count: int = 0

        # Layer 0: 硬性过滤器
        self.hard_filter = HardFilter(
            min_frame_interval=min_frame_interval,
        )

        # Layer 1: 快速触发器组
        self.fast_triggers = FastTriggers(
            motion_method=motion_method,
            motion_threshold=motion_threshold,
            scene_switch_threshold=scene_switch_threshold,
            periodic_interval=backup_interval,
        )

        # Layer 2: 精细验证器
        self.frame_validator = FrameValidator()

        logger.info(
            "MLSmartSampler 初始化完成 - 智能采样: %s, "
            "运动阈值: %.2f, 最小间隔: %.1fs, 保底间隔: %.1fs",
            "启用" if enable_smart_sampling else "禁用",
            motion_threshold,
            min_frame_interval, backup_interval,
        )

    # ── 属性 ──────────────────────────────────────────

    @property
    def frame_count(self) -> int:
        """已送入的输入帧总数。"""
        return self._input_frame_count

    # ── 核心采样接口 ──────────────────────────────────

    def sample(
        self, frames: Iterator[tuple[np.ndarray, float]],
    ) -> Iterator[Dict[str, Any]]:
        """三层漏斗过滤，接口与 SmartSampler.sample() 完全一致。

        Args:
            frames: 帧迭代器，每个元素是 (BGR图像, 时间戳) 元组。

        Yields:
            采样结果字典（格式与 SmartSampler 一致）。
        """
        if not hasattr(self, '_sample_started'):
            logger.info(
                "开始智能采样 - 模式: %s",
                "MLSmartSampler(三层漏斗)" if self.enable_smart else "时间采样",
            )
            self._sample_started = True

        for frame_np, ts in frames:
            if frame_np is None or frame_np.size == 0:
                continue

            # 全局帧序号（含被拒帧）
            current_frame_idx = self._input_frame_count
            self._input_frame_count += 1

            if not self.enable_smart:
                # 降级为纯时间采样（传统固定间隔采样）
                if self._should_emit_by_time(ts):
                    pil_image = self._numpy_to_pil(frame_np)
                    yield {
                        'image': pil_image,
                        'timestamp': ts,
                        'frame_index': current_frame_idx,
                        'significant': False,
                        'source': ['traditional'],  # 传统固定间隔采样
                        'original_frame': frame_np,
                    }
                    self._last_emit_ts = ts
                continue

            # ── Layer 0: 硬过滤 ──
            passed, reason = self.hard_filter.check(frame_np, ts)
            if not passed:
                logger.debug(
                    "[L0拒绝] 帧#%d | 原因=%s", current_frame_idx, reason,
                )
                continue

            # ── Layer 1: 快速触发 ──
            triggers = self.fast_triggers.detect(frame_np, ts)
            if not triggers:
                logger.debug(
                    "[L1无触发] 帧#%d | ts=%.3fs", current_frame_idx, ts,
                )
                continue

            # ── Layer 2: 精细验证 ──
            validation = self.frame_validator.validate(frame_np, ts, triggers)
            if not validation['passed']:
                logger.debug(
                    "[L2拒绝] 帧#%d | 分数=%.3f | 窗口均值=%.3f | 峰值=%s",
                    current_frame_idx, validation['score'],
                    validation['window_mean'],
                    '是' if validation['is_peak'] else '否',
                )
                continue

            # ── 通过：组装返回字典 ──
            result = self._build_result(
                frame_np, ts, current_frame_idx, triggers, validation,
            )
            self._last_emit_ts = ts
            self._yield_count += 1
            yield result

    # ── 状态管理 ──────────────────────────────────────

    def reset(self) -> None:
        """重置所有状态。"""
        self._last_emit_ts = -float('inf')
        self._input_frame_count = 0
        self._yield_count = 0
        self.hard_filter.reset()
        self.fast_triggers.reset()
        self.frame_validator.reset()
        logger.info("MLSmartSampler 状态已重置")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息（兼容 SmartSampler 格式 + 扩展字段）。"""
        return {
            # SmartSampler 兼容字段
            'total_frames_processed': self.frame_count,
            'smart_sampling_enabled': self.enable_smart,
            'backup_interval': self._backup_interval,
            'min_frame_interval': self._min_frame_interval,
            'motion_detector_method': self.fast_triggers.motion_method,
            # 三层统计扩展
            'layer0_reject_count': self.hard_filter.reject_count,
            'layer0_pass_rate': self.hard_filter.pass_rate,
            'layer1_trigger_count': self.fast_triggers.trigger_count,
            'layer2_pass_count': self.frame_validator.pass_count,
            'final_yield_count': self._yield_count,
        }

    # ── 内部方法 ──────────────────────────────────────

    def _build_result(
        self,
        frame_np: np.ndarray,
        ts: float,
        frame_idx: int,
        triggers: List[str],
        validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """组装与 SmartSampler 完全一致的返回字典，并打印日志。"""
        pil_image = self._numpy_to_pil(frame_np)

        # change_metrics（兼容 SmartSampler 格式）
        motion_score = self.fast_triggers.last_motion_score
        ssim_score = validation.get('ssim_score', 1.0)
        combined_score = validation.get('score', 0.0)

        # 判断是否为显著帧：非周期触发即为显著
        is_significant = 'periodic' not in triggers

        result = {
            'image': pil_image,
            'timestamp': ts,
            'frame_index': frame_idx,
            'significant': is_significant,
            'source': triggers,  # 直接使用触发器列表
            'original_frame': frame_np,
            'change_metrics': {
                'ssim_score': ssim_score,
                'combined_score': combined_score,
                'motion_score': motion_score,
            },
        }

        # 日志（扩展格式）
        has_motion = 'motion' in triggers
        is_peak = validation.get('is_peak', False)
        window_mean = validation.get('window_mean', 0.0)
        l0_rate = self.hard_filter.pass_rate * 100
        l1_rate = (
            self.fast_triggers.trigger_count / self.hard_filter.accept_count * 100
            if self.hard_filter.accept_count > 0 else 0.0
        )

        # 生成触发源标签用于日志展示
        source_label = '、'.join(triggers)

        logger.info(
            "[送VLM] 帧#%d | ts=%.3fs | "
            "L2得分=%.3f | 窗口均值=%.3f | 峰值=%s | "
            "运动=%s(得分=%.3f) | "
            "来源=%s | "
            "触发器=%s | "
            "L0通过率=%.1f%% | L1触发率=%.1f%%",
            frame_idx, ts,
            validation['score'], window_mean, '是' if is_peak else '否',
            '是' if has_motion else '否', motion_score,
            source_label,
            triggers,
            l0_rate, l1_rate,
        )

        return result

    def _should_emit_by_time(self, ts: float) -> bool:
        """保底时间间隔检查（降级模式用）。"""
        if ts < self._last_emit_ts:
            self._last_emit_ts = -float('inf')
            return True
        return (ts - self._last_emit_ts) >= self._backup_interval

    @staticmethod
    def _numpy_to_pil(frame_np: np.ndarray) -> Image.Image:
        """BGR numpy → RGB PIL。"""
        rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)
