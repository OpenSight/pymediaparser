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
from .foreground_extractor import ForegroundExtractor
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
        crop_padding_ratio: float = 0.2,
    ) -> None:
        super().__init__(
            enable_smart_sampling=enable_smart_sampling,
            backup_interval=backup_interval,
            min_frame_interval=min_frame_interval,
        )
        self._yield_count: int = 0
        self._is_initialized: bool = False  # 标记是否已用首帧初始化参考

        # Layer 0: 硬性过滤器
        self.hard_filter = HardFilter(
            min_frame_interval=min_frame_interval,
        )

        # Layer 1: 快速触发器组
        self.fast_triggers = FastTriggers(
            motion_method=motion_method,
            motion_threshold=motion_threshold,
            periodic_interval=backup_interval,
        )

        # Layer 2: 精细验证器
        self.frame_validator = FrameValidator()

        # 前景提取器（裁剪时保留边缘背景）
        self.foreground_extractor = ForegroundExtractor(padding_ratio=crop_padding_ratio)

        logger.info(
            "MLSmartSampler 初始化完成 - 智能采样: %s, "
            "运动阈值: %.2f, 最小间隔: %.1fs, 保底间隔: %.1fs, 裁剪边缘扩展: %.0f%%",
            "启用" if enable_smart_sampling else "禁用",
            motion_threshold,
            min_frame_interval, backup_interval, crop_padding_ratio * 100,
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

            # ── 首帧初始化 ──
            # 首帧初始化 L0/L1/L2 参考帧，并以周期触发模式输出
            if not self._is_initialized:
                result = self._initialize_with_first_frame(frame_np, ts, current_frame_idx)
                self._yield_count += 1
                yield result
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
        self._is_initialized = False  # 重置初始化标志
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

    def _initialize_with_first_frame(
        self,
        frame_np: np.ndarray,
        ts: float,
        frame_idx: int,
    ) -> Dict[str, Any]:
        """用首帧初始化 L0、L1、L2 层的参考帧，并以周期触发模式输出。

        首帧作为参考帧初始化各层状态，同时作为第一个周期触发帧输出到 VLM。
        适用于实时流和回放场景，保证语义一致性。
        """
        logger.info(
            "[首帧初始化] 帧#%d | ts=%.3fs | 作为参考帧初始化L0/L1/L2，以周期触发模式输出",
            frame_idx, ts,
        )

        # 初始化 L0: HardFilter 需要前一帧用于静止帧检测
        self.hard_filter.check(frame_np, ts)  # 首帧直接通过，保存为参考

        # 初始化 L1: FastTriggers 需要前一帧用于差异检测
        # 注意：detect() 会设置 _last_periodic_ts = ts，确保后续周期触发正确计时
        self.fast_triggers.detect(frame_np, ts)  # 建立 phash、直方图等参考

        # 初始化 L2: FrameValidator 需要参考帧用于 SSIM、光流等
        self.frame_validator.update_reference(frame_np)

        self._is_initialized = True
        self._last_emit_ts = ts

        # 构建首帧输出结果（周期触发模式）
        h, w = frame_np.shape[:2]
        pil_image = self._numpy_to_pil(frame_np)

        return {
            'image': pil_image,
            'timestamp': ts,
            'frame_index': frame_idx,
            'significant': False,  # 周期触发非显著帧
            'source': ['periodic'],  # 首帧为周期触发
            'original_frame': frame_np,
            'cropped_frame': frame_np,  # 不裁剪，输出整帧
            'bbox': (0, 0, w, h),
            'compression_ratio': 1.0,
            'change_metrics': {
                'ssim_score': 1.0,  # 无参考帧，默认完全相似
                'combined_score': 0.0,  # 无变化
                'motion_score': 0.0,  # 无运动
            },
        }

    def _build_result(
        self,
        frame_np: np.ndarray,
        ts: float,
        frame_idx: int,
        triggers: List[str],
        validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """组装与 SmartSampler 完全一致的返回字典，并打印日志。"""
        # 语义区域裁剪
        cropped_frame, bbox, compression_ratio = self._crop_semantic_region(
            frame_np, triggers,
        )

        pil_image = self._numpy_to_pil(cropped_frame)

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
            'cropped_frame': cropped_frame,
            'bbox': bbox,
            'compression_ratio': compression_ratio,
            'change_metrics': {
                'ssim_score': ssim_score,
                'combined_score': combined_score,
                'motion_score': motion_score,
            },
        }

        # 更新验证器参考帧
        self.frame_validator.update_reference(frame_np)

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
            "裁剪后比例=%.1f%% | "
            "触发器=%s | "
            "L0通过率=%.1f%% | L1触发率=%.1f%%",
            frame_idx, ts,
            validation['score'], window_mean, '是' if is_peak else '否',
            '是' if has_motion else '否', motion_score,
            source_label,
            compression_ratio * 100,
            triggers,
            l0_rate, l1_rate,
        )

        return result

    def _crop_semantic_region(
        self,
        frame_np: np.ndarray,
        triggers: List[str],
    ) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
        """根据触发类型裁剪语义变化区域。

        策略：
        1. motion 触发 → 使用 fg_mask（背景减除掩码）
        2. scene_switch / anomaly 触发 → 使用帧差掩码
        3. periodic 触发 → 不裁剪，输出整帧

        Returns:
            (cropped_frame, bbox, compression_ratio)
        """
        h, w = frame_np.shape[:2]
        full_bbox = (0, 0, w, h)

        # 选择掩码
        mask = None
        if 'motion' in triggers:
            mask = self.fast_triggers.last_fg_mask
        elif 'scene_switch' in triggers or 'anomaly' in triggers:
            mask = self.fast_triggers.last_diff_mask

        if mask is None:
            # 无掩码（如 periodic），返回整帧
            return frame_np, full_bbox, 1.0

        # 确保掩码与帧尺寸匹配
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 使用 ForegroundExtractor 裁剪
        cropped_frame, bbox = self.foreground_extractor.extract_foreground(frame_np, mask)
        compression_ratio = self.foreground_extractor.calculate_compression_ratio(
            frame_np.shape, cropped_frame.shape,
        )

        return cropped_frame, bbox, compression_ratio

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
