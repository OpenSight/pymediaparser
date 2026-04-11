"""SimpleSmartSampler - 简单智能采样器实现"""

from __future__ import annotations
import logging
from typing import Iterator, Dict, Any, Optional
import numpy as np
from PIL import Image
import cv2

from .base import SmartSampler
from .motion_detector import MotionDetector
from .change_analyzer import ChangeAnalyzer

logger = logging.getLogger(__name__)


class SimpleSmartSampler(SmartSampler):
    """简单智能采样器 - 基于运动检测和变化分析的自适应帧采样。

    继承自 SmartSampler 基类，实现基础的智能采样逻辑。
    """

    def __init__(
        self,
        enable_smart_sampling: bool = True,
        motion_method: str = 'MOG2',
        ssim_threshold: float = 0.80,
        motion_threshold: float = 0.1,
        backup_interval: float = 30.0,
        min_frame_interval: float = 1.0,
    ) -> None:
        super().__init__(
            enable_smart_sampling=enable_smart_sampling,
            backup_interval=backup_interval,
            min_frame_interval=min_frame_interval,
        )

        self.motion_detector = MotionDetector(method=motion_method, threshold=motion_threshold)
        self.change_analyzer = ChangeAnalyzer(ssim_threshold=ssim_threshold)

        logger.info(
            "SimpleSmartSampler 初始化完成 - 智能采样: %s, 变化阈值: %.2f, 运动阈值: %.2f, 最小间隔: %.1fs",
            "启用" if enable_smart_sampling else "禁用",
            ssim_threshold, motion_threshold, min_frame_interval,
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
        """智能采样主入口。"""
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
            
            # 递增输入帧计数器（用于统计）
            self._input_frame_count += 1
            # 使用流水线传入的帧序号
            current_frame_idx = idx
                
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
                        'source': ['traditional'],  # 传统固定间隔采样
                    }
                    self._last_emit_ts = ts

    # ── 内部方法 ──────────────────────────────────────────

    def _smart_sample_frame(
        self, frame_np: np.ndarray, ts: float, frame_idx: int,
        time_based_emit: bool,
    ) -> Optional[Dict[str, Any]]:
        change_result = self.change_analyzer.analyze_change(frame_np)
        motion_detected, motion_score, motion_mask = self.motion_detector.detect_motion(frame_np)
        
        should_emit = time_based_emit or change_result['significant'] or motion_detected
        
        # 帧间去重：时间间隔检查
        if should_emit and not time_based_emit:
            time_since_last = ts - self._last_emit_ts
            if time_since_last < self._min_frame_interval:
                logger.debug("帧#%d 检测到变化但时间间隔太短 (%.2fs < %.1fs)，跳过", 
                           frame_idx, time_since_last, self._min_frame_interval)
                return None
        
        if should_emit:
            # 采样成功，更新change_analyzer的参考帧
            self.change_analyzer.update_reference(frame_np)
            
            pil_image = self._numpy_to_pil(frame_np)
            
            # 收集触发源列表
            triggers = []
            if motion_detected:
                triggers.append('motion')
            if change_result['significant']:
                triggers.append('scene_switch')
            if time_based_emit:
                triggers.append('periodic')
            
            # 如果没有触发源（理论上不应该），默认为 periodic
            if not triggers:
                triggers = ['periodic']
            
            # 判断是否为显著帧：非周期触发即为显著
            is_significant = 'periodic' not in triggers or len(triggers) > 1
            
            result = {
                'image': pil_image,
                'timestamp': ts,
                'frame_index': frame_idx,
                'significant': is_significant,
                'source': triggers,  # 触发源列表
                'change_metrics': {
                    'ssim_score': change_result['ssim_score'],
                    'combined_score': change_result['combined_score'],
                    'motion_score': motion_score
                }
            }
            
            self._last_emit_ts = ts
            
            # 打印详细的采样决策信息
            source_label = '、'.join(triggers)
            logger.info(
                "[送VLM] 帧#%d | ts=%.3fs | "
                "相似度=%.3f(阈值<%.2f) | "
                "运动=%s(得分=%.3f) | "
                "综合=%.3f | "
                "来源=%s",
                frame_idx,
                ts,
                change_result['ssim_score'],
                self.change_analyzer.ssim_threshold,
                '是' if motion_detected else '否',
                motion_score,
                change_result['combined_score'],
                source_label,
            )
            
            return result
        
        return None

    def _should_emit_by_time(self, ts: float) -> bool:
        """检查是否满足保底时间间隔。
        
        当画面长时间无变化时，强制采样一帧避免遗漏。
        使用 backup_interval 参数控制（默认30秒）。
        """
        if ts < self._last_emit_ts:
            logger.info("检测到时间戳回跳 (%.3f -> %.3f)，重置采样状态", 
                       self._last_emit_ts, ts)
            self._last_emit_ts = -float('inf')
            return True
        
        return (ts - self._last_emit_ts) >= self._backup_interval

    # ── 状态管理 ──────────────────────────────────────────

    def reset(self) -> None:
        """重置所有状态。"""
        self._last_emit_ts = -float('inf')
        self._input_frame_count = 0
        self.motion_detector.reset()
        self.change_analyzer.reset()
        logger.info("SimpleSmartSampler 状态已重置")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息。"""
        return {
            'total_frames_processed': self.frame_count,
            'smart_sampling_enabled': self.enable_smart,
            'backup_interval': self._backup_interval,
            'min_frame_interval': self._min_frame_interval,
            'motion_detector_method': self.motion_detector.method,
            'ssim_threshold': self.change_analyzer.ssim_threshold,
        }
