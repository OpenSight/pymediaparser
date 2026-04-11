"""实时流 VLM 分析 Pipeline：将拉流、抽帧、VLM 推理串联为生产者-消费者架构。

生产者线程负责拉流解码并按频率抽帧，消费者线程负责 VLM 推理和结果输出。
通过有界队列 + 丢弃旧帧策略保证内存可控。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from PIL import Image
import numpy as np

from .frame_sampler import FrameSampler
from .pipeline_base import BasePipeline, PipelineProgress, PipelineState
from .result_handler import ConsoleResultHandler, ResultHandler
from .stream_reader import StreamReader
from .vlm_base import FrameResult, StreamConfig, VLMClient, VLMConfig, VLMResult

# 智能功能导入（可选）
try:
    from .smart_sampler import SmartSampler, BaseSamplerConfig, create_sampler
    from .frame_buffer import FrameBuffer
    _SMART_FEATURES_AVAILABLE = True
except ImportError:
    _SMART_FEATURES_AVAILABLE = False

# 图像预处理导入（可选）
try:
    from .image_processor import (
        BaseProcessor,
        ResizeConfig,
        ROICropConfig,
        create_processor,
    )
    _PREPROCESSOR_AVAILABLE = True
except ImportError:
    _PREPROCESSOR_AVAILABLE = False

logger = logging.getLogger(__name__)


# ======================================================================
# Pipeline 核心
# ======================================================================

class LivePipeline(BasePipeline):
    """实时流 VLM 分析 Pipeline（生产者-消费者模型）。

    支持两种工作模式：
    1. **传统模式**：固定频率抽帧，双线程架构
    2. **智能模式**：基于内容变化检测的自适应采样，三线程架构

    架构设计：
    - 生产者：负责拉流解码（I/O 密集），不阻塞
    - 处理器（可选）：负责智能采样（CPU 密集），不阻塞拉流
    - 消费者：负责批处理 + VLM 推理（GPU 密集）

    数据流：
    - 传统模式：生产者 → _queue（字典） → 消费者
    - 智能模式：生产者 → _processor_queue → 处理器 → _queue（字典） → 消费者

    队列满时丢弃最旧的帧，保证不会因为推理慢导致内存持续增长。

    Args:
        stream_config: 流接入配置。
        vlm_client: VLM 推理客户端实例。
        handlers: 结果处理器列表；为空时默认使用 ConsoleResultHandler。
        prompt: VLM 推理提示词；为 None 时使用 VLMConfig 中的默认 prompt。
        smart_sampler: 智能采样器类型（'simple' 或 'ml'），为 None 时不启用智能采样。
        smart_config: 智能采样器配置对象或配置字典。
        enable_batch_processing: 是否启用批量处理（需要frame_buffer模块）。
        batch_config: 批量处理配置字典。
        preprocessing: 图像预处理策略名称。
        preprocess_config: 图像预处理配置对象。
    """

    def __init__(
        self,
        stream_config: StreamConfig,
        vlm_client: VLMClient,
        handlers: Sequence[ResultHandler] | None = None,
        prompt: str | None = None,
        # 智能采样参数（新设计）
        smart_sampler: Optional[str] = None,
        smart_config: Union[BaseSamplerConfig, Dict[str, Any], None] = None,
        # 批处理参数
        enable_batch_processing: bool = False,
        batch_config: Optional[Dict[str, Any]] = None,
        # 图像预处理参数
        preprocessing: Optional[str] = None,
        preprocess_config: Union[ResizeConfig, ROICropConfig, None] = None,
    ) -> None:
        # 基本配置
        self.stream_config = stream_config
        self.vlm_client = vlm_client
        self.handlers: Sequence[ResultHandler] = (
            list(handlers) if handlers else [ConsoleResultHandler()]
        )
        self.prompt = prompt

        # 智能功能开关
        self.enable_smart_sampling = smart_sampler is not None and _SMART_FEATURES_AVAILABLE
        self.enable_batch_processing = enable_batch_processing and _SMART_FEATURES_AVAILABLE

        if smart_sampler is not None and not _SMART_FEATURES_AVAILABLE:
            logger.warning("智能功能模块未安装，回退到传统模式")

        # 初始化智能组件
        self.smart_sampler: Optional[SmartSampler] = None
        self.frame_buffer: Optional[FrameBuffer] = None
        self._processor_thread: Optional[threading.Thread] = None

        # 初始化智能采样器（使用工厂模式）
        if self.enable_smart_sampling:
            # 类型断言：enable_smart_sampling 为 True 时 smart_sampler 必不为 None
            assert smart_sampler is not None
            self.smart_sampler = create_sampler(smart_sampler, smart_config)
            logger.info("智能采样器已启用: 类型=%s", smart_sampler)

        if self.enable_batch_processing:
            config = batch_config or {}
            self.frame_buffer = FrameBuffer(
                max_size=config.get('batch_buffer_size', 5),
                max_wait_time=config.get('batch_timeout', 5.0),
            )

        # 图像预处理器初始化
        self.preprocessor: Optional[BaseProcessor] = None
        self._preprocessing_enabled = False

        if preprocessing is not None and _PREPROCESSOR_AVAILABLE:
            self.preprocessor = create_processor(preprocessing, preprocess_config)
            self._preprocessing_enabled = True
            logger.info("图像预处理器已启用: 策略=%s", preprocessing)
        elif preprocessing is not None and not _PREPROCESSOR_AVAILABLE:
            logger.warning("图像预处理模块未安装，预处理功能已禁用")

        # 判断是否需要处理器线程（智能采样或预处理任一启用）
        self._use_processor_thread = bool(
            self.enable_smart_sampling or self._preprocessing_enabled,
        )

        # 队列配置
        self._queue: queue.Queue = queue.Queue(
            maxsize=stream_config.max_queue_size,
        )
        # 处理器队列：智能采样或预处理任一启用时创建
        if self._use_processor_thread:
            self._processor_queue: queue.Queue = queue.Queue(
                maxsize=stream_config.max_queue_size * 2,
            )

        # 状态管理
        self._state = PipelineState.IDLE
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._start_time: float = 0.0
        self._processed_frames: int = 0
        self._error: Optional[Exception] = None
        self._current_timestamp: float = 0.0

        # 线程管理
        self._producer_thread: Optional[threading.Thread] = None
        self._consumer_thread: Optional[threading.Thread] = None
        self._stream_reader: Optional[StreamReader] = None

    # ==================================================================
    # 公开接口
    # ==================================================================

    def start(self) -> None:
        """启动 Pipeline（非阻塞）。"""
        with self._state_lock:
            if self._state != PipelineState.IDLE:
                if self._is_terminal_state(self._state):
                    # 终态重启前，先确保旧线程已退出
                    self._join_threads()
                    self._reset_for_restart()
                else:
                    logger.warning("Pipeline 已在运行中，忽略 start() 调用")
                    return
            self._set_state(PipelineState.STARTING)

        logger.info("Pipeline 启动中 ...")

        try:
            # 重置状态
            self._stop_event.clear()
            self._processed_frames = 0
            self._error = None
            self._start_time = time.time()
            self._current_timestamp = 0.0

            # 加载模型
            self.vlm_client.load()

            # 通知 handlers
            self._notify_handlers("on_start")

            # 启动线程
            self._start_threads()

            # 设置运行状态
            self._set_state(PipelineState.RUNNING)

            mode_str = "智能模式" if self.enable_smart_sampling else "传统模式"
            batch_str = "启用批处理" if self.enable_batch_processing else "单帧处理"
            decode_mode_str = "仅关键帧" if self.stream_config.decode_mode == "keyframe_only" else "全帧解码"
            logger.info(
                "Pipeline 已启动 [%s, %s, %s] url=%s fps=%.1f",
                mode_str, batch_str, decode_mode_str,
                self.stream_config.url, self.stream_config.target_fps,
            )

        except Exception as exc:
            logger.error("Pipeline 启动失败: %s", exc, exc_info=True)
            self._error = exc
            self._cleanup_resources()
            self._set_state(PipelineState.ERROR)
            self._stop_event.set()  # 通知 run() 退出
            self._notify_handlers("on_error", exc)

    def stop(self) -> None:
        """停止 Pipeline（幂等）。"""
        with self._state_lock:
            if self._state not in (PipelineState.RUNNING, PipelineState.STARTING):
                return
            self._set_state(PipelineState.STOPPING)

        logger.info("Pipeline 停止中 ...")

        # 设置停止信号，通知所有线程退出
        self._stop_event.set()

        # 强制关闭流连接，中断生产者线程的阻塞式网络 IO
        if self._stream_reader is not None:
            self._stream_reader.close()

        # 尝试向队列放 sentinel（作为备用手段）
        self._try_put_sentinel()

        # 等待线程退出
        self._join_threads()

        # 处理缓冲区剩余帧
        if self.frame_buffer:
            self._flush_batch_buffer()

        # 清理资源
        self._cleanup_resources()

        # 设置 STOPPED 状态
        self._set_state(PipelineState.STOPPED)

        # 通知 handlers
        self._notify_handlers("on_stop")

        logger.info("Pipeline 已停止（共处理 %d 帧）", self._processed_frames)

    def run(self) -> None:
        """阻塞式运行，支持 Ctrl+C 优雅退出。"""
        self.start()

        try:
            while self._state == PipelineState.RUNNING:
                self._stop_event.wait(timeout=1.0)
        except KeyboardInterrupt:
            logger.info("收到 Ctrl+C，正在停止 ...")
        finally:
            if self._state == PipelineState.RUNNING:
                self.stop()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """等待 Pipeline 进入终态。"""
        start = time.time()
        while True:
            if self._is_terminal_state(self._state):
                return True
            if timeout is not None:
                elapsed = time.time() - start
                if elapsed >= timeout:
                    return False
            time.sleep(0.1)

    def get_state(self) -> PipelineState:
        """获取当前运行状态。"""
        return self._state

    def get_progress(self) -> PipelineProgress:
        """获取进度信息。"""
        return PipelineProgress(
            state=self._state,
            processed_frames=self._processed_frames,
            start_time=self._start_time,
            current_timestamp=self._current_timestamp,
            error=self._error,
        )

    def is_running(self) -> bool:
        """是否运行中（RUNNING 状态）。"""
        return self._state == PipelineState.RUNNING

    def is_completed(self) -> bool:
        """是否正常完成（COMPLETED 状态）。"""
        return self._state == PipelineState.COMPLETED

    def is_stopped(self) -> bool:
        """是否被主动停止（STOPPED 状态）。"""
        return self._state == PipelineState.STOPPED

    def is_error(self) -> bool:
        """是否异常终止（ERROR 状态）。"""
        return self._state == PipelineState.ERROR

    def is_terminal(self) -> bool:
        """是否处于终态（可重新启动）。"""
        return self._is_terminal_state(self._state)

    # ==================================================================
    # 内部方法
    # ==================================================================

    def _set_state(self, state: PipelineState) -> None:
        """设置状态。"""
        old_state = self._state
        self._state = state
        logger.debug("Pipeline 状态转换: %s → %s", old_state.value, state.value)

    def _reset_for_restart(self) -> None:
        """重置状态以支持重新启动。"""
        self._processed_frames = 0
        self._error = None
        self._start_time = 0.0
        self._current_timestamp = 0.0
        self._stop_event.clear()
        self._state = PipelineState.IDLE

    def _notify_handlers(self, method: str, *args: Any) -> None:
        """通知所有 handlers 调用指定方法。"""
        for handler in self.handlers:
            try:
                func = getattr(handler, method, None)
                if func is not None:
                    func(*args)
            except Exception as exc:
                logger.error("Handler.%s 异常: %s", method, exc)

    def _start_threads(self) -> None:
        """启动工作线程。"""
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            name="producer",
            daemon=True,
        )

        if self._use_processor_thread:
            self._processor_thread = threading.Thread(
                target=self._processor_loop,
                name="processor",
                daemon=True,
            )
            self._consumer_thread = threading.Thread(
                target=self._consumer_loop,
                name="consumer",
                daemon=True,
            )
            threads = [self._producer_thread, self._processor_thread, self._consumer_thread]
        else:
            self._consumer_thread = threading.Thread(
                target=self._consumer_loop,
                name="consumer",
                daemon=True,
            )
            threads = [self._producer_thread, self._consumer_thread]

        for thread in threads:
            thread.start()

    def _try_put_sentinel(self) -> None:
        """尝试向队列放入 sentinel。"""
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._use_processor_thread:
            try:
                self._processor_queue.put_nowait(None)
            except queue.Full:
                pass

    def _join_threads(self) -> None:
        """等待工作线程退出。"""
        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=3)
        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=3)
        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=3)

    def _cleanup_resources(self) -> None:
        """清理资源。"""
        self.vlm_client.unload()

    def _handle_thread_error(self, thread_name: str, exc: Exception) -> None:
        """处理线程异常。

        1. 设置 stop_event 通知其他线程退出
        2. 尝试发送 sentinel
        3. 设置 ERROR 状态
        """
        logger.error("%s异常: %s", thread_name, exc, exc_info=True)
        self._error = exc

        # 设置 stop_event，通知其他线程退出
        self._stop_event.set()

        # 尝试发送 sentinel
        self._try_put_sentinel()

        # 清理资源
        self._cleanup_resources()

        # 设置 ERROR 状态
        with self._state_lock:
            self._set_state(PipelineState.ERROR)

        # 通知 handlers
        self._notify_handlers("on_error", exc)

    # ==================================================================
    # 生产者线程
    # ==================================================================

    def _producer_loop(self) -> None:
        """生产者：拉流 → 解码 → 入队列。

        传统模式：输出字典格式到主队列
        智能模式/预处理模式：输出原始帧到处理器队列

        队列满时丢弃旧帧，保证新帧能入队。
        """
        reader = StreamReader(self.stream_config)
        self._stream_reader = reader
        sampler = FrameSampler(target_fps=self.stream_config.target_fps)

        target_queue = self._processor_queue if self._use_processor_thread else self._queue

        try:
            for image, ts, idx in sampler.sample(reader.frames()):
                if self._stop_event.is_set():
                    break

                if self._use_processor_thread:
                    # 智能模式/预处理模式：直接传递 PIL 图像 + idx
                    item = (image, ts, idx)
                else:
                    # 传统模式：输出统一的字典格式
                    item = {
                        'image': image,
                        'timestamp': ts,
                        'frame_index': idx,
                        'significant': True,
                        'source': ['traditional'],
                    }

                # 队列满时丢弃旧帧
                if target_queue.full():
                    try:
                        target_queue.get_nowait()
                    except queue.Empty:
                        pass
                target_queue.put_nowait(item)

        except Exception as exc:
            if not self._stop_event.is_set():
                self._handle_thread_error("生产者线程", exc)
        finally:
            reader.close()
            self._stream_reader = None
            logger.info("生产者线程已退出")

    # ==================================================================
    # 消费者线程
    # ==================================================================

    def _consumer_loop(self) -> None:
        """消费者：取帧 → 批量/单帧推理 → 结果回调。"""
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=1.0)
                except queue.Empty:
                    # 超时检查：定期flush批处理缓冲区
                    if self.frame_buffer and self.frame_buffer.is_timeout():
                        self._flush_batch_buffer(reason="timeout")
                    continue

                # sentinel 检测
                if item is None:
                    break

                # 统一处理字典格式
                if not isinstance(item, dict):
                    logger.warning("收到非字典格式数据，跳过: %s", type(item))
                    continue

                self._process_frame_item(item)

        except Exception as exc:
            if not self._stop_event.is_set():
                self._handle_thread_error("消费者线程", exc)
        finally:
            # 实时流场景：停止时不需要 flush 批处理器缓存
            # - 主动停止：用户不再需要结果
            # - 异常退出：快速退出更重要
            # - 实时流本就接受丢帧（队列满时丢弃旧帧）
            logger.info("消费者线程已退出")

    def _process_frame_item(self, item: Dict[str, Any]) -> None:
        """处理单个帧数据项。"""
        image = item['image']
        ts = item['timestamp']
        idx = item['frame_index']
        batch_count = 1  # 默认单帧

        if self.frame_buffer is not None:
            # 批处理模式：add_frame 返回就绪的批次
            batch_frames = self.frame_buffer.add_frame(item)
            if batch_frames is None:
                return  # 等待更多帧

            # 批次就绪，执行批量推理
            vlm_result = self._process_batch(batch_frames)
            if vlm_result is None:
                return

            # 使用批次中最后一帧的元数据
            last_frame = batch_frames[-1]
            ts = last_frame['timestamp']
            idx = last_frame['frame_index']
            batch_count = len(batch_frames)
        else:
            # 单帧推理：直接传递帧信息字典
            vlm_result = self.vlm_client.analyze(item, self.prompt)

        # 错误处理
        if vlm_result is None:
            return

        if not isinstance(vlm_result, VLMResult):
            logger.error("无效的VLM结果类型: %s", type(vlm_result))
            vlm_result = VLMResult(text="[无效结果]", inference_time=0.0)

        # 组装帧结果
        frame_result = FrameResult(
            frame_index=idx,
            timestamp=ts,
            vlm_result=vlm_result,
        )

        # 触发所有 handlers
        for handler in self.handlers:
            try:
                handler.handle(frame_result)
            except Exception as exc:
                logger.error("ResultHandler 异常: %s", exc)

        # 更新进度
        self._processed_frames += batch_count
        self._current_timestamp = ts
        self._notify_handlers("on_progress", self.get_progress())

    def _process_batch(self, frames: List[Dict[str, Any]]) -> Optional[VLMResult]:
        """批量处理帧列表。"""
        start_time = time.time()
        significant_count = sum(1 for frame in frames if frame.get('significant'))

        # 执行批量推理：直接传递帧信息字典列表
        try:
            vlm_result = self.vlm_client.analyze_batch(frames, self.prompt)

            processing_time = time.time() - start_time
            logger.info(
                "批处理完成 - 帧数: %d, 显著帧: %d, 耗时: %.2fs",
                len(frames), significant_count, processing_time,
            )
            return vlm_result

        except Exception as e:
            logger.error("批处理失败: %s", e, exc_info=True)
            return None

    def _flush_batch_buffer(self, reason: str = "shutdown") -> None:
        """清空批处理缓冲区，处理剩余帧。
        
        Args:
            reason: flush原因，"timeout"=运行中超时触发，"shutdown"=停止时清理
        """
        if not self.frame_buffer:
            return

        batch_frames = self.frame_buffer.flush()
        if batch_frames:
            if reason == "timeout":
                logger.info("[超时flush] 处理缓冲区帧 - 帧数: %d", len(batch_frames))
            else:
                logger.info("处理缓冲区剩余帧 - 帧数: %d", len(batch_frames))
            
            vlm_result = self._process_batch(batch_frames)
            if vlm_result:
                last_frame = batch_frames[-1]
                frame_result = FrameResult(
                    frame_index=last_frame['frame_index'],
                    timestamp=last_frame['timestamp'],
                    vlm_result=vlm_result,
                )
                for handler in self.handlers:
                    try:
                        handler.handle(frame_result)
                    except Exception as exc:
                        logger.error("ResultHandler 异常: %s", exc)

                # 进度跟踪
                self._processed_frames += len(batch_frames)

    # ==================================================================
    # 处理器线程（智能模式专用）
    # ==================================================================

    def _processor_loop(self) -> None:
        """处理器：智能采样（可选）+ 预处理（可选）。

        从处理器队列取原始帧 → 智能采样（可选）→ 预处理（可选）→ 输出字典格式到主队列
        """
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._processor_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if item is None:
                    # 传递 sentinel 给消费者
                    self._enqueue_frame(None)
                    break

                pil_image, ts, idx = item

                # 统一获取帧数据迭代器
                if self.smart_sampler:
                    frame_iter = self.smart_sampler.sample(iter([(pil_image, ts, idx)]))
                else:
                    frame_iter = iter([{
                        'image': pil_image,
                        'timestamp': ts,
                        'frame_index': idx,
                        'significant': True,
                        'source': ['traditional'],
                    }])

                # 统一处理流程：预处理 → 入队
                for frame_data in frame_iter:
                    if self._stop_event.is_set():
                        break
                    if self.preprocessor:
                        frame_data = self.preprocessor.process_frame(frame_data)
                    self._enqueue_frame(frame_data)

        except Exception as exc:
            if not self._stop_event.is_set():
                self._handle_thread_error("处理器线程", exc)
        finally:
            logger.info("处理器线程已退出")

    def _enqueue_frame(self, frame_data: Optional[Dict[str, Any]]) -> None:
        """将帧数据入队到主队列。"""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put_nowait(frame_data)

    @staticmethod
    def _numpy_to_pil(frame_np: np.ndarray) -> Image.Image:
        """将 numpy 数组转换为 PIL 图像。"""
        import cv2
        rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

