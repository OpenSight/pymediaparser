"""文件回放 VLM 分析 Pipeline：读取视频/图片文件，按频率抽帧，送入 VLM 推理。

与 LivePipeline 镜像架构（2 线程 / 3 线程），核心差异：
- 队列策略：阻塞 put（不丢帧），而非丢弃旧帧
- 完成检测：文件读完后 Pipeline 自动退出
- 进度跟踪：显示处理进度

典型用法::

    from pymediaparser import ReplayPipeline, StreamConfig, create_vlm_client

    cfg = StreamConfig(url="/path/to/video.mp4", target_fps=1.0)
    vlm_client = create_vlm_client("openai_api", api_cfg)
    pipeline = ReplayPipeline(cfg, vlm_client)
    pipeline.run()
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from .file_reader import FileReader
from .frame_sampler import FrameSampler
from .result_handler import ConsoleResultHandler, ResultHandler
from .vlm_base import FrameResult, StreamConfig, VLMClient, VLMResult

# 智能功能导入（可选）
try:
    from .smart_sampler import SmartSampler, MLSmartSampler
    from .frame_buffer import FrameBuffer
    _SMART_FEATURES_AVAILABLE = True
except ImportError:
    _SMART_FEATURES_AVAILABLE = False

logger = logging.getLogger(__name__)

# 阻塞 put 超时（秒），短超时保证线程能及时响应退出信号
_PUT_TIMEOUT = 0.5


# ======================================================================
# Pipeline 核心
# ======================================================================

class ReplayPipeline:
    """文件回放 VLM 分析 Pipeline（生产者-消费者模型）。

    与 LivePipeline 架构一致，支持两种工作模式：
    1. **传统模式**：固定频率抽帧，双线程架构
    2. **智能模式**：基于内容变化检测的自适应采样，三线程架构

    与 LivePipeline 的关键差异：
    - 队列策略：阻塞 put，保证不丢帧
    - 完成检测：文件处理完毕后自动退出
    - 进度跟踪：显示处理进度信息

    Args:
        stream_config: 配置。``url`` 字段为文件路径或网络 URL。
        vlm_client: VLM 推理客户端实例。
        handlers: 结果处理器列表；为空时默认使用 ConsoleResultHandler。
        prompt: VLM 推理提示词；为 None 时使用 VLMConfig 中的默认 prompt。
        enable_smart_sampling: 是否启用智能采样。
        enable_batch_processing: 是否启用批量处理。
        smart_config: 智能功能配置字典。
    """

    def __init__(
        self,
        stream_config: StreamConfig,
        vlm_client: VLMClient,
        handlers: Sequence[ResultHandler] | None = None,
        prompt: str | None = None,
        # 智能功能参数
        enable_smart_sampling: bool = False,
        enable_batch_processing: bool = False,
        smart_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.stream_config = stream_config
        self.vlm_client = vlm_client
        self.handlers: Sequence[ResultHandler] = list(handlers) if handlers else [ConsoleResultHandler()]
        self.prompt = prompt

        # 智能功能开关
        self.enable_smart_sampling = enable_smart_sampling and _SMART_FEATURES_AVAILABLE
        self.enable_batch_processing = enable_batch_processing and _SMART_FEATURES_AVAILABLE

        if (enable_smart_sampling or enable_batch_processing) and not _SMART_FEATURES_AVAILABLE:
            logger.warning("智能功能模块未安装，回退到传统模式")

        # 初始化智能组件
        self.smart_sampler: Optional[SmartSampler] = None
        self.frame_buffer: Optional[FrameBuffer] = None
        self._processor_thread: Optional[threading.Thread] = None

        if self.enable_smart_sampling:
            config = smart_config or {}
            self.smart_sampler = MLSmartSampler(
                enable_smart_sampling=True,
                motion_method=config.get('motion_method', 'MOG2'),
                motion_threshold=config.get('motion_threshold', 0.1),
                backup_interval=config.get('backup_interval', 30.0),
                min_frame_interval=config.get('min_frame_interval', 1.0),
            )

        if self.enable_batch_processing:
            config = smart_config or {}
            self.frame_buffer = FrameBuffer(
                max_size=config.get('batch_buffer_size', 5),
                max_wait_time=config.get('batch_timeout', 5.0),
            )

        # 队列配置（与 LivePipeline 结构一致）
        self._queue: queue.Queue = queue.Queue(
            maxsize=stream_config.max_queue_size,
        )
        if self.enable_smart_sampling:
            self._processor_queue: queue.Queue = queue.Queue(
                maxsize=stream_config.max_queue_size * 2,
            )

        self._stop_event = threading.Event()
        self._done_event = threading.Event()  # 回放完成事件
        self._producer_thread: Optional[threading.Thread] = None
        self._consumer_thread: Optional[threading.Thread] = None
        self._file_reader: Optional[FileReader] = None

        # 进度跟踪
        self._processed_count: int = 0

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动 Pipeline（非阻塞）。"""
        logger.info("回放 Pipeline 启动中 ...")
        self._stop_event.clear()
        self._done_event.clear()
        self._processed_count = 0

        # 加载模型
        self.vlm_client.load()

        # 通知 handlers
        for h in self.handlers:
            h.on_start()

        # 启动生产者线程
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            name="replay-producer",
            daemon=True,
        )

        # 根据模式启动其他线程
        if self.enable_smart_sampling:
            self._processor_thread = threading.Thread(
                target=self._processor_loop,
                name="replay-processor",
                daemon=True,
            )
            self._consumer_thread = threading.Thread(
                target=self._consumer_loop,
                name="replay-consumer",
                daemon=True,
            )
            threads = [self._producer_thread, self._processor_thread, self._consumer_thread]
        else:
            self._consumer_thread = threading.Thread(
                target=self._consumer_loop,
                name="replay-consumer",
                daemon=True,
            )
            threads = [self._producer_thread, self._consumer_thread]

        for thread in threads:
            thread.start()

        mode_str = "智能模式" if self.enable_smart_sampling else "传统模式"
        batch_str = "启用批处理" if self.enable_batch_processing else "单帧处理"
        logger.info(
            "回放 Pipeline 已启动 [%s, %s] file=%s fps=%.1f",
            mode_str, batch_str, self.stream_config.url, self.stream_config.target_fps,
        )

    def stop(self) -> None:
        """优雅停止 Pipeline。"""
        if self._stop_event.is_set():
            return
        logger.info("回放 Pipeline 停止中 ...")
        self._stop_event.set()

        # 向队列放 sentinel 让消费者/处理器线程退出
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self.enable_smart_sampling:
            try:
                self._processor_queue.put_nowait(None)
            except queue.Full:
                pass

        # 等待生产者线程退出（生产者会自己关闭 file_reader）
        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=5)

        # 如果生产者线程仍未退出，设置 file_reader 为 None 避免重复关闭
        # 注意：不主动关闭 file_reader，因为可能正在解码中
        if self._file_reader is not None:
            logger.warning("生产者线程未能在超时内退出，放弃等待")
            self._file_reader = None

        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=3)
        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=3)

        # 处理缓冲区剩余帧
        if self.frame_buffer:
            self._flush_batch_buffer()

        # 通知 handlers
        for h in self.handlers:
            h.on_stop()

        # 卸载模型
        self.vlm_client.unload()
        logger.info("回放 Pipeline 已停止（共处理 %d 帧）", self._processed_count)

    def run(self) -> None:
        """阻塞式运行，文件处理完毕或 Ctrl+C 时退出。"""
        self.start()

        try:
            while not self._done_event.is_set() and not self._stop_event.is_set():
                self._done_event.wait(timeout=1.0)
        except KeyboardInterrupt:
            logger.info("收到 Ctrl+C，正在停止 ...")
        finally:
            self.stop()

    # ------------------------------------------------------------------
    # 生产者线程
    # ------------------------------------------------------------------

    def _producer_loop(self) -> None:
        """生产者：读取文件 → 解码 → 阻塞入队。

        传统模式：输出字典格式到主队列
        智能模式：输出原始帧到处理器队列

        队列满时阻塞等待（不丢帧）。
        """
        reader = FileReader(self.stream_config)
        self._file_reader = reader
        sampler = FrameSampler(target_fps=self.stream_config.target_fps)

        target_queue = self._processor_queue if self.enable_smart_sampling else self._queue

        try:
            reader.open()

            # 打印文件信息
            if reader.is_image:
                logger.info("文件类型: 图片")
            else:
                total = reader.total_frames
                dur = reader.duration_seconds
                logger.info(
                    "文件类型: 视频 (总帧数=%s, 时长=%.1fs)",
                    total if total > 0 else "未知", dur,
                )

            for image, ts, idx in sampler.sample(reader.frames(), stop_event=self._stop_event):
                if self._stop_event.is_set():
                    break

                if self.enable_smart_sampling:
                    # 智能模式：PIL 转 numpy(BGR) 入处理器队列
                    import cv2
                    frame_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    item = (frame_np, ts, idx)
                else:
                    # 传统模式：输出统一的字典格式
                    item = {
                        'image': image,
                        'timestamp': ts,
                        'frame_index': idx,
                        'significant': True,
                        'source': ['traditional'],  # 传统固定间隔采样
                    }

                # 阻塞入队（短超时 + 循环检查 stop_event）
                self._blocking_put(target_queue, item)

        except Exception as exc:
            if not self._stop_event.is_set():
                logger.error("生产者线程异常: %s", exc, exc_info=True)
        finally:
            # 发送 sentinel 通知下游
            try:
                target_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
            reader.close()
            self._file_reader = None
            logger.info("生产者线程已退出")

    # ------------------------------------------------------------------
    # 消费者线程
    # ------------------------------------------------------------------

    def _consumer_loop(self) -> None:
        """消费者：取帧 → 批量/单帧推理 → 结果回调。"""
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # sentinel 检测
                if item is None:
                    break

                if not isinstance(item, dict):
                    logger.warning("收到非字典格式数据，跳过: %s", type(item))
                    continue

                self._process_frame_item(item)

        except Exception as exc:
            if not self._stop_event.is_set():
                logger.error("消费者线程异常: %s", exc, exc_info=True)
        finally:
            # 处理缓冲区剩余帧
            if self.frame_buffer:
                self._flush_batch_buffer()
            # 通知主线程：回放完成
            self._done_event.set()
            logger.info("消费者线程已退出")

    def _process_frame_item(self, item: Dict[str, Any]) -> None:
        """处理单个帧数据项。"""
        image = item['image']
        ts = item['timestamp']
        idx = item['frame_index']
        batch_count = 0  # 批处理帧数

        if self.frame_buffer is not None:
            # 批处理模式：add_frame 返回就绪的批次
            batch_frames = self.frame_buffer.add_frame(item)
            if batch_frames is None:
                return

            vlm_result = self._process_batch(batch_frames)
            if vlm_result is None:
                return

            last_frame = batch_frames[-1]
            ts = last_frame['timestamp']
            idx = last_frame['frame_index']
            batch_count = len(batch_frames)
            item = last_frame  # 使用最后一帧记录进度
        else:
            # 单帧推理
            vlm_result = self.vlm_client.analyze(image, self.prompt)
            batch_count = 1

        if vlm_result is None:
            return

        if not isinstance(vlm_result, VLMResult):
            logger.error("无效的VLM结果类型: %s", type(vlm_result))
            vlm_result = VLMResult(text="[无效结果]", inference_time=0.0)

        frame_result = FrameResult(
            frame_index=idx,
            timestamp=ts,
            vlm_result=vlm_result,
        )

        for handler in self.handlers:
            try:
                handler.handle(frame_result)
            except Exception as exc:
                logger.error("ResultHandler 异常: %s", exc)

        # 进度跟踪
        self._processed_count += batch_count
        self._log_progress(item)

    def _log_progress(self, item: Dict[str, Any]) -> None:
        """输出回放进度日志。"""
        ts = item.get('timestamp', 0.0)
        frame_idx = item.get('frame_index', self._processed_count)
        reader = self._file_reader

        if reader is not None and reader.total_frames > 0:
            dur = reader.duration_seconds
            if dur > 0:
                # 计算预计的总采样帧数（基于目标帧率和视频时长）
                estimated_total = int(dur * self.stream_config.target_fps)
                pct = (ts / dur) * 100.0
                logger.info(
                    "[回放进度] 帧 %d/%d (采样) | 视频 %.1fs/%.1fs | %.1f%%",
                    frame_idx + 1, estimated_total, ts, dur, pct,
                )
            else:
                logger.info(
                    "[回放进度] 帧 %d (采样) | 视频 %.1fs",
                    frame_idx + 1, ts,
                )
        else:
            logger.info(
                "[回放进度] 已处理 %d 帧 | 视频 %.1fs",
                self._processed_count, ts,
            )

    def _process_batch(self, frames: List[Dict[str, Any]]) -> Optional[VLMResult]:
        """批量处理帧列表。"""
        start_time = time.time()

        images = [frame['image'] for frame in frames]
        significant_count = sum(1 for frame in frames if frame.get('significant'))

        prompt = self._build_batch_prompt(frames, significant_count)

        try:
            vlm_result = self.vlm_client.analyze_batch(images, prompt)

            processing_time = time.time() - start_time
            logger.info(
                "批处理完成 - 帧数: %d, 显著帧: %d, 耗时: %.2fs",
                len(frames), significant_count, processing_time,
            )
            return vlm_result

        except Exception as e:
            logger.error("批处理失败: %s", e, exc_info=True)
            return None

    def _build_batch_prompt(self, frames: List[Dict[str, Any]], significant_count: int) -> str:
        """构建批处理提示词。"""
        total_frames = len(frames)

        if significant_count > 0:
            return (
                f"请分析这{total_frames}帧图像序列，其中包含{significant_count}帧显著变化。"
                f"请描述：1)主要的场景变化和事件发展；2)关键时间点的重要活动；"
                f"3)整体趋势和可能的后续发展。"
            )
        else:
            return (
                f"请分析这{total_frames}帧连续图像，描述场景内容的演变趋势和发展规律。"
                f"重点关注画面中的主体活动和环境变化。"
            )

    def _flush_batch_buffer(self) -> None:
        """清空批处理缓冲区，处理剩余帧。"""
        if not self.frame_buffer:
            return

        batch_frames = self.frame_buffer.flush()
        if batch_frames:
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
                self._processed_count += len(batch_frames)
                self._log_progress(last_frame)

    # ------------------------------------------------------------------
    # 处理器线程（智能模式专用）
    # ------------------------------------------------------------------

    def _processor_loop(self) -> None:
        """处理器：智能采样（CPU 密集）。

        从处理器队列取原始帧 → 智能采样 → 阻塞入主队列
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

                frame_np, ts, idx = item

                if self.smart_sampler:
                    for sampled_data in self.smart_sampler.sample(iter([(frame_np, ts)])):
                        if self._stop_event.is_set():
                            break
                        self._enqueue_frame(sampled_data)
                else:
                    pil_image = self._numpy_to_pil(frame_np)
                    frame_data = {
                        'image': pil_image,
                        'timestamp': ts,
                        'frame_index': idx,
                        'significant': True,
                        'source': ['processor'],  # 处理器模式
                    }
                    self._enqueue_frame(frame_data)

        except Exception as exc:
            if not self._stop_event.is_set():
                logger.error("处理器线程异常: %s", exc, exc_info=True)
        finally:
            logger.info("处理器线程已退出")

    def _enqueue_frame(self, frame_data: Optional[Dict[str, Any]]) -> None:
        """将帧数据阻塞入队到主队列（不丢帧）。"""
        if frame_data is None:
            # sentinel：使用阻塞式发送，确保一定能送达
            self._blocking_put(self._queue, None)
            return

        self._blocking_put(self._queue, frame_data)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _blocking_put(self, q: queue.Queue, item: Any) -> None:
        """阻塞入队，短超时 + 循环检查退出信号。"""
        while not self._stop_event.is_set():
            try:
                q.put(item, timeout=_PUT_TIMEOUT)
                return
            except queue.Full:
                continue

    @staticmethod
    def _numpy_to_pil(frame_np: np.ndarray) -> Image.Image:
        """将 numpy 数组转换为 PIL 图像。"""
        import cv2
        rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)
