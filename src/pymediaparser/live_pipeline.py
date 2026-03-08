"""实时流 VLM 分析 Pipeline：将拉流、抽帧、VLM 推理串联为生产者-消费者架构。

生产者线程负责拉流解码并按频率抽帧，消费者线程负责 VLM 推理和结果输出。
通过有界队列 + 丢弃旧帧策略保证内存可控。

CLI 用法::

    python -m pymediaparser.live_pipeline \\
        --url "rtmp://host/live/stream" \\
        --fps 1.0 \\
        --device cuda:0 \\
        --prompt "请描述画面中的内容。"
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from typing import Sequence, Dict, Any, Optional, List

from PIL import Image
import numpy as np

from .frame_sampler import FrameSampler
from .result_handler import ConsoleResultHandler, ResultHandler
from .stream_reader import StreamReader
from .vlm_base import FrameResult, StreamConfig, VLMClient, VLMConfig, VLMResult

# 智能功能导入（可选）
try:
    from .smart_sampler import SmartSampler, SimpleSmartSampler, MLSmartSampler
    from .frame_buffer import FrameBuffer
    _SMART_FEATURES_AVAILABLE = True
except ImportError:
    _SMART_FEATURES_AVAILABLE = False

logger = logging.getLogger(__name__)


# ======================================================================
# Pipeline 核心
# ======================================================================

class LivePipeline:
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
        enable_smart_sampling: 是否启用智能采样（需要smart_sampler模块）。
        enable_batch_processing: 是否启用批量处理（需要frame_buffer模块）。
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
            # 暂时注释 SimpleSmartSampler，改用 MLSmartSampler
            # self.smart_sampler = SimpleSmartSampler(
            #     enable_smart_sampling=True,
            #     motion_method=config.get('motion_method', 'MOG2'),
            #     ssim_threshold=config.get('ssim_threshold', 0.80),
            #     motion_threshold=config.get('motion_threshold', 0.1),
            #     backup_interval=config.get('backup_interval', 30.0),
            #     min_frame_interval=config.get('min_frame_interval', 1.0),
            # )
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

        # 队列配置
        # 传统模式：生产者（字典） → _queue → 消费者
        # 智能模式：生产者（原始帧） → _processor_queue → 处理器 → _queue（字典） → 消费者
        self._queue: queue.Queue = queue.Queue(
            maxsize=stream_config.max_queue_size,
        )
        if self.enable_smart_sampling:
            self._processor_queue: queue.Queue = queue.Queue(
                maxsize=stream_config.max_queue_size * 2,
            )
            
        self._stop_event = threading.Event()
        self._producer_thread: Optional[threading.Thread] = None
        self._consumer_thread: Optional[threading.Thread] = None
        self._stream_reader: Optional[StreamReader] = None

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动 Pipeline（非阻塞）。

        根据配置自动选择传统模式或智能模式启动。
        """
        logger.info("Pipeline 启动中 ...")
        self._stop_event.clear()

        # 加载模型
        self.vlm_client.load()

        # 通知 handlers
        for h in self.handlers:
            h.on_start()

        # 启动生产者线程
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            name="producer",
            daemon=True,
        )
        
        # 根据模式启动其他线程
        if self.enable_smart_sampling:
            # 智能模式：三线程架构（生产者 + 处理器 + 消费者）
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
            # 传统模式：双线程架构（生产者 + 消费者）
            self._consumer_thread = threading.Thread(
                target=self._consumer_loop,
                name="consumer",
                daemon=True,
            )
            threads = [self._producer_thread, self._consumer_thread]
        
        # 启动所有线程
        for thread in threads:
            thread.start()

        mode_str = "智能模式" if self.enable_smart_sampling else "传统模式"
        batch_str = "启用批处理" if self.enable_batch_processing else "单帧处理"
        logger.info(
            "Pipeline 已启动 [%s, %s] url=%s fps=%.1f",
            mode_str, batch_str, self.stream_config.url, self.stream_config.target_fps,
        )

    def stop(self) -> None:
        """优雅停止 Pipeline。

        设置停止信号、强制关闭流连接以中断阻塞 IO、等待线程退出、卸载模型。
        """
        logger.info("Pipeline 停止中 ...")
        self._stop_event.set()

        # 强制关闭流连接，中断生产者线程的阻塞式网络 IO
        if self._stream_reader is not None:
            self._stream_reader.close()

        # 向队列放 sentinel 让消费者线程退出阻塞
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        
        # 智能模式下还需向处理器队列放 sentinel
        if self.enable_smart_sampling:
            try:
                self._processor_queue.put_nowait(None)
            except queue.Full:
                pass

        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=3)
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
        logger.info("Pipeline 已停止")

    def run(self) -> None:
        """阻塞式运行，支持 Ctrl+C 优雅退出。"""
        self.start()

        # 主线程等待停止事件
        try:
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=1.0)
        except KeyboardInterrupt:
            logger.info("收到 Ctrl+C，正在停止 ...")
        finally:
            if not self._stop_event.is_set():
                self.stop()

    # ------------------------------------------------------------------
    # 生产者线程
    # ------------------------------------------------------------------

    def _producer_loop(self) -> None:
        """生产者：拉流 → 解码 → 入队列。

        传统模式：输出字典格式到主队列
        智能模式：输出原始帧到处理器队列
        
        队列满时丢弃旧帧，保证新帧能入队。
        """
        reader = StreamReader(self.stream_config)
        self._stream_reader = reader
        sampler = FrameSampler(target_fps=self.stream_config.target_fps)

        try:
            for image, ts, idx in sampler.sample(reader.frames()):
                if self._stop_event.is_set():
                    break

                if self.enable_smart_sampling:
                    # 智能模式：PIL 转 numpy(BGR) 入处理器队列
                    import cv2
                    frame_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    target_queue = self._processor_queue
                    item = (frame_np, ts, idx)
                else:
                    # 传统模式：输出统一的字典格式
                    target_queue = self._queue
                    item = {
                        'image': image,
                        'timestamp': ts,
                        'frame_index': idx,
                        'significant': True,
                        'source': ['traditional'],  # 传统固定间隔采样
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
                logger.error("生产者线程异常: %s", exc, exc_info=True)
        finally:
            reader.close()
            self._stream_reader = None
            logger.info("生产者线程已退出")

    # ------------------------------------------------------------------
    # 消费者线程
    # ------------------------------------------------------------------

    def _consumer_loop(self) -> None:
        """消费者：取帧 → 批量/单帧推理 → 结果回调。
        
        统一处理字典格式的帧数据：
        1. 如果启用批处理，帧先入缓冲区
        2. 批次就绪时，调用 VLM 批量推理
        3. 未启用批处理时，直接单帧推理
        """
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=1.0)
                except queue.Empty:
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
                logger.error("消费者线程异常: %s", exc, exc_info=True)
        finally:
            logger.info("消费者线程已退出")

    def _process_frame_item(self, item: Dict[str, Any]) -> None:
        """处理单个帧数据项。
        
        Args:
            item: 帧数据字典，包含 image, timestamp, frame_index 等
        """
        image = item['image']
        ts = item['timestamp']
        idx = item['frame_index']

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
        else:
            # 单帧推理
            vlm_result = self.vlm_client.analyze(image, self.prompt)

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

    def _process_batch(self, frames: List[Dict[str, Any]]) -> Optional[VLMResult]:
        """批量处理帧列表。
        
        Args:
            frames: 帧数据列表，每个包含 image, timestamp, frame_index 等
            
        Returns:
            VLM 推理结果
        """
        start_time = time.time()
        
        images = [frame['image'] for frame in frames]
        significant_count = sum(1 for frame in frames if frame.get('significant'))
        
        # 构建批处理提示词
        prompt = self._build_batch_prompt(frames, significant_count)
        
        # 执行批量推理（基类提供默认逐帧实现，子类可覆盖）
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

    # ------------------------------------------------------------------
    # 处理器线程（智能模式专用）
    # ------------------------------------------------------------------

    def _processor_loop(self) -> None:
        """处理器：智能采样（CPU 密集）。
        
        从处理器队列取原始帧 → 智能采样 → 输出字典格式到主队列
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
                    # 智能采样处理
                    for sampled_data in self.smart_sampler.sample(iter([(frame_np, ts)])):
                        self._enqueue_frame(sampled_data)
                else:
                    # 无智能采样器，直接转换入队（防御性代码，理论上不应执行到此分支）
                    pil_image = self._numpy_to_pil(frame_np)
                    frame_data = {
                        'image': pil_image,
                        'timestamp': ts,
                        'frame_index': idx,
                        'significant': True,
                        'source': ['traditional'],  # 传统固定间隔采样
                    }
                    self._enqueue_frame(frame_data)

        except Exception as exc:
            if not self._stop_event.is_set():
                logger.error("处理器线程异常: %s", exc, exc_info=True)
        finally:
            logger.info("处理器线程已退出")

    def _enqueue_frame(self, frame_data: Optional[Dict[str, Any]]) -> None:
        """将帧数据入队到主队列。
        
        Args:
            frame_data: 帧数据字典，None 时入队 sentinel
        """
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


# ======================================================================
# CLI 入口
# ======================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="实时流 VLM 分析 Pipeline —— 从 RTMP/HTTP-FLV/HTTP-TS 拉流，"
                    "按频率抽帧并送入 VLM 大模型进行理解。",
    )
    # 流配置
    parser.add_argument("--url", required=True, help="流地址 (rtmp:// / http://*.flv / http://*.ts)")
    parser.add_argument("--format", default=None, help="强制容器格式: flv / mpegts (默认自动检测)")
    parser.add_argument("--fps", type=float, default=1.0, help="目标抽帧频率 (默认 1.0)")
    parser.add_argument("--queue-size", type=int, default=3, help="帧缓冲队列大小 (默认 3)")
    parser.add_argument("--reconnect", type=float, default=3.0, help="断线重连间隔秒数 (默认 3.0)")

    # VLM 配置
    parser.add_argument("--model-path", default=None, help="VLM 模型路径 (默认项目内置)")
    parser.add_argument("--device", default="cuda:0", help="推理设备 (默认 cuda:0)")
    parser.add_argument("--dtype", default="float16", help="推理精度 (默认 float16)")
    parser.add_argument("--max-tokens", type=int, default=256, help="最大生成 token 数 (默认 256)")
    parser.add_argument("--prompt", default=None, help="VLM 提示词 (默认 '请描述当前画面中的人物活动。')")

    # VLM 后端选择
    parser.add_argument("--vlm-backend", default="qwen3",
                        help="VLM 后端名称: qwen2 / qwen3 / openai_api (默认 qwen3)")
    parser.add_argument("--api-base-url", default=None,
                        help="API 服务地址 (用于 openai_api 后端)")
    parser.add_argument("--api-key", default=None,
                        help="API 密钥 (用于 openai_api 后端)")
    parser.add_argument("--api-model", default=None,
                        help="API 模型名称 (用于 openai_api 后端)")

    # 智能功能
    parser.add_argument("--smart-sampling", action="store_true", help="启用智能采样")
    parser.add_argument("--batch-processing", action="store_true", help="启用批量处理")

    # 日志
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别 (默认 INFO)")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI 主函数。"""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 构建配置
    stream_cfg = StreamConfig(
        url=args.url,
        format=args.format,
        target_fps=args.fps,
        reconnect_interval=args.reconnect,
        max_queue_size=args.queue_size,
    )

    vlm_kwargs: dict = {
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_tokens,
    }
    if args.model_path:
        vlm_kwargs["model_path"] = args.model_path
    if args.prompt:
        vlm_kwargs["default_prompt"] = args.prompt

    # 根据后端类型创建客户端
    from .vlm.factory import create_vlm_client

    backend = args.vlm_backend
    if backend == "openai_api":
        from .vlm.configs import APIVLMConfig
        api_kwargs: dict = {}
        if args.api_base_url:
            api_kwargs["base_url"] = args.api_base_url
        if args.api_key:
            api_kwargs["api_key"] = args.api_key
        if args.api_model:
            api_kwargs["model_name"] = args.api_model
        if args.prompt:
            api_kwargs["default_prompt"] = args.prompt
        api_kwargs["max_new_tokens"] = args.max_tokens
        vlm_client = create_vlm_client(backend, APIVLMConfig(**api_kwargs))
    else:
        vlm_cfg = VLMConfig(**vlm_kwargs)
        vlm_client = create_vlm_client(backend, vlm_cfg)

    # 构建并运行 Pipeline
    pipeline = LivePipeline(
        stream_config=stream_cfg,
        vlm_client=vlm_client,
        prompt=args.prompt,
        enable_smart_sampling=args.smart_sampling,
        enable_batch_processing=args.batch_processing,
    )

    logger.info(
        "配置: url=%s  fps=%.1f  backend=%s  device=%s  dtype=%s  max_tokens=%d",
        stream_cfg.url, stream_cfg.target_fps, backend,
        args.device, args.dtype, args.max_tokens,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
