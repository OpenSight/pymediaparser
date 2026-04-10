#!/usr/bin/env python3
"""媒体流解析启动脚本。

支持两种运行模式：
- live（默认）：从实时流（RTMP / HTTP-FLV / HTTP-TS）拉流抽帧分析
- replay：从视频文件 / 图片文件读取并分析

用法示例::

    # 实时流模式（默认）
    python scripts/run_parser.py --url rtmp://host/live/stream

    # 文件回放模式
    python scripts/run_parser.py --mode replay --url /path/to/video.mp4
    python scripts/run_parser.py --mode replay --url /path/to/image.jpg

    # 使用 Qwen3-VL 后端
    python scripts/run_parser.py \\
        --url rtmp://host/live/stream \\
        --vlm-backend qwen3 \\
        --model-path /path/to/Qwen3-VL-2B

    # 使用 OpenAI 兼容 API（vLLM / Ollama 等）
    python scripts/run_parser.py \
        --url rtmp://host/live/stream \
        --vlm-backend openai_api \
        --api-base-url http://localhost:8000/v1 \
        --api-model Qwen2-VL-2B-Instruct
    
    # 使用 BMP 调试后端（保存帧为BMP文件，不调用模型）
    python scripts/run_parser.py \
        --url rtmp://host/live/stream \
        --vlm-backend bmp \
        --model-path /tmp/debug_frames

    # HTTP-FLV 流，每 2 秒 1 帧，自定义 prompt
    python scripts/run_parser.py \\
        --url http://host/live/stream.flv \\
        --fps 0.5 \\
        --prompt "识别画面中的人物并描述他们的行为。"

    # CPU 模式（无 GPU 时）
    python scripts/run_parser.py \\
        --url rtmp://host/live/stream \\
        --device cpu --dtype float32 --fps 0.2
"""

import argparse
import logging
import os
import sys
import time

# 确保项目 src 目录在搜索路径中（支持直接 python scripts/run_parser.py 运行）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pymediaparser.vlm_base import StreamConfig, VLMConfig
from pymediaparser.vlm.factory import create_vlm_client
from pymediaparser.vlm.configs import APIVLMConfig
from pymediaparser.result_handler import ConsoleResultHandler, HttpCallbackHandler
from pymediaparser.live_pipeline import LivePipeline
from pymediaparser.pipeline_base import PipelineState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLM 视频/图片分析 —— 支持实时流和文件回放两种模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  实时流模式（默认）:
    %(prog)s --url rtmp://192.168.1.100/live/stream
    %(prog)s --url http://host/live/stream.flv --fps 0.5
  
  文件回放模式:
    %(prog)s --mode replay --url /path/to/video.mp4
    %(prog)s --mode replay --url /path/to/image.jpg
    %(prog)s --mode replay --url http://example.com/video.mp4
  
使用不同 VLM 后端:
  %(prog)s --url rtmp://host/live/stream --vlm-backend qwen3 --model-path /path/to/Qwen3-VL-2B
  %(prog)s --url rtmp://host/live/stream --vlm-backend openai_api --api-base-url http://localhost:8000/v1
  %(prog)s --url rtmp://host/live/stream --vlm-backend bmp --model-path /tmp/debug_frames
  
智能采样模式:
  %(prog)s --url rtmp://host/live/stream --smart-sampler simple
  %(prog)s --url rtmp://host/live/stream --smart-sampler ml --motion-threshold 0.2
  
批量处理模式（可独立使用或配合智能采样）:
  %(prog)s --url rtmp://host/live/stream --batch-processing
  %(prog)s --url rtmp://host/live/stream --smart-sampler ml --batch-processing
""",
    )

    # ── 运行模式 ──────────────────────────────────────────────
    parser.add_argument(
        "--mode", default="live", choices=["live", "replay"],
        help="运行模式: live=实时流 / replay=文件回放（默认: live）",
    )

    # ── 流/文件配置 ──────────────────────────────────────────
    stream_group = parser.add_argument_group("流/文件配置")
    stream_group.add_argument(
        "--url", required=True,
        help="live: 流地址 (rtmp:// / http://*.flv 等); replay: 文件路径或URL (/path/to/video.mp4)",
    )
    stream_group.add_argument(
        "--format", default=None,
        choices=["flv", "mpegts"],
        help="强制指定容器格式（默认根据 URL 自动检测）",
    )
    stream_group.add_argument(
        "--fps", type=float, default=1.0,
        help="目标抽帧频率，帧/秒（默认: 1.0）",
    )
    stream_group.add_argument(
        "--queue-size", type=int, default=3,
        help="帧缓冲队列最大长度，超出时丢弃旧帧（默认: 3）",
    )
    stream_group.add_argument(
        "--reconnect", type=float, default=3.0,
        help="断线重连等待秒数（默认: 3.0）",
    )
    stream_group.add_argument(
        "--timeout", type=float, default=10.0,
        help="流读取超时秒数（默认: 10.0）",
    )
    stream_group.add_argument(
        "--decode-mode", default="all",
        choices=["all", "keyframe_only"],
        help="解码模式: all=全帧解码 / keyframe_only=仅解码关键帧（默认: all）",
    )

    # ── VLM 配置 ─────────────────────────────────────────────
    vlm_group = parser.add_argument_group("VLM 模型配置")
    vlm_group.add_argument(
        "--vlm-backend", default="qwen35",
        help="VLM 后端名称: qwen2 / qwen3 / qwen35 / openai_api / bmp（默认: qwen35）",
    )
    vlm_group.add_argument(
        "--model-path", default=None,
        help="本地模型路径（默认: 项目内置 models/Qwen/Qwen3.5-0.8B）",
    )
    vlm_group.add_argument(
        "--device", default="cuda:0",
        help="推理设备（默认: cuda:0）",
    )
    vlm_group.add_argument(
        "--dtype", default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="推理精度（默认: float16）",
    )
    vlm_group.add_argument(
        "--max-tokens", type=int, default=256,
        help="单次生成最大 token 数（默认: 256）",
    )
    vlm_group.add_argument(
        "--prompt", default=None,
        help="VLM 提示词（默认: '请描述当前画面中的人物活动。'）",
    )
    vlm_group.add_argument(
        "--no-flash-attn", action="store_true",
        help="禁用 Flash Attention 2",
    )

    # ── API 后端配置 ──────────────────────────────────────────
    api_group = parser.add_argument_group("API 后端配置（--vlm-backend openai_api 时使用）")
    api_group.add_argument(
        "--api-base-url", default=None,
        help="API 服务地址，如 http://localhost:8000/v1",
    )
    api_group.add_argument(
        "--api-key", default=None,
        help="API 密钥（本地服务通常不需要）",
    )
    api_group.add_argument(
        "--api-model", default=None,
        help="API 模型名称，如 Qwen2-VL-2B-Instruct",
    )
    
    # ── 智能抽帧配置 ─────────────────────────────────────────
    smart_group = parser.add_argument_group("智能抽帧配置")
    smart_group.add_argument(
        "--smart-sampler",
        choices=["simple", "ml"],
        default=None,
        help="启用智能抽帧并指定采样器类型: simple=基础采样器, ml=三层漏斗采样器 (默认: 不启用)",
    )
    # 通用参数
    smart_group.add_argument(
        "--motion-method", default="MOG2",
        choices=["MOG2", "KNN"],
        help="[simple/ml] 运动检测方法 (默认: MOG2)",
    )
    smart_group.add_argument(
        "--motion-threshold", type=float, default=0.1,
        help="[simple/ml] 运动检测阈值 (默认: 0.1)",
    )
    smart_group.add_argument(
        "--backup-interval", type=float, default=30.0,
        help="[simple/ml] 保底/周期采样间隔秒数 (默认: 30.0)",
    )
    smart_group.add_argument(
        "--min-frame-interval", type=float, default=1.0,
        help="[simple/ml] 最小帧间隔秒数 (默认: 1.0)",
    )
    # Simple 专属参数
    smart_group.add_argument(
        "--ssim-threshold", type=float, default=0.80,
        help="[simple] SSIM相似度阈值 (默认: 0.80)",
    )
    # ML 专属参数
    smart_group.add_argument(
        "--scene-switch-threshold", type=float, default=0.5,
        help="[ml] 场景切换阈值，值越高越敏感 (默认: 0.5)",
    )
    
    # ── 批量处理配置 ─────────────────────────────────────────
    batch_group = parser.add_argument_group("批量处理配置（可独立使用）")
    batch_group.add_argument(
        "--batch-processing", action="store_true",
        help="启用批量处理模式：积累多帧后批量送入VLM理解",
    )
    batch_group.add_argument(
        "--batch-buffer-size", type=int, default=5,
        help="批量处理缓冲区大小（默认: 5）",
    )
    batch_group.add_argument(
        "--batch-timeout", type=float, default=5.0,
        help="批量处理帧时间戳最大跨度（秒，默认: 5.0）",
    )

    # ── 图像预处理配置 ───────────────────────────────────────
    preprocess_group = parser.add_argument_group("图像预处理配置")
    preprocess_group.add_argument(
        "--preprocessing",
        choices=["resize", "roi_crop"],
        default=None,
        help="启用图像预处理并指定策略: resize=缩放到指定尺寸, roi_crop=对非周期触发帧进行ROI裁剪（默认: 不启用）",
    )
    preprocess_group.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="[resize策略] 图像最大边长（像素），超过时等比缩放（默认: 1024）",
    )
    preprocess_group.add_argument(
        "--roi-method",
        choices=["motion", "saliency"],
        default="motion",
        help="[roi_crop策略] ROI检测方法: motion=帧差法, saliency=显著性检测（默认: motion）",
    )
    preprocess_group.add_argument(
        "--roi-padding",
        type=float,
        default=0.2,
        help="[roi_crop策略] 边界扩展比例（默认: 0.2）",
    )
    preprocess_group.add_argument(
        "--min-roi-ratio",
        type=float,
        default=0.2,
        help="[roi_crop策略] 最小占比阈值，ROI区域过小时扩展到该比例（默认: 0.2）",
    )

    # ── 输出配置 ─────────────────────────────────────────────
    output_group = parser.add_argument_group("输出配置")
    output_group.add_argument(
        "--callback-url", default=None,
        help="HTTP 回调地址（可选），设置后将同时通过 POST 推送结果",
    )
    output_group.add_argument(
        "--quiet", action="store_true",
        help="静默模式：不在控制台打印分析结果（仅配合 --callback-url 使用）",
    )

    # ── 日志 ─────────────────────────────────────────────────
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认: INFO）",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 配置日志 ──────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("run_live_vlm")

    # ── 构建 StreamConfig ─────────────────────────────────────
    stream_cfg = StreamConfig(
        url=args.url,
        format=args.format,
        target_fps=args.fps,
        reconnect_interval=args.reconnect,
        timeout=args.timeout,
        max_queue_size=args.queue_size,
        decode_mode=args.decode_mode,
    )

    # ── 构建 VLM 客户端 ─────────────────────────────────────
    backend = args.vlm_backend
    if backend == "openai_api":
        api_kwargs = {
            "max_new_tokens": args.max_tokens,
        }
        if args.api_base_url:
            api_kwargs["base_url"] = args.api_base_url
        if args.api_key:
            api_kwargs["api_key"] = args.api_key
        if args.api_model:
            api_kwargs["model_name"] = args.api_model
        if args.prompt:
            api_kwargs["default_prompt"] = args.prompt
        vlm_cfg = APIVLMConfig(**api_kwargs)
        vlm_client = create_vlm_client(backend, vlm_cfg)
    else:
        vlm_kwargs = {
            "device": args.device,
            "dtype": args.dtype,
            "max_new_tokens": args.max_tokens,
            "use_flash_attn": not args.no_flash_attn,
        }
        if args.model_path:
            vlm_kwargs["model_path"] = args.model_path
        if args.prompt:
            vlm_kwargs["default_prompt"] = args.prompt
        vlm_cfg = VLMConfig(**vlm_kwargs)
        vlm_client = create_vlm_client(backend, vlm_cfg)

    # ── 构建 ResultHandler 列表 ───────────────────────────────
    handlers = []
    if not args.quiet:
        handlers.append(ConsoleResultHandler(verbose=True))
    if args.callback_url:
        handlers.append(HttpCallbackHandler(callback_url=args.callback_url))
    if not handlers:
        # quiet 模式下也没有 callback，至少保留控制台输出
        logger.warning("未配置任何输出通道，将默认使用控制台输出")
        handlers.append(ConsoleResultHandler(verbose=True))

    # ── 打印启动信息 ──────────────────────────────────────────
    is_replay = args.mode == "replay"
    logger.info("=" * 50)
    if is_replay:
        logger.info("文件回放 VLM 分析")
    else:
        logger.info("实时流 VLM 分析")
    logger.info("=" * 50)
    if is_replay:
        logger.info("文件路径:   %s", stream_cfg.url)
    else:
        logger.info("流地址:     %s", stream_cfg.url)
    logger.info("抽帧频率:   %.2f fps", stream_cfg.target_fps)
    logger.info("VLM 后端:   %s", backend)
    if backend == "openai_api":
        logger.info("API 地址:   %s", vlm_cfg.base_url)
        logger.info("API 模型:   %s", vlm_cfg.model_name)
    else:
        logger.info("推理设备:   %s", vlm_cfg.device)
        logger.info("推理精度:   %s", vlm_cfg.dtype)
        logger.info("模型路径:   %s", vlm_cfg.model_path)
    logger.info("最大 tokens: %d", vlm_cfg.max_new_tokens)
    logger.info("提示词:     %s", vlm_cfg.default_prompt)
    logger.info("队列大小:   %d", stream_cfg.max_queue_size)
    logger.info("解码模式:   %s", "仅关键帧" if stream_cfg.decode_mode == "keyframe_only" else "全帧解码")
    if not is_replay and args.callback_url:
        logger.info("HTTP 回调:  %s", args.callback_url)
    elif args.callback_url:
        logger.info("HTTP 回调:  %s", args.callback_url)
    logger.info("=" * 50)

    # ── 构建 Pipeline ────────────────────────────────────────
    # 构建智能采样器配置
    smart_config = None
    smart_sampler = None
    if args.smart_sampler is not None:
        from pymediaparser.smart_sampler import SimpleSamplerConfig, MLSamplerConfig
        smart_sampler = args.smart_sampler

        common_config = {
            'backup_interval': args.backup_interval,
            'min_frame_interval': args.min_frame_interval,
            'motion_method': args.motion_method,
            'motion_threshold': args.motion_threshold,
        }

        if args.smart_sampler == 'simple':
            smart_config = SimpleSamplerConfig(
                **common_config,
                ssim_threshold=args.ssim_threshold,
            )
        elif args.smart_sampler == 'ml':
            smart_config = MLSamplerConfig(
                **common_config,
                scene_switch_threshold=args.scene_switch_threshold,
            )

    # 构建批量处理配置
    batch_config = None
    if args.batch_processing:
        batch_config = {
            'batch_buffer_size': args.batch_buffer_size,
            'batch_timeout': args.batch_timeout,
        }

    # 构建预处理配置
    preprocess_config = None
    if args.preprocessing is not None:
        from pymediaparser.image_processor import ResizeConfig, ROICropConfig

        if args.preprocessing == 'resize':
            preprocess_config = ResizeConfig(max_size=args.max_size)
        elif args.preprocessing == 'roi_crop':
            preprocess_config = ROICropConfig(
                method=args.roi_method,
                padding_ratio=args.roi_padding,
                min_roi_ratio=args.min_roi_ratio,
            )

    if is_replay:
        # 文件回放模式
        from pymediaparser.replay_pipeline import ReplayPipeline
        pipeline = ReplayPipeline(
            stream_config=stream_cfg,
            vlm_client=vlm_client,
            handlers=handlers,
            prompt=args.prompt,
            smart_sampler=smart_sampler,
            smart_config=smart_config,
            enable_batch_processing=args.batch_processing,
            batch_config=batch_config,
            preprocessing=args.preprocessing,
            preprocess_config=preprocess_config,
        )
    else:
        # 实时流模式
        pipeline = LivePipeline(
            stream_config=stream_cfg,
            vlm_client=vlm_client,
            handlers=handlers,
            prompt=args.prompt,
            smart_sampler=smart_sampler,
            smart_config=smart_config,
            enable_batch_processing=args.batch_processing,
            batch_config=batch_config,
            preprocessing=args.preprocessing,
            preprocess_config=preprocess_config,
        )
    
    # 显示运行模式信息
    if is_replay:
        mode_parts = ["文件回放"]
        if args.smart_sampler:
            mode_parts.append(f"智能采样({args.smart_sampler})")
        if args.batch_processing:
            mode_parts.append("批量处理")
        logger.info("运行模式:   %s", "+".join(mode_parts))
    elif args.smart_sampler or args.batch_processing:
        mode_info = []
        if args.smart_sampler:
            mode_info.append(f"智能采样({args.smart_sampler})")
        if args.batch_processing:
            mode_info.append("批量处理")
        logger.info("运行模式:   %s", "+".join(mode_info))
    else:
        logger.info("运行模式:   传统固定频率抽帧")

    # 显示智能功能详情
    if args.smart_sampler:
        logger.info("采样器类型: %s", args.smart_sampler)
        logger.info("运动检测:   %s", args.motion_method)
        logger.info("运动阈值:   %.2f", args.motion_threshold)
        logger.info("保底间隔:   %.1f秒", args.backup_interval)
        logger.info("最小帧间隔: %.1f秒", args.min_frame_interval)
        if args.smart_sampler == 'simple':
            logger.info("SSIM阈值:   %.2f", args.ssim_threshold)
        elif args.smart_sampler == 'ml':
            logger.info("场景切换阈值: %.2f", args.scene_switch_threshold)
    if args.batch_processing:
        logger.info("批缓冲区:   %d", args.batch_buffer_size)
        logger.info("帧时间戳跨度上限: %.1f秒", args.batch_timeout)

    # 显示预处理配置
    if args.preprocessing:
        logger.info("预处理策略: %s", args.preprocessing)
        if args.preprocessing == 'resize':
            logger.info("最大边长:   %d像素", args.max_size)
        elif args.preprocessing == 'roi_crop':
            logger.info("ROI方法:    %s", args.roi_method)
            logger.info("边界扩展:   %.0f%%", args.roi_padding * 100)
            logger.info("最小占比:   %.0f%%", args.min_roi_ratio * 100)

    logger.info("=" * 50)
    
    # ── 运行 Pipeline ────────────────────────────────────────
    pipeline.start()

    try:
        # 轮询进度
        while pipeline.is_running():
            progress = pipeline.get_progress()
            if progress.duration:
                # ReplayPipeline: 显示百分比
                pct = progress.progress_percent
                if pct is not None:
                    print(f"\r进度: {pct:.1f}% | "
                          f"{progress.current_timestamp:.1f}s / {progress.duration:.1f}s | "
                          f"已处理 {progress.processed_frames} 帧", end="")
                else:
                    print(f"\r已处理: {progress.processed_frames} 帧 | "
                          f"运行时间: {progress.elapsed_time:.0f}s", end="")
            else:
                # LivePipeline: 显示帧数和运行时间
                print(f"\r已处理: {progress.processed_frames} 帧 | "
                      f"运行时间: {progress.elapsed_time:.0f}s", end="")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在停止...")
        pipeline.stop()
    
    # 显示最终状态
    final_state = pipeline.get_state()
    if final_state == PipelineState.COMPLETED:
        print(f"\n处理完成，共处理 {pipeline.get_progress().processed_frames} 帧")
    elif final_state == PipelineState.ERROR:
        print(f"\n处理异常: {pipeline.get_progress().error}")

    # ── 显式清理资源，避免程序退出时 C++ 析构异常 ─────────────
    # 问题：PyTorch/transformers 在程序退出时可能抛出
    # "terminate called without an active exception"
    del pipeline
    del vlm_client
    handlers.clear()

    # 确保 CUDA 资源完全释放
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        # 强制立即退出，跳过所有 Python/C++ 清理过程
        # 避免 PyTorch CUDA 析构时的 "terminate called" 错误
        os._exit(0)
