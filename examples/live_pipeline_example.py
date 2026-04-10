#!/usr/bin/env python3
"""LivePipeline 实时流分析示例

演示如何使用 LivePipeline 对实时视频流进行 VLM 分析。
支持传统固定频率采样和智能自适应采样两种模式。

VLM 后端选择：
- bmp: 虚拟调试后端，保存图像为 BMP 文件（无需 GPU，推荐用于入门和调试）
- qwen35: Qwen3.5 本地模型推理（需要 GPU 和模型文件）
- openai_api: OpenAI 兼容 API 服务（需要 API 服务地址）
"""

import logging
import os
import tempfile
import time
from pymediaparser import (
    LivePipeline,
    StreamConfig,
    VLMConfig,
    create_vlm_client,
    ConsoleResultHandler,
    HttpCallbackHandler,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# ============================================================================
# BMP 后端示例（推荐入门使用）
# ============================================================================

def example_basic_live_stream():
    """示例1: 基础实时流分析（BMP 调试后端）
    
    使用 BMP 虚拟后端，将抽取的帧保存为 BMP 文件。
    无需 GPU 和模型文件，适合入门学习和调试。
    """
    print("\n" + "=" * 60)
    print("示例1: 基础实时流分析（BMP 调试后端）")
    print("=" * 60)
    
    # 1. 配置流接入参数
    stream_config = StreamConfig(
        url="rtmp://your-stream-address/live/stream",  # 替换为实际流地址
        target_fps=1.0,           # 每秒抽取1帧
        max_queue_size=3,         # 帧缓冲队列大小
        reconnect_interval=3.0,   # 断线重连间隔（秒）
        decode_mode="all",        # 解码模式: all=全帧, keyframe_only=仅关键帧
    )
    
    # 2. 配置 BMP 后端（model_path 为输出目录）
    output_dir = tempfile.mkdtemp(prefix="live_pipeline_")
    vlm_config = VLMConfig(model_path=output_dir)
    
    # 3. 创建 VLM 客户端（使用 BMP 虚拟后端）
    vlm_client = create_vlm_client("bmp", vlm_config)
    
    # 4. 创建 Pipeline
    pipeline = LivePipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
        prompt="调试帧",  # BMP 后端会忽略此参数
    )
    
    print(f"配置完成:")
    print(f"  流地址: {stream_config.url}")
    print(f"  抽帧频率: {stream_config.target_fps} fps")
    print(f"  输出目录: {output_dir}")
    print(f"  后端类型: BMP（调试模式）")
    print("\n执行: pipeline.run()  # 阻塞式运行，按 Ctrl+C 停止")
    print("结果: 帧图像将保存为 BMP 文件到输出目录")
    
    # 5. 运行 Pipeline（阻塞式，支持 Ctrl+C 优雅退出）
    # pipeline.run()


def example_smart_sampling_stream():
    """示例2: 智能采样模式（BMP 后端）
    
    启用智能采样后，Pipeline 会基于画面内容变化自动决定是否保留帧，
    减少相似帧的重复保存。
    """
    print("\n" + "=" * 60)
    print("示例2: 智能采样模式（BMP 后端）")
    print("=" * 60)
    
    stream_config = StreamConfig(
        url="rtmp://your-stream-address/live/stream",
        target_fps=2.0,           # 原始抽帧频率（智能采样会在此基础上筛选）
        max_queue_size=5,
    )
    
    # BMP 后端配置
    output_dir = tempfile.mkdtemp(prefix="smart_sampling_")
    vlm_client = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    
    # 智能采样配置
    smart_config = {
        "motion_method": "MOG2",       # 运动检测算法: MOG2 / KNN
        "motion_threshold": 0.1,       # 运动检测阈值
        "backup_interval": 30.0,       # 最大备份间隔（秒）
        "min_frame_interval": 1.0,     # 最小帧间隔（秒）
    }
    
    pipeline = LivePipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
        smart_sampler='ml',              # 启用智能采样（'simple' 或 'ml'）
        enable_batch_processing=False,
        smart_config=smart_config,
    )
    
    print(f"配置完成:")
    print(f"  输出目录: {output_dir}")
    print(f"  智能采样: 已启用")
    print(f"  运动检测: {smart_config['motion_method']}")
    print(f"  备份间隔: {smart_config['backup_interval']}秒")
    print("\n执行: pipeline.run()")
    
    # pipeline.run()


def example_batch_processing_stream():
    """示例3: 批量处理模式（BMP 后端）
    
    启用批量处理后，Pipeline 会将多帧图像打包一起处理。
    BMP 后端会将批量帧保存为多个 BMP 文件。
    """
    print("\n" + "=" * 60)
    print("示例3: 批量处理模式（BMP 后端）")
    print("=" * 60)
    
    stream_config = StreamConfig(
        url="rtmp://your-stream-address/live/stream",
        target_fps=5.0,           # 较高的抽帧频率
        max_queue_size=10,
    )
    
    # BMP 后端配置
    output_dir = tempfile.mkdtemp(prefix="batch_processing_")
    vlm_client = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    
    smart_config = {
        "batch_buffer_size": 5,    # 批缓冲区大小
        "batch_timeout": 5.0,      # 批处理超时（秒）
    }
    
    pipeline = LivePipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
        smart_sampler=None,
        enable_batch_processing=True,   # 启用批量处理
        batch_config=smart_config,
    )
    
    print(f"配置完成:")
    print(f"  输出目录: {output_dir}")
    print(f"  批量处理: 已启用")
    print(f"  批缓冲区: {smart_config['batch_buffer_size']}帧")
    print(f"  批超时: {smart_config['batch_timeout']}秒")
    
    # pipeline.run()


def example_full_smart_stream():
    """示例4: 完整智能模式（BMP 后端）
    
    同时启用智能采样和批量处理，最大程度优化帧处理效率。
    """
    print("\n" + "=" * 60)
    print("示例4: 完整智能模式（BMP 后端）")
    print("=" * 60)
    
    stream_config = StreamConfig(
        url="rtmp://your-stream-address/live/stream",
        target_fps=2.0,
        max_queue_size=5,
    )
    
    # BMP 后端配置
    output_dir = tempfile.mkdtemp(prefix="full_smart_")
    vlm_client = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    
    smart_config = {
        "motion_method": "MOG2",
        "motion_threshold": 0.1,
        "backup_interval": 30.0,
        "min_frame_interval": 1.0,
        "batch_buffer_size": 5,
        "batch_timeout": 5.0,
    }
    
    pipeline = LivePipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
        smart_sampler='ml',
        enable_batch_processing=True,
        smart_config=smart_config,
    )
    
    print(f"配置完成:")
    print(f"  输出目录: {output_dir}")
    print(f"  智能采样: 已启用")
    print(f"  批量处理: 已启用")
    
    # pipeline.run()


# ============================================================================
# Qwen3.5 本地模型示例（需要 GPU）
# ============================================================================

def example_qwen35_local():
    """示例5: Qwen3.5 本地模型推理
    
    使用 Qwen3.5-0.8B 本地模型进行实时 VLM 分析。
    需要安装 vlm-qwen 依赖并下载模型文件。
    """
    print("\n" + "=" * 60)
    print("示例5: Qwen3.5 本地模型推理")
    print("=" * 60)
    
    stream_config = StreamConfig(
        url="rtmp://your-stream-address/live/stream",
        target_fps=1.0,
        max_queue_size=3,
    )
    
    # Qwen3.5 配置（显存占用约 1.4GB，适合资源受限场景）
    vlm_config = VLMConfig(
        model_path="/path/to/Qwen3.5-0.8B",  # 替换为实际模型路径
        device="cuda:0",
        dtype="float16",
        max_new_tokens=256,
    )
    
    vlm_client = create_vlm_client("qwen35", vlm_config)
    
    pipeline = LivePipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
        prompt="请描述当前画面中的人物活动。",
    )
    
    print(f"配置完成:")
    print(f"  后端类型: Qwen3.5 本地模型")
    print(f"  模型路径: {vlm_config.model_path}")
    print(f"  推理设备: {vlm_config.device}")
    print(f"  显存占用: 约 1.4GB")
    print("\n安装依赖: pip install 'pymediaparser[vlm-qwen]'")
    
    # pipeline.run()


# ============================================================================
# OpenAI API 后端示例
# ============================================================================

def example_openai_api():
    """示例6: OpenAI 兼容 API 后端
    
    通过 OpenAI 兼容 API 调用远程 VLM 服务（如 vLLM、Ollama、OpenAI 等），
    无需本地 GPU 即可运行。
    """
    print("\n" + "=" * 60)
    print("示例6: OpenAI 兼容 API 后端")
    print("=" * 60)
    
    from pymediaparser import APIVLMConfig
    
    stream_config = StreamConfig(
        url="http://example.com/live/stream.flv",
        target_fps=1.0,
    )
    
    # 配置 API 后端
    api_config = APIVLMConfig(
        base_url="http://localhost:8000/v1",  # vLLM 服务地址
        api_key="your-api-key",               # API 密钥（如需要）
        model_name="Qwen2-VL-7B-Instruct",    # 模型名称
        max_new_tokens=256,
    )
    
    vlm_client = create_vlm_client("openai_api", api_config)
    
    pipeline = LivePipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
        prompt="请描述画面内容。",
    )
    
    print(f"配置完成:")
    print(f"  后端类型: OpenAI 兼容 API")
    print(f"  API地址: {api_config.base_url}")
    print(f"  模型: {api_config.model_name}")
    print("\n支持的 API 服务:")
    print("  - vLLM (http://localhost:8000/v1)")
    print("  - Ollama (http://localhost:11434/v1)")
    print("  - OpenAI (https://api.openai.com/v1)")
    print("  - 通义千问、智谱 GLM 等")
    
    # pipeline.run()


# ============================================================================
# 其他高级示例
# ============================================================================

def example_custom_handler_stream():
    """示例7: 自定义结果处理器
    
    除了默认的控制台输出，还可以添加 HTTP 回调等自定义处理器。
    """
    print("\n" + "=" * 60)
    print("示例7: 自定义结果处理器")
    print("=" * 60)
    
    stream_config = StreamConfig(
        url="rtmp://your-stream-address/live/stream",
        target_fps=1.0,
    )
    
    # BMP 后端配置
    output_dir = tempfile.mkdtemp(prefix="custom_handler_")
    vlm_client = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    
    # 配置多个结果处理器
    handlers = [
        ConsoleResultHandler(verbose=True),  # 控制台输出
        # HttpCallbackHandler(
        #     callback_url="http://your-server.com/callback",
        #     timeout=5.0,
        # ),
    ]
    
    pipeline = LivePipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
        handlers=handlers,
    )
    
    print(f"配置完成:")
    print(f"  输出目录: {output_dir}")
    print(f"  处理器数量: {len(handlers)}")
    for i, handler in enumerate(handlers):
        print(f"    [{i+1}] {handler.__class__.__name__}")
    
    # pipeline.run()


def example_non_blocking_control():
    """示例8: 非阻塞控制（程序化启停）
    
    使用 start() / stop() 方法实现非阻塞控制，适合集成到现有系统中。
    """
    print("\n" + "=" * 60)
    print("示例8: 非阻塞控制")
    print("=" * 60)
    
    stream_config = StreamConfig(
        url="rtmp://your-stream-address/live/stream",
        target_fps=1.0,
    )
    
    # BMP 后端配置
    output_dir = tempfile.mkdtemp(prefix="non_blocking_")
    vlm_client = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    
    pipeline = LivePipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
    )
    
    print("非阻塞控制示例代码:")
    print("-" * 40)
    code = '''
    # 启动 Pipeline（非阻塞）
    pipeline.start()
    
    # 检查状态
    print(f"状态: {pipeline.get_state()}")  # RUNNING
    print(f"运行中: {pipeline.is_running()}")  # True
    
    # 获取进度
    progress = pipeline.get_progress()
    print(f"已处理帧数: {progress.processed_frames}")
    
    # 等待一段时间
    time.sleep(60)
    
    # 停止 Pipeline
    pipeline.stop()
    
    # 等待进入终态
    pipeline.wait(timeout=10.0)
    
    # 检查最终状态
    print(f"已停止: {pipeline.is_stopped()}")  # True
    '''
    print(code)


def example_image_preprocessing():
    """示例9: 图像预处理
    
    在送入 VLM 前对图像进行预处理（缩放、ROI裁剪等）。
    """
    print("\n" + "=" * 60)
    print("示例9: 图像预处理")
    print("=" * 60)
    
    from pymediaparser.image_processor import ResizeConfig, ROICropConfig
    
    stream_config = StreamConfig(
        url="rtmp://your-stream-address/live/stream",
        target_fps=1.0,
    )
    
    # BMP 后端配置
    output_dir = tempfile.mkdtemp(prefix="preprocessing_")
    vlm_client = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    
    # 方式1: 等比缩放（控制最大边长）
    print("方式1: 等比缩放")
    pipeline_resize = LivePipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
        preprocessing="resize",
        preprocess_config=ResizeConfig(max_size=1024),
    )
    print(f"  配置: max_size=1024")
    print(f"  输出目录: {output_dir}")
    
    # 方式2: ROI 裁剪（仅对非周期触发帧）
    print("\n方式2: ROI 裁剪")
    output_dir2 = tempfile.mkdtemp(prefix="roi_crop_")
    vlm_client2 = create_vlm_client("bmp", VLMConfig(model_path=output_dir2))
    pipeline_roi = LivePipeline(
        stream_config=stream_config,
        vlm_client=vlm_client2,
        preprocessing="roi_crop",
        preprocess_config=ROICropConfig(
            method="motion",       # 运动检测方法
            padding_ratio=0.2,     # 边界扩展比例
            min_roi_ratio=0.2,     # 最小 ROI 占比
        ),
    )
    print(f"  配置: method=motion, padding=0.2")
    print(f"  输出目录: {output_dir2}")


if __name__ == "__main__":
    # 运行所有示例（仅打印配置，不实际执行）
    print("\n" + "=" * 70)
    print("LivePipeline 实时流分析示例")
    print("=" * 70)
    print("\n【VLM 后端选择指南】")
    print("-" * 70)
    print("| 后端       | 说明                          | 安装要求          |")
    print("|------------|-------------------------------|-------------------|")
    print("| bmp        | 虚拟调试，保存图像为 BMP      | 核心包（推荐入门）|")
    print("| qwen35     | Qwen3.5 本地 GPU 推理         | [vlm-qwen]        |")
    print("| openai_api | vLLM/Ollama/OpenAI API 服务   | 核心包            |")
    print("-" * 70)
    
    # BMP 后端示例（推荐入门）
    example_basic_live_stream()
    example_smart_sampling_stream()
    example_batch_processing_stream()
    example_full_smart_stream()
    
    # Qwen3.5 和 OpenAI API 示例
    example_qwen35_local()
    example_openai_api()
    
    # 其他高级示例
    example_custom_handler_stream()
    example_non_blocking_control()
    example_image_preprocessing()
    
    print("\n" + "=" * 70)
    print("所有示例配置展示完成")
    print("=" * 70)
    print("\n【快速开始】")
    print("1. 使用 BMP 后端测试（无需 GPU）:")
    print("   - 将示例中的流地址替换为实际的 RTMP/HTTP-FLV 地址")
    print("   - 取消注释 pipeline.run() 行以实际运行")
    print("   - 帧图像将保存到临时目录")
    print("\n2. 使用 Qwen3.5 本地模型:")
    print("   - 安装依赖: pip install 'pymediaparser[vlm-qwen]'")
    print("   - 下载模型文件到 models/ 目录")
    print("   - 配置 model_path 参数")
    print("\n3. 使用 OpenAI API 服务:")
    print("   - 启动 vLLM 或 Ollama 服务")
    print("   - 配置 base_url 和 model_name 参数")
    print("\n【运行提示】")
    print("- 运行时使用 Ctrl+C 可以优雅停止 Pipeline")
    print("- 根据 GPU 显存调整 device 和 dtype 参数")
    print("- BMP 后端适合验证帧采集流程和图像质量")
