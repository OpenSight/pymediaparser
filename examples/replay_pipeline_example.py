#!/usr/bin/env python3
"""ReplayPipeline 文件回放分析示例

演示如何使用 ReplayPipeline 对视频/图片文件进行 VLM 分析。
与 LivePipeline 不同，ReplayPipeline 会在文件处理完毕后自动退出。

VLM 后端选择：
- bmp: 虚拟调试后端，保存图像为 BMP 文件（无需 GPU，推荐用于入门和调试）
- qwen35: Qwen3.5 本地模型推理（需要 GPU 和模型文件）
- openai_api: OpenAI 兼容 API 服务（需要 API 服务地址）
"""

import logging
import os
import tempfile
from pymediaparser import (
    ReplayPipeline,
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

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# BMP 后端示例（推荐入门使用）
# ============================================================================

def example_basic_video_replay():
    """示例1: 基础视频文件分析（BMP 后端）
    
    使用 BMP 虚拟后端，将抽取的帧保存为 BMP 文件。
    无需 GPU 和模型文件，适合入门学习和调试。
    """
    print("\n" + "=" * 60)
    print("示例1: 基础视频文件分析（BMP 后端）")
    print("=" * 60)
    
    # 视频文件路径（请替换为实际路径）
    video_path = os.path.join(PROJECT_ROOT, "resource", "test_video.mp4")
    
    # 1. 配置流参数（url 可以是本地文件路径或网络 URL）
    stream_config = StreamConfig(
        url=video_path,           # 视频文件路径
        target_fps=1.0,           # 每秒抽取1帧
        max_queue_size=5,         # 帧缓冲队列大小
        decode_mode="all",        # 解码模式
    )
    
    # 2. 配置 BMP 后端（model_path 为输出目录）
    output_dir = tempfile.mkdtemp(prefix="replay_pipeline_")
    vlm_client = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    
    # 3. 创建 Pipeline
    pipeline = ReplayPipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
    )
    
    print(f"配置完成:")
    print(f"  文件路径: {stream_config.url}")
    print(f"  抽帧频率: {stream_config.target_fps} fps")
    print(f"  输出目录: {output_dir}")
    print(f"  后端类型: BMP（调试模式）")
    print("\n执行: pipeline.run()  # 阻塞式运行，处理完成后自动退出")
    print("结果: 帧图像将保存为 BMP 文件到输出目录")
    
    # 4. 运行 Pipeline（阻塞式，文件处理完毕后自动退出）
    # pipeline.run()


def example_image_analysis():
    """示例2: 单张图片分析（BMP 后端）
    
    ReplayPipeline 也支持单张图片输入，适合批量图片分析场景。
    """
    print("\n" + "=" * 60)
    print("示例2: 单张图片分析（BMP 后端）")
    print("=" * 60)
    
    image_path = os.path.join(PROJECT_ROOT, "resource", "test_img1.png")
    
    stream_config = StreamConfig(
        url=image_path,           # 图片文件路径
        target_fps=1.0,           # 对单张图片无效，但需配置
    )
    
    # BMP 后端配置
    output_dir = tempfile.mkdtemp(prefix="image_analysis_")
    vlm_client = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    
    pipeline = ReplayPipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
    )
    
    print(f"配置完成:")
    print(f"  图片路径: {stream_config.url}")
    print(f"  输出目录: {output_dir}")
    print(f"  后端类型: BMP（调试模式）")
    
    # pipeline.run()


def example_smart_sampling_replay():
    """示例3: 智能采样模式分析视频（BMP 后端）
    
    对视频文件启用智能采样，只保存内容发生变化的帧。
    """
    print("\n" + "=" * 60)
    print("示例3: 智能采样模式分析视频（BMP 后端）")
    print("=" * 60)
    
    video_path = os.path.join(PROJECT_ROOT, "resource", "test_video.mp4")
    
    stream_config = StreamConfig(
        url=video_path,
        target_fps=2.0,           # 原始抽帧频率
        max_queue_size=10,
    )
    
    # BMP 后端配置
    output_dir = tempfile.mkdtemp(prefix="smart_replay_")
    vlm_client = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    
    # 智能采样配置
    smart_config = {
        "motion_method": "MOG2",
        "motion_threshold": 0.1,
        "backup_interval": 30.0,   # 最长30秒至少保存一帧
        "min_frame_interval": 0.5, # 最短0.5秒一帧
    }
    
    pipeline = ReplayPipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
        smart_sampler='ml',
        enable_batch_processing=False,
        smart_config=smart_config,
    )
    
    print(f"配置完成:")
    print(f"  输出目录: {output_dir}")
    print(f"  智能采样: 已启用")
    print(f"  运动检测: {smart_config['motion_method']}")
    print(f"  备份间隔: {smart_config['backup_interval']}秒")
    
    # pipeline.run()


def example_batch_processing_replay():
    """示例4: 批量处理模式分析视频（BMP 后端）
    
    将多帧打包批量处理，BMP 后端会将批量帧保存为多个 BMP 文件。
    """
    print("\n" + "=" * 60)
    print("示例4: 批量处理模式分析视频（BMP 后端）")
    print("=" * 60)
    
    video_path = os.path.join(PROJECT_ROOT, "resource", "test_video.mp4")
    
    stream_config = StreamConfig(
        url=video_path,
        target_fps=5.0,           # 较高的抽帧频率
        max_queue_size=20,
    )
    
    # BMP 后端配置
    output_dir = tempfile.mkdtemp(prefix="batch_replay_")
    vlm_client = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    
    smart_config = {
        "batch_buffer_size": 5,    # 每5帧一批
        "batch_timeout": 3.0,      # 最多等待3秒
    }
    
    pipeline = ReplayPipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
        smart_sampler=None,
        enable_batch_processing=True,
        batch_config=smart_config,
    )
    
    print(f"配置完成:")
    print(f"  输出目录: {output_dir}")
    print(f"  批量处理: 已启用")
    print(f"  批大小: {smart_config['batch_buffer_size']}帧")
    print(f"  批超时: {smart_config['batch_timeout']}秒")
    
    # pipeline.run()


def example_full_smart_replay():
    """示例5: 完整智能模式分析视频（BMP 后端）
    
    同时启用智能采样和批量处理，最大化处理效率。
    """
    print("\n" + "=" * 60)
    print("示例5: 完整智能模式分析视频（BMP 后端）")
    print("=" * 60)
    
    video_path = os.path.join(PROJECT_ROOT, "resource", "test_video.mp4")
    
    stream_config = StreamConfig(
        url=video_path,
        target_fps=2.0,
        max_queue_size=10,
    )
    
    # BMP 后端配置
    output_dir = tempfile.mkdtemp(prefix="full_smart_replay_")
    vlm_client = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    
    smart_config = {
        "motion_method": "MOG2",
        "motion_threshold": 0.1,
        "backup_interval": 30.0,
        "min_frame_interval": 1.0,
        "batch_buffer_size": 5,
        "batch_timeout": 5.0,
    }
    
    pipeline = ReplayPipeline(
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
    """示例6: Qwen3.5 本地模型推理
    
    使用 Qwen3.5-0.8B 本地模型进行视频分析。
    需要安装 vlm-qwen 依赖并下载模型文件。
    """
    print("\n" + "=" * 60)
    print("示例6: Qwen3.5 本地模型推理")
    print("=" * 60)
    
    video_path = os.path.join(PROJECT_ROOT, "resource", "test_video.mp4")
    
    stream_config = StreamConfig(
        url=video_path,
        target_fps=1.0,
    )
    
    # Qwen3.5 配置（显存占用约 1.4GB）
    vlm_config = VLMConfig(
        model_path="/path/to/Qwen3.5-0.8B",  # 替换为实际模型路径
        device="cuda:0",
        dtype="float16",
        max_new_tokens=256,
    )
    
    vlm_client = create_vlm_client("qwen35", vlm_config)
    
    pipeline = ReplayPipeline(
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
    """示例7: OpenAI 兼容 API 后端
    
    通过 OpenAI 兼容 API 调用远程 VLM 服务分析视频。
    """
    print("\n" + "=" * 60)
    print("示例7: OpenAI 兼容 API 后端")
    print("=" * 60)
    
    from pymediaparser import APIVLMConfig
    
    video_path = os.path.join(PROJECT_ROOT, "resource", "test_video.mp4")
    
    stream_config = StreamConfig(
        url=video_path,
        target_fps=1.0,
    )
    
    # 配置 API 后端
    api_config = APIVLMConfig(
        base_url="http://localhost:8000/v1",
        api_key="your-api-key",
        model_name="Qwen2-VL-7B-Instruct",
        max_new_tokens=256,
    )
    
    vlm_client = create_vlm_client("openai_api", api_config)
    
    pipeline = ReplayPipeline(
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
    
    # pipeline.run()


# ============================================================================
# 其他高级示例
# ============================================================================

def example_progress_tracking():
    """示例8: 进度跟踪与状态监控
    
    ReplayPipeline 提供详细的进度信息，适合长时间任务监控。
    """
    print("\n" + "=" * 60)
    print("示例8: 进度跟踪与状态监控")
    print("=" * 60)
    
    video_path = os.path.join(PROJECT_ROOT, "resource", "test_video.mp4")
    
    stream_config = StreamConfig(
        url=video_path,
        target_fps=1.0,
    )
    
    # BMP 后端配置
    output_dir = tempfile.mkdtemp(prefix="progress_tracking_")
    vlm_client = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    
    pipeline = ReplayPipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
    )
    
    print(f"配置完成:")
    print(f"  输出目录: {output_dir}")
    
    print("\n进度跟踪示例代码:")
    print("-" * 40)
    code = '''
    import time
    
    # 启动 Pipeline
    pipeline.start()
    
    # 循环监控进度
    while pipeline.is_running():
        progress = pipeline.get_progress()
        
        # 打印进度信息
        print(f"状态: {progress.state}")
        print(f"已处理帧数: {progress.processed_frames}")
        print(f"当前时间戳: {progress.current_timestamp:.2f}s")
        
        if progress.duration:
            print(f"总时长: {progress.duration:.2f}s")
            pct = (progress.current_timestamp / progress.duration) * 100
            print(f"进度: {pct:.1f}%")
        
        time.sleep(1.0)
    
    # 等待完成
    pipeline.wait()
    
    # 检查结果
    if pipeline.is_completed():
        print("处理正常完成！")
    elif pipeline.is_error():
        print(f"处理出错: {progress.error}")
    elif pipeline.is_stopped():
        print("处理被手动停止")
    '''
    print(code)


def example_keyframe_only_decode():
    """示例9: 仅解码关键帧（加速处理）
    
    对于只需要大致了解视频内容的场景，可以仅解码关键帧，大幅提升处理速度。
    """
    print("\n" + "=" * 60)
    print("示例9: 仅解码关键帧（加速处理）")
    print("=" * 60)
    
    video_path = os.path.join(PROJECT_ROOT, "resource", "test_video.mp4")
    
    stream_config = StreamConfig(
        url=video_path,
        target_fps=10.0,          # 设置较高的目标频率
        decode_mode="keyframe_only",  # 仅解码关键帧
    )
    
    # BMP 后端配置
    output_dir = tempfile.mkdtemp(prefix="keyframe_only_")
    vlm_client = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    
    pipeline = ReplayPipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
    )
    
    print(f"配置完成:")
    print(f"  输出目录: {output_dir}")
    print(f"  解码模式: keyframe_only（仅关键帧）")
    print(f"  说明: 只处理视频中的 I 帧，跳过 P/B 帧")
    print(f"  适用场景: 快速预览、内容摘要")
    
    # pipeline.run()


def example_multiple_handlers():
    """示例10: 多结果处理器组合
    
    同时使用控制台输出和 HTTP 回调，实现多渠道结果推送。
    """
    print("\n" + "=" * 60)
    print("示例10: 多结果处理器组合")
    print("=" * 60)
    
    video_path = os.path.join(PROJECT_ROOT, "resource", "test_video.mp4")
    
    stream_config = StreamConfig(
        url=video_path,
        target_fps=1.0,
    )
    
    # BMP 后端配置
    output_dir = tempfile.mkdtemp(prefix="multiple_handlers_")
    vlm_client = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    
    # 配置多个处理器
    handlers = [
        ConsoleResultHandler(verbose=True),
        # HttpCallbackHandler(
        #     callback_url="http://your-server.com/callback",
        #     timeout=5.0,
        # ),
    ]
    
    pipeline = ReplayPipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
        handlers=handlers,
    )
    
    print(f"配置完成:")
    print(f"  输出目录: {output_dir}")
    print(f"  处理器列表:")
    for handler in handlers:
        print(f"    - {handler.__class__.__name__}")
    
    # pipeline.run()


def example_different_models():
    """示例11: 切换不同 VLM 模型
    
    展示如何在不同模型间切换。
    """
    print("\n" + "=" * 60)
    print("示例11: 切换不同 VLM 模型")
    print("=" * 60)
    
    video_path = os.path.join(PROJECT_ROOT, "resource", "test_video.mp4")
    
    stream_config = StreamConfig(url=video_path, target_fps=1.0)
    
    # BMP 后端（推荐入门）
    print("\n1. BMP 虚拟后端（推荐入门）:")
    output_dir = tempfile.mkdtemp(prefix="bmp_backend_")
    client_bmp = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    pipeline_bmp = ReplayPipeline(stream_config, client_bmp)
    print(f"   后端: bmp")
    print(f"   输出目录: {output_dir}")
    print(f"   特点: 无需 GPU，保存图像为 BMP 文件")
    
    # Qwen3.5 模型
    print("\n2. Qwen3.5 本地模型:")
    vlm_config_qwen35 = VLMConfig(
        model_path="/path/to/Qwen3.5-0.8B",
        device="cuda:0",
    )
    client_qwen35 = create_vlm_client("qwen35", vlm_config_qwen35)
    pipeline_qwen35 = ReplayPipeline(stream_config, client_qwen35)
    print(f"   后端: qwen35")
    print(f"   显存占用: 约 1.4GB")
    print(f"   特点: 适合资源受限场景")
    
    # OpenAI API
    print("\n3. OpenAI 兼容 API:")
    from pymediaparser import APIVLMConfig
    api_config = APIVLMConfig(
        base_url="http://localhost:8000/v1",
        model_name="Qwen2-VL-7B-Instruct",
    )
    client_api = create_vlm_client("openai_api", api_config)
    pipeline_api = ReplayPipeline(stream_config, client_api)
    print(f"   后端: openai_api")
    print(f"   API地址: {api_config.base_url}")
    print(f"   特点: 无需本地 GPU，调用远程服务")
    
    print("\n提示: 根据模型大小和 GPU 显存选择合适的模型")


if __name__ == "__main__":
    # 运行所有示例（仅打印配置，不实际执行）
    print("\n" + "=" * 70)
    print("ReplayPipeline 文件回放分析示例")
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
    example_basic_video_replay()
    example_image_analysis()
    example_smart_sampling_replay()
    example_batch_processing_replay()
    example_full_smart_replay()
    
    # Qwen3.5 和 OpenAI API 示例
    example_qwen35_local()
    example_openai_api()
    
    # 其他高级示例
    example_progress_tracking()
    example_keyframe_only_decode()
    example_multiple_handlers()
    example_different_models()
    
    print("\n" + "=" * 70)
    print("所有示例配置展示完成")
    print("=" * 70)
    print("\n【快速开始】")
    print("1. 使用 BMP 后端测试（无需 GPU）:")
    print("   - 将示例中的文件路径替换为实际的视频/图片路径")
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
    print("- ReplayPipeline 会在文件处理完毕后自动退出")
    print("- 可以使用 Ctrl+C 提前停止处理")
    print("- 支持的格式: MP4, AVI, MKV, FLV, MOV, PNG, JPG, JPEG, BMP, WebP")
