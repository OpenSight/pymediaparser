#!/usr/bin/env python3
"""智能Pipeline完整使用示例

演示如何使用 LivePipeline 的智能采样和批量处理功能。

VLM 后端选择：
- bmp: 虚拟调试后端，保存图像为 BMP 文件（无需 GPU，推荐用于入门和调试）
- qwen35: Qwen3.5 本地模型推理（需要 GPU 和模型文件）
- openai_api: OpenAI 兼容 API 服务（需要 API 服务地址）
"""

import logging
import tempfile
from pymediaparser import (
    LivePipeline,
    StreamConfig,
    VLMConfig,
    create_vlm_client,
    ConsoleResultHandler,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def demo_traditional_vs_smart():
    """传统模式 vs 智能模式对比演示"""
    print("=" * 60)
    print("智能Pipeline功能演示")
    print("=" * 60)
    
    # 基础配置
    stream_url = "rtmp://your-actual-stream-address/live/stream"  # 请替换为实际流地址
    stream_config = StreamConfig(url=stream_url, target_fps=1.0, max_queue_size=5)
    
    # BMP 后端（无需 GPU）
    output_dir = tempfile.mkdtemp(prefix="smart_demo_")
    vlm_client = create_vlm_client("bmp", VLMConfig(model_path=output_dir))
    
    print("演示配置:")
    print(f"  流地址: {stream_config.url}")
    print(f"  抽帧频率: {stream_config.target_fps} fps")
    print(f"  输出目录: {output_dir}")
    print()
    
    # 1. 传统模式演示
    print("1️⃣ 传统模式 (固定频率采样)")
    print("-" * 40)
    traditional_pipeline = LivePipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
        handlers=[ConsoleResultHandler()],
    )
    print("✅ 配置完成 - 处理所有帧，无优化")
    
    # 2. 智能采样模式演示
    print("\n2️⃣ 智能采样模式")
    print("-" * 40)
    smart_config = {
        "motion_threshold": 0.1,
        "backup_interval": 30.0,
        "min_frame_interval": 1.0,
    }
    smart_pipeline = LivePipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
        handlers=[ConsoleResultHandler()],
        smart_sampler='ml',
        smart_config=smart_config,
    )
    print("✅ 配置完成 - 基于内容变化智能采样")
    print(f"   备份间隔: {smart_config['backup_interval']}秒")
    print(f"   最小帧间隔: {smart_config['min_frame_interval']}秒")
    
    # 3. 完整智能模式演示
    print("\n3️⃣ 完整智能模式")
    print("-" * 40)
    full_smart_pipeline = LivePipeline(
        stream_config=stream_config,
        vlm_client=vlm_client,
        handlers=[ConsoleResultHandler()],
        smart_sampler='ml',
        enable_batch_processing=True,
        smart_config={
            "motion_threshold": 0.1,
            "backup_interval": 30.0,
            "min_frame_interval": 1.0,
        },
        batch_config={
            "batch_buffer_size": 5,
            "batch_timeout": 5.0,
        },
    )
    print("✅ 配置完成 - 智能采样 + 批量处理")
    
    print("\n" + "=" * 60)
    print("🎯 性能优化预期:")
    print("=" * 60)
    print("传统模式    : 100% 帧处理, 100% VLM调用 (基准)")
    print("智能采样    : 30-70% 帧处理, 30-70% VLM调用 (降低30-70%)")
    print("批量处理    : 100% 帧处理, 30-50% VLM调用 (降低50-70%)") 
    print("完整智能    : 30-70% 帧处理, 10-30% VLM调用 (降低70-90%)")
    print("=" * 60)
    
    print("\n💡 使用说明:")
    print("1. 请将上面的流地址替换为真实的RTMP/HTTP流地址")
    print("2. 首次使用建议用BMP后端测试基础功能（无需GPU）")
    print("3. 生产环境推荐使用GPU模式获得最佳性能")
    print("4. 可通过 smart_sampler 和 enable_batch_processing 参数控制功能开关")
    
    # 实际运行示例（注释掉避免意外执行）
    # print("\n🚀 开始运行完整智能模式演示...")
    # try:
    #     full_smart_pipeline.run()
    # except KeyboardInterrupt:
    #     print("演示已停止")


if __name__ == "__main__":
    demo_traditional_vs_smart()
