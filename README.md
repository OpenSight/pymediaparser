# pymediaparser
 
媒体流解析与视觉语言模型工具包，提供实时视频流接入、智能帧采样、VLM 推理等功能。

## 功能特性

- **实时流接入**：支持 RTMP/HTTP-FLV/HTTP-TS/HLS 等协议的视频流拉取
- **多后端 VLM 推理**：
  - 本地模型：Qwen2-VL / Qwen3-VL / Qwen3.5
  - API 服务：vLLM / Ollama / OpenAI / 通义千问等 OpenAI 兼容接口
  - 调试模式：BMP 后端（保存图像，不调用模型）
- **智能帧采样**：基于运动检测和内容变化分析的自适应采样，减少无效帧处理
- **批量处理**：支持帧缓冲和批量推理，提升 GPU 利用率
- **灵活扩展**：模块化设计，支持自定义采样器和结果处理器
- **轻量部署**：核心包无需 GPU 依赖，按需安装可选功能

## 安装

### 核心安装（轻量级）

核心包仅包含视频流解析和帧处理功能，无需 GPU 依赖：

```bash
pip install pymediaparser
```

### 可选功能安装

根据使用场景选择安装对应的可选依赖：

| 功能 | 安装命令 | 说明 |
|------|----------|------|
| **Qwen 本地推理** | `pip install 'pymediaparser[vlm-qwen]'` | Qwen2-VL / Qwen3-VL / Qwen3.5 本地模型 |
| **智能采样** | `pip install 'pymediaparser[smart-sampling]'` | 高级智能帧过滤功能 |
| **开发测试** | `pip install 'pymediaparser[dev]'` | pytest 等开发工具 |
| **全部安装** | `pip install 'pymediaparser[all]'` | 所有可选功能 |

### 常见安装场景

```bash
# 场景 1：使用 OpenAI API 推理（无需本地 GPU）
pip install pymediaparser

# 场景 2：本地 Qwen 模型推理
pip install 'pymediaparser[vlm-qwen]'

# 场景 3：完整功能（本地模型 + 智能采样）
pip install 'pymediaparser[vlm-qwen,smart-sampling]'

# 场景 4：开发环境
pip install -e '.[dev]'
```

### 依赖缺失时的提示

当使用未安装依赖的功能时，会自动提示安装方法：

```
============================================================
VLM 后端 'qwen2' 缺少必要的依赖包
============================================================
导入错误: No module named 'torch'

请运行以下命令安装依赖:
    pip install 'pymediaparser[vlm-qwen]'
============================================================
```

## 快速开始

### 方式 1：命令行脚本

使用 `scripts/run_parser.py` 脚本快速启动媒体流解析：

```bash
# RTMP 流，每秒 1 帧，使用 GPU 推理
python scripts/run_parser.py --url rtmp://host/live/stream

# HTTP-FLV 流，自定义 prompt
python scripts/run_parser.py \
    --url http://host/live/stream.flv \
    --fps 0.5 \
    --prompt "识别画面中的人物并描述他们的行为。"

# 启用智能采样模式
python scripts/run_parser.py \
    --url rtmp://host/live/stream \
    --smart-sampling \
    --ssim-threshold 0.8

# 启用批量处理模式
python scripts/run_parser.py \
    --url rtmp://host/live/stream \
    --batch-processing

# 组合使用智能采样 + 批量处理
python scripts/run_parser.py \
    --url rtmp://host/live/stream \
    --smart-sampling \
    --batch-processing

# CPU 模式（无 GPU 时）
python scripts/run_parser.py \
    --url rtmp://host/live/stream \
    --device cpu --dtype float32 --fps 0.2
```

#### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--url` | 必填 | 视频流地址 (RTMP/HTTP-FLV/HTTP-TS) |
| `--fps` | 1.0 | 目标抽帧频率 |
| `--device` | cuda:0 | 推理设备 |
| `--prompt` | 内置默认 | VLM 提示词 |
| `--model-path` | 内置模型 | Qwen2-VL 模型路径 |
| `--smart-sampling` | False | 启用智能采样 |
| `--batch-processing` | False | 启用批量处理 |
| `--callback-url` | None | HTTP 回调地址 |

## 核心 Pipeline 类

本项目提供两个核心 Pipeline 类，分别用于实时流分析和文件回放分析：

| Pipeline | 用途 | 输入源 | 终止条件 |
|----------|------|--------|----------|
| **LivePipeline** | 实时流分析 | RTMP/HTTP-FLV/HTTP-TS/HLS | 手动停止 (Ctrl+C) |
| **ReplayPipeline** | 文件回放分析 | 本地视频文件 / 网络视频 URL | 文件处理完毕 |

### LivePipeline（实时流分析）

用于分析实时视频流，持续运行直到手动停止。

#### 阻塞模式

```python
from pymediaparser import LivePipeline, StreamConfig, create_vlm_client
from pymediaparser.vlm.configs import LocalVLMConfig

# 配置视频流
stream_cfg = StreamConfig(
    url="rtmp://host/live/stream",
    target_fps=1.0,  # 每秒采样 1 帧
)

# 方式 1：使用本地 Qwen 模型（需要 [vlm-qwen] 依赖）
vlm_client = create_vlm_client("qwen35", LocalVLMConfig(device="cuda:0"))

# 方式 2：使用 OpenAI 兼容 API（无需额外依赖）
# from pymediaparser.vlm.configs import APIVLMConfig
# vlm_client = create_vlm_client("openai_api", APIVLMConfig(
#     base_url="http://localhost:8000/v1",
#     model_name="Qwen2-VL-2B-Instruct",
# ))

# 创建并运行 Pipeline（阻塞直到 Ctrl+C）
pipeline = LivePipeline(
    stream_cfg,
    vlm_client,
    prompt="请描述画面中的内容。",
)
pipeline.run()
```

#### 异步模式（外部系统集成）

```python
import time
from pymediaparser import LivePipeline, StreamConfig, PipelineState, create_vlm_client
from pymediaparser.vlm.configs import APIVLMConfig

stream_cfg = StreamConfig(url="rtmp://host/live/stream", target_fps=1.0)
vlm_client = create_vlm_client("openai_api", APIVLMConfig(
    base_url="http://localhost:8000/v1",
))

pipeline = LivePipeline(stream_cfg, vlm_client)

# 启动（立即返回，后台运行）
pipeline.start()

# 轮询进度
while pipeline.is_running():
    progress = pipeline.get_progress()
    print(f"已处理: {progress.processed_frames} 帧 | 运行时间: {progress.elapsed_time:.0f}s")
    time.sleep(5)

# 主动停止
pipeline.stop()
```

### ReplayPipeline（文件回放分析）

用于分析本地视频文件或网络视频，处理完毕后自动结束。

#### 阻塞模式

```python
from pymediaparser import ReplayPipeline, StreamConfig, create_vlm_client
from pymediaparser.vlm.configs import LocalVLMConfig

# 配置视频文件（支持本地路径和网络 URL）
stream_cfg = StreamConfig(
    url="/path/to/video.mp4",  # 或 "http://example.com/video.mp4"
    target_fps=0.5,  # 每 2 秒采样 1 帧
)

vlm_client = create_vlm_client("qwen35", LocalVLMConfig(device="cuda:0"))

# 创建并运行 Pipeline（阻塞直到处理完毕）
pipeline = ReplayPipeline(
    stream_cfg,
    vlm_client,
    prompt="请描述画面中的人物活动。",
)
pipeline.run()
print("处理完成！")
```

#### 异步模式（带进度监控）

```python
import time
from pymediaparser import ReplayPipeline, StreamConfig, PipelineState, create_vlm_client
from pymediaparser.vlm.configs import APIVLMConfig
from pymediaparser.result_handler import ResultHandler

# 自定义结果处理器
class MyHandler(ResultHandler):
    def handle(self, result):
        # 处理每帧结果：写入数据库、发送消息等
        print(f"帧 #{result.frame_index}: {result.vlm_result.text[:50]}...")
    
    def on_complete(self):
        print("✅ 文件处理完成")
    
    def on_error(self, error):
        print(f"❌ 处理异常: {error}")

stream_cfg = StreamConfig(url="/path/to/video.mp4", target_fps=1.0)
vlm_client = create_vlm_client("openai_api", APIVLMConfig(
    base_url="http://localhost:8000/v1",
))

pipeline = ReplayPipeline(stream_cfg, vlm_client, handlers=[MyHandler()])

# 启动（立即返回）
pipeline.start()

# 轮询进度
while pipeline.is_running():
    progress = pipeline.get_progress()
    if progress.duration:
        print(f"进度: {progress.progress_percent:.1f}% | "
              f"{progress.current_timestamp:.1f}s / {progress.duration:.1f}s")
    time.sleep(2)

# 检查最终状态
match pipeline.get_state():
    case PipelineState.COMPLETED:
        print("处理完成")
    case PipelineState.STOPPED:
        print("被主动停止")
    case PipelineState.ERROR:
        print(f"异常: {pipeline.get_progress().error}")
```

### Pipeline 运行模式对比

| 模式 | 方法 | 特点 | 适用场景 |
|------|------|------|----------|
| **阻塞模式** | `pipeline.run()` | 阻塞当前线程 | 简单脚本、独立运行 |
| **异步模式** | `pipeline.start()` + `pipeline.stop()` | 后台运行，可控启停 | 外部系统集成、服务化 |

### Pipeline 状态说明

| 状态 | 含义 | 资源状态 |
|------|------|----------|
| `IDLE` | 初始状态，未启动 | 未分配 |
| `STARTING` | 启动中（加载模型） | 部分分配 |
| `RUNNING` | 正常运行中 | 已分配 |
| `STOPPING` | 停止中（清理资源） | 清理中 |
| `COMPLETED` | 正常完成（仅 ReplayPipeline） | 已清理 |
| `STOPPED` | 被主动停止 | 已清理 |
| `ERROR` | 异常终止 | 已清理 |

**核心契约**：进入任何终态（COMPLETED/STOPPED/ERROR）时，资源已清理完毕，调用者无需额外操作。

### 高级功能

#### 智能采样模式

启用智能采样后，系统会自动识别画面变化，只在有意义的时刻采样，大幅减少无效帧处理：

```python
from pymediaparser import LivePipeline, StreamConfig, create_vlm_client
from pymediaparser.vlm.configs import LocalVLMConfig

stream_cfg = StreamConfig(url="rtmp://host/live/stream", target_fps=30.0)
vlm_client = create_vlm_client("qwen35", LocalVLMConfig(device="cuda:0"))

pipeline = LivePipeline(
    stream_cfg,
    vlm_client,
    prompt="请描述画面中的内容。",
    # 启用智能采样
    enable_smart_sampling=True,
    smart_config={
        'motion_threshold': 0.3,      # 运动检测阈值（0-1）
        'backup_interval': 30.0,      # 保底采样间隔（秒）
        'min_frame_interval': 1.0,    # 最小帧间隔（秒）
    },
)
pipeline.run()
```

#### 批量处理模式

启用批量处理后，系统会缓存多个帧后一次性送入 VLM，提升 GPU 利用率：

```python
pipeline = LivePipeline(
    stream_cfg,
    vlm_client,
    prompt="请分析这些帧的内容变化。",
    # 启用批量处理
    enable_batch_processing=True,
    batch_config={
        'max_size': 5,        # 批次最大帧数
        'max_wait_time': 5.0, # 最大等待时间（秒）
    },
)
pipeline.run()
```

#### HTTP 回调输出

将分析结果通过 HTTP POST 推送到指定地址：

```python
from pymediaparser import ReplayPipeline, StreamConfig, HttpCallbackHandler, create_vlm_client

handlers = [
    HttpCallbackHandler(callback_url="http://your-server/callback", timeout=5.0),
]

pipeline = ReplayPipeline(stream_cfg, vlm_client, handlers=handlers)
pipeline.run()
```
 
## 主要组件

### Pipeline

| 组件 | 说明 |
|------|------|
| `LivePipeline` | 实时流 VLM 分析 Pipeline |
| `ReplayPipeline` | 文件回放 VLM 分析 Pipeline |
| `PipelineState` | Pipeline 运行状态枚举 |
| `PipelineProgress` | Pipeline 进度信息 |

### 流接入

| 组件 | 说明 |
|------|------|
| `StreamReader` | 实时流解码器，支持 RTMP/HTTP-FLV/HTTP-TS |

### 帧采样

| 组件 | 说明 |
|------|------|
| `FrameSampler` | 固定频率采样器 |
| `SimpleSmartSampler` | 简单智能采样器，基于运动检测和 SSIM 变化分析 |
| `MLSmartSampler` | 多层漏斗型智能采样器，三层过滤架构 |

### VLM 客户端

| 组件 | 说明 | 依赖 |
|------|------|------|
| `VLMClient` | VLM 客户端抽象基类 | - |
| `Qwen2VLClient` | Qwen2-VL 本地推理 | `[vlm-qwen]` |
| `Qwen3VLClient` | Qwen3-VL 本地推理 | `[vlm-qwen]` |
| `Qwen35Client` | Qwen3.5 本地推理 | `[vlm-qwen]` |
| `OpenAIAPIClient` | OpenAI 兼容 API（vLLM/Ollama/通义千问） | 核心包 |
| `BMPVLMClient` | 调试后端（保存图像为 BMP） | 核心包 |

#### VLM 后端选择指南

| 后端 | 适用场景 | 安装要求 |
|------|----------|----------|
| `qwen2` | Qwen2-VL 本地 GPU 推理 | `[vlm-qwen]` |
| `qwen3` | Qwen3-VL 本地 GPU 推理 | `[vlm-qwen]` |
| `qwen35` | Qwen3.5 本地 GPU 推理（显存更低） | `[vlm-qwen]` |
| `openai_api` | vLLM / Ollama / OpenAI API 服务 | 核心包 |
| `bmp` | 调试模式（不调用模型） | 核心包 |

### 结果处理

| 组件 | 说明 |
|------|------|
| `ResultHandler` | 结果处理器抽象基类 |
| `ConsoleResultHandler` | 控制台输出处理器 |
| `HttpCallbackHandler` | HTTP 回调处理器 |

### 配置类

| 组件 | 说明 |
|------|------|
| `StreamConfig` | 流配置（URL、帧率、队列大小等） |
| `VLMConfig` | VLM 配置（模型路径、设备等） |

## 智能采样器对比

| 特性 | FrameSampler | SimpleSmartSampler | MLSmartSampler |
|------|--------------|-------------------|----------------|
| 采样策略 | 固定频率 | 运动检测 + 变化分析 | 三层漏斗过滤 |
| 计算开销 | 无 | 低 | 中 |
| 过滤精度 | 无 | 中 | 高 |
| 适用场景 | 均匀采样 | 一般智能采样 | 高精度场景 |

## 依赖

### 核心依赖（必须）

| 包 | 用途 |
|----|------|
| Python >= 3.10 | 运行环境 |
| numpy | 数值计算 |
| pillow | 图像处理 |
| av (PyAV) | 视频流解码 |
| opencv-python | 图像处理 / 智能采样 |
| requests | HTTP 请求 |

### 可选依赖

| 包 | 功能组 | 用途 |
|----|--------|------|
| torch | `[vlm-qwen]` | 深度学习框架 |
| torchvision | `[vlm-qwen]` | 图像预处理 |
| transformers | `[vlm-qwen]` | 模型加载 |
| qwen-vl-utils | `[vlm-qwen]` | Qwen-VL 工具 |
| imagehash | `[smart-sampling]` | 图像哈希 |
| scikit-image | `[smart-sampling]` | SSIM 计算 |

## 许可证

MIT License
