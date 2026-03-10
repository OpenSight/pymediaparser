# LivePipeline 用户使用手册

## 概述

`LivePipeline` 是一个实时流 VLM（视觉语言模型）分析 Pipeline，采用生产者-消费者架构，将拉流、抽帧、VLM 推理串联为完整的处理流程。适用于实时视频流分析、监控场景理解、直播内容审核等业务场景。

### 核心特性

- **多协议支持**：支持 RTMP、HTTP-FLV、HTTP-TS、HLS 等主流流媒体协议
- **双模式架构**：传统模式（固定频率抽帧）与智能模式（自适应采样）
- **内存可控**：有界队列 + 丢弃旧帧策略，防止内存持续增长
- **异步非阻塞**：`start()` 非阻塞启动，支持服务化集成
- **状态管理**：完善的状态机设计，支持优雅启停和异常恢复
- **可扩展结果处理**：支持控制台输出、HTTP 回调等多种输出方式

---

## 快速开始

### 基本使用

```python
from pymediaparser import (
    LivePipeline,
    StreamConfig,
    VLMConfig,
    create_vlm_client,
    ConsoleResultHandler,
)

# 1. 配置流接入参数
stream_config = StreamConfig(
    url="rtmp://example.com/live/stream",
    target_fps=1.0,  # 每秒抽取 1 帧
)

# 2. 创建 VLM 客户端
vlm_config = VLMConfig(device="cuda:0", max_new_tokens=256)
vlm_client = create_vlm_client("qwen3", vlm_config)

# 3. 创建 Pipeline
pipeline = LivePipeline(
    stream_config=stream_config,
    vlm_client=vlm_client,
    prompt="请描述当前画面中的人物活动。",
)

# 4. 阻塞式运行（支持 Ctrl+C 优雅退出）
pipeline.run()
```

### CLI 命令行运行

```bash
python -m pymediaparser.live_pipeline \
    --url "rtmp://host/live/stream" \
    --fps 1.0 \
    --device cuda:0 \
    --prompt "请描述画面中的内容。"
```

---

## 核心概念

### 架构设计

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   生产者     │ --> │   队列      │ --> │   消费者     │
│ (拉流解码)   │     │ (有界队列)  │     │ (VLM 推理)  │
└─────────────┘     └─────────────┘     └─────────────┘
```

| 组件 | 职责 | 特点 |
|------|------|------|
| 生产者 | 拉流解码、按频率抽帧 | I/O 密集，不阻塞 |
| 队列 | 帧数据缓冲 | 有界队列，满时丢弃旧帧 |
| 消费者 | VLM 推理、结果输出 | GPU 密集，批处理可选 |

### 智能模式架构（可选）

当启用智能采样或预处理时，会增加处理器线程：

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   生产者     │ --> │ 处理器队列   │ --> │   处理器     │ --> │   主队列     │
│ (拉流解码)   │     │             │     │ (智能采样)   │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                  │
                                                                  v
                                                           ┌─────────────┐
                                                           │   消费者     │
                                                           │ (VLM 推理)  │
                                                           └─────────────┘
```

### 状态流转

```
IDLE → STARTING → RUNNING → STOPPING → STOPPED
                      │
                      ├── 正常完成 → COMPLETED（仅 ReplayPipeline）
                      │
                      └── 异常 → ERROR
```

**状态说明：**

| 状态 | 说明 | 资源状态 |
|------|------|----------|
| `IDLE` | 初始状态，未启动 | 未加载 |
| `STARTING` | 启动中（加载模型） | 加载中 |
| `RUNNING` | 正常运行中 | 已加载 |
| `STOPPING` | 停止中（清理资源） | 清理中 |
| `STOPPED` | 被主动停止 | 已清理 |
| `ERROR` | 异常终止 | 已清理 |

**重要约定：** 进入任何终态（`STOPPED`/`ERROR`）时，资源已清理完毕，调用者无需再调用 `stop()`。

---

## 配置详解

### StreamConfig - 流接入配置

```python
from pymediaparser import StreamConfig

stream_config = StreamConfig(
    url="rtmp://example.com/live/stream",  # 流地址
    format=None,                            # 容器格式：None(自动检测) / "flv" / "mpegts"
    target_fps=1.0,                         # 目标抽帧频率（帧/秒）
    reconnect_interval=3.0,                 # 断线重连间隔（秒）
    timeout=10.0,                           # 读取超时（秒）
    max_queue_size=3,                       # 帧缓冲队列大小
    decode_mode="all",                      # 解码模式："all" / "keyframe_only"
)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `url` | `str` | 必填 | 流地址，支持 `rtmp://`、`http://*.flv`、`http://*.ts`、`.m3u8` |
| `format` | `str \| None` | `None` | 显式指定容器格式，`None` 表示自动检测 |
| `target_fps` | `float` | `1.0` | 目标抽帧频率，如 `0.5` 表示每 2 秒一帧 |
| `reconnect_interval` | `float` | `3.0` | 断线后等待重连秒数 |
| `timeout` | `float` | `10.0` | 读取超时秒数 |
| `max_queue_size` | `int` | `3` | 帧缓冲队列最大长度，超出时丢弃旧帧 |
| `decode_mode` | `str` | `"all"` | 解码模式：`"all"` 全帧解码，`"keyframe_only"` 仅解码关键帧 |

### VLMConfig - VLM 模型配置

```python
from pymediaparser import VLMConfig

vlm_config = VLMConfig(
    model_path="/path/to/model",           # 本地模型路径（可选）
    device="cuda:0",                        # 推理设备
    dtype="float16",                        # 推理精度
    max_new_tokens=256,                     # 最大生成 token 数
    default_prompt="请描述画面内容。",       # 默认提示词
    use_flash_attn=True,                    # 是否使用 Flash Attention
    max_pixels=512 * 28 * 28,               # 最大像素数（控制显存）
    min_pixels=256 * 28 * 28,               # 最小像素数
)
```

### APIVLMConfig - API 后端配置

```python
from pymediaparser.vlm.configs import APIVLMConfig

api_config = APIVLMConfig(
    base_url="https://api.example.com/v1",
    api_key="your-api-key",
    model_name="gpt-4-vision-preview",
    max_new_tokens=256,
    default_prompt="请描述画面内容。",
)
```

---

## 使用模式

### 模式一：阻塞式运行

适用于脚本工具、一次性任务场景。

```python
pipeline = LivePipeline(stream_config, vlm_client)
pipeline.run()  # 阻塞，支持 Ctrl+C 优雅退出
```

### 模式二：非阻塞启动 + 轮询状态

适用于需要同时执行其他任务的场景。

```python
pipeline = LivePipeline(stream_config, vlm_client)
pipeline.start()  # 非阻塞

# 执行其他任务...
while pipeline.is_running():
    progress = pipeline.get_progress()
    print(f"已处理 {progress.processed_frames} 帧")
    time.sleep(1)

# 等待进入终态
pipeline.wait(timeout=60.0)
```

### 模式三：服务化集成

适用于 Web 服务、微服务架构。

```python
class StreamAnalysisService:
    def __init__(self):
        self.pipeline: Optional[LivePipeline] = None
        self.results: List[FrameResult] = []
    
    def start_stream(self, url: str) -> str:
        """启动流分析"""
        if self.pipeline and self.pipeline.is_running():
            raise RuntimeError("已有流在运行中")
        
        # 自定义结果处理器
        class Collector(ResultHandler):
            def __init__(self, results):
                self.results = results
            def handle(self, result):
                self.results.append(result)
        
        stream_config = StreamConfig(url=url)
        self.pipeline = LivePipeline(
            stream_config=stream_config,
            vlm_client=create_vlm_client("qwen3"),
            handlers=[Collector(self.results)],
        )
        self.pipeline.start()
        return "started"
    
    def stop_stream(self) -> str:
        """停止流分析"""
        if self.pipeline:
            self.pipeline.stop()
        return "stopped"
    
    def get_status(self) -> dict:
        """获取状态"""
        if not self.pipeline:
            return {"state": "idle"}
        return self.pipeline.get_progress().to_dict()
```

---

## 状态管理 API

### 状态查询方法

```python
# 获取当前状态
state = pipeline.get_state()  # 返回 PipelineState 枚举

# 获取进度信息
progress = pipeline.get_progress()
print(f"状态: {progress.state}")
print(f"已处理帧数: {progress.processed_frames}")
print(f"运行时长: {progress.elapsed_time:.1f}s")
print(f"当前时间戳: {progress.current_timestamp:.3f}s")

# 状态判断方法
pipeline.is_running()    # 是否运行中
pipeline.is_stopped()    # 是否被主动停止
pipeline.is_error()      # 是否异常终止
pipeline.is_terminal()   # 是否处于终态（可重新启动）
```

### 状态转换规则

| 当前状态 | 可执行操作 | 目标状态 |
|----------|------------|----------|
| `IDLE` | `start()` | `STARTING` → `RUNNING` |
| `STARTING` | `stop()` | `STOPPING` → `STOPPED` |
| `RUNNING` | `stop()` | `STOPPING` → `STOPPED` |
| `STOPPED` | `start()` | `STARTING` → `RUNNING` |
| `ERROR` | `start()` | `STARTING` → `RUNNING` |

**注意：** 终态（`STOPPED`/`ERROR`）下调用 `start()` 会自动重置状态并重新启动。

---

## 结果处理

### 内置处理器

#### ConsoleResultHandler

将结果打印到控制台。

```python
from pymediaparser import ConsoleResultHandler

handler = ConsoleResultHandler(verbose=True)  # verbose=True 显示详细信息
```

输出示例：
```
[2026-03-10 14:30:01] Frame #0 | ts=1.000s | 推理耗时=0.82s | tokens=128
>>> 画面中有一个人正在走路...
────────────────────────────────────────────────────────────
```

#### HttpCallbackHandler

通过 HTTP POST 推送结果。

```python
from pymediaparser import HttpCallbackHandler

handler = HttpCallbackHandler(
    callback_url="https://your-server.com/callback",
    timeout=5.0,
)
```

### 自定义处理器

继承 `ResultHandler` 实现自定义逻辑：

```python
from pymediaparser import ResultHandler, FrameResult, PipelineProgress

class DatabaseHandler(ResultHandler):
    """将结果存储到数据库"""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def handle(self, result: FrameResult) -> None:
        self.db.insert({
            "frame_index": result.frame_index,
            "timestamp": result.timestamp,
            "vlm_text": result.vlm_result.text,
            "inference_time": result.vlm_result.inference_time,
        })
    
    def on_start(self) -> None:
        print("Pipeline 启动，开始记录...")
    
    def on_stop(self) -> None:
        print("Pipeline 停止，记录结束。")
    
    def on_error(self, error: Exception) -> None:
        self.db.log_error(str(error))
    
    def on_progress(self, progress: PipelineProgress) -> None:
        # 可选：记录进度信息
        pass
```

### 多处理器组合

```python
handlers = [
    ConsoleResultHandler(verbose=True),
    HttpCallbackHandler("https://api.example.com/callback"),
    DatabaseHandler(db_conn),
]

pipeline = LivePipeline(
    stream_config=stream_config,
    vlm_client=vlm_client,
    handlers=handlers,  # 传入多个处理器
)
```

---

## 高级功能

### 智能采样

基于内容变化检测的自适应采样，减少冗余帧处理。

```python
pipeline = LivePipeline(
    stream_config=stream_config,
    vlm_client=vlm_client,
    enable_smart_sampling=True,
    smart_config={
        'motion_method': 'MOG2',        # 运动检测方法：'MOG2' / 'frame_diff'
        'motion_threshold': 0.1,        # 运动检测阈值
        'backup_interval': 30.0,        # 备份帧间隔（秒）
        'min_frame_interval': 1.0,      # 最小帧间隔（秒）
    },
)
```

### 批量处理

将多帧合并为批次进行推理，提高 GPU 利用率。

```python
pipeline = LivePipeline(
    stream_config=stream_config,
    vlm_client=vlm_client,
    enable_batch_processing=True,
    smart_config={
        'batch_buffer_size': 5,         # 批次大小
        'batch_timeout': 5.0,           # 批次超时（秒）
    },
)
```

### 图像预处理

支持缩放和 ROI 裁剪预处理。

```python
from pymediaparser.image_processor import ResizeConfig, ROICropConfig

# 缩放预处理
resize_config = ResizeConfig(max_size=1024)

# ROI 裁剪预处理
roi_config = ROICropConfig(
    method="motion",        # 检测方法：'motion' / 'saliency'
    padding_ratio=0.2,      # 边界扩展比例
    min_roi_ratio=0.2,      # 最小 ROI 占比
)

pipeline = LivePipeline(
    stream_config=stream_config,
    vlm_client=vlm_client,
    preprocessing="resize",           # 或 "roi_crop"
    preprocess_config=resize_config,  # 或 roi_config
)
```

### 完整智能模式

智能采样 + 批量处理 + 预处理的组合：

```python
pipeline = LivePipeline(
    stream_config=stream_config,
    vlm_client=vlm_client,
    enable_smart_sampling=True,
    enable_batch_processing=True,
    preprocessing="roi_crop",
    preprocess_config=ROICropConfig(method="motion"),
    smart_config={
        'motion_method': 'MOG2',
        'motion_threshold': 0.1,
        'batch_buffer_size': 5,
        'batch_timeout': 5.0,
    },
)
```

---

## VLM 后端选择

### 内置后端

| 后端名称 | 说明 | 配置类型 |
|----------|------|----------|
| `qwen2` | Qwen2-VL 本地模型 | `VLMConfig` / `LocalVLMConfig` |
| `qwen3` | Qwen3-VL 本地模型 | `VLMConfig` / `LocalVLMConfig` |
| `qwen35` | Qwen3.5 本地模型 | `VLMConfig` / `LocalVLMConfig` |
| `openai_api` | OpenAI 兼容 API | `APIVLMConfig` |
| `bmp` | BMP 平台 API | `APIVLMConfig` |

### 创建 VLM 客户端

```python
from pymediaparser import create_vlm_client, VLMConfig
from pymediaparser.vlm.configs import APIVLMConfig

# 本地模型
vlm_client = create_vlm_client("qwen3", VLMConfig(device="cuda:0"))

# API 后端
vlm_client = create_vlm_client("openai_api", APIVLMConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-key",
    model_name="gpt-4-vision-preview",
))
```

### 注册自定义后端

```python
from pymediaparser import register_vlm_backend

register_vlm_backend(
    name="my_vlm",
    module_path="my_package.vlm_impl",
    class_name="MyVLMClient",
)

# 使用自定义后端
vlm_client = create_vlm_client("my_vlm", my_config)
```

---

## 最佳实践

### 1. 资源管理

```python
# 推荐：使用上下文管理器
with vlm_client:
    pipeline = LivePipeline(stream_config, vlm_client)
    pipeline.run()

# 或手动管理
try:
    vlm_client.load()
    pipeline = LivePipeline(stream_config, vlm_client)
    pipeline.run()
finally:
    vlm_client.unload()
```

### 2. 异常处理

```python
pipeline = LivePipeline(stream_config, vlm_client)
pipeline.start()

try:
    pipeline.wait(timeout=3600)  # 最多等待 1 小时
except KeyboardInterrupt:
    pipeline.stop()

# 检查最终状态
if pipeline.is_error():
    progress = pipeline.get_progress()
    print(f"Pipeline 异常终止: {progress.error}")
```

### 3. 性能调优

| 场景 | 推荐配置 |
|------|----------|
| 低延迟优先 | `decode_mode="keyframe_only"`, `max_queue_size=1` |
| 吞吐量优先 | `enable_batch_processing=True`, `batch_buffer_size=10` |
| 显存受限 | `max_pixels=256*28*28`, `dtype="float16"` |
| CPU 推理 | `device="cpu"`, `dtype="float32"` |

### 4. 监控与日志

```python
import logging

# 配置日志级别
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# 自定义监控处理器
class MetricsHandler(ResultHandler):
    def __init__(self):
        self.frame_count = 0
        self.total_inference_time = 0.0
    
    def handle(self, result: FrameResult) -> None:
        self.frame_count += 1
        self.total_inference_time += result.vlm_result.inference_time
        
        if self.frame_count % 100 == 0:
            avg_time = self.total_inference_time / self.frame_count
            print(f"[监控] 已处理 {self.frame_count} 帧, 平均推理耗时 {avg_time:.2f}s")
```

---

## CLI 参数参考

```bash
python -m pymediaparser.live_pipeline [OPTIONS]

# 流配置
--url TEXT              流地址（必填）
--format TEXT           容器格式：flv / mpegts
--fps FLOAT             目标抽帧频率（默认 1.0）
--queue-size INT        帧缓冲队列大小（默认 3）
--reconnect FLOAT       断线重连间隔秒数（默认 3.0）
--decode-mode TEXT      解码模式：all / keyframe_only（默认 all）

# VLM 配置
--model-path TEXT       VLM 模型路径
--device TEXT           推理设备（默认 cuda:0）
--dtype TEXT            推理精度（默认 float16）
--max-tokens INT        最大生成 token 数（默认 256）
--prompt TEXT           VLM 提示词

# VLM 后端
--vlm-backend TEXT      VLM 后端：qwen2 / qwen3 / openai_api（默认 qwen3）
--api-base-url TEXT     API 服务地址（openai_api 后端）
--api-key TEXT          API 密钥（openai_api 后端）
--api-model TEXT        API 模型名称（openai_api 后端）

# 智能功能
--smart-sampling        启用智能采样
--batch-processing      启用批量处理

# 图像预处理
--preprocessing TEXT    预处理策略：resize / roi_crop
--max-size INT          [resize] 图像最大边长（默认 1024）
--roi-method TEXT       [roi_crop] ROI 检测方法：motion / saliency（默认 motion）
--roi-padding FLOAT     [roi_crop] 边界扩展比例（默认 0.2）
--min-roi-ratio FLOAT   [roi_crop] 最小 ROI 占比（默认 0.2）

# 日志
--log-level TEXT        日志级别：DEBUG / INFO / WARNING / ERROR（默认 INFO）
```

---

## 常见问题

### Q: 如何处理断流重连？

Pipeline 内置断线重连机制，通过 `reconnect_interval` 配置重连间隔。如果流中断，生产者线程会自动尝试重新连接。

### Q: 队列满了会怎样？

当队列满时，Pipeline 会丢弃最旧的帧，保证新帧能够入队。这确保了内存不会持续增长，但也意味着在推理速度跟不上抽帧速度时会丢帧。

### Q: 如何在运行时更换 prompt？

目前不支持运行时更换 prompt。如需更换，需要停止当前 Pipeline 并创建新实例。

### Q: ERROR 状态后如何恢复？

ERROR 状态是终态，资源已自动清理。可以直接调用 `start()` 重新启动，Pipeline 会自动重置状态。

```python
if pipeline.is_error():
    # 可以直接重新启动
    pipeline.start()
```

### Q: 如何获取每一帧的原始图像？

可以在自定义 `ResultHandler` 中保存图像：

```python
class ImageSaver(ResultHandler):
    def handle(self, result: FrameResult) -> None:
        # 注意：FrameResult 不包含原始图像
        # 如需保存图像，需要在 Pipeline 外部实现
        pass
```

当前版本 `FrameResult` 不包含原始图像数据。如需保存图像，建议在自定义 Pipeline 子类中实现。

---

## 相关文档

- [ReplayPipeline 用户手册](./replay_pipeline_guide.md) - 视频文件回放分析
- [VLM 后端开发指南](../design/vlm_backend_design.md) - 自定义 VLM 后端
- [智能采样器设计](../design/smart_sampler_design.md) - 智能采样原理与配置
