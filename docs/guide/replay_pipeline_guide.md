# ReplayPipeline 用户使用手册

## 概述

`ReplayPipeline` 是一个文件回放 VLM（视觉语言模型）分析 Pipeline，用于处理视频文件或图片文件。与 `LivePipeline` 采用镜像架构，核心差异在于队列策略和处理完成检测。

### 核心特性

- **多格式支持**：支持 mp4、mov、mkv、avi、flv、ts 等主流视频格式
- **图片处理**：支持 bmp、jpg、png、gif、webp、tiff 等图片格式
- **不丢帧保证**：阻塞式队列策略，确保所有帧都被处理
- **进度跟踪**：实时显示处理进度百分比
- **自动完成**：文件处理完毕后自动退出并触发完成回调
- **智能功能**：支持智能采样、批量处理、图像预处理

### 与 LivePipeline 的关键差异

| 特性 | ReplayPipeline | LivePipeline |
|------|----------------|--------------|
| 输入源 | 视频/图片文件 | 实时流（RTMP/HTTP） |
| 队列策略 | 阻塞 put，不丢帧 | 满时丢弃旧帧 |
| 完成检测 | 文件读完自动退出 | 无限运行，需手动停止 |
| 进度信息 | 包含总帧数、时长、百分比 | 仅当前时间戳 |
| 终态 | `COMPLETED` / `STOPPED` / `ERROR` | `STOPPED` / `ERROR` |

---

## 快速开始

### 基本使用

```python
from pymediaparser import (
    ReplayPipeline,
    StreamConfig,
    VLMConfig,
    create_vlm_client,
    ConsoleResultHandler,
)

# 1. 配置文件路径
stream_config = StreamConfig(
    url="/path/to/video.mp4",
    target_fps=1.0,  # 每秒抽取 1 帧
)

# 2. 创建 VLM 客户端
vlm_config = VLMConfig(device="cuda:0", max_new_tokens=256)
vlm_client = create_vlm_client("qwen3", vlm_config)

# 3. 创建 Pipeline
pipeline = ReplayPipeline(
    stream_config=stream_config,
    vlm_client=vlm_client,
    prompt="请描述当前画面中的内容。",
)

# 4. 阻塞式运行
pipeline.run()

# 5. 检查完成状态
if pipeline.is_completed():
    print("文件处理完成")
```

### 处理图片文件

```python
stream_config = StreamConfig(
    url="/path/to/image.jpg",
    target_fps=1.0,  # 图片文件此参数无效
)

pipeline = ReplayPipeline(stream_config, vlm_client)
pipeline.run()
```

### CLI 命令行运行

```bash
python -m pymediaparser.replay_pipeline \
    --url "/path/to/video.mp4" \
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
│ (文件读取)   │     │ (阻塞入队)  │     │ (VLM 推理)  │
└─────────────┘     └─────────────┘     └─────────────┘
```

| 组件 | 职责 | 特点 |
|------|------|------|
| 生产者 | 读取文件、解码、按频率抽帧 | 阻塞入队，不丢帧 |
| 队列 | 帧数据缓冲 | 有界队列，满时阻塞 |
| 消费者 | VLM 推理、结果输出 | GPU 密集 |

### 状态流转

```
IDLE → STARTING → RUNNING → STOPPING → STOPPED
                      │
                      ├── 正常完成 → COMPLETED
                      │
                      └── 异常 → ERROR
```

**状态说明：**

| 状态 | 说明 | 触发条件 |
|------|------|----------|
| `IDLE` | 初始状态，未启动 | 创建实例后 |
| `STARTING` | 启动中（加载模型） | 调用 `start()` |
| `RUNNING` | 正常运行中 | 启动成功 |
| `STOPPING` | 停止中（清理资源） | 调用 `stop()` |
| `COMPLETED` | 正常完成 | 文件处理完毕 |
| `STOPPED` | 被主动停止 | 调用 `stop()` |
| `ERROR` | 异常终止 | 发生错误 |

**重要约定：** 进入任何终态（`COMPLETED`/`STOPPED`/`ERROR`）时，资源已清理完毕。

---

## 配置详解

### StreamConfig - 文件配置

```python
from pymediaparser import StreamConfig

stream_config = StreamConfig(
    url="/path/to/video.mp4",    # 文件路径或网络 URL
    format=None,                  # 容器格式：None(自动检测)
    target_fps=1.0,               # 目标抽帧频率（帧/秒）
    timeout=10.0,                 # 网络文件读取超时（秒）
    max_queue_size=3,             # 帧缓冲队列大小
    decode_mode="all",            # 解码模式："all" / "keyframe_only"
)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `url` | `str` | 必填 | 文件路径或网络 URL |
| `target_fps` | `float` | `1.0` | 目标抽帧频率，`0.5` 表示每 2 秒一帧 |
| `max_queue_size` | `int` | `3` | 帧缓冲队列大小 |
| `decode_mode` | `str` | `"all"` | 解码模式：`"all"` 全帧解码，`"keyframe_only"` 仅关键帧 |

**支持的文件格式：**

| 类型 | 支持格式 |
|------|----------|
| 视频 | mp4, mov, mkv, avi, flv, ts, m3u8, ps 等 |
| 图片 | bmp, jpg, jpeg, png, gif, webp, tiff, tif |

---

## 使用模式

### 模式一：阻塞式运行

适用于脚本工具、批处理任务。

```python
pipeline = ReplayPipeline(stream_config, vlm_client)
pipeline.run()  # 阻塞，文件处理完毕或 Ctrl+C 时退出

if pipeline.is_completed():
    print("处理完成")
elif pipeline.is_stopped():
    print("被主动停止")
elif pipeline.is_error():
    print(f"异常终止: {pipeline.get_progress().error}")
```

### 模式二：非阻塞启动 + 轮询进度

适用于需要显示进度或执行其他任务的场景。

```python
pipeline = ReplayPipeline(stream_config, vlm_client)
pipeline.start()  # 非阻塞

# 轮询进度
while pipeline.is_running():
    progress = pipeline.get_progress()
    pct = progress.progress_percent
    if pct is not None:
        print(f"进度: {pct:.1f}% ({progress.processed_frames} 帧)")
    time.sleep(1)

# 等待进入终态
pipeline.wait()
print(f"最终状态: {pipeline.get_state().value}")
```

### 模式三：服务化集成

适用于 Web 服务、异步任务队列。

```python
class FileAnalysisService:
    def __init__(self):
        self.pipeline: Optional[ReplayPipeline] = None
        self.results: List[FrameResult] = []
    
    def submit_file(self, file_path: str) -> str:
        """提交文件分析任务"""
        if self.pipeline and self.pipeline.is_running():
            raise RuntimeError("已有任务在运行中")
        
        class Collector(ResultHandler):
            def __init__(self, results):
                self.results = results
            def handle(self, result):
                self.results.append(result)
        
        stream_config = StreamConfig(url=file_path)
        self.pipeline = ReplayPipeline(
            stream_config=stream_config,
            vlm_client=create_vlm_client("qwen3"),
            handlers=[Collector(self.results)],
        )
        self.pipeline.start()
        return "submitted"
    
    def get_status(self) -> dict:
        """获取任务状态"""
        if not self.pipeline:
            return {"state": "idle"}
        
        progress = self.pipeline.get_progress()
        return {
            "state": progress.state.value,
            "processed_frames": progress.processed_frames,
            "progress_percent": progress.progress_percent,
            "elapsed_time": progress.elapsed_time,
        }
    
    def cancel(self) -> str:
        """取消任务"""
        if self.pipeline and self.pipeline.is_running():
            self.pipeline.stop()
        return "cancelled"
```

---

## 进度跟踪

### 获取进度信息

```python
progress = pipeline.get_progress()

# 基本信息
print(f"状态: {progress.state.value}")
print(f"已处理帧数: {progress.processed_frames}")
print(f"运行时长: {progress.elapsed_time:.1f}s")

# 视频文件特有信息
if progress.total_frames is not None:
    print(f"总帧数: {progress.total_frames}")

if progress.duration is not None:
    print(f"视频时长: {progress.duration:.1f}s")

if progress.progress_percent is not None:
    print(f"进度: {progress.progress_percent:.1f}%")

# 当前处理位置
print(f"当前时间戳: {progress.current_timestamp:.3f}s")

# 错误信息
if progress.error:
    print(f"错误: {progress.error}")
```

### 进度百分比计算

进度百分比基于视频时间戳计算：

```python
progress_percent = (current_timestamp / duration) * 100.0
```

**注意：** 图片文件的 `progress_percent` 和 `duration` 均为 `None`。

### 序列化进度信息

```python
progress_dict = progress.to_dict()
# {
#     "state": "running",
#     "processed_frames": 100,
#     "start_time": 1700000000.0,
#     "elapsed_time": 120.5,
#     "total_frames": 1000,
#     "current_timestamp": 100.0,
#     "duration": 1000.0,
#     "progress_percent": 10.0,
#     "error": None,
#     "extra": {}
# }
```

---

## 结果处理

### 自定义处理器

```python
from pymediaparser import ResultHandler, FrameResult, PipelineProgress

class JSONFileHandler(ResultHandler):
    """将结果保存为 JSON 文件"""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.results: List[dict] = []
    
    def handle(self, result: FrameResult) -> None:
        self.results.append({
            "frame_index": result.frame_index,
            "timestamp": result.timestamp,
            "vlm_text": result.vlm_result.text,
            "inference_time": result.vlm_result.inference_time,
        })
    
    def on_complete(self) -> None:
        import json
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到 {self.output_path}")
    
    def on_error(self, error: Exception) -> None:
        print(f"处理出错: {error}")

# 使用自定义处理器
pipeline = ReplayPipeline(
    stream_config=stream_config,
    vlm_client=vlm_client,
    handlers=[JSONFileHandler("output.json")],
)
pipeline.run()
```

### 多处理器组合

```python
handlers = [
    ConsoleResultHandler(verbose=True),
    JSONFileHandler("results.json"),
    HttpCallbackHandler("https://api.example.com/callback"),
]

pipeline = ReplayPipeline(
    stream_config=stream_config,
    vlm_client=vlm_client,
    handlers=handlers,
)
```

---

## 高级功能

### 智能采样

基于内容变化检测的自适应采样，减少冗余帧处理。

```python
pipeline = ReplayPipeline(
    stream_config=stream_config,
    vlm_client=vlm_client,
    enable_smart_sampling=True,
    smart_config={
        'motion_method': 'MOG2',        # 运动检测方法
        'motion_threshold': 0.1,        # 运动检测阈值
        'backup_interval': 30.0,        # 备份帧间隔（秒）
        'min_frame_interval': 1.0,      # 最小帧间隔（秒）
    },
)
```

### 批量处理

将多帧合并为批次进行推理，提高 GPU 利用率。

```python
pipeline = ReplayPipeline(
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
    method="motion",
    padding_ratio=0.2,
    min_roi_ratio=0.2,
)

pipeline = ReplayPipeline(
    stream_config=stream_config,
    vlm_client=vlm_client,
    preprocessing="resize",           # 或 "roi_crop"
    preprocess_config=resize_config,
)
```

### 仅解码关键帧

对于长视频，可仅解码关键帧以加速处理：

```python
stream_config = StreamConfig(
    url="/path/to/long_video.mp4",
    target_fps=0.2,  # 每 5 秒一帧
    decode_mode="keyframe_only",  # 仅解码 I 帧
)
```

---

## VLM 后端选择

### 本地模型

```python
from pymediaparser import create_vlm_client, VLMConfig

vlm_client = create_vlm_client("qwen3", VLMConfig(
    model_path="/path/to/model",
    device="cuda:0",
    dtype="float16",
    max_new_tokens=256,
))
```

### API 后端

```python
from pymediaparser.vlm.configs import APIVLMConfig

vlm_client = create_vlm_client("openai_api", APIVLMConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4-vision-preview",
    max_new_tokens=256,
))
```

---

## 最佳实践

### 1. 大文件处理

```python
# 长视频推荐配置
stream_config = StreamConfig(
    url="/path/to/long_video.mp4",
    target_fps=0.5,              # 降低抽帧频率
    decode_mode="keyframe_only", # 仅解码关键帧
    max_queue_size=5,            # 适当增大队列
)

pipeline = ReplayPipeline(
    stream_config=stream_config,
    vlm_client=vlm_client,
    enable_smart_sampling=True,  # 启用智能采样
)
```

### 2. 批量文件处理

```python
import os
from pathlib import Path

def process_directory(directory: str, vlm_client):
    """批量处理目录下的所有视频文件"""
    results = {}
    
    for file_path in Path(directory).glob("*.mp4"):
        print(f"处理文件: {file_path}")
        
        pipeline = ReplayPipeline(
            StreamConfig(url=str(file_path), target_fps=1.0),
            vlm_client=vlm_client,
        )
        pipeline.run()
        
        results[str(file_path)] = {
            "completed": pipeline.is_completed(),
            "processed_frames": pipeline.get_progress().processed_frames,
        }
    
    return results
```

### 3. 错误处理与重试

```python
from pymediaparser import ReplayPipeline, StreamConfig

def process_with_retry(file_path: str, vlm_client, max_retries: int = 3):
    """带重试的文件处理"""
    for attempt in range(max_retries):
        pipeline = ReplayPipeline(
            StreamConfig(url=file_path),
            vlm_client=vlm_client,
        )
        
        try:
            pipeline.run()
            
            if pipeline.is_completed():
                return True, pipeline.get_progress()
            elif pipeline.is_error():
                error = pipeline.get_progress().error
                print(f"尝试 {attempt + 1} 失败: {error}")
            else:
                return False, pipeline.get_progress()
                
        except Exception as e:
            print(f"尝试 {attempt + 1} 异常: {e}")
    
    return False, None
```

### 4. 资源管理

```python
# 推荐：使用上下文管理器
with vlm_client:
    pipeline = ReplayPipeline(stream_config, vlm_client)
    pipeline.run()

# 或在服务中长期持有 VLM 客户端
class AnalysisService:
    def __init__(self):
        self.vlm_client = create_vlm_client("qwen3", VLMConfig())
        self.vlm_client.load()  # 预加载模型
    
    def process(self, file_path: str):
        pipeline = ReplayPipeline(
            StreamConfig(url=file_path),
            self.vlm_client,  # 复用已加载的客户端
        )
        pipeline.run()
    
    def shutdown(self):
        self.vlm_client.unload()
```

---

## CLI 参数参考

```bash
python -m pymediaparser.replay_pipeline [OPTIONS]

# 文件配置
--url TEXT              文件路径或网络 URL（必填）
--fps FLOAT             目标抽帧频率（默认 1.0）
--queue-size INT        帧缓冲队列大小（默认 3）
--decode-mode TEXT      解码模式：all / keyframe_only（默认 all）

# VLM 配置
--model-path TEXT       VLM 模型路径
--device TEXT           推理设备（默认 cuda:0）
--dtype TEXT            推理精度（默认 float16）
--max-tokens INT        最大生成 token 数（默认 256）
--prompt TEXT           VLM 提示词

# VLM 后端
--vlm-backend TEXT      VLM 后端：qwen2 / qwen3 / openai_api（默认 qwen3）
--api-base-url TEXT     API 服务地址
--api-key TEXT          API 密钥
--api-model TEXT        API 模型名称

# 智能功能
--smart-sampling        启用智能采样
--batch-processing      启用批量处理

# 图像预处理
--preprocessing TEXT    预处理策略：resize / roi_crop
--max-size INT          [resize] 图像最大边长
--roi-method TEXT       [roi_crop] ROI 检测方法
--roi-padding FLOAT     [roi_crop] 边界扩展比例
--min-roi-ratio FLOAT   [roi_crop] 最小 ROI 占比

# 日志
--log-level TEXT        日志级别（默认 INFO）
```

---

## 常见问题

### Q: ReplayPipeline 和 LivePipeline 该选哪个？

| 场景 | 推荐 |
|------|------|
| 实时流分析（监控、直播） | `LivePipeline` |
| 视频文件分析 | `ReplayPipeline` |
| 图片分析 | `ReplayPipeline` |
| 需要进度信息 | `ReplayPipeline` |
| 需要保证不丢帧 | `ReplayPipeline` |

### Q: 如何判断文件处理是否完成？

```python
pipeline.run()

if pipeline.is_completed():
    print("文件处理完成")
elif pipeline.is_stopped():
    print("被主动停止")
elif pipeline.is_error():
    print(f"异常终止: {pipeline.get_progress().error}")
```

### Q: 处理大文件时内存不足怎么办？

1. 降低抽帧频率：`target_fps=0.5`
2. 使用关键帧模式：`decode_mode="keyframe_only"`
3. 减小队列大小：`max_queue_size=2`
4. 启用智能采样过滤冗余帧

### Q: 如何获取视频的元信息？

```python
from pymediaparser import FileReader, StreamConfig

with FileReader(StreamConfig(url="/path/to/video.mp4")) as reader:
    print(f"总帧数: {reader.total_frames}")
    print(f"时长: {reader.duration_seconds}秒")
    print(f"是否为图片: {reader.is_image}")
```

### Q: ERROR 状态后如何恢复？

ERROR 状态是终态，资源已自动清理。可以直接调用 `start()` 重新启动：

```python
if pipeline.is_error():
    # 可以直接重新启动
    pipeline.start()
```

### Q: 如何处理网络视频文件？

```python
stream_config = StreamConfig(
    url="https://example.com/video.mp4",
    timeout=30.0,  # 设置较长的超时时间
)
pipeline = ReplayPipeline(stream_config, vlm_client)
pipeline.run()
```

---

## 相关文档

- [LivePipeline 用户手册](./live_pipeline_guide.md) - 实时流分析
- [VLM 后端开发指南](../design/vlm_backend_design.md) - 自定义 VLM 后端
- [智能采样器设计](../design/smart_sampler_design.md) - 智能采样原理
