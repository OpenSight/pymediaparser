# VLM 时序信息输入改造方案

## Context

**问题**：当前 pipeline（live_pipeline/replay_pipeline）的消费者将图像送入 VLM 时，没有传递帧序号和时间戳信息。这导致 VLM 无法理解图像的时序关系，只能针对单帧图像进行独立分析，无法通过图像的前后关系分析时间连续的活动情况。在批处理场景下问题尤为严重。

**目标**：改造 VLM 模块接口，支持传入帧信息字典（与 pipeline 队列格式一致），在消息中以穿插方式将时序信息与图像结合，让 VLM 能够理解帧的时序关系。

**设计原则**：
- **接口简洁统一**：接口只接收帧信息（列表），不再单独传入 image
- **数据结构一致性**：帧信息字典格式与 pipeline 队列保持一致
- **无需向后兼容**：直接修改接口，不保留旧参数

**帧信息字典格式**（pipeline 中已使用）：
```python
{
    'image': PIL.Image,      # RGB 图像
    'timestamp': float,      # 时间戳（秒）
    'frame_index': int,      # 帧序号
    'significant': bool,     # 是否显著帧
    'source': list,          # 触发源
}
```

---

## Implementation Plan

### Step 1: 修改 VLMClient 抽象接口

**文件**: `src/pymediaparser/vlm_base.py`

修改 `analyze()` 抽象方法签名（约第160行）：

```python
@abstractmethod
def analyze(
    self,
    frame: Dict[str, Any],  # 单个帧信息字典
    prompt: str | None = None,
) -> VLMResult:
    """对单帧图像进行理解分析。

    Args:
        frame: 帧信息字典，包含：
            - image: RGB 格式的 PIL 图像
            - timestamp: 帧时间戳（秒）
            - frame_index: 帧序号
            - significant: 是否显著帧
            - source: 触发源
        prompt: 文本提示词。
    """
```

修改 `analyze_batch()` 方法签名（约第177行）：

```python
def analyze_batch(
    self,
    frames: Sequence[Dict[str, Any]],  # 帧信息字典列表
    prompt: str | None = None,
) -> VLMResult:
    """对多帧图像进行批量理解分析。

    Args:
        frames: 帧信息字典列表，每个字典包含 image、timestamp、frame_index 等。
        prompt: 文本提示词。
    """
```

**注意**：删除默认实现中逐帧调用 `analyze()` 的逻辑，改为抽象方法或抛出 `NotImplementedError`，由子类实现。

### Step 2: 扩展 VLMConfig 配置

**文件**: `src/pymediaparser/vlm_base.py` - `VLMConfig` 类

新增时序消息格式配置：

```python
@dataclass
class VLMConfig:
    # ... 现有字段 ...

    # 时序消息格式配置
    timing_prefix: str = "以下是连续的视频帧："
    timing_format: str = "[Frame #{index}, t={timestamp:.2f}s]"
```

### Step 3: 扩展 APIVLMConfig 配置

**文件**: `src/pymediaparser/vlm/configs.py` - `APIVLMConfig` 类

新增相同的时序消息格式配置：

```python
@dataclass
class APIVLMConfig:
    # ... 现有字段 ...

    # 时序消息格式配置
    timing_prefix: str = "以下是连续的视频帧："
    timing_format: str = "[Frame #{index}, t={timestamp:.2f}s]"
```

### Step 4: 实现消息构建辅助方法（本地模型）

**文件**: `src/pymediaparser/vlm/_local_base.py`

在 `_LocalTransformersBase` 类中新增辅助方法：

```python
from typing import Any, Dict, List, Sequence

def _build_content_with_timing(
    self,
    frames: Sequence[Dict[str, Any]],
    prompt: str,
) -> List[Dict[str, Any]]:
    """构建带时序标签的消息内容（穿插方式）。

    Args:
        frames: 帧信息字典列表，包含 image、frame_index 和 timestamp。
        prompt: 提示词。

    Returns:
        消息内容列表，图像与时序标签穿插排列。
    """
    content: List[Dict[str, Any]] = []

    # 多帧时添加前缀
    if len(frames) > 1:
        content.append({
            "type": "text",
            "text": self.config.timing_prefix
        })

    # 穿插方式：图像 → 时序标签
    for frame in frames:
        img = frame.get('image')
        timing_text = self.config.timing_format.format(
            index=frame.get('frame_index', 0),
            timestamp=frame.get('timestamp', 0.0)
        )
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": timing_text})

    content.append({"type": "text", "text": prompt})
    return content
```

### Step 5: 修改本地模型 analyze() 方法

**文件**: `src/pymediaparser/vlm/_local_base.py`

修改 `analyze()` 方法（约第149行）：

```python
def analyze(
    self,
    frame: Dict[str, Any],
    prompt: str | None = None,
) -> VLMResult:
    """对单帧图像进行 VLM 推理。"""
    if self._model is None or self._processor is None:
        raise RuntimeError("模型尚未加载，请先调用 load()")

    prompt = prompt or self.config.default_prompt
    t0 = time.perf_counter()

    # 构造消息（单帧也有时序信息）
    content = self._build_content_with_timing([frame], prompt)
    messages = [{"role": "user", "content": content}]

    # ... 后续推理逻辑不变 ...
```

### Step 6: 修改本地模型 analyze_batch() 方法

**文件**: `src/pymediaparser/vlm/_local_base.py`

修改 `analyze_batch()` 方法（约第209行）：

```python
def analyze_batch(
    self,
    frames: Sequence[Dict[str, Any]],
    prompt: str | None = None,
) -> VLMResult:
    """对多张图像进行批量 VLM 推理。"""
    if not frames:
        raise ValueError("帧信息列表不能为空")

    if self._model is None or self._processor is None:
        raise RuntimeError("模型尚未加载，请先调用 load()")

    prompt = prompt or self.config.default_prompt
    start_time = time.perf_counter()

    # 使用辅助方法构建消息
    content = self._build_content_with_timing(frames, prompt)
    messages = [{"role": "user", "content": content}]

    # ... 后续推理逻辑不变 ...
```

### Step 7: 修改 OpenAI API 客户端

**文件**: `src/pymediaparser/vlm/openai_api.py`

新增 `_build_content_with_timing()` 方法，适配 OpenAI API 格式：

```python
def _build_content_with_timing(
    self,
    frames: Sequence[Dict[str, Any]],
    prompt: str,
) -> List[Dict[str, Any]]:
    """构建带时序标签的消息内容（OpenAI API 格式）。"""
    content: List[Dict[str, Any]] = []

    # 多帧时添加前缀
    if len(frames) > 1:
        content.append({
            "type": "text",
            "text": self.config.timing_prefix
        })

    for frame in frames:
        img = frame.get('image')
        timing_text = self.config.timing_format.format(
            index=frame.get('frame_index', 0),
            timestamp=frame.get('timestamp', 0.0)
        )
        # OpenAI API 使用 image_url 格式
        image_b64 = self._encode_image(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
        })
        content.append({"type": "text", "text": timing_text})

    content.append({"type": "text", "text": prompt})
    return content
```

修改 `analyze()` 方法：

```python
def analyze(
    self,
    frame: Dict[str, Any],
    prompt: str | None = None,
) -> VLMResult:
    """对单帧图像调用 API 推理。"""
    if self._session is None:
        raise RuntimeError("客户端尚未初始化，请先调用 load()")

    prompt = prompt or self.config.default_prompt
    t0 = time.perf_counter()

    content = self._build_content_with_timing([frame], prompt)
    messages = [{"role": "user", "content": content}]

    # ... 后续请求逻辑不变 ...
```

修改 `analyze_batch()` 方法：

```python
def analyze_batch(
    self,
    frames: Sequence[Dict[str, Any]],
    prompt: str | None = None,
) -> VLMResult:
    """对多张图像调用 API 批量推理。"""
    if not frames:
        raise ValueError("帧信息列表不能为空")

    if self._session is None:
        raise RuntimeError("客户端尚未初始化，请先调用 load()")

    prompt = prompt or self.config.default_prompt
    start_time = time.perf_counter()

    content = self._build_content_with_timing(frames, prompt)
    messages = [{"role": "user", "content": content}]

    # ... 后续请求逻辑不变 ...
```

### Step 8: 修改 LivePipeline 消费者

**文件**: `src/pymediaparser/live_pipeline.py`

修改 `_process_frame_item()` 单帧分支（约第556行）：

```python
else:
    # 单帧推理：直接传递帧信息字典
    vlm_result = self.vlm_client.analyze(item, self.prompt)
```

修改 `_process_batch()` 方法（约第585行）：

```python
def _process_batch(self, frames: List[Dict[str, Any]]) -> Optional[VLMResult]:
    """批量处理帧列表。"""
    start_time = time.time()
    significant_count = sum(1 for frame in frames if frame.get('significant'))

    # 直接传递帧信息字典列表
    vlm_result = self.vlm_client.analyze_batch(frames, self.prompt)

    # ... 后续逻辑不变 ...
```

### Step 9: 修改 ReplayPipeline 消费者

**文件**: `src/pymediaparser/replay_pipeline.py`

与 LivePipeline 做相同的修改：
- `_process_frame_item()` 单帧分支直接传递帧信息字典
- `_process_batch()` 直接传递帧列表

---

## Critical Files

| 文件 | 修改内容 |
|------|---------|
| `src/pymediaparser/vlm_base.py` | 修改 VLMClient 接口签名，扩展 VLMConfig |
| `src/pymediaparser/vlm/configs.py` | 扩展 APIVLMConfig |
| `src/pymediaparser/vlm/_local_base.py` | 新增辅助方法，修改 analyze/analyze_batch |
| `src/pymediaparser/vlm/openai_api.py` | 新增辅助方法，修改 analyze/analyze_batch |
| `src/pymediaparser/live_pipeline.py` | 消费者传递帧信息字典 |
| `src/pymediaparser/replay_pipeline.py` | 消费者传递帧信息字典 |

---

## Message Format Example

**单帧处理**：
```
content = [
    {"type": "image", "image": img0},
    {"type": "text", "text": "[Frame #0, t=0.00s]"},
    {"type": "text", "text": prompt}
]
```

**多帧批处理**：
```
content = [
    {"type": "text", "text": "以下是连续的视频帧："},
    {"type": "image", "image": img0},
    {"type": "text", "text": "[Frame #0, t=0.00s]"},
    {"type": "image", "image": img1},
    {"type": "text", "text": "[Frame #1, t=1.50s]"},
    {"type": "text", "text": prompt}
]
```

---

## Data Flow

```
Pipeline 队列                     消费者                        VLM
     │                             │                           │
     ▼                             ▼                           │
帧信息字典                        _process_batch()             │
{                                │                            │
  'image': PIL.Image,            │                            │
  'timestamp': float,   ───────► │ analyze_batch(frames,       │
  'frame_index': int,            │               prompt)       │
  'significant': bool,           │                            │
  'source': list,                ▼                            │
}                          frames_info 直接传入                │
                                        │                     │
                                        ▼                     │
                                  ┌─────────────┐             │
                                  │ 构建消息     │             │
                                  │ 穿插时序标签 │             │
                                  └──────┬──────┘             │
                                         │                    │
                                         └────────────────────┘
```

---

## Interface Changes Summary

| 原接口 | 新接口 |
|--------|--------|
| `analyze(image, prompt)` | `analyze(frame, prompt)` |
| `analyze_batch(images, prompt)` | `analyze_batch(frames, prompt)` |

**参数命名**（与 pipeline 保持一致）：
- `frame`: 单个帧信息字典
- `frames`: 帧信息字典列表

**关键变化**：
- 不再单独传入 `image` 参数，从 `frame['image']` 获取
- 单帧和多帧接口统一使用帧信息字典格式
- 多帧时自动添加前缀，单帧时不添加

---

## Verification

1. **单元测试**：为消息构建逻辑添加测试
2. **集成测试**：
   ```python
   # 测试批处理时序信息
   pipeline = LivePipeline(
       stream_config=StreamConfig(url="test.mp4"),
       vlm_client=vlm_client,
       enable_batch_processing=True,
   )
   # 验证 VLM 推理时消息格式正确
   ```

3. **向后兼容验证**：不传 frame_info 时确认原有功能正常

4. **日志检查**：确认消息构建正确
