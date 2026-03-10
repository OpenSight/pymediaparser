# 图像预处理模块设计方案

## Context

当前项目在处理器线程调用智能采样器（ml_smart_sampler）后，直接将图像送到消费者队列，然后单张或批量送入大模型（VLM）。为了降低大模型开销，需要在智能采样器之后、消费者队列之前增加预处理步骤。

预处理模块需要与智能采样器**完全解耦**，每个策略有独立的配置类，便于扩展：

**当前支持的策略**：
1. **缩放策略（resize）**：对所有帧进行缩放处理
2. **ROI 裁剪策略（roi_crop）**：对非周期触发帧提取感兴趣区域，周期触发帧跳过

**策略选择方式**：通过参数配置选择单一策略，Pipeline 根据策略名称创建对应的处理器对象。

**支持范围**：LivePipeline 和 ReplayPipeline 均需支持。

---

## Architecture

### 模块结构

```
src/pymediaparser/
├── image_processor/
│   ├── __init__.py                    # 模块导出
│   ├── foreground_extractor.py        # 现有：前景提取器
│   ├── base.py                        # 新增：基类定义
│   ├── resize_processor.py            # 新增：缩放处理器 + 配置类
│   └── roi_crop_processor.py          # 新增：ROI 裁剪处理器 + 配置类
├── live_pipeline.py                   # 修改：集成预处理步骤
└── replay_pipeline.py                 # 修改：集成预处理步骤
```

### 设计思路

```
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline 初始化                           │
├─────────────────────────────────────────────────────────────┤
│  strategy = 'resize' 或 'roi_crop'                          │
│  config = 对应的配置类实例                                    │
│                                                             │
│  processor = create_processor(strategy, config)             │
└───────────────────────┬─────────────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│  ResizeProcessor    │     │  ROICropProcessor   │
│  + ResizeConfig     │     │  + ROICropConfig    │
└─────────────────────┘     └─────────────────────┘
```

### 数据流

```
智能采样器输出 → 预处理器 → 消费者队列
                    │
                    ├─ strategy='resize'
                    │     → 对所有帧执行缩放
                    │
                    └─ strategy='roi_crop'
                          → 检查是否周期触发帧
                                ├─ 周期触发帧：跳过
                                └─ 非周期触发帧：执行 ROI 裁剪
```

---

## Core Components

### 1. 基类定义

文件：`src/pymediaparser/image_processor/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict
from PIL import Image


@dataclass
class BaseProcessorConfig:
    """处理器配置基类"""
    enabled: bool = True
    """是否启用"""
    fallback_on_error: bool = True
    """处理失败时是否降级为原图"""


class BaseProcessor(ABC):
    """图像处理器基类"""

    def __init__(self, config: BaseProcessorConfig):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """处理器名称"""
        pass

    @abstractmethod
    def should_apply(self, frame_data: Dict[str, Any]) -> bool:
        """判断是否应该对该帧应用此处理器"""
        pass

    @abstractmethod
    def process(self, image: Image.Image) -> Image.Image:
        """执行图像处理"""
        pass

    def process_frame(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理帧数据（包含判断和处理逻辑）"""
        if not self.config.enabled:
            return frame_data

        if not self.should_apply(frame_data):
            return frame_data

        try:
            image = frame_data['image']
            processed_image = self.process(image)
            result = frame_data.copy()
            result['image'] = processed_image
            result['preprocessed'] = True
            result['preprocess_strategy'] = self.name
            return result
        except Exception as e:
            logger.warning("预处理失败: %s", e)
            if self.config.fallback_on_error:
                return frame_data
            raise
```

### 2. 缩放处理器

文件：`src/pymediaparser/image_processor/resize_processor.py`

```python
from dataclasses import dataclass
from typing import Any, Dict
from PIL import Image
import logging

from .base import BaseProcessor, BaseProcessorConfig

logger = logging.getLogger(__name__)


@dataclass
class ResizeConfig(BaseProcessorConfig):
    """缩放处理器配置"""
    max_size: int = 1024
    """图像最大边长（像素），超过时等比缩放"""


class ResizeProcessor(BaseProcessor):
    """缩放处理器 - 对所有帧执行等比缩放"""

    def __init__(self, config: ResizeConfig):
        super().__init__(config)
        self._resize_config = config

    @property
    def name(self) -> str:
        return 'resize'

    def should_apply(self, frame_data: Dict[str, Any]) -> bool:
        """对所有帧都应用"""
        return True

    def process(self, image: Image.Image) -> Image.Image:
        """等比缩放图像"""
        w, h = image.size
        max_size = self._resize_config.max_size

        if max(w, h) <= max_size:
            return image

        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)

        logger.debug("缩放图像: %dx%d -> %dx%d", w, h, new_w, new_h)
        return image.resize((new_w, new_h), Image.LANCZOS)
```

### 3. ROI 裁剪处理器

文件：`src/pymediaparser/image_processor/roi_crop_processor.py`

```python
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from PIL import Image
import cv2
import numpy as np
import logging

from .base import BaseProcessor, BaseProcessorConfig

logger = logging.getLogger(__name__)


@dataclass
class ROICropConfig(BaseProcessorConfig):
    """ROI 裁剪处理器配置"""
    method: str = 'motion'
    """ROI 检测方法: 'motion'(运动检测) | 'saliency'(显著性检测)"""

    padding_ratio: float = 0.2
    """ROI 区域边界扩展比例"""

    min_roi_ratio: float = 0.2
    """最小 ROI 占比阈值，低于此值时扩大到该比例"""


class ROICropProcessor(BaseProcessor):
    """ROI 裁剪处理器 - 仅对非周期触发帧执行"""

    def __init__(self, config: ROICropConfig):
        super().__init__(config)
        self._roi_config = config
        self._prev_gray: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return 'roi_crop'

    def should_apply(self, frame_data: Dict[str, Any]) -> bool:
        """仅对非周期触发帧应用

        判断逻辑：source 列表只有 'periodic' 时跳过
        """
        source = frame_data.get('source', [])
        if source == ['periodic']:
            return False
        return True

    def process(self, image: Image.Image) -> Image.Image:
        """执行 ROI 检测并裁剪"""
        # 转换为 numpy (RGB)
        frame_np = np.array(image)

        # 检测 ROI
        bbox = self._detect_roi(frame_np)

        # 检查最小占比
        bbox = self._ensure_min_ratio(bbox, frame_np.shape)

        # 裁剪
        x, y, w, h = bbox
        cropped = frame_np[y:y+h, x:x+w]

        logger.debug("ROI裁剪: 原图%dx%d -> 裁剪后%dx%d",
                     frame_np.shape[1], frame_np.shape[0], w, h)
        return Image.fromarray(cropped)

    def _detect_roi(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """检测 ROI 区域"""
        if self._roi_config.method == 'motion':
            return self._detect_by_motion(frame)
        else:
            return self._detect_by_saliency(frame)

    def _detect_by_motion(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """基于帧差法的 ROI 检测"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return (0, 0, frame.shape[1], frame.shape[0])

        # 帧差
        diff = cv2.absdiff(gray, self._prev_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 轮廓检测
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self._prev_gray = gray
            return (0, 0, frame.shape[1], frame.shape[0])

        # 合并所有轮廓的边界框
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # 添加 padding
        x, y, w, h = self._add_padding(x, y, w, h, frame.shape)

        self._prev_gray = gray
        return (x, y, w, h)

    def _detect_by_saliency(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """基于显著性检测的 ROI 检测"""
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(frame)

        if not success:
            return (0, 0, frame.shape[1], frame.shape[0])

        # 二值化
        _, thresh = cv2.threshold(
            (saliency_map * 255).astype(np.uint8),
            127, 255, cv2.THRESH_BINARY
        )

        # 轮廓检测 + 取最大轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return (0, 0, frame.shape[1], frame.shape[0])

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        return self._add_padding(x, y, w, h, frame.shape)

    def _add_padding(self, x, y, w, h, frame_shape):
        """添加边界扩展"""
        pad_w = int(w * self._roi_config.padding_ratio)
        pad_h = int(h * self._roi_config.padding_ratio)

        x = max(0, x - pad_w)
        y = max(0, y - pad_h)
        w = min(frame_shape[1] - x, w + 2 * pad_w)
        h = min(frame_shape[0] - y, h + 2 * pad_h)

        return (x, y, w, h)

    def _ensure_min_ratio(self, bbox, frame_shape):
        """确保 ROI 占比不低于 min_roi_ratio"""
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]
        frame_area = frame_h * frame_w

        current_area = w * h
        min_area = frame_area * self._roi_config.min_roi_ratio

        if current_area < min_area:
            # 计算需要扩展的比例
            scale = (min_area / current_area) ** 0.5
            new_w = int(w * scale)
            new_h = int(h * scale)

            # 以中心为基准扩展
            cx, cy = x + w // 2, y + h // 2
            new_x = max(0, cx - new_w // 2)
            new_y = max(0, cy - new_h // 2)
            new_w = min(frame_w - new_x, new_w)
            new_h = min(frame_h - new_y, new_h)

            return (new_x, new_y, new_w, new_h)
        return bbox
```

---

## Integration

### Pipeline 工作模式

Pipeline 支持以下四种工作模式：

| 模式 | 智能采样器 | 预处理器 | 说明 |
|------|-----------|---------|------|
| 传统模式 | ❌ | ❌ | 生产者 → 消费者 |
| 仅预处理 | ❌ | ✅ | 生产者 → 处理器(预处理) → 消费者 |
| 仅智能采样 | ✅ | ❌ | 生产者 → 处理器(采样) → 消费者 |
| 完整模式 | ✅ | ✅ | 生产者 → 处理器(采样+预处理) → 消费者 |

### Pipeline 工厂函数

```python
# image_processor/__init__.py
from typing import Union
from .base import BaseProcessor, BaseProcessorConfig
from .resize_processor import ResizeProcessor, ResizeConfig
from .roi_crop_processor import ROICropProcessor, ROICropConfig


def create_processor(
    strategy: str,
    config: Union[ResizeConfig, ROICropConfig, None] = None,
) -> BaseProcessor:
    """创建图像处理器

    Args:
        strategy: 策略名称 'resize' 或 'roi_crop'
        config: 对应策略的配置对象

    Returns:
        处理器实例
    """
    if strategy == 'resize':
        config = config or ResizeConfig()
        return ResizeProcessor(config)
    elif strategy == 'roi_crop':
        config = config or ROICropConfig()
        return ROICropProcessor(config)
    else:
        raise ValueError(f"未知的预处理策略: {strategy}")


__all__ = [
    'BaseProcessor',
    'BaseProcessorConfig',
    'ResizeProcessor',
    'ResizeConfig',
    'ROICropProcessor',
    'ROICropConfig',
    'create_processor',
]
```

### LivePipeline 修改

```python
from typing import Optional, Union
from pymediaparser.image_processor import create_processor, ResizeConfig, ROICropConfig


class LivePipeline:
    def __init__(
        self,
        # ... 现有参数 ...
        enable_smart_sampling: bool = False,
        preprocessing: Optional[str] = None,  # None/resize/roi_crop
        preprocess_config: Union[ResizeConfig, ROICropConfig, None] = None,
        # ... 其他参数 ...
    ):
        # ... 现有初始化 ...

        # 智能采样器
        self.smart_sampler = None
        if enable_smart_sampling:
            # ... 创建智能采样器 ...

        # 预处理器（preprocessing 参数同时表示使能和策略）
        self.preprocessor = None
        if preprocessing is not None:
            self.preprocessor = create_processor(preprocessing, preprocess_config)

        # 判断是否需要处理器线程
        self._use_processor_thread = bool(self.smart_sampler or self.preprocessor)


def _processor_loop(self) -> None:
    """处理器线程：智能采样（可选）+ 预处理（可选）"""
    try:
        while not self._stop_event.is_set():
            try:
                item = self._processor_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                self._enqueue_frame(None)
                break

            frame_np, ts, idx = item

            # 情况1: 智能采样器 + 预处理器
            if self.smart_sampler:
                for sampled_data in self.smart_sampler.sample(iter([(frame_np, ts)])):
                    if self._stop_event.is_set():
                        break
                    # 预处理采样后的帧
                    if self.preprocessor:
                        sampled_data = self.preprocessor.process_frame(sampled_data)
                    self._enqueue_frame(sampled_data)

            # 情况2: 仅预处理器（无智能采样器）
            elif self.preprocessor:
                # 直接对原始帧进行预处理
                pil_image = self._numpy_to_pil(frame_np)
                frame_data = {
                    'image': pil_image,
                    'timestamp': ts,
                    'frame_index': idx,
                    'significant': True,  # 无智能采样器时，所有帧都视为有效
                    'source': ['direct'],  # 标记为直接投递
                }
                processed_data = self.preprocessor.process_frame(frame_data)
                self._enqueue_frame(processed_data)

            # 情况3: 无处理器（不应该进入此线程）
            else:
                # 直接转发
                pil_image = self._numpy_to_pil(frame_np)
                frame_data = {
                    'image': pil_image,
                    'timestamp': ts,
                    'frame_index': idx,
                    'significant': True,
                    'source': ['direct'],
                }
                self._enqueue_frame(frame_data)

    except Exception as exc:
        if not self._stop_event.is_set():
            self._handle_thread_error("处理器线程", exc)
    finally:
        logger.info("处理器线程已退出")
```

### ReplayPipeline 修改

与 LivePipeline 类似，在对应的处理循环中实现相同逻辑。

---

## Files to Modify

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/pymediaparser/image_processor/base.py` | 新建 | 基类定义 |
| `src/pymediaparser/image_processor/resize_processor.py` | 新建 | 缩放处理器 + ResizeConfig |
| `src/pymediaparser/image_processor/roi_crop_processor.py` | 新建 | ROI 裁剪处理器 + ROICropConfig |
| `src/pymediaparser/image_processor/__init__.py` | 修改 | 导出 + create_processor 工厂函数 |
| `src/pymediaparser/live_pipeline.py` | 修改 | 集成预处理器 |
| `src/pymediaparser/replay_pipeline.py` | 修改 | 集成预处理器 |
| `scripts/run_parser.py` | 修改 | 添加预处理相关命令行参数 |

---

## run_parser.py 修改详情

### 新增命令行参数

```python
# ── 图像预处理参数 ──
preprocess_group = parser.add_argument_group('图像预处理选项')

preprocess_group.add_argument(
    '--preprocessing',
    choices=['resize', 'roi_crop'],
    default=None,
    help='启用图像预处理并指定策略: resize=缩放到指定尺寸, roi_crop=对非周期触发帧进行ROI裁剪 (默认: 不启用)',
)

# ── 缩放策略参数 ──
preprocess_group.add_argument(
    '--max-size',
    type=int,
    default=1024,
    help='[resize策略] 图像最大边长（像素），超过时等比缩放 (默认: 1024)',
)

# ── ROI 裁剪策略参数 ──
preprocess_group.add_argument(
    '--roi-method',
    choices=['motion', 'saliency'],
    default='motion',
    help='[roi_crop策略] ROI检测方法: motion=帧差法, saliency=显著性检测 (默认: motion)',
)

preprocess_group.add_argument(
    '--roi-padding',
    type=float,
    default=0.2,
    help='[roi_crop策略] 边界扩展比例 (默认: 0.2)',
)

preprocess_group.add_argument(
    '--min-roi-ratio',
    type=float,
    default=0.2,
    help='[roi_crop策略] 最小占比阈值，ROI区域过小时扩展到该比例 (默认: 0.2)',
)
```

### 构建配置并传递给 Pipeline

```python
# 根据策略构建对应的配置对象
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

# 创建 Pipeline 时传递
pipeline = LivePipeline(
    stream_config=stream_cfg,
    vlm_client=vlm_client,
    prompt=prompt,
    enable_smart_sampling=args.smart_sampling,
    smart_config=smart_config,
    # 预处理参数（一个参数同时表示使能和策略）
    preprocessing=args.preprocessing,
    preprocess_config=preprocess_config,
)
```

### 使用示例

```bash
# 模式1: 传统模式（无智能采样，无预处理）
python scripts/run_parser.py \
    --url rtmp://test-stream

# 模式2: 仅预处理（无智能采样）
python scripts/run_parser.py \
    --url rtmp://test-stream \
    --preprocessing resize \
    --max-size 768

# 模式3: 仅智能采样（无预处理）
python scripts/run_parser.py \
    --url rtmp://test-stream \
    --smart-sampling

# 模式4: 完整模式（智能采样 + 预处理）
python scripts/run_parser.py \
    --url rtmp://test-stream \
    --smart-sampling \
    --preprocessing roi_crop \
    --roi-method motion \
    --min-roi-ratio 0.2
```

---

## Verification

### 单元测试

1. **ResizeProcessor 测试**
   - 测试图像缩放
   - 测试边界条件（小图不缩放）

2. **ROICropProcessor 测试**
   - 测试 should_apply 逻辑（周期帧跳过）
   - 测试 ROI 检测方法
   - 测试最小占比扩展

3. **create_processor 测试**
   - 测试根据策略创建正确的处理器

### 集成测试

```bash
# 运行现有测试
pytest tests/

# 手动测试缩放策略
python scripts/run_parser.py \
    --url rtmp://test-stream \
    --preprocessing resize \
    --max-size 512

# 手动测试 ROI 策略
python scripts/run_parser.py \
    --url rtmp://test-stream \
    --smart-sampling \
    --preprocessing roi_crop \
    --roi-method motion \
    --min-roi-ratio 0.2
```

### 验证点

1. 每个策略有独立的配置类
2. Pipeline 根据策略名称创建对应的处理器
3. 预处理模块与智能采样器完全解耦
4. 缩放策略对所有帧执行
5. ROI 策略仅对非周期触发帧执行
6. ROI 区域过小时自动扩展到 min_roi_ratio
7. LivePipeline 和 ReplayPipeline 均支持
8. **支持仅启用预处理器（无智能采样器）的模式**
9. **支持四种工作模式的正确切换**

---

## Implementation Steps

1. **创建基类** - `image_processor/base.py`
2. **创建缩放处理器** - `image_processor/resize_processor.py`
3. **创建 ROI 裁剪处理器** - `image_processor/roi_crop_processor.py`
4. **更新模块导出** - `image_processor/__init__.py`（含 create_processor 工厂函数）
5. **集成到 LivePipeline** - 修改初始化和 `_processor_loop`
6. **集成到 ReplayPipeline** - 同步修改
7. **添加命令行参数** - `scripts/run_parser.py`
8. **编写单元测试**
