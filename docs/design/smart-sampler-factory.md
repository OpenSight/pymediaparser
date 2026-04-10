# Smart Sampler 可扩展架构优化方案

## Context

当前项目中的 `smart_sampler` 模块采用固定实现方式，Pipeline 中直接实例化 `MLSmartSampler`，无法扩展支持其他智能分析算法。后期需要支持大小模型模式，实现更多个性化分析功能。

**目标**：
1. 将 smart-sampler 模块优化为可扩展架构，Pipeline 按配置创建对应的 sampler 对象
2. 不同 sampler 有独立的配置参数，用户不提供参数时采用默认值
3. 命令行参数参考 `--preprocessing` 方式，不指定 `--smart-sampler` 时不启用该功能

---

## Implementation Plan

### Step 1: 创建配置类文件

**新建文件**: `src/pymediaparser/smart_sampler/configs.py`

```python
from dataclasses import dataclass
from typing import Literal
from .base import BaseSamplerConfig

@dataclass
class SimpleSamplerConfig(BaseSamplerConfig):
    """SimpleSmartSampler 配置 - 基础二层架构
    
    基于运动检测 + SSIM变化分析的简单智能采样器。
    """
    motion_method: Literal['MOG2', 'KNN'] = 'MOG2'
    """运动检测方法：MOG2 或 KNN 背景减除"""
    
    ssim_threshold: float = 0.80
    """SSIM 相似度阈值，低于此值认为有显著变化"""
    
    motion_threshold: float = 0.1
    """运动检测阈值（运动像素比例）"""


@dataclass
class MLSamplerConfig(BaseSamplerConfig):
    """MLSmartSampler 配置 - 三层漏斗架构
    
    Layer 0 (硬过滤): 快速排除无价值帧，90%+ 拒绝率
    Layer 1 (快速触发): 多路 OR 并行检测，高召回率
    Layer 2 (精细验证): 多特征融合打分 + 峰值检测
    
    注：内部技术参数采用最优默认值，用户无需关心。
    """
    motion_method: Literal['MOG2', 'KNN'] = 'MOG2'
    """运动检测方法：MOG2 或 KNN 背景减除"""
    
    motion_threshold: float = 0.1
    """运动检测阈值（运动像素比例）"""
    
    scene_switch_threshold: float = 0.5
    """场景切换阈值，HSV直方图最小相关系数 < 此值视为场景切换
    
    - 值越高：越敏感，更容易触发场景切换，采样更多帧
    - 值越低：越不敏感，可能漏掉场景切换，采样更少帧
    """
```

**配置参数对比表：**

| 参数 | SimpleSamplerConfig | MLSamplerConfig | 说明 |
|------|:-------------------:|:---------------:|------|
| `motion_method` | ✓ | ✓ | 运动检测方法 |
| `motion_threshold` | ✓ | ✓ | 运动检测阈值 |
| `backup_interval` | ✓ | ✓ | 保底/周期采样间隔 |
| `min_frame_interval` | ✓ | ✓ | 最小帧间隔 |
| `ssim_threshold` | ✓ | ✗ | Simple专属：SSIM阈值 |
| `scene_switch_threshold` | ✗ | ✓ | ML专属：场景切换阈值 |

---

### Step 2: 修改基类添加配置基类

**修改文件**: `src/pymediaparser/smart_sampler/base.py`

在文件开头添加配置基类：

```python
from dataclasses import dataclass

@dataclass
class BaseSamplerConfig:
    """采样器配置基类"""
    enable_smart_sampling: bool = True
    backup_interval: float = 30.0
    min_frame_interval: float = 1.0
```

---

### Step 3: 创建工厂模块

**新建文件**: `src/pymediaparser/smart_sampler/factory.py`

参考 `image_processor/__init__.py` 的工厂模式：

```python
from typing import Union, Dict, Any, Optional
from .base import SmartSampler, BaseSamplerConfig
from .configs import SimpleSamplerConfig, MLSamplerConfig

def create_sampler(
    name: str,
    config: Union[BaseSamplerConfig, Dict[str, Any], None] = None,
) -> SmartSampler:
    """创建采样器实例"""
    if name == 'simple':
        from .simple_smart_sampler import SimpleSmartSampler
        if config is None:
            config = SimpleSamplerConfig()
        elif isinstance(config, dict):
            config = SimpleSamplerConfig(**config)
        return SimpleSmartSampler(**vars(config))
    elif name == 'ml':
        from .ml_smart_sampler import MLSmartSampler
        if config is None:
            config = MLSamplerConfig()
        elif isinstance(config, dict):
            config = MLSamplerConfig(**config)
        return MLSmartSampler(**vars(config))
    else:
        raise ValueError(f"未知的采样器类型: {name}")

def list_samplers() -> list[str]:
    """列出所有可用的采样器"""
    return ['simple', 'ml']
```

---

### Step 4: 更新模块导出

**修改文件**: `src/pymediaparser/smart_sampler/__init__.py`

```python
from .base import SmartSampler, BaseSamplerConfig
from .configs import SimpleSamplerConfig, MLSamplerConfig
from .factory import create_sampler, list_samplers
from .simple_smart_sampler import SimpleSmartSampler
from .ml_smart_sampler import MLSmartSampler
from .motion_detector import MotionDetector
from .change_analyzer import ChangeAnalyzer

__all__ = [
    'SmartSampler', 'BaseSamplerConfig',
    'SimpleSamplerConfig', 'MLSamplerConfig',
    'SimpleSmartSampler', 'MLSmartSampler',
    'create_sampler', 'list_samplers',
    'MotionDetector', 'ChangeAnalyzer',
]
```

---

### Step 5: 修改命令行参数

**修改文件**: `scripts/run_parser.py`

#### 5.1 替换智能抽帧参数组（约186-207行）

将 `--smart-sampling` 参数替换为 `--smart-sampler`：

```python
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
```

#### 5.2 修改配置构建逻辑（约373-382行）

```python
# 构建智能采样器配置
smart_config = None
smart_sampler_type = None
if args.smart_sampler is not None:
    from pymediaparser.smart_sampler import SimpleSamplerConfig, MLSamplerConfig
    smart_sampler_type = args.smart_sampler

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
```

#### 5.3 修改 Pipeline 创建（约401-424行）

```python
# LivePipeline
pipeline = LivePipeline(
    stream_config=stream_cfg,
    vlm_client=vlm_client,
    handlers=handlers,
    prompt=args.prompt,
    smart_sampler=smart_sampler_type,  # 新参数
    smart_config=smart_config,
    enable_batch_processing=args.batch_processing,
    preprocessing=args.preprocessing,
    preprocess_config=preprocess_config,
)
```

#### 5.4 更新启动信息显示（约427-463行）

将 `args.smart_sampling` 替换为 `args.smart_sampler` 相关判断。

---

### Step 6: 修改 Pipeline 类

**修改文件**: `src/pymediaparser/live_pipeline.py`

#### 6.1 修改构造函数签名（约88-101行）

```python
def __init__(
    self,
    stream_config: StreamConfig,
    vlm_client: VLMClient,
    handlers: Sequence[ResultHandler] | None = None,
    prompt: str | None = None,
    # 智能采样参数（新设计）
    smart_sampler: Optional[str] = None,
    smart_config: Union[BaseSamplerConfig, Dict[str, Any], None] = None,
    # 批处理参数
    enable_batch_processing: bool = False,
    # 图像预处理参数
    preprocessing: Optional[str] = None,
    preprocess_config: Union[ResizeConfig, ROICropConfig, None] = None,
) -> None:
```

#### 6.2 修改智能采样器初始化（约117-130行）

```python
# 智能功能开关
self.enable_smart_sampling = smart_sampler is not None and _SMART_FEATURES_AVAILABLE

# 初始化智能采样器（使用工厂模式）
self.smart_sampler_instance: Optional[SmartSampler] = None
if self.enable_smart_sampling:
    from .smart_sampler import create_sampler
    self.smart_sampler_instance = create_sampler(smart_sampler, smart_config)
    logger.info("智能采样器已启用: 类型=%s", smart_sampler)
```

**同样修改**: `src/pymediaparser/replay_pipeline.py`

---

## Critical Files

| 文件路径 | 操作 | 说明 |
|---------|------|------|
| `src/pymediaparser/smart_sampler/configs.py` | 新建 | 配置类定义 |
| `src/pymediaparser/smart_sampler/factory.py` | 新建 | 工厂函数 |
| `src/pymediaparser/smart_sampler/base.py` | 修改 | 添加 BaseSamplerConfig |
| `src/pymediaparser/smart_sampler/__init__.py` | 修改 | 更新导出 |
| `scripts/run_parser.py` | 修改 | 命令行参数 |
| `src/pymediaparser/live_pipeline.py` | 修改 | 使用工厂创建 |
| `src/pymediaparser/replay_pipeline.py` | 修改 | 使用工厂创建 |

---

## Usage Examples

```bash
# 不启用智能采样（默认）
python scripts/run_parser.py --url rtmp://host/live/stream

# 使用 simple 采样器（默认参数）
python scripts/run_parser.py --url rtmp://host/live/stream --smart-sampler simple

# 使用 ml 采样器（默认参数）
python scripts/run_parser.py --url rtmp://host/live/stream --smart-sampler ml

# 使用 simple 采样器 + 自定义参数
python scripts/run_parser.py --url rtmp://host/live/stream \
    --smart-sampler simple \
    --motion-method KNN \
    --ssim-threshold 0.75 \
    --backup-interval 20.0

# 使用 ml 采样器 + 自定义参数
python scripts/run_parser.py --url rtmp://host/live/stream \
    --smart-sampler ml \
    --motion-threshold 0.15 \
    --scene-switch-threshold 0.6 \
    --min-frame-interval 2.0
```

---

## Verification

1. **单元测试**: 验证工厂函数能正确创建两种采样器
2. **集成测试**: 运行 `python scripts/run_parser.py --smart-sampler simple --url ...` 验证功能
3. **回归测试**: 确保不指定 `--smart-sampler` 时行为与原来一致（不启用智能采样）
4. **参数验证**: 测试各采样器专属参数是否正确传递

```bash
# 测试命令
python scripts/run_parser.py --url rtmp://... --smart-sampler simple --help
python scripts/run_parser.py --url rtmp://... --smart-sampler ml --motion-threshold 0.2
```
