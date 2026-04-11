# SimpleSmartSampler 优化方案

## Context

当前 MLSmartSampler 采用三层漏斗架构（HardFilter → FastTriggers → FrameValidator），过滤过于激进，一次活动事件仅抽取 1-2 帧送入 VLM 分析，丢失了大量活动细节，导致 VLM 识别准确性不足。

本次优化目标：改造 SimpleSmartSampler，从"少采样"模式转变为**"高召回 + 智能排除"**模式，确保有意义的动物活动（人、宠物、飞鸟、牲畜等）和场景变化期间的所有帧都被采样，同时精准过滤静态/准静态画面。

## 需求

1. **排除**：静态画面、异常画面、光线微变、风吹草动等准静态画面
2. **全采样**：有**有意义活动**的画面（人/宠物/飞鸟/牲畜等），每一帧都采样
3. **全采样**：场景变化（镜头移动/云台旋转）过程中每一帧都采样
4. **约束**：CPU 环境，上游送帧率 1fps，单帧处理预算几十ms量级

## 问题根因

| 瓶颈 | 影响 |
|------|------|
| `min_frame_interval=1.0s` 直接拦截连续活动帧 | 最严重 |
| `motion_threshold=0.1` 需要 10% 像素运动才触发 | 严重 |
| SSIM 全局相似度对局部变化不敏感 | 中等 |
| 无法区分动物活动 vs 风吹植物 | 中等 |
| 无场景变化连续性追踪 | 中等 |

## 设计方案

### 架构：活动状态机 + 多策略活动检测

```
输入帧 → [静态/准静态过滤器] → 排除静态帧
                            ↓
                    [活动状态机]
              IDLE ──活动/场景触发──→ ACTIVE
               ↑                        │
               └── 静默N帧/超时 ────────┘
              
              ACTIVE: 所有帧 emit
              IDLE: 需触发条件才 emit
```

### 核心改动

#### 1. 新增 `MeaningfulActivityDetector`（有意义的活动检测器）

检测画面中所有有意义的动物活动（人、宠物、飞鸟、牲畜等），采用**多策略融合**方案：

**策略 A: MOG2 前景掩码 + 形态学分析（基础层，~2ms）**
- 获取 MOG2 前景掩码
- 分析连通域特征：
  - 数量：动物活动通常 1-5 个连通域，植物/雨点通常 10+ 个碎区域
  - 面积占比：动物活动通常 2%-40% 画面
  - 高宽比：动物/肢体通常 0.2-2.0（适应飞鸟展翅、宠物奔跑等）
  - 重心位置：动物通常在画面中下部
  - 实心度：动物 > 0.3，树枝摇曳 < 0.3

**策略 B: 稀疏光流分析（增强层，~10-15ms）**
- 使用 Lucas-Kanade 稀疏光流跟踪特征点
- 分析运动模式：
  - 动物活动：特征点运动方向不一致（有自主运动），速度变化大
  - 风吹植物：特征点运动方向一致（被风吹动），速度均匀
  - 光线变化：几乎无特征点运动

**策略 C: 运动持续性追踪（时间层，~1ms）**
- 维护运动历史窗口（5-10帧）
- 动物活动特征：运动持续多帧，运动区域有连续性
- 瞬时干扰特征：仅1-2帧有运动，随后消失

**综合判定规则**:
```
meaningful_activity = (
    (形态学评分 > 阈值 AND 光流显示自主运动) OR
    (形态学评分 > 高阈值 AND 运动持续3帧+) OR
    (光流显示强自主运动 AND 运动持续2帧+)
)
```

**性能估算**: ~15-20ms/帧（在 1fps 场景下完全可接受）

#### 2. 新增 `SceneTransitionTracker`（场景连续变化追踪）

- 计算相邻帧 HSV 3x3 分块直方图相关性
- 维护滑动窗口(5帧)，检测相关性是否持续单调变化
- 触发条件：连续3帧相关性在 [0.6, 0.95] + 帧间差分在 [5, 50] + 趋势单调
- 一旦触发，设置 grace period（3秒或3帧），期间所有帧直接 emit

#### 3. 重构 `SimpleSmartSampler` 核心逻辑

**状态机流程**：
- **首帧**：按时间触发逻辑处理（等待 backup_interval 或检测到变化），作为参考帧，不直接进入 ACTIVE
- **ACTIVE 状态**：所有帧直接 emit，同时运行 MOG2 和静默计数器；连续 N 帧无活动 → 退回 IDLE
- **IDLE 状态**：执行静态过滤 + 活动检测 + 场景变化检测；任一触发 → emit 并进入 ACTIVE
- **保底触发**：超过 backup_interval（30秒）无变化，强制 emit

**关键变更**：
- **移除** `min_frame_interval` 参数和相关的帧间隔拦截逻辑
- ACTIVE 状态下继续运行 MOG2，用于判断何时活动结束
- `source` 标签沿用原有：`motion`、`scene_switch`、`periodic`（不增加新标签）

**触发源映射**：
- 有意义的活动检测触发 → `source: ['motion']`
- 场景连续变化触发 → `source: ['scene_switch']`
- 保底时间触发 → `source: ['periodic']`
- 多种触发同时存在 → `source: ['motion', 'scene_switch']` 等

#### 4. 静态/准静态过滤（多维度 Z-Score 异常检测）

**在 IDLE 状态下的静态过滤器**：

```
is_static_frame 判定（多维度 Z-Score 分析）:

维护滑动窗口（10帧）的三维度特征：
1. 帧间差分均值（检测画面变化）
2. 颜色方差（检测光线变化）
3. 边缘密度（检测结构变化）

计算当前帧特征的 Z-Score：
- 所有维度 |z| < 1.0 → 静态帧，排除
- 仅颜色方差 |z| > 1.0 但其他维度正常 → 光线微变，排除
- 运动像素占比 < 0.02 且 不符合动物活动模式 → 风吹植物，排除
```

这个多维度分析确保：
- **纯静态画面**（无变化）: Z-Score 低 → 排除
- **微光变化**（光线缓慢变化）: 仅颜色维度异常 → 排除
- **风吹植物/雨点**: 运动模式不匹配动物 → 排除

### 参数设计

#### 对外暴露的用户配置参数（仅 4 个核心参数）

```python
class SimpleSmartSampler:
    def __init__(
        self,
        # 核心效果参数（用户可配置）
        sensitivity: float = 0.5,              # 活动检测灵敏度 (0.0-1.0)
        activity_duration: float = 3.0,        # 活动期持续时间（秒）
        quiet_frames_threshold: int = 10,      # 静默帧阈值（连续无活动帧数）
        backup_interval: float = 30.0,         # 保底采样间隔（秒）
        
        # 兼容参数
        enable_smart_sampling: bool = True,
    ) -> None:
```

**参数说明**：

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| `sensitivity` | 0.0-1.0 | 0.5 | 活动检测灵敏度。值越低越敏感（更多帧被采样），值越高越严格（仅明显活动被采样） |
| `activity_duration` | 1.0-10.0 | 3.0 | 活动期持续时间（秒）。检测到活动后，此期间内所有帧都采样 |
| `quiet_frames_threshold` | 3-30 | 10 | 连续多少帧无活动后退出活动期 |
| `backup_interval` | 10.0-120.0 | 30.0 | 保底采样间隔。长时间无活动时的强制采样频率 |

**sensitivity 参数内部映射**：

用户配置的 `sensitivity` 会自动映射到内部各检测器的阈值：

```python
# 内部映射逻辑（用户不可见）
def _apply_sensitivity(self, sensitivity: float):
    """将用户灵敏度映射到内部参数"""
    
    # 形态学分析阈值
    self._morph_threshold = 0.3 + (1.0 - sensitivity) * 0.4  # 0.3-0.7
    
    # 光流分析阈值
    self._flow_threshold = 0.02 + (1.0 - sensitivity) * 0.03  # 0.02-0.05
    
    # 静态过滤 Z-Score 阈值
    self._zscore_threshold = 0.5 + sensitivity * 1.0  # 0.5-1.5
    
    # 场景变化相关性阈值
    self._scene_corr_threshold = 0.7 - sensitivity * 0.2  # 0.5-0.7
```

#### 内部隐藏参数（用户不可见，使用默认值）

以下参数对用户隐藏，使用合理的默认值：

```python
# 形态学分析
min_components: int = 1
max_components: int = 15
min_area_ratio: float = 0.01
max_area_ratio: float = 0.40
min_aspect_ratio: float = 0.2
max_aspect_ratio: float = 2.0
min_solidity: float = 0.3
min_centroid_y_ratio: float = 0.35

# 光流分析
max_corners: int = 100
quality_level: float = 0.01
min_distance: float = 5.0
displacement_threshold: float = 1.0
direction_consistency_threshold: float = 0.5

# 运动持续性
motion_history_window: int = 5
min_consecutive_motion_frames: int = 2

# 场景追踪
scene_window_size: int = 5
scene_min_consecutive: int = 3
scene_diff_low: float = 5.0
scene_diff_high: float = 50.0

# 静态过滤
zscore_window_size: int = 10
```

### 性能预算

| 组件 | 耗时 | 条件 |
|------|------|------|
| MOG2 背景减除 | 10-15ms | 每帧（ACTIVE/IDLE 都运行） |
| 形态学分析 | 2-3ms | 仅 IDLE |
| 稀疏光流分析 | 10-15ms | 仅 IDLE |
| HSV 直方图相关性 | 1-2ms | 每帧 |
| Z-Score 静态过滤 | 2-3ms | 仅 IDLE |
| **ACTIVE 状态总计** | **~12-17ms** | MOG2 + HSV 直方图 |
| **IDLE 状态总计** | **~25-35ms** | 全量检测 |

> 注：上游送帧率 1fps，单帧处理 35ms 完全可接受（仅占 3.5% 时间）

## 变更文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/pymediaparser/smart_sampler/simple_smart_sampler.py` | 重构 | 核心逻辑重写，引入状态机 |
| `src/pymediaparser/smart_sampler/meaningful_activity_detector.py` | 新增 | 有意义的活动检测器（形态学+光流+持续性） |
| `src/pymediaparser/smart_sampler/scene_transition_tracker.py` | 新增 | 场景连续变化追踪器 |
| `src/pymediaparser/smart_sampler/static_filter.py` | 新增 | 多维度 Z-Score 静态过滤器 |
| `src/pymediaparser/smart_sampler/configs.py` | 修改 | 更新 SimpleSamplerConfig 参数 |
| `src/pymediaparser/smart_sampler/__init__.py` | 修改 | 导出新组件 |

## 实施步骤

1. 创建 `MeaningfulActivityDetector` 类（形态学 + 光流 + 持续性）
2. 创建 `SceneTransitionTracker` 类
3. 创建 `StaticFilter` 类（Z-Score 多维度分析）
4. 更新 `SimpleSamplerConfig` 添加新参数，移除 `min_frame_interval`
5. 更新 `__init__.py` 导出
6. 重写 `SimpleSmartSampler._smart_sample_frame()` 实现状态机逻辑
7. 更新 `SimpleSmartSampler.sample()` 和 `reset()` 适配新流程
8. 更新 `get_statistics()` 增加新统计字段
9. 运行测试验证

## 验证方案

1. **功能测试**：
   - 准备含人物活动的视频片段 → 验证活动期间所有帧都被采样
   - 准备含宠物/飞鸟活动的视频 → 验证活动期间所有帧都被采样
   - 准备云台旋转视频 → 验证旋转过程中所有帧都被采样
   - 准备静态/微光变/风吹植物视频 → 验证这些帧被排除

2. **性能测试**：
   - 测量单帧处理时间，确认 <50ms

3. **回归测试**：
   - 确保向后兼容，现有调用代码无需修改
