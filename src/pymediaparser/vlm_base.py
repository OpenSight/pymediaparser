"""VLM 基础定义：配置数据类、结果数据类、VLM 客户端抽象接口。

所有 VLM 实现（Qwen2-VL、llama.cpp 等）都应继承 VLMClient 基类。
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Sequence

from PIL import Image


# ---------------------------------------------------------------------------
# 项目级常量
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 默认模型路径：优先使用环境变量，否则按顺序尝试已存在的模型
def _get_default_model_path() -> str:
    """获取默认模型路径，按优先级尝试已存在的模型目录。"""
    # 优先使用环境变量
    env_path = os.environ.get("VLM_MODEL_PATH") or os.environ.get("QWEN_VL_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    
    # 备选模型路径（按优先级排序）
    candidates = [
        os.path.join(_PROJECT_ROOT, "models", "Qwen", "Qwen3.5-0.8B"),
        os.path.join(_PROJECT_ROOT, "models", "Qwen", "Qwen3-VL-2B-Instruct"),
        os.path.join(_PROJECT_ROOT, "models", "Qwen", "Qwen2-VL-2B-Instruct"),
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
    
    # 都不存在时返回第一个（保持向后兼容）
    return candidates[0]

DEFAULT_MODEL_PATH = _get_default_model_path()


# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------

@dataclass
class StreamConfig:
    """实时视频流接入配置。

    Attributes:
        url: 流地址，支持 rtmp:// / http://*.flv / http://*.ts / .m3u8
        format: 显式指定容器格式（"flv"/"mpegts"/None 表示自动检测）。
        target_fps: 目标抽帧频率（帧/秒），例如 1.0 表示每秒一帧。
        reconnect_interval: 断线后等待重连秒数。
        timeout: 读取超时秒数。
        max_queue_size: 帧缓冲队列最大长度（超出时丢弃旧帧）。
        decode_mode: 解码模式。"all"=全帧解码，"keyframe_only"=仅解码关键帧(I帧)。
    """

    url: str
    format: str | None = None
    target_fps: float = 1.0
    reconnect_interval: float = 3.0
    timeout: float = 10.0
    max_queue_size: int = 3
    decode_mode: str = "all"  # "all" | "keyframe_only"


@dataclass
class VLMConfig:
    """VLM 模型加载/推理配置。

    Attributes:
        model_path: 本地模型目录路径。
        device: 推理设备，如 "cuda:0" / "cpu"。
        dtype: 推理精度："float16" / "bfloat16" / "float32"。
        max_new_tokens: 单次生成最大 token 数。
        default_prompt: 没有提供 prompt 时使用的默认问题。
        use_flash_attn: 是否尝试使用 Flash Attention 2。
        max_pixels: processor 最大像素数（控制显存）。
        min_pixels: processor 最小像素数。
        timing_prefix: 多帧批处理时的时序前缀文本。
        timing_format: 时序标签格式，支持 {index} 和 {timestamp} 占位符。
    """

    model_path: str = DEFAULT_MODEL_PATH
    device: str = "cuda:0"
    dtype: str = "float16"
    max_new_tokens: int = 256
    default_prompt: str = "请描述当前画面中的人物活动。"
    use_flash_attn: bool = True
    max_pixels: int = 512 * 28 * 28
    min_pixels: int = 256 * 28 * 28
    # 时序消息格式配置
    timing_prefix: str = "以下是连续的视频帧："
    timing_format: str = "[Frame #{index}, t={timestamp:.2f}s]"


# ---------------------------------------------------------------------------
# 结果数据类
# ---------------------------------------------------------------------------

@dataclass
class VLMResult:
    """VLM 推理结果。

    Attributes:
        text: 模型生成的主文本回复。
        tokens_used: 生成消耗的 token 数（已废弃，使用 output_tokens）。
        inference_time: 推理耗时（秒）。
        input_tokens: 输入 token 数。
        output_tokens: 输出 token 数。
        raw_output: 原始模型输出（可选，用于调试）。
        meta: 额外元信息。
    """

    text: str
    tokens_used: int = 0
    inference_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    raw_output: Any = None
    meta: dict = field(default_factory=dict)


@dataclass
class FrameResult:
    """单帧分析结果。

    Attributes:
        frame_index: 全局帧序号（从 0 开始）。
        timestamp: 帧在流中的时间戳（秒）。
        vlm_result: VLM 推理结果。
        capture_time: 帧被采集时的系统时间戳（Unix epoch 秒）。
    """

    frame_index: int
    timestamp: float
    vlm_result: VLMResult
    capture_time: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# VLM 客户端抽象基类
# ---------------------------------------------------------------------------

class VLMClient(ABC):
    """VLM 推理客户端抽象基类。

    所有 VLM 实现（Qwen2-VL、llama.cpp、vLLM 等）都需要继承此类并实现
    ``load`` / ``analyze`` / ``unload`` 三个方法。
    """

    @abstractmethod
    def load(self) -> None:
        """加载模型到显存/内存。应在调用 ``analyze`` 前执行。"""

    @abstractmethod
    def analyze(
        self,
        frame: Dict[str, Any],
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
            prompt: 文本提示词；为 ``None`` 时使用配置中的默认 prompt。

        Returns:
            VLMResult 包含文本回复及推理统计信息。
        """

    @abstractmethod
    def unload(self) -> None:
        """释放模型资源、清空显存。"""

    # 批量处理接口（可选） --------------------------------------------------

    def analyze_batch(
        self,
        frames: Sequence[Dict[str, Any]],
        prompt: str | None = None,
    ) -> VLMResult:
        """对多帧图像进行批量理解分析（可选实现）。

        默认实现为逐帧调用 analyze() 后合并结果。
        子类可覆盖此方法以提供更高效的批量处理实现。

        Args:
            frames: 帧信息字典列表，每个字典包含 image、timestamp、frame_index 等。
            prompt: 文本提示词；为 ``None`` 时使用配置中的默认 prompt。

        Returns:
            VLMResult 包含合并后的文本回复及推理统计信息。

        Raises:
            NotImplementedError: 如果子类未实现此方法且需要显式报错。
        """
        # 默认实现：逐帧推理后合并结果
        if not frames:
            return VLMResult(text="[空帧列表]", inference_time=0.0)

        results_text: List[str] = []
        total_time = 0.0

        for i, frame in enumerate(frames):
            frame_prompt = f"{prompt or self._get_default_prompt()}\n\n这是第{i+1}/{len(frames)}帧的内容分析："
            result = self.analyze(frame, frame_prompt)
            results_text.append(result.text)
            total_time += result.inference_time

        combined_text = "\n\n".join([f"帧{i+1}: {text}" for i, text in enumerate(results_text)])

        return VLMResult(
            text=combined_text,
            inference_time=total_time,
            input_tokens=0,
            output_tokens=0,
        )

    def _get_default_prompt(self) -> str:
        """获取默认提示词（供子类覆盖）。"""
        return "请描述当前画面中的人物活动。"

    def supports_batch(self) -> bool:
        """检查客户端是否支持原生批量处理。

        Returns:
            True 表示支持原生批量处理，False 表示使用默认逐帧实现。
        """
        # 默认返回 False，子类覆盖此方法以表明支持批量处理
        return False

    # 上下文管理器支持 --------------------------------------------------

    def __enter__(self) -> "VLMClient":
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()
