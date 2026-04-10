"""VLM 配置类：本地模型配置与 API 服务配置。

LocalVLMConfig 继承 VLMConfig，用于本地 transformers 模型。
APIVLMConfig 独立定义，用于 OpenAI 兼容 API 服务。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..vlm_base import VLMConfig


@dataclass
class LocalVLMConfig(VLMConfig):
    """本地 transformers 模型配置。

    继承 VLMConfig 的全部字段，当前无额外字段。
    用于类型区分和未来扩展。
    """


@dataclass
class APIVLMConfig:
    """OpenAI 兼容 API 服务配置。

    适用于 vLLM、Ollama、TGI、OpenAI、通义千问等提供
    /v1/chat/completions 接口的服务。

    Attributes:
        base_url: API 基础地址，如 "http://localhost:8000/v1"。
        api_key: API 密钥（可选，部分本地服务不需要）。
        model_name: 模型名称，如 "Qwen2-VL-2B-Instruct"。
        timeout: 单次请求超时秒数。
        max_retries: 请求失败最大重试次数。
        image_max_size: 图像编码前的最大边长（像素），超过时等比缩放。
        image_quality: JPEG 编码质量（1-100）。
        max_new_tokens: 单次生成最大 token 数。
        default_prompt: 没有提供 prompt 时使用的默认问题。
        timing_prefix: 多帧批处理时的时序前缀文本。
        timing_format: 时序标签格式，支持 {index} 和 {timestamp} 占位符。
    """

    base_url: str = "http://localhost:8000/v1"
    api_key: str | None = None
    model_name: str = "default"
    timeout: float = 60.0
    max_retries: int = 3
    image_max_size: int = 1024
    image_quality: int = 85
    max_new_tokens: int = 256
    default_prompt: str = "请描述当前画面中的人物活动。"
    # 时序消息格式配置
    timing_prefix: str = "以下是连续的视频帧："
    timing_format: str = "[Frame #{index}, t={timestamp:.2f}s]"
