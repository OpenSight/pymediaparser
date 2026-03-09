"""Qwen3.5 本地推理客户端。

基于 ``transformers`` 的本地推理实现，
继承 :class:`_LocalTransformersBase` 抽象基类。

Qwen3.5 使用混合注意力机制（linear_attention + full_attention），
通过 AutoModelForImageTextToText 自动加载正确的模型架构。
处理器复用 Qwen3VLProcessor。

典型用法::

    from pymediaparser.vlm_base import VLMConfig
    from pymediaparser.vlm.qwen35 import Qwen35Client

    config = VLMConfig(model_path="/path/to/Qwen3.5-0.8B", device="cuda:0")
    with Qwen35Client(config) as client:
        result = client.analyze(pil_image, "请描述画面内容。")
        print(result.text)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..vlm_base import VLMConfig
from ._local_base import _LocalTransformersBase

logger = logging.getLogger(__name__)


class Qwen35Client(_LocalTransformersBase):
    """Qwen3.5 本地推理客户端。

    Qwen3.5 使用 AutoModelForImageTextToText 加载，
    架构为 Qwen3_5ForConditionalGeneration（混合注意力机制）。
    处理器复用 Qwen3VLProcessor，输入准备逻辑与 Qwen3-VL 相同。

    相比 Qwen3-VL-2B，Qwen3.5-0.8B 显存占用更低（约 1.4GB），
    适合资源受限场景。

    Args:
        config: VLM 配置（模型路径、设备、精度等）。
    """

    def _load_model_and_processor(
        self, dtype: torch.dtype, load_kwargs: Dict[str, Any],
    ) -> Tuple[Any, Any]:
        """加载 Qwen3.5 模型和处理器。"""
        model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_path, **load_kwargs,
        )

        processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
        )

        return model, processor

    def _prepare_inputs(
        self, messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Qwen3.5 输入准备：直接使用 processor.apply_chat_template。

        Qwen3.5 的 processor 已集成视觉信息处理，
        apply_chat_template 可直接接受包含 PIL Image 的消息。
        """
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # Qwen3.5 直接从消息中提取图像，不需要 process_vision_info
        image_inputs = []
        for msg in messages:
            for item in msg.get("content", []):
                if isinstance(item, dict) and item.get("type") == "image":
                    image_inputs.append(item["image"])

        inputs = self._processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            padding=True,
            return_tensors="pt",
        )

        return inputs
