"""Qwen3-VL 本地推理客户端。

基于 ``transformers`` 的本地推理实现（无需 qwen-vl-utils），
继承 :class:`_LocalTransformersBase` 抽象基类。

典型用法::

    from pymediaparser.vlm_base import VLMConfig
    from pymediaparser.vlm.qwen3 import Qwen3VLClient

    config = VLMConfig(model_path="/path/to/Qwen3-VL-2B", device="cuda:0")
    with Qwen3VLClient(config) as client:
        result = client.analyze(pil_image, "请描述画面内容。")
        print(result.text)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

# 延迟导入：torch 和 transformers 是可选依赖
# 当用户尝试使用此后端但缺少依赖时，给出友好的安装提示
try:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    _DEPENDENCIES_AVAILABLE = True
except ImportError as _e:
    _IMPORT_ERROR = _e
    _DEPENDENCIES_AVAILABLE = False
    # 创建占位符，避免类型检查报错
    torch = None  # type: ignore[assignment,misc]
    AutoModelForImageTextToText = None  # type: ignore[assignment,misc]
    AutoProcessor = None  # type: ignore[assignment,misc]

from ..vlm_base import VLMConfig
from ._local_base import _LocalTransformersBase

logger = logging.getLogger(__name__)


def _check_dependencies() -> None:
    """检查依赖是否可用，不可用时抛出友好的错误提示。"""
    if not _DEPENDENCIES_AVAILABLE:
        raise ImportError(
            f"\n{'='*60}\n"
            f"Qwen3-VL 后端缺少必要的依赖包\n"
            f"{'='*60}\n"
            f"导入错误: {_IMPORT_ERROR}\n\n"
            f"请运行以下命令安装依赖:\n"
            f"    pip install 'pymediaparser[vlm-qwen]'\n"
            f"{'='*60}"
        )


class Qwen3VLClient(_LocalTransformersBase):
    """Qwen3-VL 本地推理客户端。

    Qwen3-VL 使用 AutoModelForImageTextToText 加载，
    processor 的 apply_chat_template 已内置视觉信息处理，
    不需要 qwen_vl_utils。

    Args:
        config: VLM 配置（模型路径、设备、精度等）。
    """

    def __init__(self, config: VLMConfig | None = None) -> None:
        _check_dependencies()  # 检查依赖是否可用
        super().__init__(config)

    def _load_model_and_processor(
        self, dtype: torch.dtype, load_kwargs: Dict[str, Any],
    ) -> Tuple[Any, Any]:
        """加载 Qwen3-VL 模型和处理器。"""
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
        """Qwen3-VL 输入准备：直接使用 processor.apply_chat_template。

        Qwen3-VL 的 processor 已集成视觉信息处理，
        apply_chat_template 可直接接受包含 PIL Image 的消息。
        """
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # Qwen3-VL 直接从消息中提取图像，不需要 process_vision_info
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
