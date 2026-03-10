"""Qwen2-VL 本地推理客户端。

基于 ``transformers`` + ``qwen-vl-utils`` 的本地推理实现，
继承 :class:`_LocalTransformersBase` 抽象基类。

典型用法::

    from pymediaparser.vlm_base import VLMConfig
    from pymediaparser.vlm.qwen2 import Qwen2VLClient

    config = VLMConfig(device="cuda:0")
    with Qwen2VLClient(config) as client:
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
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    _DEPENDENCIES_AVAILABLE = True
except ImportError as _e:
    _IMPORT_ERROR = _e
    _DEPENDENCIES_AVAILABLE = False
    # 创建占位符，避免类型检查报错
    torch = None  # type: ignore[assignment,misc]
    Qwen2VLForConditionalGeneration = None  # type: ignore[assignment,misc]
    AutoProcessor = None  # type: ignore[assignment,misc]
    process_vision_info = None  # type: ignore[assignment,misc]

from ..vlm_base import VLMConfig
from ._local_base import _LocalTransformersBase

logger = logging.getLogger(__name__)


def _check_dependencies() -> None:
    """检查依赖是否可用，不可用时抛出友好的错误提示。"""
    if not _DEPENDENCIES_AVAILABLE:
        raise ImportError(
            f"\n{'='*60}\n"
            f"Qwen2-VL 后端缺少必要的依赖包\n"
            f"{'='*60}\n"
            f"导入错误: {_IMPORT_ERROR}\n\n"
            f"请运行以下命令安装依赖:\n"
            f"    pip install 'pymediaparser[vlm-qwen]'\n"
            f"{'='*60}"
        )


class Qwen2VLClient(_LocalTransformersBase):
    """Qwen2-VL 本地推理客户端。

    在 T4 GPU 上使用 float16 精度，Qwen2-VL-2B-Instruct 模型
    约占 4~5 GB 显存，单帧推理耗时约 0.5~1.5 秒。

    Args:
        config: VLM 配置（模型路径、设备、精度等）。
    """

    def __init__(self, config: VLMConfig | None = None) -> None:
        _check_dependencies()  # 检查依赖是否可用
        super().__init__(config)

    def _load_model_and_processor(
        self, dtype: torch.dtype, load_kwargs: Dict[str, Any],
    ) -> Tuple[Any, Any]:
        """加载 Qwen2-VL 模型和处理器。"""
        model = Qwen2VLForConditionalGeneration.from_pretrained(
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
        """Qwen2-VL 输入准备：使用 qwen_vl_utils.process_vision_info。"""
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)  # type: ignore[assignment]

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return inputs
