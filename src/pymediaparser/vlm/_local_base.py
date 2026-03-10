"""本地 transformers 模型抽象基类。

提取 Qwen2VLClient / Qwen3VLClient 等本地 transformers 模型的
公共逻辑，子类只需实现 ``_load_model_and_processor`` 和
``_prepare_inputs`` 两个方法。
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from typing import Any, Dict, List, Sequence, Tuple

# 延迟导入：torch 是可选依赖（仅本地 VLM 后端需要）
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False

from PIL import Image

from ..vlm_base import VLMClient, VLMConfig, VLMResult

logger = logging.getLogger(__name__)

# torch dtype 映射（统一维护，子类共用）
# 仅在 torch 可用时定义
if _TORCH_AVAILABLE:
    _DTYPE_MAP: dict[str, Any] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
else:
    _DTYPE_MAP = {}


class _LocalTransformersBase(VLMClient):
    """本地 transformers 模型推理基类。

    封装了模型加载、推理、卸载等公共流程，子类通过覆盖
    ``_load_model_and_processor`` 和 ``_prepare_inputs`` 来适配不同模型。

    Args:
        config: VLM 配置（模型路径、设备、精度等）。
    """

    def __init__(self, config: VLMConfig | None = None) -> None:
        self.config = config or VLMConfig()
        self._model: Any = None
        self._processor: Any = None

    # ------------------------------------------------------------------
    # 子类必须实现的抽象方法
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_model_and_processor(
        self, dtype: torch.dtype, load_kwargs: Dict[str, Any],
    ) -> Tuple[Any, Any]:
        """加载模型和处理器。

        Args:
            dtype: 推理精度对应的 torch dtype。
            load_kwargs: 预构建的 from_pretrained 关键字参数，包含
                torch_dtype、device_map、attn_implementation 等。

        Returns:
            (model, processor) 元组。
        """

    @abstractmethod
    def _prepare_inputs(
        self, messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """将多模态消息转换为模型输入 tensor。

        Args:
            messages: OpenAI 格式的多模态消息列表。

        Returns:
            可直接传入 model.generate() 的 inputs 字典。
        """

    # ------------------------------------------------------------------
    # VLMClient 接口实现（公共逻辑）
    # ------------------------------------------------------------------

    def load(self) -> None:
        """加载模型和处理器到指定设备。"""
        cfg = self.config
        dtype = _DTYPE_MAP.get(cfg.dtype, torch.float16)

        logger.info(
            "加载模型: %s  device=%s  dtype=%s",
            cfg.model_path, cfg.device, cfg.dtype,
        )

        # 构建 from_pretrained 参数
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }

        if "cuda" in cfg.device:
            load_kwargs["device_map"] = cfg.device

        if cfg.use_flash_attn and "cuda" in cfg.device:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("尝试启用 Flash Attention 2 ...")

        # 调用子类实现加载模型和处理器
        try:
            self._model, self._processor = self._load_model_and_processor(
                dtype, load_kwargs,
            )
        except Exception as exc:
            # Flash Attention 不可用时回退
            if "flash" in str(exc).lower() and "attn_implementation" in load_kwargs:
                logger.warning("Flash Attention 2 不可用，回退到默认注意力机制")
                load_kwargs.pop("attn_implementation", None)
                self._model, self._processor = self._load_model_and_processor(
                    dtype, load_kwargs,
                )
            else:
                raise

        # CPU 模式手动移动
        if "cuda" not in cfg.device:
            assert self._model is not None
            self._model = self._model.to(torch.device(cfg.device))

        self._model.eval()

        # 打印 GPU 信息
        if "cuda" in cfg.device and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info("GPU: %s (%.1f GB)", gpu_name, gpu_mem)

        logger.info("模型加载完成")

    def analyze(self, image: Image.Image, prompt: str | None = None) -> VLMResult:
        """对单帧图像进行 VLM 推理。"""
        if self._model is None or self._processor is None:
            raise RuntimeError("模型尚未加载，请先调用 load()")

        prompt = prompt or self.config.default_prompt
        t0 = time.perf_counter()

        # 构造多模态消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 子类实现输入准备
        inputs = self._prepare_inputs(messages)

        # 移动到模型所在设备
        device = next(self._model.parameters()).device
        inputs = inputs.to(device)

        # 推理
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        # 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        elapsed = time.perf_counter() - t0
        tokens_used = generated_ids_trimmed[0].shape[0] if generated_ids_trimmed else 0

        logger.debug(
            "推理完成: %.2fs, %d tokens — %s",
            elapsed, tokens_used, output_text[:80],
        )

        return VLMResult(
            text=output_text.strip(),
            tokens_used=int(tokens_used),
            inference_time=elapsed,
        )

    def analyze_batch(self, images: Sequence[Image.Image], prompt: str | None = None) -> VLMResult:
        """对多张图像进行批量 VLM 推理。"""
        if not images:
            raise ValueError("图像列表不能为空")

        if self._model is None or self._processor is None:
            raise RuntimeError("模型尚未加载，请先调用 load()")

        prompt = prompt or self.config.default_prompt
        start_time = time.perf_counter()

        # 构造多图像消息
        content: List[Dict[str, Any]] = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        # 子类实现输入准备
        inputs = self._prepare_inputs(messages)

        # 移动到设备
        device = next(self._model.parameters()).device
        inputs = inputs.to(device)

        # 推理
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=getattr(self.config, "temperature", 0.7),
                do_sample=getattr(self.config, "do_sample", True),
                use_cache=True,
            )

        # 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        inference_time = time.perf_counter() - start_time
        input_tokens = len(inputs.input_ids[0])
        output_tokens = len(generated_ids_trimmed[0])

        logger.debug(
            "批量处理完成 - 图像数: %d, 输入tokens: %d, 输出tokens: %d, 耗时: %.2fs",
            len(images), input_tokens, output_tokens, inference_time,
        )

        return VLMResult(
            text=output_text,
            inference_time=inference_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def unload(self) -> None:
        """释放模型，清空 GPU 显存。"""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("模型已卸载，显存已释放")

    # ------------------------------------------------------------------
    # 上下文分析（公共实现）
    # ------------------------------------------------------------------

    def analyze_with_context(
        self,
        images: List[Image.Image],
        context_prompt: str | None = None,
    ) -> VLMResult:
        """带上下文的批量分析。

        Args:
            images: PIL 图像列表。
            context_prompt: 上下文提示词，为 None 时自动生成。

        Returns:
            分析结果 VLMResult。
        """
        if context_prompt is None:
            context_prompt = self._build_context_prompt(len(images))
        return self.analyze_batch(images, context_prompt)

    @staticmethod
    def _build_context_prompt(image_count: int) -> str:
        """根据图像数量构建上下文提示词。"""
        if image_count == 1:
            return "请详细分析这张图像的内容。"
        elif image_count <= 3:
            return f"请分析这{image_count}张图像，描述它们之间的关系和变化。"
        else:
            return (
                f"请分析这{image_count}帧连续图像序列，"
                f"描述场景的发展变化和重要事件。"
            )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _get_default_prompt(self) -> str:
        """获取默认提示词。"""
        return self.config.default_prompt

    def supports_batch(self) -> bool:
        """本地 transformers 模型支持原生批量处理。"""
        return True
