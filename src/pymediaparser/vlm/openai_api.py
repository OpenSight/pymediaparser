"""OpenAI 兼容 API 推理客户端。

通过 HTTP 请求调用 /v1/chat/completions 接口，
支持 vLLM、Ollama、TGI、OpenAI、通义千问等服务。
仅使用 requests 库，不依赖 openai SDK。

典型用法::

    from pymediaparser.vlm.openai_api import OpenAIAPIClient
    from pymediaparser.vlm.configs import APIVLMConfig
    from PIL import Image

    config = APIVLMConfig(
        base_url="http://localhost:8000/v1",
        model_name="Qwen2-VL-2B-Instruct",
    )
    with OpenAIAPIClient(config) as client:
        frame = {'image': Image.open('test.jpg'), 'frame_index': 0, 'timestamp': 0.0}
        result = client.analyze(frame, "请描述画面内容。")
        print(result.text)
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Dict, List, Sequence

import requests
from PIL import Image

from ..vlm_base import VLMClient, VLMResult
from .configs import APIVLMConfig

logger = logging.getLogger(__name__)


class OpenAIAPIClient(VLMClient):
    """OpenAI 兼容 API 推理客户端。

    通过标准的 /v1/chat/completions 接口进行多模态推理，
    图像以 base64 编码传输。

    Args:
        config: API 服务配置。
    """

    def __init__(self, config: APIVLMConfig | None = None) -> None:
        self.config = config or APIVLMConfig()
        self._session: requests.Session | None = None

    # ------------------------------------------------------------------
    # VLMClient 接口实现
    # ------------------------------------------------------------------

    def load(self) -> None:
        """创建 HTTP 会话并配置认证头。"""
        self._session = requests.Session()

        if self.config.api_key:
            self._session.headers["Authorization"] = f"Bearer {self.config.api_key}"

        self._session.headers["Content-Type"] = "application/json"

        logger.info(
            "API 客户端就绪: base_url=%s  model=%s",
            self.config.base_url, self.config.model_name,
        )

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

        # 构建消息（单帧也有时序信息）
        content = self._build_content_with_timing([frame], prompt)
        messages = [{"role": "user", "content": content}]

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": self.config.max_new_tokens,
            "temperature": 0,
        }

        # 发送请求（带重试）
        response_data = self._request_with_retry(payload)

        # 解析结果
        result = self._parse_response(response_data)
        result.inference_time = time.perf_counter() - t0

        logger.debug(
            "API 推理完成: %.2fs — %s",
            result.inference_time, result.text[:80],
        )

        return result

    def analyze_batch(
        self,
        frames: Sequence[Dict[str, Any]],
        prompt: str | None = None,
    ) -> VLMResult:
        """对多张图像调用 API 批量推理。

        将多张图像放入同一条消息的 content 数组中。
        """
        if not frames:
            raise ValueError("帧信息列表不能为空")

        if self._session is None:
            raise RuntimeError("客户端尚未初始化，请先调用 load()")

        prompt = prompt or self.config.default_prompt
        start_time = time.perf_counter()

        # 使用辅助方法构建消息
        content = self._build_content_with_timing(frames, prompt)
        messages = [{"role": "user", "content": content}]

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": self.config.max_new_tokens,
            "temperature": 0,
        }

        response_data = self._request_with_retry(payload)
        result = self._parse_response(response_data)
        result.inference_time = time.perf_counter() - start_time

        logger.debug(
            "API 批量推理完成: %d 帧, %.2fs",
            len(frames), result.inference_time,
        )

        return result

    def unload(self) -> None:
        """关闭 HTTP 会话。"""
        if self._session is not None:
            self._session.close()
            self._session = None

        logger.info("API 客户端已关闭")

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

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
            # 类型安全转换，避免 None 或非数值类型导致 format 异常
            try:
                index = int(frame.get('frame_index', 0) or 0)
            except (TypeError, ValueError):
                index = 0
            try:
                timestamp = float(frame.get('timestamp', 0.0) or 0.0)
            except (TypeError, ValueError):
                timestamp = 0.0

            timing_text = self.config.timing_format.format(
                index=index,
                timestamp=timestamp
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

    def _encode_image(self, image: Image.Image) -> str:
        """将 PIL 图像编码为 base64 字符串。

        超过 image_max_size 时等比缩放。
        """
        cfg = self.config

        # 等比缩放
        w, h = image.size
        max_side = max(w, h)
        if max_side > cfg.image_max_size:
            scale = cfg.image_max_size / max_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)

        # 编码为 JPEG base64
        buffer = io.BytesIO()
        # 确保 RGB 模式（RGBA 不支持 JPEG）
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=cfg.image_quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _request_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发送请求并在失败时重试。"""
        assert self._session is not None

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self._session.post(
                    url, json=payload, timeout=self.config.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "API 请求失败 (第 %d/%d 次): %s",
                    attempt, self.config.max_retries, exc,
                )
                if attempt < self.config.max_retries:
                    time.sleep(min(2 ** attempt, 10))

        raise RuntimeError(
            f"API 请求在 {self.config.max_retries} 次重试后仍失败: {last_error}"
        ) from last_error

    @staticmethod
    def _parse_response(data: Dict[str, Any]) -> VLMResult:
        """解析 OpenAI 格式的响应数据。"""
        choices = data.get("choices", [])
        if not choices:
            return VLMResult(text="[API 返回空结果]")

        message = choices[0].get("message", {})
        text = message.get("content", "").strip()

        # 提取 token 用量
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return VLMResult(
            text=text,
            tokens_used=output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _get_default_prompt(self) -> str:
        """获取默认提示词。"""
        return self.config.default_prompt

    def supports_batch(self) -> bool:
        """API 客户端支持批量处理。"""
        return True
