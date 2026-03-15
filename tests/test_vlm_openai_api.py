"""OpenAI 兼容 API 客户端的单元测试。

使用 mock 替代真实 HTTP 请求。
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from PIL import Image

from pymediaparser.vlm_base import VLMResult
from pymediaparser.vlm.configs import APIVLMConfig
from pymediaparser.vlm.openai_api import OpenAIAPIClient


def _make_image(width: int = 64, height: int = 64) -> Image.Image:
    """创建测试用纯色图像。"""
    return Image.new("RGB", (width, height), color=(128, 128, 128))


def _make_frame(
    image: Image.Image | None = None,
    frame_index: int = 0,
    timestamp: float = 0.0,
) -> dict:
    """创建测试用帧信息字典。"""
    if image is None:
        image = _make_image()
    return {
        'image': image,
        'frame_index': frame_index,
        'timestamp': timestamp,
        'significant': True,
        'source': ['test'],
    }


def _mock_api_response(text: str = "这是一个测试回复",
                       prompt_tokens: int = 100,
                       completion_tokens: int = 20) -> dict:
    """构建模拟的 OpenAI API 响应。"""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class TestOpenAIAPIClientInit:
    """初始化测试。"""

    def test_default_config(self):
        """默认配置初始化。"""
        client = OpenAIAPIClient()
        assert client.config.base_url == "http://localhost:8000/v1"
        assert client._session is None

    def test_custom_config(self):
        """自定义配置初始化。"""
        cfg = APIVLMConfig(
            base_url="http://my-server:9000/v1",
            api_key="test-key",
            model_name="my-model",
        )
        client = OpenAIAPIClient(cfg)
        assert client.config.base_url == "http://my-server:9000/v1"
        assert client.config.api_key == "test-key"


class TestOpenAIAPIClientLoad:
    """load() / unload() 测试。"""

    def test_load_creates_session(self):
        """load() 创建 HTTP session。"""
        client = OpenAIAPIClient()
        client.load()
        assert client._session is not None
        client.unload()

    def test_load_sets_auth_header(self):
        """配置 api_key 时设置 Authorization 头。"""
        cfg = APIVLMConfig(api_key="sk-test-123")
        client = OpenAIAPIClient(cfg)
        client.load()
        assert "Authorization" in client._session.headers
        assert client._session.headers["Authorization"] == "Bearer sk-test-123"
        client.unload()

    def test_load_no_auth_header_without_key(self):
        """无 api_key 时不设置 Authorization 头。"""
        cfg = APIVLMConfig(api_key=None)
        client = OpenAIAPIClient(cfg)
        client.load()
        assert "Authorization" not in client._session.headers
        client.unload()

    def test_unload_closes_session(self):
        """unload() 关闭并清除 session。"""
        client = OpenAIAPIClient()
        client.load()
        client.unload()
        assert client._session is None


class TestOpenAIAPIClientAnalyze:
    """analyze() 测试。"""

    def test_analyze_without_load_raises(self):
        """未 load 时调用 analyze 抛出 RuntimeError。"""
        client = OpenAIAPIClient()
        with pytest.raises(RuntimeError, match="尚未初始化"):
            client.analyze(_make_frame())

    @patch("pymediaparser.vlm.openai_api.requests.Session")
    def test_analyze_returns_vlm_result(self, MockSession):
        """analyze() 返回正确的 VLMResult。"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_api_response("画面中有一个人")
        mock_resp.raise_for_status.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_session.headers = {}
        MockSession.return_value = mock_session

        client = OpenAIAPIClient(APIVLMConfig(model_name="test-model"))
        client.load()
        result = client.analyze(_make_frame(), "描述画面")

        assert isinstance(result, VLMResult)
        assert result.text == "画面中有一个人"
        assert result.input_tokens == 100
        assert result.output_tokens == 20
        assert result.inference_time > 0

        # 验证请求体
        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]
        assert payload["model"] == "test-model"
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

        client.unload()

    @patch("pymediaparser.vlm.openai_api.requests.Session")
    def test_analyze_uses_default_prompt(self, MockSession):
        """不传 prompt 时使用默认提示词。"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_api_response()
        mock_resp.raise_for_status.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_session.headers = {}
        MockSession.return_value = mock_session

        cfg = APIVLMConfig(default_prompt="自定义默认提示")
        client = OpenAIAPIClient(cfg)
        client.load()
        client.analyze(_make_frame())

        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]
        content = payload["messages"][0]["content"]
        # 最后一个 content 元素是 text
        text_item = [c for c in content if c["type"] == "text"][-1]
        assert text_item["text"] == "自定义默认提示"

        client.unload()


class TestOpenAIAPIClientAnalyzeBatch:
    """analyze_batch() 测试。"""

    def test_empty_frames_raises(self):
        """空帧列表抛出 ValueError。"""
        client = OpenAIAPIClient()
        client.load()
        with pytest.raises(ValueError, match="不能为空"):
            client.analyze_batch([])
        client.unload()

    @patch("pymediaparser.vlm.openai_api.requests.Session")
    def test_batch_sends_multiple_images(self, MockSession):
        """批量处理发送多张图像。"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_api_response("批量分析结果")
        mock_resp.raise_for_status.return_value = None

        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_session.headers = {}
        MockSession.return_value = mock_session

        client = OpenAIAPIClient()
        client.load()

        frames = [_make_frame(frame_index=i, timestamp=float(i)) for i in range(3)]
        result = client.analyze_batch(frames, "分析这些图像")

        assert isinstance(result, VLMResult)
        assert result.text == "批量分析结果"

        # 验证发送了 3 张图像 + 时序标签 + 文本
        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]
        content = payload["messages"][0]["content"]
        image_items = [c for c in content if c["type"] == "image_url"]
        text_items = [c for c in content if c["type"] == "text"]
        assert len(image_items) == 3
        # 多帧时有前缀 + 每帧一个时序标签 + prompt
        assert len(text_items) >= 1

        client.unload()


class TestOpenAIAPIClientEncodeImage:
    """_encode_image() 测试。"""

    def test_small_image_not_resized(self):
        """小于 max_size 的图像不缩放。"""
        cfg = APIVLMConfig(image_max_size=1024)
        client = OpenAIAPIClient(cfg)
        img = _make_image(100, 100)
        b64 = client._encode_image(img)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_large_image_resized(self):
        """大于 max_size 的图像会缩放。"""
        cfg = APIVLMConfig(image_max_size=64)
        client = OpenAIAPIClient(cfg)
        img = _make_image(200, 100)
        b64 = client._encode_image(img)
        assert isinstance(b64, str)

    def test_rgba_image_converted(self):
        """RGBA 图像转换为 RGB 后编码。"""
        cfg = APIVLMConfig()
        client = OpenAIAPIClient(cfg)
        img = Image.new("RGBA", (64, 64), color=(128, 128, 128, 255))
        b64 = client._encode_image(img)
        assert isinstance(b64, str)


class TestOpenAIAPIClientParseResponse:
    """_parse_response() 测试。"""

    def test_normal_response(self):
        """正常响应解析。"""
        data = _mock_api_response("测试回复", 50, 10)
        result = OpenAIAPIClient._parse_response(data)
        assert result.text == "测试回复"
        assert result.input_tokens == 50
        assert result.output_tokens == 10

    def test_empty_choices(self):
        """空 choices 返回占位文本。"""
        data = {"choices": []}
        result = OpenAIAPIClient._parse_response(data)
        assert result.text == "[API 返回空结果]"

    def test_no_usage(self):
        """缺少 usage 字段时 token 为 0。"""
        data = {
            "choices": [{"message": {"content": "ok"}}],
        }
        result = OpenAIAPIClient._parse_response(data)
        assert result.text == "ok"
        assert result.input_tokens == 0
        assert result.output_tokens == 0


class TestOpenAIAPIClientSupportsBatch:
    """supports_batch() 测试。"""

    def test_supports_batch(self):
        """API 客户端声明支持批量处理。"""
        client = OpenAIAPIClient()
        assert client.supports_batch() is True
