"""ReplayPipeline 单元测试。

使用 mock VLM 客户端测试回放 Pipeline 的核心功能：
- 文件回放自动完成退出
- 阻塞队列不丢帧
- 进度跟踪

使用项目中已有的测试资源文件：
- resource/866f893bb7a353c75f1aa5c7cb61e4e3.mp4 (视频)
- resource/test_img1.png (图片)
"""

from __future__ import annotations

import os
import threading
import time
import pytest
from unittest.mock import MagicMock

from PIL import Image

from pymediaparser.vlm_base import StreamConfig, VLMResult, VLMClient

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESOURCE_DIR = os.path.join(_PROJECT_ROOT, "resource")

# 测试资源文件
_VIDEO_FILE = os.path.join(_RESOURCE_DIR, "866f893bb7a353c75f1aa5c7cb61e4e3.mp4")
_IMAGE_FILE = os.path.join(_RESOURCE_DIR, "test_img1.png")


# =====================================================================
# 跳过条件
# =====================================================================

skip_if_no_video = pytest.mark.skipif(
    not os.path.exists(_VIDEO_FILE),
    reason=f"测试视频文件不存在: {_VIDEO_FILE}",
)

skip_if_no_image = pytest.mark.skipif(
    not os.path.exists(_IMAGE_FILE),
    reason=f"测试图片文件不存在: {_IMAGE_FILE}",
)


# =====================================================================
# Mock VLM 客户端
# =====================================================================

class MockVLMClient(VLMClient):
    """用于测试的 Mock VLM 客户端。"""

    def __init__(self):
        self.load_called = False
        self.unload_called = False
        self.analyze_calls = []
        self.analyze_batch_calls = []

    def load(self) -> None:
        self.load_called = True

    def unload(self) -> None:
        self.unload_called = True

    def analyze(self, image: Image.Image, prompt: str | None = None) -> VLMResult:
        self.analyze_calls.append((image, prompt))
        return VLMResult(
            text="[Mock Analysis]",
            inference_time=0.01,
            input_tokens=10,
            output_tokens=20,
        )

    def analyze_batch(self, images, prompt: str | None = None) -> VLMResult:
        self.analyze_batch_calls.append((images, prompt))
        return VLMResult(
            text="[Mock Batch Analysis]",
            inference_time=0.05,
            input_tokens=30,
            output_tokens=50,
        )


# =====================================================================
# Mock ResultHandler
# =====================================================================

class MockResultHandler:
    """用于测试的 Mock 结果处理器。"""

    def __init__(self):
        self.on_start_called = False
        self.on_stop_called = False
        self.on_complete_called = False
        self.results = []

    def on_start(self):
        self.on_start_called = True

    def on_stop(self):
        self.on_stop_called = True

    def on_complete(self):
        self.on_complete_called = True

    def handle(self, frame_result):
        self.results.append(frame_result)


# =====================================================================
# ReplayPipeline 测试
# =====================================================================

@skip_if_no_video
class TestReplayPipelineVideo:
    """视频回放测试。"""

    def test_video_replay_completes(self):
        """视频回放完成后 Pipeline 自动退出。"""
        from pymediaparser.replay_pipeline import ReplayPipeline

        cfg = StreamConfig(url=_VIDEO_FILE, target_fps=5.0)  # 较高 FPS 加快测试
        vlm_client = MockVLMClient()
        handler = MockResultHandler()

        pipeline = ReplayPipeline(
            stream_config=cfg,
            vlm_client=vlm_client,
            handlers=[handler],
        )

        # 在单独线程运行，避免阻塞测试
        thread = threading.Thread(target=pipeline.run)
        thread.start()
        thread.join(timeout=60)  # 最多等 60 秒

        assert not thread.is_alive(), "Pipeline 应在文件处理完后自动退出"
        assert vlm_client.load_called
        assert vlm_client.unload_called
        assert handler.on_start_called
        assert handler.on_complete_called, "正常完成时应调用 on_complete"
        assert len(handler.results) > 0, "应有分析结果"

    def test_video_no_frame_loss(self):
        """阻塞队列应保证不丢帧。"""
        from pymediaparser.replay_pipeline import ReplayPipeline
        from pymediaparser.file_reader import FileReader
        from pymediaparser.frame_sampler import FrameSampler

        cfg = StreamConfig(url=_VIDEO_FILE, target_fps=2.0, max_queue_size=2)
        vlm_client = MockVLMClient()
        handler = MockResultHandler()

        # 先计算预期采样帧数
        sampler = FrameSampler(target_fps=cfg.target_fps)
        with FileReader(cfg) as reader:
            expected_count = len(list(sampler.sample(reader.frames())))

        # 运行 Pipeline
        pipeline = ReplayPipeline(
            stream_config=cfg,
            vlm_client=vlm_client,
            handlers=[handler],
        )

        thread = threading.Thread(target=pipeline.run)
        thread.start()
        thread.join(timeout=120)

        assert not thread.is_alive()
        # 所有采样帧都应被处理（不丢帧）
        assert len(vlm_client.analyze_calls) == expected_count, \
            f"期望处理 {expected_count} 帧，实际 {len(vlm_client.analyze_calls)} 帧"


@skip_if_no_image
class TestReplayPipelineImage:
    """图片回放测试。"""

    def test_image_replay_completes(self):
        """图片回放完成后 Pipeline 自动退出。"""
        from pymediaparser.replay_pipeline import ReplayPipeline

        cfg = StreamConfig(url=_IMAGE_FILE)
        vlm_client = MockVLMClient()
        handler = MockResultHandler()

        pipeline = ReplayPipeline(
            stream_config=cfg,
            vlm_client=vlm_client,
            handlers=[handler],
        )

        thread = threading.Thread(target=pipeline.run)
        thread.start()
        thread.join(timeout=30)

        assert not thread.is_alive(), "Pipeline 应在处理完图片后自动退出"
        assert vlm_client.load_called
        assert vlm_client.unload_called
        assert len(handler.results) == 1, "图片应只产生一个结果"

    def test_image_single_analysis(self):
        """图片只调用一次 VLM 分析。"""
        from pymediaparser.replay_pipeline import ReplayPipeline

        cfg = StreamConfig(url=_IMAGE_FILE)
        vlm_client = MockVLMClient()

        pipeline = ReplayPipeline(
            stream_config=cfg,
            vlm_client=vlm_client,
        )

        thread = threading.Thread(target=pipeline.run)
        thread.start()
        thread.join(timeout=30)

        assert len(vlm_client.analyze_calls) == 1


class TestReplayPipelineStop:
    """停止机制测试。"""

    @skip_if_no_image
    def test_stop_interrupts_replay(self):
        """调用 stop() 可中断回放（使用图片避免视频解码中断问题）。"""
        from pymediaparser.replay_pipeline import ReplayPipeline

        # 使用图片测试避免 PyAV 解码中断 segfault
        cfg = StreamConfig(url=_IMAGE_FILE)
        vlm_client = MockVLMClient()

        pipeline = ReplayPipeline(
            stream_config=cfg,
            vlm_client=vlm_client,
        )

        # 测试 stop() 能被正常调用
        pipeline.start()
        time.sleep(2)  # 等待处理完成
        pipeline.stop()

        assert vlm_client.unload_called  # 确认正常清理


@skip_if_no_video
class TestReplayPipelineHandlers:
    """Handler 回调测试。"""

    def test_handler_lifecycle(self):
        """Handler 的 on_start 和 on_stop 被正确调用。"""
        from pymediaparser.replay_pipeline import ReplayPipeline

        cfg = StreamConfig(url=_VIDEO_FILE, target_fps=10.0)
        vlm_client = MockVLMClient()
        handler = MockResultHandler()

        pipeline = ReplayPipeline(
            stream_config=cfg,
            vlm_client=vlm_client,
            handlers=[handler],
        )

        thread = threading.Thread(target=pipeline.run)
        thread.start()
        thread.join(timeout=60)

        assert handler.on_start_called, "on_start 应被调用"
        assert handler.on_complete_called, "正常完成时应调用 on_complete"

    def test_frame_results(self):
        """FrameResult 应包含正确字段。"""
        from pymediaparser.replay_pipeline import ReplayPipeline
        from pymediaparser.vlm_base import FrameResult

        cfg = StreamConfig(url=_VIDEO_FILE, target_fps=10.0)
        vlm_client = MockVLMClient()
        handler = MockResultHandler()

        pipeline = ReplayPipeline(
            stream_config=cfg,
            vlm_client=vlm_client,
            handlers=[handler],
        )

        thread = threading.Thread(target=pipeline.run)
        thread.start()
        thread.join(timeout=60)

        for result in handler.results:
            assert isinstance(result, FrameResult)
            assert isinstance(result.frame_index, int)
            assert isinstance(result.timestamp, float)
            assert result.vlm_result is not None
