"""BMP虚拟VLM后端：用于调试分析的图像保存后端。

该后端不实际调用大模型进行帧分析，而是将收到的图像保存为BMP文件。
主要用于调试、验证帧采集流程、检查图像质量等场景。

文件命名格式：``frame_{序号:06d}_t{时间戳:.2f}s.bmp``，例如 ``frame_000001_t1.50s.bmp``。

典型用法::

    from pymediaparser.vlm import create_vlm_client
    from pymediaparser.vlm_base import VLMConfig

    config = VLMConfig(model_path="/tmp/debug_frames")
    client = create_vlm_client("bmp", config)

    with client:
        frame = {'image': pil_image, 'timestamp': 1.5, 'frame_index': 0}
        result = client.analyze(frame, "调试帧")
        # result.text = '{"files": ["/tmp/debug_frames/frame_000001_t1.50s.bmp"]}'
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import List, Sequence

from PIL import Image

from ..vlm_base import VLMClient, VLMConfig, VLMResult

logger = logging.getLogger(__name__)


class BMPVLMClient(VLMClient):
    """BMP虚拟VLM后端客户端。

    将图像保存为BMP文件，不进行实际的VLM推理。
    适用于调试分析、帧采集验证等场景。

    文件命名格式：``frame_{序号:06d}_t{时间戳:.2f}s.bmp``

    Args:
        config: VLM配置，其中 model_path 用作BMP文件保存目录。

    Attributes:
        config.model_path: BMP文件保存目录路径。
    """

    def __init__(self, config: VLMConfig | None = None) -> None:
        self.config = config or VLMConfig()
        self._output_dir: str = ""

    def load(self) -> None:
        """初始化后端：创建输出目录。"""
        self._output_dir = self.config.model_path

        # 创建输出目录
        os.makedirs(self._output_dir, exist_ok=True)

        # 扫描已有文件，记录日志
        existing_count = self._count_existing_frames()
        logger.info("BMP后端初始化: 输出目录=%s, 已有帧数=%d", self._output_dir, existing_count)

    def _count_existing_frames(self) -> int:
        """统计目录中已有的帧文件数量。

        Returns:
            目录中已有的帧文件数量。
        """
        count = 0
        pattern = re.compile(r"^frame_\d{6}_t[\d.]+s\.bmp$")

        try:
            for filename in os.listdir(self._output_dir):
                if pattern.match(filename):
                    count += 1
        except OSError as e:
            logger.warning("扫描目录失败: %s", e)

        return count

    def _get_frame_path(self, frame_index: int, timestamp: float) -> str:
        """获取帧的文件路径。

        文件名格式：frame_{序号:06d}_t{时间戳:.2f}s.bmp

        Args:
            frame_index: 帧序号。
            timestamp: 帧时间戳（秒）。

        Returns:
            完整的BMP文件路径。
        """
        filename = f"frame_{frame_index:06d}_t{timestamp:.2f}s.bmp"
        return os.path.join(self._output_dir, filename)

    def analyze(
        self,
        frame: dict,
        prompt: str | None = None,
    ) -> VLMResult:
        """保存单帧图像为BMP文件。

        Args:
            frame: 帧信息字典，包含 image、timestamp、frame_index 等。
            prompt: 文本提示词（记录到元信息中，不参与实际处理）。

        Returns:
            VLMResult，text字段为JSON列表格式的文件路径。
        """
        t0 = time.perf_counter()
        image = frame.get('image')
        frame_index = frame.get('frame_index', 0)
        timestamp = frame.get('timestamp', 0.0)

        # 获取输出路径并保存图像
        file_path = self._get_frame_path(frame_index, timestamp)
        image.save(file_path, format="BMP")

        elapsed = time.perf_counter() - t0
        image_size = image.size

        logger.debug(
            "BMP保存: %s (%dx%d) %.3fs",
            file_path, image_size[0], image_size[1], elapsed
        )

        # 构建返回结果
        files = [file_path]
        result = VLMResult(
            text=json.dumps({"files": files}),
            inference_time=elapsed,
            meta={
                "backend": "bmp",
                "files": files,
                "frame_index": frame_index,
                "timestamp": timestamp,
                "image_size": image_size,
                "prompt": prompt or self.config.default_prompt,
            }
        )

        return result

    def analyze_batch(
        self,
        frames: Sequence[dict],
        prompt: str | None = None,
    ) -> VLMResult:
        """批量保存多帧图像为BMP文件。

        Args:
            frames: 帧信息字典列表，每个字典包含 image、timestamp、frame_index 等。
            prompt: 文本提示词（记录到元信息中）。

        Returns:
            VLMResult，text字段为JSON列表格式的所有文件路径。
        """
        if not frames:
            return VLMResult(
                text=json.dumps({"files": []}),
                inference_time=0.0,
                meta={"backend": "bmp", "files": [], "frame_indices": []}
            )

        t0 = time.perf_counter()

        files: List[str] = []
        frame_indices: List[int] = []
        timestamps: List[float] = []
        image_sizes: List[tuple] = []

        for frame in frames:
            img = frame.get('image')
            frame_index = frame.get('frame_index', 0)
            timestamp = frame.get('timestamp', 0.0)

            file_path = self._get_frame_path(frame_index, timestamp)
            img.save(file_path, format="BMP")

            files.append(file_path)
            frame_indices.append(frame_index)
            timestamps.append(timestamp)
            image_sizes.append(img.size)

        elapsed = time.perf_counter() - t0

        logger.debug(
            "BMP批量保存: %d帧 -> %s (%.3fs)",
            len(frames), self._output_dir, elapsed
        )

        return VLMResult(
            text=json.dumps({"files": files}),
            inference_time=elapsed,
            meta={
                "backend": "bmp",
                "files": files,
                "frame_indices": frame_indices,
                "timestamps": timestamps,
                "image_sizes": image_sizes,
                "prompt": prompt or self.config.default_prompt,
            }
        )

    def unload(self) -> None:
        """清理资源，记录统计信息。"""
        # 统计当前保存的帧数
        saved_count = self._count_existing_frames()
        logger.info("BMP后端卸载: 共保存 %d 帧 -> %s", saved_count, self._output_dir)
        self._output_dir = ""

    def _get_default_prompt(self) -> str:
        """获取默认提示词。"""
        return self.config.default_prompt

    def supports_batch(self) -> bool:
        """BMP后端支持批量处理。"""
        return True
