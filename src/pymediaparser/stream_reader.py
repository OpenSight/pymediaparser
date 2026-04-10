"""视频流读取器：从 RTMP / HTTP-FLV / HTTP-TS 拉流并解码为视频帧。

基于 PyAV（FFmpeg）实现，支持断线自动重连。

典型用法::

    from pymediaparser.vlm_base import StreamConfig
    from pymediaparser.stream_reader import StreamReader

    cfg = StreamConfig(url="rtmp://host/live/stream")
    with StreamReader(cfg) as reader:
        for frame, ts in reader.frames():
            # frame: av.VideoFrame, ts: float (秒)
            ...
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

import av
from av.container import InputContainer

from .vlm_base import StreamConfig

logger = logging.getLogger(__name__)


class StreamReader:
    """从实时流拉取并解码视频帧。

    支持协议：
    - RTMP:     ``rtmp://host/app/stream``
    - HTTP-FLV: ``http://host/live/stream.flv``
    - HTTP-TS:  ``http://host/live/stream.ts``
    - HLS:      ``http://host/live/stream.m3u8``

    Args:
        config: 流配置（URL、格式、超时等）。
    """

    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self._container: InputContainer | None = None

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def open(self) -> None:
        """打开流连接。

        Raises:
            av.AVError: 连接或打开失败时抛出。
        """
        fmt = self._detect_format()
        is_rtmp = self.config.url.lower().startswith("rtmp")

        logger.info("正在连接流: %s (format=%s)", self.config.url, fmt or "auto")

        # PyAV 17 / libavformat 62: passing a non-empty options dict to av.open()
        # for RTMP URLs triggers a segfault in the bundled libavformat.  Use
        # PyAV's native timeout= parameter for RTMP; keep the options dict only
        # for non-RTMP protocols where it is safe.
        if is_rtmp:
            kwargs: dict = {"timeout": self.config.timeout}
        else:
            options = self._build_options()
            timeout_us = int(self.config.timeout * 1_000_000)
            options.setdefault("stimeout", str(timeout_us))
            options.setdefault("rw_timeout", str(timeout_us))
            kwargs = {"options": options}
        if fmt:
            kwargs["format"] = fmt

        self._container = av.open(self.config.url, **kwargs)
        logger.info("流连接成功: %s", self.config.url)

    def frames(self) -> Iterator[tuple[av.VideoFrame, float]]:
        """迭代解码视频帧，自动处理断线重连。

        Yields:
            ``(video_frame, timestamp_seconds)`` 元组。

        每次 yield 一个解码后的 ``av.VideoFrame`` 及其对应的流内时间戳（秒）。
        当连接中断时，会按照 ``config.reconnect_interval`` 间隔自动重连。
        """
        while True:
            try:
                if self._container is None:
                    self.open()
                assert self._container is not None

                stream = self._container.streams.video[0]
                # 不进行线程级解码（实时流通常不需要）
                stream.thread_type = "AUTO"

                # 设置解码模式
                if self.config.decode_mode == "keyframe_only":
                    stream.codec_context.skip_frame = "NONKEY"

                for frame in self._container.decode(video=0):
                    ts = self._frame_timestamp(frame)
                    yield frame, ts

                # 流正常结束（不常见于实时流）
                logger.warning("流已结束: %s", self.config.url)
                break

            except GeneratorExit:
                # 调用方主动关闭生成器
                self._close_container()
                return

            except Exception as exc:
                logger.error("流读取错误: %s — %s 秒后重连", exc, self.config.reconnect_interval)
                self._close_container()
                time.sleep(self.config.reconnect_interval)

    def close(self) -> None:
        """关闭流连接并释放资源。"""
        self._close_container()
        logger.info("流连接已关闭")

    # ------------------------------------------------------------------
    # 上下文管理器
    # ------------------------------------------------------------------

    def __enter__(self) -> "StreamReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _detect_format(self) -> str | None:
        """根据 URL 或用户配置推断容器格式。

        基于 URL 最后一段路径的文件扩展名判断，而非子串匹配，
        避免路径中包含 ``/flv/`` 等字样时误判。
        """
        if self.config.format:
            return self.config.format

        from urllib.parse import urlparse
        path = urlparse(self.config.url).path.lower()
        if path.endswith(".flv"):
            return "flv"
        if path.endswith(".ts"):
            return "mpegts"
        if path.endswith(".m3u8"):
            return None  # HLS 由 FFmpeg 自动处理
        # RTMP / 其他由 FFmpeg 自动检测
        return None

    def _build_options(self) -> dict[str, str]:
        """构建 FFmpeg 选项字典。"""
        opts: dict[str, str] = {
            # 降低延迟
            "fflags": "nobuffer",
            "flags": "low_delay",
            "analyzeduration": "500000",  # 0.5 秒
            "probesize": "500000",
        }
        # RTMP 特殊参数
        if self.config.url.lower().startswith("rtmp"):
            opts["live"] = "1"
        return opts

    @staticmethod
    def _frame_timestamp(frame: av.VideoFrame) -> float:
        """从帧中提取时间戳（秒）。"""
        if frame.pts is not None and frame.time_base is not None:
            return float(frame.pts * frame.time_base)
        # 回退：使用 frame.time（部分流可能为 None）
        if frame.time is not None:
            return float(frame.time)
        return 0.0

    def _close_container(self) -> None:
        """安全关闭容器。"""
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None
