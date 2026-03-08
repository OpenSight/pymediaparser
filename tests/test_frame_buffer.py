"""FrameBuffer 单元测试"""

from __future__ import annotations
import pytest

from pymediaparser.frame_buffer import FrameBuffer, BufferedFrame


def _make_frame_data(timestamp: float, is_periodic: bool = False) -> dict:
    """生成测试帧数据"""
    source = ['periodic'] if is_periodic else ['motion']
    return {
        'timestamp': timestamp,
        'image': None,  # 简化测试
        'source': source,
    }


class TestFrameBuffer:
    """FrameBuffer 基础测试"""

    def test_buffer_initialization(self):
        """缓冲区初始化应正确"""
        fb = FrameBuffer(max_size=5, max_wait_time=10.0)
        assert fb.max_size == 5
        assert fb.max_wait_time == 10.0
        assert len(fb) == 0

    def test_add_frame_returns_none_when_not_ready(self):
        """未满足条件时应返回 None"""
        fb = FrameBuffer(max_size=5, max_wait_time=10.0)
        result = fb.add_frame(_make_frame_data(0.0))
        assert result is None
        assert len(fb) == 1

    def test_buffer_full_returns_batch(self):
        """缓冲区满时应返回批次"""
        fb = FrameBuffer(max_size=3, max_wait_time=100.0)
        
        fb.add_frame(_make_frame_data(0.0))  # 入队
        fb.add_frame(_make_frame_data(1.0))  # 入队
        result = fb.add_frame(_make_frame_data(2.0))  # 满，应返回批次
        
        assert result is not None
        assert len(result) == 3
        assert len(fb) == 0  # 缓冲区已清空

    def test_time_span_exceeds_returns_batch(self):
        """时间跨度超限应返回批次"""
        fb = FrameBuffer(max_size=10, max_wait_time=5.0)
        
        fb.add_frame(_make_frame_data(0.0))  # 入队
        fb.add_frame(_make_frame_data(1.0))  # 入队
        fb.add_frame(_make_frame_data(2.0))  # 入队
        # 时间跨度 6.0 - 0.0 = 6.0 > 5.0，应返回批次
        result = fb.add_frame(_make_frame_data(6.0))
        
        assert result is not None
        assert len(result) == 3  # 前三帧为一批
        assert len(fb) == 1  # 新帧已入队

    def test_flush_returns_all_frames(self):
        """flush 应返回所有帧"""
        fb = FrameBuffer(max_size=10, max_wait_time=100.0)
        
        fb.add_frame(_make_frame_data(0.0))
        fb.add_frame(_make_frame_data(1.0))
        fb.add_frame(_make_frame_data(2.0))
        
        result = fb.flush()
        assert result is not None
        assert len(result) == 3
        assert len(fb) == 0

    def test_flush_empty_buffer_returns_none(self):
        """flush 空缓冲区应返回 None"""
        fb = FrameBuffer()
        result = fb.flush()
        assert result is None

    def test_clear_empties_buffer(self):
        """clear 应清空缓冲区"""
        fb = FrameBuffer()
        fb.add_frame(_make_frame_data(0.0))
        fb.add_frame(_make_frame_data(1.0))
        
        fb.clear()
        assert len(fb) == 0

    def test_get_status(self):
        """get_status 应返回正确状态"""
        fb = FrameBuffer(max_size=5, max_wait_time=10.0)
        
        status = fb.get_status()
        assert status['size'] == 0
        assert status['ready'] is False
        
        fb.add_frame(_make_frame_data(0.0))
        status = fb.get_status()
        assert status['size'] == 1
        assert status['timestamp_span'] == 0.0


class TestFrameBufferPeriodicBoundary:
    """FrameBuffer 周期边界帧测试"""

    def test_periodic_boundary_splits_batch(self):
        """周期边界帧应分割批次"""
        fb = FrameBuffer(max_size=10, max_wait_time=100.0)
        
        fb.add_frame(_make_frame_data(0.0))  # 入队
        fb.add_frame(_make_frame_data(1.0))  # 入队
        fb.add_frame(_make_frame_data(2.0))  # 入队
        
        # 周期边界帧到达，应返回当前批次
        result = fb.add_frame(_make_frame_data(3.0, is_periodic=True))
        
        assert result is not None
        assert len(result) == 3  # 前三帧为一批
        assert len(fb) == 1  # 周期边界帧已入队

    def test_periodic_boundary_with_empty_buffer(self):
        """周期边界帧到达时缓冲区为空，不应分割"""
        fb = FrameBuffer(max_size=10, max_wait_time=100.0)
        
        # 缓冲区为空，周期边界帧入队
        result = fb.add_frame(_make_frame_data(0.0, is_periodic=True))
        
        assert result is None  # 无批次返回
        assert len(fb) == 1  # 帧已入队

    def test_periodic_boundary_priority_over_time_span(self):
        """周期边界优先于时间跨度判断"""
        fb = FrameBuffer(max_size=10, max_wait_time=2.0)  # 短等待时间
        
        fb.add_frame(_make_frame_data(0.0))  # 入队
        fb.add_frame(_make_frame_data(1.0))  # 入队
        
        # 时间跨度未超限，但周期边界帧到达
        result = fb.add_frame(_make_frame_data(1.5, is_periodic=True))
        
        # 应因周期边界分割，而非时间跨度
        assert result is not None
        assert len(result) == 2
        assert len(fb) == 1

    def test_non_periodic_frame_does_not_split(self):
        """非周期边界帧不应分割批次"""
        fb = FrameBuffer(max_size=10, max_wait_time=100.0)
        
        fb.add_frame(_make_frame_data(0.0))  # 入队
        fb.add_frame(_make_frame_data(1.0))  # 入队
        
        # 非周期边界帧（source 不包含 'periodic'）
        result = fb.add_frame(_make_frame_data(2.0, is_periodic=False))
        
        assert result is None  # 不分割
        assert len(fb) == 3

    def test_consecutive_periodic_boundaries(self):
        """连续周期边界帧处理"""
        fb = FrameBuffer(max_size=10, max_wait_time=100.0)
        
        # 第一个周期边界帧
        result1 = fb.add_frame(_make_frame_data(0.0, is_periodic=True))
        assert result1 is None  # 缓冲区为空，不分割
        assert len(fb) == 1
        
        # 第二个周期边界帧
        result2 = fb.add_frame(_make_frame_data(5.0, is_periodic=True))
        assert result2 is not None  # 分割前一帧
        assert len(result2) == 1
        assert len(fb) == 1  # 新帧已入队

    def test_periodic_boundary_after_full_buffer(self):
        """缓冲区满后再收到周期边界帧"""
        fb = FrameBuffer(max_size=3, max_wait_time=100.0)
        
        # 填满缓冲区
        result1 = fb.add_frame(_make_frame_data(0.0))
        result2 = fb.add_frame(_make_frame_data(1.0))
        result3 = fb.add_frame(_make_frame_data(2.0))  # 满，返回批次
        
        assert result1 is None
        assert result2 is None
        assert result3 is not None
        assert len(result3) == 3
        
        # 周期边界帧入队
        result4 = fb.add_frame(_make_frame_data(3.0, is_periodic=True))
        assert result4 is None  # 缓冲区为空，不分割
        assert len(fb) == 1


class TestBufferedFrame:
    """BufferedFrame 数据结构测试"""

    def test_buffered_frame_creation(self):
        """BufferedFrame 创建应正确"""
        frame_data = {'timestamp': 1.5, 'image': None}
        bf = BufferedFrame(frame_data=frame_data, timestamp=1.5)
        
        assert bf.frame_data == frame_data
        assert bf.timestamp == 1.5
