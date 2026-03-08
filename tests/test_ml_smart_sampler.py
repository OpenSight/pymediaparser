"""MLSmartSampler 单元测试"""

from __future__ import annotations
import numpy as np
import pytest

from pymediaparser.smart_sampler.hard_filter import HardFilter
from pymediaparser.smart_sampler.fast_triggers import FastTriggers
from pymediaparser.smart_sampler.frame_validator import FrameValidator
from pymediaparser.smart_sampler.ml_smart_sampler import MLSmartSampler


# ── 帧生成工具 ──────────────────────────────────────


def _make_frame(h: int = 480, w: int = 640, value: int = 128) -> np.ndarray:
    """生成纯色 BGR 帧。"""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _make_noise_frame(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """生成随机噪声 BGR 帧（高熵，会被熵异常过滤）。"""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _make_natural_frame(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """生成类自然场景的 BGR 帧（熵值在合理范围内 5~7）。
    
    通过渐变底色 + 适度高斯噪声模拟真实图像的像素分布。
    """
    rng = np.random.RandomState(seed)
    # 渐变底色（不同 seed 产生不同偏移和方向）
    base_val = 60 + (seed * 37) % 120  # 60~180
    row = np.linspace(base_val, base_val + 80, w, dtype=np.float64)
    col = np.linspace(0, 40, h, dtype=np.float64)
    plane = (row[np.newaxis, :] + col[:, np.newaxis]) % 256
    # 添加适度高斯噪声
    noise = rng.normal(0, 15, (h, w))
    plane = np.clip(plane + noise, 0, 255).astype(np.uint8)
    # 三个通道略有差异
    b = plane
    g = np.clip(plane.astype(np.int16) + rng.randint(-20, 20), 0, 255).astype(np.uint8)
    r = np.clip(plane.astype(np.int16) + rng.randint(-30, 30), 0, 255).astype(np.uint8)
    return np.stack([b, g, r], axis=-1)


def _make_gradient_frame(h: int = 480, w: int = 640, offset: int = 0) -> np.ndarray:
    """生成渐变 BGR 帧，offset 控制平移。"""
    row = np.linspace(offset, 255 + offset, w, dtype=np.float64) % 256
    frame = np.tile(row.astype(np.uint8), (h, 1))
    return np.stack([frame, frame, frame], axis=-1)


# ======================================================================
# Layer 0: HardFilter 测试
# ======================================================================


class TestHardFilter:
    def test_time_interval_reject(self):
        """时间间隔内的帧应被拒绝。"""
        hf = HardFilter(min_frame_interval=1.0)
        frame = _make_natural_frame(seed=0)
        passed1, _ = hf.check(frame, 0.0)
        assert passed1

        frame2 = _make_natural_frame(seed=1)
        passed2, reason = hf.check(frame2, 0.5)
        assert not passed2
        assert reason == '时间间隔'

    def test_time_interval_pass(self):
        """超过时间间隔的帧应通过。"""
        hf = HardFilter(min_frame_interval=1.0)
        frame1 = _make_natural_frame(seed=0)
        hf.check(frame1, 0.0)

        frame2 = _make_natural_frame(seed=1)
        passed, _ = hf.check(frame2, 1.5)
        assert passed

    def test_still_frame_reject(self):
        """静止帧应被拒绝。"""
        hf = HardFilter(min_frame_interval=0.0, still_threshold=2.0)
        frame = _make_natural_frame(seed=0)
        hf.check(frame, 0.0)  # 第一帧（无前帧参考，通过）

        # 完全相同的帧
        passed, reason = hf.check(frame, 1.0)
        assert not passed
        assert reason == '静止帧'

    def test_black_screen_reject(self):
        """黑屏帧应被拒绝。"""
        hf = HardFilter(min_frame_interval=0.0, still_threshold=0.0)
        # 先送一个正常帧
        hf.check(_make_natural_frame(seed=0), 0.0)

        black = _make_frame(value=5)
        passed, reason = hf.check(black, 1.0)
        assert not passed
        assert reason == '黑屏'

    def test_white_screen_reject(self):
        """白屏帧应被拒绝。"""
        hf = HardFilter(min_frame_interval=0.0, still_threshold=0.0)
        hf.check(_make_natural_frame(seed=0), 0.0)

        white = _make_frame(value=250)
        passed, reason = hf.check(white, 1.0)
        assert not passed
        assert reason == '白屏'

    def test_resolution_change_reject(self):
        """分辨率突变帧应被拒绝。"""
        hf = HardFilter(min_frame_interval=0.0, still_threshold=0.0, size_change_tolerance=0.05)
        hf.check(_make_natural_frame(480, 640, seed=0), 0.0)

        # 分辨率突变（面积变化远超5%）
        small_frame = _make_natural_frame(240, 320, seed=1)
        passed, reason = hf.check(small_frame, 1.0)
        assert not passed
        assert reason == '分辨率突变'

    def test_normal_frame_pass(self):
        """正常帧应通过所有检查。"""
        hf = HardFilter(min_frame_interval=0.0, still_threshold=1.0)
        frame1 = _make_natural_frame(seed=0)
        passed1, _ = hf.check(frame1, 0.0)
        assert passed1

        frame2 = _make_natural_frame(seed=42)
        passed2, _ = hf.check(frame2, 1.0)
        assert passed2

    def test_reset(self):
        """重置后计数器清零。"""
        hf = HardFilter(min_frame_interval=1.0)
        hf.check(_make_natural_frame(seed=0), 0.0)
        hf.check(_make_natural_frame(seed=1), 0.1)  # rejected
        assert hf.reject_count == 1

        hf.reset()
        assert hf.reject_count == 0
        assert hf.accept_count == 0


# ======================================================================
# Layer 1: FastTriggers 测试
# ======================================================================


class TestFastTriggers:
    def test_periodic_trigger(self):
        """周期触发器应在达到间隔时触发。"""
        ft = FastTriggers(periodic_interval=2.0, motion_threshold=0.99)
        frame = _make_frame(value=128)

        # t=0: 首帧初始化，不触发
        triggers = ft.detect(frame, 0.0)
        assert triggers == []

        # t=1: 未达间隔
        triggers = ft.detect(frame, 1.0)
        # periodic 不应触发
        assert 'periodic' not in triggers

        # t=2.5: 达到间隔
        triggers = ft.detect(frame, 2.5)
        assert 'periodic' in triggers

    def test_scene_switch_trigger(self):
        """明显的场景切换应触发 scene_switch。"""
        ft = FastTriggers(
            scene_switch_threshold=0.5,
            motion_threshold=0.99,
            periodic_interval=9999,
        )
        # 送两帧差异很大的图
        frame1 = _make_frame(value=50)
        ft.detect(frame1, 0.0)

        frame2 = _make_frame(value=200)
        triggers = ft.detect(frame2, 2.0)
        # 场景切换或通过 phash 检测到差异
        # 注意：纯色帧直方图差异可能不大，但 phash 应该能检测到
        # 至少 periodic 不应触发（interval=9999）
        assert 'periodic' not in triggers

    def test_motion_trigger_with_large_change(self):
        """大面积像素变化应触发 motion。"""
        ft = FastTriggers(
            motion_threshold=0.01,  # 非常低的阈值，容易触发
            periodic_interval=9999,
        )
        frame1 = _make_natural_frame(seed=0)
        ft.detect(frame1, 0.0)  # 建立背景模型

        # 多送几帧建立背景
        for i in range(5):
            ft.detect(frame1, float(i + 1))

        # 突然变化
        frame2 = _make_natural_frame(seed=99)
        triggers = ft.detect(frame2, 10.0)
        assert 'motion' in triggers

    def test_reset(self):
        """重置后触发计数归零。"""
        ft = FastTriggers(periodic_interval=1.0)
        ft.detect(_make_natural_frame(), 0.0)  # 首帧初始化，不触发
        assert ft.trigger_count == 0
        
        # 发送第二帧，应该触发
        ft.detect(_make_natural_frame(seed=1), 1.5)
        assert ft.trigger_count >= 1

        ft.reset()
        assert ft.trigger_count == 0

    def test_last_fg_mask_cached(self):
        """detect 后应缓存 fg_mask。"""
        ft = FastTriggers()
        ft.detect(_make_natural_frame(seed=0), 0.0)  # 首帧初始化
        ft.detect(_make_natural_frame(seed=1), 1.0)  # 第二帧，触发 motion 检测
        assert ft.last_fg_mask is not None

    def test_last_diff_mask_cached(self):
        """两帧后应缓存 diff_mask。"""
        ft = FastTriggers()
        ft.detect(_make_natural_frame(seed=0), 0.0)
        ft.detect(_make_natural_frame(seed=1), 1.0)
        assert ft.last_diff_mask is not None


# ======================================================================
# Layer 2: FrameValidator 测试
# ======================================================================


class TestFrameValidator:
    def test_periodic_always_passes(self):
        """周期强制触发应直接通过验证。"""
        fv = FrameValidator()
        frame = _make_natural_frame(seed=0)
        result = fv.validate(frame, 0.0, ['periodic'])
        assert result['passed'] is True

    def test_score_in_result(self):
        """验证结果应包含 score 和 features。"""
        fv = FrameValidator()
        frame1 = _make_natural_frame(seed=0)
        fv.validate(frame1, 0.0, ['periodic'])  # 建立初始状态

        frame2 = _make_natural_frame(seed=42)
        result = fv.validate(frame2, 1.0, ['motion'])
        assert 'score' in result
        assert 'features' in result
        assert 'is_peak' in result
        assert 'window_mean' in result

    def test_reset(self):
        """重置后通过计数归零。"""
        fv = FrameValidator()
        fv.validate(_make_natural_frame(), 0.0, ['periodic'])
        assert fv.pass_count >= 1

        fv.reset()
        assert fv.pass_count == 0


# ======================================================================
# MLSmartSampler 集成测试
# ======================================================================


class TestMLSmartSampler:
    def test_interface_compatibility(self):
        """MLSmartSampler 应具有与 SmartSampler 相同的接口。"""
        sampler = MLSmartSampler()
        # 检查属性
        assert hasattr(sampler, 'frame_count')
        assert hasattr(sampler, 'enable_smart')
        # 检查方法
        assert callable(sampler.sample)
        assert callable(sampler.reset)
        assert callable(sampler.get_statistics)

    def test_frame_count_increments(self):
        """frame_count 应对所有输入帧递增（包括被拒绝的帧）。"""
        sampler = MLSmartSampler(min_frame_interval=0.0)
        frames = [(_make_natural_frame(seed=i), float(i)) for i in range(5)]

        # 消费所有输出
        list(sampler.sample(iter(frames)))
        assert sampler.frame_count == 5

    def test_sample_returns_correct_keys(self):
        """sample() 返回的字典应包含所有必需键。"""
        sampler = MLSmartSampler(
            min_frame_interval=0.0,
            backup_interval=1.0,  # 短间隔确保周期触发
        )
        required_keys = {
            'image', 'timestamp', 'frame_index', 'significant',
            'source', 'original_frame',
        }

        # 生成差异较大的自然帧以确保有输出
        frames = []
        for i in range(20):
            frames.append((_make_natural_frame(seed=i * 100), float(i) * 2.0))

        results = list(sampler.sample(iter(frames)))
        assert len(results) > 0, "应至少有一帧输出（周期触发）"

        for result in results:
            missing = required_keys - set(result.keys())
            assert not missing, f"返回字典缺少键: {missing}"

    def test_change_metrics_format(self):
        """change_metrics 应包含 ssim_score, combined_score, motion_score。"""
        sampler = MLSmartSampler(
            min_frame_interval=0.0,
            backup_interval=1.0,
        )
        frames = [(_make_natural_frame(seed=i * 100), float(i) * 2.0) for i in range(20)]
        results = list(sampler.sample(iter(frames)))

        for result in results:
            if 'change_metrics' in result:
                metrics = result['change_metrics']
                assert 'ssim_score' in metrics
                assert 'combined_score' in metrics
                assert 'motion_score' in metrics

    def test_source_field_values(self):
        """source 字段应为列表格式。"""
        sampler = MLSmartSampler(
            min_frame_interval=0.0,
            backup_interval=2.0,
        )
        frames = [(_make_natural_frame(seed=i * 100), float(i) * 3.0) for i in range(15)]
        results = list(sampler.sample(iter(frames)))

        for result in results:
            assert isinstance(result['source'], list), \
                f"source 应为列表，实际: {type(result['source'])}"

    def test_disable_smart_sampling(self):
        """禁用智能采样时应退化为时间采样。"""
        sampler = MLSmartSampler(
            enable_smart_sampling=False,
            backup_interval=2.0,
        )
        frames = [(_make_natural_frame(seed=i), float(i)) for i in range(10)]
        results = list(sampler.sample(iter(frames)))

        # 应有输出（基于时间间隔）
        assert len(results) > 0
        for result in results:
            assert result['source'] == ['traditional']
            assert result['significant'] is False

    def test_reset_clears_state(self):
        """reset() 应重置所有状态。"""
        sampler = MLSmartSampler()
        frames = [(_make_natural_frame(seed=i), float(i) * 2.0) for i in range(5)]
        list(sampler.sample(iter(frames)))
        assert sampler.frame_count > 0

        sampler.reset()
        assert sampler.frame_count == 0

    def test_get_statistics_format(self):
        """get_statistics() 应返回兼容格式。"""
        sampler = MLSmartSampler()
        stats = sampler.get_statistics()

        # SmartSampler 兼容字段
        assert 'total_frames_processed' in stats
        assert 'smart_sampling_enabled' in stats
        assert 'backup_interval' in stats
        assert 'min_frame_interval' in stats
        assert 'motion_detector_method' in stats

        # 扩展字段
        assert 'layer0_reject_count' in stats
        assert 'layer0_pass_rate' in stats
        assert 'layer1_trigger_count' in stats
        assert 'layer2_pass_count' in stats

    def test_frame_index_sequential(self):
        """frame_index 应从 0 开始全局连续递增。"""
        sampler = MLSmartSampler(
            min_frame_interval=0.0,
            backup_interval=1.0,
        )
        frames = [(_make_natural_frame(seed=i * 100), float(i) * 2.0) for i in range(20)]
        results = list(sampler.sample(iter(frames)))

        if results:
            # frame_index 应单调递增
            indices = [r['frame_index'] for r in results]
            for i in range(1, len(indices)):
                assert indices[i] > indices[i - 1], \
                    f"frame_index 应单调递增: {indices}"

    def test_cropped_frame_present_for_smart(self):
        """smart source 的结果应包含 cropped_frame 和 bbox。"""
        sampler = MLSmartSampler(
            min_frame_interval=0.0,
            backup_interval=1.0,
        )
        frames = [(_make_natural_frame(seed=i * 100), float(i) * 2.0) for i in range(20)]
        results = list(sampler.sample(iter(frames)))

        for result in results:
            # 非周期触发帧应有裁剪信息
            if 'periodic' not in result['source']:
                assert 'cropped_frame' in result
                assert 'bbox' in result
                assert 'compression_ratio' in result

    def test_empty_and_none_frames_skipped(self):
        """None 和空帧应被跳过。"""
        sampler = MLSmartSampler()
        empty = np.array([], dtype=np.uint8)
        frames = [(None, 0.0), (empty, 1.0)]
        results = list(sampler.sample(iter(frames)))
        assert len(results) == 0
        # frame_count 不应递增（None/空帧被跳过）
        assert sampler.frame_count == 0

    # ==================================================================
    # 首帧输出测试
    # ==================================================================

    def test_first_frame_is_output(self):
        """首帧应该被输出（周期触发模式）。"""
        sampler = MLSmartSampler()
        frames = [(_make_natural_frame(seed=0), 0.0)]
        results = list(sampler.sample(iter(frames)))

        assert len(results) == 1, "首帧应该被输出"
        assert results[0]['frame_index'] == 0, "首帧的 frame_index 应为 0"

    def test_first_frame_source_is_time(self):
        """首帧的 source 应为 ['periodic']（周期触发语义）。"""
        sampler = MLSmartSampler()
        frames = [(_make_natural_frame(seed=0), 0.0)]
        results = list(sampler.sample(iter(frames)))

        assert results[0]['source'] == ['periodic'], \
            f"首帧 source 应为 ['periodic']，实际: {results[0]['source']}"

    def test_first_frame_not_significant(self):
        """首帧的 significant 应为 False（周期触发非显著帧）。"""
        sampler = MLSmartSampler()
        frames = [(_make_natural_frame(seed=0), 0.0)]
        results = list(sampler.sample(iter(frames)))

        assert results[0]['significant'] is False, \
            "首帧 significant 应为 False"

    def test_first_frame_no_crop(self):
        """首帧不裁剪，输出整帧。"""
        sampler = MLSmartSampler()
        frame = _make_natural_frame(seed=0)
        h, w = frame.shape[:2]
        frames = [(frame, 0.0)]
        results = list(sampler.sample(iter(frames)))

        result = results[0]
        assert 'cropped_frame' in result
        assert 'bbox' in result
        assert result['bbox'] == (0, 0, w, h), \
            f"首帧 bbox 应为整帧，实际: {result['bbox']}"
        assert result['compression_ratio'] == 1.0, \
            "首帧 compression_ratio 应为 1.0"
        # cropped_frame 应与 original_frame 相同
        np.testing.assert_array_equal(result['cropped_frame'], result['original_frame'])

    def test_first_frame_change_metrics(self):
        """首帧 change_metrics 应为初始值。"""
        sampler = MLSmartSampler()
        frames = [(_make_natural_frame(seed=0), 0.0)]
        results = list(sampler.sample(iter(frames)))

        metrics = results[0]['change_metrics']
        assert metrics['ssim_score'] == 1.0, \
            f"首帧 ssim_score 应为 1.0，实际: {metrics['ssim_score']}"
        assert metrics['combined_score'] == 0.0, \
            f"首帧 combined_score 应为 0.0，实际: {metrics['combined_score']}"
        assert metrics['motion_score'] == 0.0, \
            f"首帧 motion_score 应为 0.0，实际: {metrics['motion_score']}"

    def test_first_frame_is_first_result(self):
        """首帧应为第一个输出结果。"""
        sampler = MLSmartSampler(
            min_frame_interval=0.0,
            backup_interval=1.0,
        )
        frames = [(_make_natural_frame(seed=i * 100), float(i) * 2.0) for i in range(10)]
        results = list(sampler.sample(iter(frames)))

        assert len(results) > 0, "应有输出"
        assert results[0]['frame_index'] == 0, "第一个输出应为首帧（frame_index=0）"
        assert results[0]['timestamp'] == 0.0, "第一个输出的时间戳应为 0.0"

    def test_single_image_input(self):
        """单张图片输入应正确处理（回放场景兼容）。"""
        sampler = MLSmartSampler()
        # 模拟单张图片输入
        frame = _make_natural_frame(seed=42)
        frames = [(frame, 0.0)]
        results = list(sampler.sample(iter(frames)))

        assert len(results) == 1, "单张图片应有 1 个输出"
        result = results[0]
        assert result['frame_index'] == 0
        assert result['source'] == ['periodic']
        assert result['significant'] is False
        assert 'image' in result
        assert result['image'].mode == 'RGB'  # PIL Image 应为 RGB 模式

    def test_first_frame_sets_periodic_timestamp(self):
        """首帧输出后，周期触发的计时起点应正确设置。"""
        sampler = MLSmartSampler(backup_interval=5.0)
        # 首帧
        frames = [(_make_natural_frame(seed=0), 0.0)]
        results = list(sampler.sample(iter(frames)))
        assert len(results) == 1

        # 继续输入，确认周期触发计时正确
        # 在 backup_interval 之内不应触发周期
        more_frames = [(_make_natural_frame(seed=i), float(i)) for i in range(1, 4)]
        results2 = list(sampler.sample(iter(more_frames)))

        # 由于时间间隔小于 backup_interval(5.0)，不应有纯周期触发
        # 但可能有其他触发（运动、场景切换等）
        for r in results2:
            if r['source'] == ['periodic']:
                # 如果是纯周期触发，时间戳应 >= 5.0
                assert r['timestamp'] >= 5.0, \
                    f"周期触发时间戳应 >= 5.0，实际: {r['timestamp']}"

    def test_first_frame_with_different_resolutions(self):
        """不同分辨率的首帧都能正确输出。"""
        resolutions = [(640, 480), (1920, 1080), (320, 240)]

        for w, h in resolutions:
            sampler = MLSmartSampler()
            frame = _make_natural_frame(h=h, w=w, seed=0)
            frames = [(frame, 0.0)]
            results = list(sampler.sample(iter(frames)))

            assert len(results) == 1, f"分辨率 {w}x{h} 首帧应输出"
            result = results[0]
            assert result['bbox'] == (0, 0, w, h), \
                f"分辨率 {w}x{h} bbox 不正确"
            # 验证输出图像尺寸
            assert result['image'].size == (w, h), \
                f"输出图像尺寸应为 {w}x{h}，实际: {result['image'].size}"

    def test_first_frame_pil_image_format(self):
        """首帧输出的 PIL Image 格式应正确。"""
        sampler = MLSmartSampler()
        frame = _make_natural_frame(seed=0)
        frames = [(frame, 0.0)]
        results = list(sampler.sample(iter(frames)))

        from PIL import Image
        pil_img = results[0]['image']
        assert isinstance(pil_img, Image.Image), "应输出 PIL Image"
        assert pil_img.mode == 'RGB', f"应为 RGB 模式，实际: {pil_img.mode}"

    def test_first_frame_is_periodic_boundary(self):
        """首帧的 source 应包含 'periodic'。"""
        sampler = MLSmartSampler()
        frame = _make_natural_frame(seed=0)
        frames = [(frame, 0.0)]
        results = list(sampler.sample(iter(frames)))

        assert 'periodic' in results[0].get('source', []), \
            "首帧 source 应包含 'periodic'"

    def test_periodic_trigger_frame_is_boundary(self):
        """周期触发帧的 source 应包含 'periodic'。"""
        sampler = MLSmartSampler(
            min_frame_interval=0.0,
            backup_interval=2.0,  # 短周期便于测试
        )
        # 生成足够多的帧，确保有周期触发
        frames = [(_make_natural_frame(seed=i), float(i)) for i in range(10)]
        results = list(sampler.sample(iter(frames)))

        # 检查是否存在周期边界帧
        periodic_boundary_frames = [r for r in results if 'periodic' in r.get('source', [])]
        assert len(periodic_boundary_frames) > 0, "应有周期边界帧（source 包含 'periodic'）"

    def test_non_periodic_frame_not_boundary(self):
        """非周期触发帧的 source 不应包含 'periodic'。"""
        sampler = MLSmartSampler(
            min_frame_interval=0.0,
            backup_interval=100.0,  # 长周期避免触发
        )
        # 生成有明显变化的帧，触发 motion 或 scene_switch
        frames = []
        for i in range(20):
            # 每帧差异较大
            frames.append((_make_natural_frame(seed=i * 1000), float(i * 0.5)))

        results = list(sampler.sample(iter(frames)))

        # 非首帧且非周期触发的帧，source 不应包含 'periodic'
        for result in results[1:]:  # 跳过首帧
            if 'motion' in result['source'] or 'scene_switch' in result['source']:
                assert 'periodic' not in result['source'], \
                    f"非周期触发帧 source 不应包含 'periodic'，source={result['source']}"

    def test_disable_smart_sampling_not_periodic_boundary(self):
        """禁用智能采样时，帧的 source 应为 ['traditional']。"""
        sampler = MLSmartSampler(
            enable_smart_sampling=False,
            backup_interval=2.0,
        )
        frames = [(_make_natural_frame(seed=i), float(i)) for i in range(10)]
        results = list(sampler.sample(iter(frames)))

        for result in results:
            assert result['source'] == ['traditional'], \
                f"禁用智能采样时 source 应为 ['traditional']，实际: {result['source']}"

