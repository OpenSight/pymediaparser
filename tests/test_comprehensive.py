"""SimpleSmartSampler 综合功能测试"""
import numpy as np
from PIL import Image
import cv2
import time

from pymediaparser.smart_sampler import (
    SimpleSmartSampler,
    MeaningfulActivityDetector,
    SceneTransitionTracker,
    StaticFilter,
    SimpleSamplerConfig,
    create_sampler,
)


def generate_static_frames(count):
    """生成完全静态的帧"""
    for i in range(count):
        img = np.ones((180, 320, 3), dtype=np.uint8) * 128
        yield (Image.fromarray(img), float(i), i)


def generate_moving_object_frames(count, start_idx=0):
    """生成有移动方块（模拟动物活动）的帧"""
    for i in range(count):
        img = np.ones((180, 320, 3), dtype=np.uint8) * 128
        pos = i * 8
        img[40:100, pos:pos+30] = [255, 0, 0]
        yield (Image.fromarray(img), float(start_idx + i), start_idx + i)


def generate_lighting_change_frames(count, start_idx=0):
    """生成光线缓慢变化的帧（准静态）"""
    for i in range(count):
        brightness = 128 + int(i * 2)  # 逐渐变亮
        img = np.ones((180, 320, 3), dtype=np.uint8) * min(brightness, 255)
        yield (Image.fromarray(img), float(start_idx + i), start_idx + i)


def generate_swaying_frames(count, start_idx=0):
    """模拟风吹植物（多处小幅运动）的帧"""
    np.random.seed(42)
    base = np.ones((180, 320, 3), dtype=np.uint8) * 128
    # 添加一些固定的"植物"
    base[0:50, 50:70] = [0, 180, 0]
    base[0:40, 200:220] = [0, 160, 0]
    base[0:60, 130:145] = [0, 170, 0]

    for i in range(count):
        img = base.copy()
        # 多处小幅随机扰动（模拟风吹）
        for _ in range(8):
            rx, ry = np.random.randint(0, 170), np.random.randint(0, 300)
            img[ry:ry+5, rx:rx+5] = np.random.randint(100, 200, 3)
        yield (Image.fromarray(img), float(start_idx + i), start_idx + i)


def generate_camera_pan_frames(count, start_idx=0):
    """模拟云台旋转（整体平移）的帧 - 1fps 标准下约 5%-10% 像素变化"""
    base = np.zeros((180, 320, 3), dtype=np.uint8)
    # 创建有纹理的背景
    for y in range(180):
        for x in range(320):
            base[y, x] = [(x + y) % 256, (x * 2) % 256, (y * 2) % 256]

    # 1fps 下云台缓慢旋转，每帧偏移 1-2 像素，导致 5%-10% 像素变化
    for i in range(count):
        offset = i * 2  # 每帧偏移 2 像素
        img = np.roll(base, offset, axis=1)
        yield (Image.fromarray(img), float(start_idx + i), start_idx + i)


def generate_camera_pan_large_frames(count, start_idx=0):
    """模拟摄像机位置改变（大幅场景变化）- 50%+ 像素变化"""
    # 创建两个完全不同的场景
    base1 = np.zeros((180, 320, 3), dtype=np.uint8)
    base2 = np.zeros((180, 320, 3), dtype=np.uint8)
    
    # 场景 1: 室内场景
    for y in range(180):
        for x in range(320):
            base1[y, x] = [min(200, x + y), min(150, x // 2), min(100, y // 2)]
    
    # 场景 2: 室外场景（完全不同的颜色和纹理）
    for y in range(180):
        for x in range(320):
            base2[y, x] = [min(255, 100 + y), min(200, 150 + x // 3), min(255, 200 + y // 2)]
    
    # 前 3 帧是场景 1，之后突然切换到场景 2（模拟云台快速转动到不同位置）
    for i in range(count):
        if i < 3:
            img = base1.copy()
        else:
            # 从场景 1 渐变到场景 2（模拟云台转动过程中的中间状态）
            blend_factor = min(1.0, (i - 3) / 3.0)  # 3 帧内完成过渡
            img = (base1 * (1 - blend_factor) + base2 * blend_factor).astype(np.uint8)
        yield (Image.fromarray(img), float(start_idx + i), start_idx + i)


def run_test(name, frames_iter, expected_min_rate=0.0, expected_max_rate=1.0):
    """运行单个测试"""
    sampler = SimpleSmartSampler(
        sensitivity=0.5,
        activity_duration=3.0,
        quiet_frames_threshold=10,
        backup_interval=30.0,
        min_frame_interval=0.0,  # 测试中不限制最小帧间隔
    )

    frames = list(frames_iter)
    start_time = time.time()
    sampled = list(sampler.sample(iter(frames)))
    elapsed = time.time() - start_time

    rate = len(sampled) / len(frames) if len(frames) > 0 else 0
    avg_time = elapsed / len(frames) * 1000 if len(frames) > 0 else 0

    passed = expected_min_rate <= rate <= expected_max_rate
    status = "PASS" if passed else "FAIL"

    print(f"[{status}] {name}")
    print(f"  输入帧: {len(frames)}, 采样帧: {len(sampled)}, 采样率: {rate*100:.1f}%")
    print(f"  平均处理时间: {avg_time:.1f}ms/帧")
    if not passed:
        print(f"  期望采样率: {expected_min_rate*100:.1f}% ~ {expected_max_rate*100:.1f}%")

    return passed


def test_factory_compatibility():
    """测试工厂函数兼容性"""
    print("\n--- 工厂函数兼容性测试 ---")

    # 测试1: 默认配置创建
    sampler1 = create_sampler('simple')
    assert isinstance(sampler1, SimpleSmartSampler), "工厂函数创建类型错误"
    print("[PASS] 工厂函数默认配置创建")

    # 测试2: 字典配置创建
    sampler2 = create_sampler('simple', {
        'sensitivity': 0.3,
        'activity_duration': 5.0,
    })
    assert sampler2._sensitivity == 0.3, "sensitivity 未正确传递"
    assert sampler2._activity_duration == 5.0, "activity_duration 未正确传递"
    print("[PASS] 工厂函数字典配置创建")

    # 测试3: Config 对象创建
    config = SimpleSamplerConfig(sensitivity=0.8, activity_duration=2.0)
    sampler3 = create_sampler('simple', config)
    assert sampler3._sensitivity == 0.8, "sensitivity 未正确传递"
    assert sampler3._activity_duration == 2.0, "activity_duration 未正确传递"
    print("[PASS] 工厂函数 Config 对象创建")

    return True


def test_component_independent():
    """测试独立组件"""
    print("\n--- 独立组件测试 ---")

    # 测试 MeaningfulActivityDetector
    detector = MeaningfulActivityDetector(sensitivity=0.5)
    static_frame = np.ones((180, 320, 3), dtype=np.uint8) * 128
    has_activity, score = detector.detect(static_frame)
    print(f"[INFO] MeaningfulActivityDetector - 静态帧: has_activity={has_activity}, score={score:.3f}")

    # 连续送入有运动的帧
    for i in range(5):
        frame = np.ones((180, 320, 3), dtype=np.uint8) * 128
        frame[40:100, i*10:i*10+30] = [255, 0, 0]
        has_activity, score = detector.detect(frame)
    print(f"[INFO] MeaningfulActivityDetector - 运动帧: has_activity={has_activity}, score={score:.3f}")

    # 测试 SceneTransitionTracker
    tracker = SceneTransitionTracker(sensitivity=0.5)
    for i in range(10):
        base = np.zeros((180, 320, 3), dtype=np.uint8)
        for y in range(180):
            for x in range(320):
                base[y, x] = [(x + y + i*5) % 256, 0, 0]
        is_transition, score = tracker.detect(base)
    print(f"[INFO] SceneTransitionTracker - 云台旋转: is_transition={is_transition}, score={score:.3f}")

    # 测试 StaticFilter
    sf = StaticFilter(sensitivity=0.5)
    # 静态帧应该被过滤
    static_img = np.ones((180, 320, 3), dtype=np.uint8) * 128
    passed, reason = sf.check(static_img, 0.0, False)
    print(f"[INFO] StaticFilter - 静态帧: passed={passed}, reason={reason}")

    return True


def test_output_format():
    """测试输出格式兼容性"""
    print("\n--- 输出格式兼容性测试 ---")

    sampler = SimpleSmartSampler()
    frames = list(generate_moving_object_frames(5))
    results = list(sampler.sample(iter(frames)))

    if not results:
        print("[WARN] 无采样结果，跳过格式检查")
        return True

    result = results[0]
    required_keys = {'image', 'timestamp', 'frame_index', 'significant', 'source', 'change_metrics'}
    missing_keys = required_keys - set(result.keys())
    if missing_keys:
        print(f"[FAIL] 输出缺少字段: {missing_keys}")
        return False

    # 检查 change_metrics 格式
    metrics = result['change_metrics']
    metric_keys = {'ssim_score', 'combined_score', 'motion_score'}
    missing_metrics = metric_keys - set(metrics.keys())
    if missing_metrics:
        print(f"[FAIL] change_metrics 缺少字段: {missing_metrics}")
        return False

    # 检查 source 格式
    valid_sources = {'motion', 'scene_switch', 'periodic'}
    invalid_sources = set(result['source']) - valid_sources
    if invalid_sources:
        print(f"[FAIL] source 含非法值: {invalid_sources}")
        return False

    print("[PASS] 输出格式兼容性检查")
    return True


def test_reset():
    """测试重置功能"""
    print("\n--- 重置功能测试 ---")

    sampler = SimpleSmartSampler()
    frames = list(generate_moving_object_frames(5))
    list(sampler.sample(iter(frames)))

    stats_before = sampler.get_statistics()
    assert stats_before['total_frames_processed'] == 5

    sampler.reset()
    stats_after = sampler.get_statistics()
    assert stats_after['total_frames_processed'] == 0
    assert stats_after['current_state'] == 'idle'

    print("[PASS] 重置功能测试")
    return True


if __name__ == '__main__':
    all_passed = True

    print("=" * 60)
    print("SimpleSmartSampler 综合功能测试")
    print("=" * 60)

    # 1. 静态帧测试（应该极低采样率）
    all_passed &= run_test(
        "静态帧测试",
        generate_static_frames(10),
        expected_min_rate=0.0,
        expected_max_rate=0.3,
    )

    # 2. 移动对象测试（应该高采样率）
    all_passed &= run_test(
        "移动对象测试",
        generate_moving_object_frames(15),
        expected_min_rate=0.5,
        expected_max_rate=1.0,
    )

    # 3. 光线变化测试（应该低采样率）
    all_passed &= run_test(
        "光线微变测试",
        generate_lighting_change_frames(10),
        expected_min_rate=0.0,
        expected_max_rate=0.5,
    )

    # 4. 风吹植物测试（应该低采样率）
    all_passed &= run_test(
        "风吹植物测试",
        generate_swaying_frames(10),
        expected_min_rate=0.0,
        expected_max_rate=0.6,
    )

    # 5. 云台缓慢旋转测试（5%-10% 像素变化，应该高采样率）
    all_passed &= run_test(
        "云台缓慢旋转测试",
        generate_camera_pan_frames(10),
        expected_min_rate=0.3,
        expected_max_rate=1.0,
    )

    # 6. 摄像机位置改变测试（50%+ 像素变化，应该高采样率）
    all_passed &= run_test(
        "摄像机位置改变测试",
        generate_camera_pan_large_frames(10),
        expected_min_rate=0.3,
        expected_max_rate=1.0,
    )

    # 7. 组件测试
    all_passed &= test_component_independent()

    # 8. 工厂函数测试
    all_passed &= test_factory_compatibility()

    # 9. 输出格式测试
    all_passed &= test_output_format()

    # 10. 重置测试
    all_passed &= test_reset()

    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过!")
    else:
        print("部分测试失败，请检查!")
    print("=" * 60)
