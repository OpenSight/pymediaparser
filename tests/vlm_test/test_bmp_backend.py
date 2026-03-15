"""BMP虚拟VLM后端测试脚本。

测试BMPVLMClient的基本功能：
- 单帧保存（文件名包含帧序号和时间戳）
- 批量保存
- 断点续编
- 返回格式验证
"""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from PIL import Image

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pymediaparser.vlm import create_vlm_client, list_backends
from pymediaparser.vlm_base import VLMConfig


def create_test_image(width: int = 640, height: int = 480, color: tuple = (255, 0, 0)) -> Image.Image:
    """创建测试用纯色图像。"""
    return Image.new("RGB", (width, height), color)


def create_test_frame(
    image: Image.Image,
    frame_index: int = 0,
    timestamp: float = 0.0,
) -> dict:
    """创建测试用帧信息字典。"""
    return {
        'image': image,
        'frame_index': frame_index,
        'timestamp': timestamp,
        'significant': True,
        'source': ['test'],
    }


def test_bmp_backend():
    """测试BMP后端基本功能。"""
    # 使用临时目录
    output_dir = tempfile.mkdtemp(prefix="bmp_test_")
    print(f"输出目录: {output_dir}")

    try:
        # 验证后端已注册
        backends = list_backends()
        assert "bmp" in backends, f"bmp后端未注册，可用后端: {backends}"
        print(f"✓ bmp后端已注册，可用后端: {backends}")

        # 创建客户端
        config = VLMConfig(model_path=output_dir)
        client = create_vlm_client("bmp", config)
        print(f"✓ 创建BMP客户端成功")

        with client:
            # 测试1：单帧保存
            print("\n--- 测试1: 单帧保存 ---")
            img1 = create_test_image(800, 600, (255, 0, 0))
            frame1 = create_test_frame(img1, frame_index=0, timestamp=0.0)
            result1 = client.analyze(frame1, "测试帧1")

            # 验证返回格式
            data1 = json.loads(result1.text)
            assert "files" in data1, "返回结果缺少files字段"
            assert len(data1["files"]) == 1, "单帧应返回1个文件"
            assert os.path.exists(data1["files"][0]), f"文件不存在: {data1['files'][0]}"
            # 验证文件名格式：frame_000000_t0.00s.bmp
            assert "frame_000000_t0.00s.bmp" in data1["files"][0], f"文件名格式错误: {data1['files'][0]}"
            print(f"✓ 单帧保存成功: {data1['files'][0]}")
            print(f"  meta: {result1.meta}")

            # 测试2：批量保存
            print("\n--- 测试2: 批量保存 ---")
            frames = [
                create_test_frame(create_test_image(640, 480, (0, 255, 0)), frame_index=1, timestamp=1.5),
                create_test_frame(create_test_image(640, 480, (0, 0, 255)), frame_index=2, timestamp=3.0),
                create_test_frame(create_test_image(640, 480, (255, 255, 0)), frame_index=3, timestamp=4.5),
            ]
            result2 = client.analyze_batch(frames, "批量测试")

            data2 = json.loads(result2.text)
            assert len(data2["files"]) == 3, f"批量应返回3个文件，实际: {len(data2['files'])}"
            for fp in data2["files"]:
                assert os.path.exists(fp), f"文件不存在: {fp}"
            print(f"✓ 批量保存成功: {len(data2['files'])}个文件")
            for fp in data2["files"]:
                print(f"  - {fp}")

            # 验证文件名包含时间戳
            assert "frame_000001_t1.50s.bmp" in data2["files"][0], f"文件名格式错误: {data2['files'][0]}"
            assert "frame_000002_t3.00s.bmp" in data2["files"][1], f"文件名格式错误: {data2['files'][1]}"
            assert "frame_000003_t4.50s.bmp" in data2["files"][2], f"文件名格式错误: {data2['files'][2]}"
            print("✓ 文件名包含正确的帧序号和时间戳")

            # 测试3：验证文件数量
            print("\n--- 测试3: 验证文件数量 ---")
            saved_files = sorted(os.listdir(output_dir))
            print(f"已保存文件: {saved_files}")
            assert len(saved_files) == 4, f"应有4个文件，实际: {len(saved_files)}"
            print("✓ 文件数量正确")

        # 测试4：断点续编（使用更大的帧序号）
        print("\n--- 测试4: 断点续编 ---")
        with client:
            img4 = create_test_image(320, 240, (128, 128, 128))
            frame4 = create_test_frame(img4, frame_index=100, timestamp=10.0)
            result4 = client.analyze(frame4, "续编测试")

            data4 = json.loads(result4.text)
            # 文件名应使用帧信息中的 frame_index
            assert "frame_000100_t10.00s.bmp" in data4["files"][0], f"断点续编错误: {data4['files'][0]}"
            print(f"✓ 断点续编成功: {data4['files'][0]}")

        # 测试5：验证supports_batch
        print("\n--- 测试5: 验证supports_batch ---")
        assert client.supports_batch() == True, "BMP后端应支持批量处理"
        print("✓ supports_batch返回True")

        # 测试6：验证元数据
        print("\n--- 测试6: 验证元数据 ---")
        assert result1.meta.get('frame_index') == 0, "元数据缺少frame_index"
        assert result1.meta.get('timestamp') == 0.0, "元数据缺少timestamp"
        assert result2.meta.get('timestamps') == [1.5, 3.0, 4.5], "元数据timestamps错误"
        print("✓ 元数据正确")

        print("\n" + "=" * 50)
        print("所有测试通过！")

    finally:
        # 清理临时目录
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"\n已清理临时目录: {output_dir}")


if __name__ == "__main__":
    test_bmp_backend()
