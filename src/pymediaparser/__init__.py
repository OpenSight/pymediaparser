"""pymediaparser - 人物检测与视觉语言模型工具包

包含：
- vlm_base: VLM 配置/结果数据类与抽象接口
- vlm: 多后端 VLM 推理子包（Qwen2-VL、Qwen3-VL、OpenAI API 等）
- stream_reader: RTMP/HTTP-FLV/HTTP-TS 实时流接入
- file_reader: 视频/图片文件读取
- frame_sampler: 按频率抽帧
- result_handler: 结果输出处理器
- live_pipeline: 实时流 VLM 分析 Pipeline
- replay_pipeline: 文件回放 VLM 分析 Pipeline

使用示例::

    # 方式 1：CLI 命令行直接运行
    # 实时流模式
    # python scripts/run_parser.py --url rtmp://host/live/stream --fps 1
    # 文件回放模式
    # python scripts/run_parser.py --mode replay --url /path/to/video.mp4

    # 方式 2：Python API（工厂模式）
    from pymediaparser import LivePipeline, StreamConfig, VLMConfig, create_vlm_client

    stream_cfg = StreamConfig(url="rtmp://host/live/stream", target_fps=1.0)
    vlm_client = create_vlm_client("qwen2", VLMConfig(device="cuda:0"))
    pipeline = LivePipeline(stream_cfg, vlm_client)
    pipeline.run()

    # 方式 3：文件回放
    from pymediaparser import ReplayPipeline, StreamConfig, create_vlm_client

    cfg = StreamConfig(url="/path/to/video.mp4", target_fps=1.0)
    vlm_client = create_vlm_client("openai_api", api_cfg)
    pipeline = ReplayPipeline(cfg, vlm_client)
    pipeline.run()
"""

# 延迟导入：避免 python -m pymediaparser.live_pipeline 时的 RuntimeWarning
# 用户通过 from pymediaparser import XXX 时会触发 __getattr__ 按需加载。

__all__ = [
    # 配置与数据类
    'StreamConfig',
    'VLMConfig',
    'VLMResult',
    'FrameResult',
    # Pipeline 状态与进度
    'PipelineState',
    'PipelineProgress',
    # VLM 客户端
    'VLMClient',
    'Qwen2VLClient',
    'Qwen3VLClient',
    'Qwen35Client',
    'OpenAIAPIClient',
    # VLM 配置
    'LocalVLMConfig',
    'APIVLMConfig',
    # VLM 工厂
    'create_vlm_client',
    'register_vlm_backend',
    'list_backends',
    # 流与采样
    'StreamReader',
    'FileReader',
    'FrameSampler',
    'SmartSampler',
    'SimpleSmartSampler',
    'MLSmartSampler',
    'MotionDetector',
    'ChangeAnalyzer',
    'ForegroundExtractor',
    # 帧缓存
    'FrameBuffer',
    # 结果处理
    'ResultHandler',
    'ConsoleResultHandler',
    'HttpCallbackHandler',
    # Pipeline（统一入口，支持传统/智能双模式）
    'LivePipeline',
    'ReplayPipeline',
]


def __getattr__(name: str):
    """延迟导入：按需加载子模块符号。"""
    _import_map = {
        'StreamConfig':          ('.vlm_base', 'StreamConfig'),
        'VLMConfig':             ('.vlm_base', 'VLMConfig'),
        'VLMResult':             ('.vlm_base', 'VLMResult'),
        'FrameResult':           ('.vlm_base', 'FrameResult'),
        'VLMClient':             ('.vlm_base', 'VLMClient'),
        'StreamReader':          ('.stream_reader', 'StreamReader'),
        'FileReader':            ('.file_reader', 'FileReader'),
        'FrameSampler':          ('.frame_sampler', 'FrameSampler'),
        # Pipeline 状态与进度
        'PipelineState':         ('.pipeline_base', 'PipelineState'),
        'PipelineProgress':      ('.pipeline_base', 'PipelineProgress'),
        # VLM 客户端（从 vlm 子包加载）
        'Qwen2VLClient':         ('.vlm.qwen2', 'Qwen2VLClient'),
        'Qwen3VLClient':         ('.vlm.qwen3', 'Qwen3VLClient'),
        'Qwen35Client':          ('.vlm.qwen35', 'Qwen35Client'),
        'OpenAIAPIClient':       ('.vlm.openai_api', 'OpenAIAPIClient'),
        # VLM 配置
        'LocalVLMConfig':        ('.vlm.configs', 'LocalVLMConfig'),
        'APIVLMConfig':          ('.vlm.configs', 'APIVLMConfig'),
        # VLM 工厂
        'create_vlm_client':     ('.vlm.factory', 'create_vlm_client'),
        'register_vlm_backend':  ('.vlm.factory', 'register_vlm_backend'),
        'list_backends':         ('.vlm.factory', 'list_backends'),
        # 结果处理
        'ResultHandler':         ('.result_handler', 'ResultHandler'),
        'ConsoleResultHandler':  ('.result_handler', 'ConsoleResultHandler'),
        'HttpCallbackHandler':   ('.result_handler', 'HttpCallbackHandler'),
        'LivePipeline':          ('.live_pipeline', 'LivePipeline'),
        'ReplayPipeline':        ('.replay_pipeline', 'ReplayPipeline'),
        # 智能采样相关
        'SmartSampler':          ('.smart_sampler.base', 'SmartSampler'),
        'SimpleSmartSampler':    ('.smart_sampler.simple_smart_sampler', 'SimpleSmartSampler'),
        'MLSmartSampler':        ('.smart_sampler.ml_smart_sampler', 'MLSmartSampler'),
        'MotionDetector':        ('.smart_sampler.motion_detector', 'MotionDetector'),
        'ChangeAnalyzer':        ('.smart_sampler.change_analyzer', 'ChangeAnalyzer'),
        'ForegroundExtractor':   ('.smart_sampler.foreground_extractor', 'ForegroundExtractor'),
        # 帧缓存
        'FrameBuffer':           ('.frame_buffer', 'FrameBuffer'),
    }
    if name in _import_map:
        module_path, attr = _import_map[name]
        import importlib
        module = importlib.import_module(module_path, __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
