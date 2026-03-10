"""VLM 后端工厂：注册表 + 工厂函数。

通过字符串元组 (module_path, class_name) 实现延迟导入，
避免在 import 时加载 transformers 等重型依赖。

典型用法::

    from pymediaparser.vlm.factory import create_vlm_client
    from pymediaparser.vlm.configs import LocalVLMConfig

    client = create_vlm_client("qwen2", LocalVLMConfig(device="cuda:0"))
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Tuple, Type

from ..vlm_base import VLMClient

logger = logging.getLogger(__name__)

# 后端注册表：backend_name -> (module_path, class_name, install_hint)
# module_path 使用绝对路径格式，在 import 时解析
# install_hint 用于在缺少依赖时提示用户安装命令
_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    "qwen2": (
        "pymediaparser.vlm.qwen2",
        "Qwen2VLClient",
        "pip install 'pymediaparser[vlm-qwen]'",
    ),
    "qwen3": (
        "pymediaparser.vlm.qwen3",
        "Qwen3VLClient",
        "pip install 'pymediaparser[vlm-qwen]'",
    ),
    "qwen35": (
        "pymediaparser.vlm.qwen35",
        "Qwen35Client",
        "pip install 'pymediaparser[vlm-qwen]'",
    ),
    "openai_api": (
        "pymediaparser.vlm.openai_api",
        "OpenAIAPIClient",
        "pip install 'pymediaparser[vlm-api]'",
    ),
    "bmp": (
        "pymediaparser.vlm.bmp",
        "BMPVLMClient",
        "",  # BMP 后端无额外依赖
    ),
}


def register_vlm_backend(
    name: str,
    module_path: str,
    class_name: str,
    install_hint: str = "",
) -> None:
    """注册自定义 VLM 后端。

    Args:
        name: 后端名称（用于 create_vlm_client 的 backend 参数）。
        module_path: 模块的完整导入路径，如 "my_package.my_vlm"。
        class_name: 模块中的 VLMClient 子类名称。
        install_hint: 安装提示信息（可选），在缺少依赖时显示。

    Raises:
        ValueError: 如果 name 已被注册。

    Example::

        register_vlm_backend("my_vlm", "my_package.vlm_impl", "MyVLMClient")
        client = create_vlm_client("my_vlm", my_config)
    """
    if name in _REGISTRY:
        raise ValueError(
            f"后端 {name!r} 已注册为 {_REGISTRY[name]}，"
            f"不能重复注册"
        )
    _REGISTRY[name] = (module_path, class_name, install_hint)
    logger.info("注册自定义 VLM 后端: %s -> %s.%s", name, module_path, class_name)


def list_backends() -> List[str]:
    """列出所有已注册的后端名称。

    Returns:
        已注册的后端名称列表。
    """
    return list(_REGISTRY.keys())


def create_vlm_client(backend: str, config: Any = None) -> VLMClient:
    """根据后端名称创建 VLM 客户端实例。

    通过注册表延迟导入对应模块，避免未使用的后端依赖被加载。

    Args:
        backend: 后端名称，如 "qwen2"、"qwen3"、"openai_api"。
        config: 配置对象（LocalVLMConfig / APIVLMConfig / VLMConfig）。

    Returns:
        VLMClient 子类实例。

    Raises:
        ValueError: 如果 backend 未注册。
        ImportError: 如果后端模块导入失败（缺少依赖等），提示安装方法。
    """
    if backend not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"未知的 VLM 后端: {backend!r}。可用后端: {available}"
        )

    module_path, class_name, install_hint = _REGISTRY[backend]

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        # 构建友好的错误提示
        error_msg = f"\n{'='*60}\n"
        error_msg += f"VLM 后端 '{backend}' 缺少必要的依赖包\n"
        error_msg += f"{'='*60}\n"
        error_msg += f"导入错误: {exc}\n\n"
        if install_hint:
            error_msg += f"请运行以下命令安装依赖:\n"
            error_msg += f"    {install_hint}\n"
        else:
            error_msg += f"请检查该后端所需的依赖是否已安装。\n"
        error_msg += f"{'='*60}"
        raise ImportError(error_msg) from exc

    client_cls: Type[VLMClient] = getattr(module, class_name)

    logger.info("创建 VLM 客户端: backend=%s  class=%s", backend, class_name)

    if config is not None:
        return client_cls(config)
    return client_cls()
