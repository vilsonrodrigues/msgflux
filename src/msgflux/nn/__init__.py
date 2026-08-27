from importlib import import_module

_MODULE_EXPORTS = [
    "Agent",
    "Embedder",
    "Generator",
    "LocalTool",
    "MCPTool",
    "MediaMaker",
    "Module",
    "ModuleDict",
    "ModuleList",
    "Predictor",
    "Searcher",
    "Sequential",
    "Speaker",
    "Tool",
    "ToolLibrary",
    "Transcriber",
]

__all__ = [
    "AgentExtension",
    "AgentExtensionHandle",
    "CurrentDateExtension",
    "Parameter",
    "functional",
    "modules",
    "parameter",
    "Agent",
    "Embedder",
    "Generator",
    "LocalTool",
    "MCPTool",
    "MediaMaker",
    "Module",
    "ModuleDict",
    "ModuleList",
    "Predictor",
    "Searcher",
    "Sequential",
    "Speaker",
    "SkillsExtension",
    "ToolUsageGuidanceExtension",
    "Tool",
    "ToolLibrary",
    "Transcriber",
]


def __getattr__(name: str):
    if name in {"functional", "modules", "parameter"}:
        value = import_module(f"msgflux.nn.{name}")
    elif name in {
        "AgentExtension",
        "AgentExtensionHandle",
        "CurrentDateExtension",
        "SkillsExtension",
        "ToolUsageGuidanceExtension",
    }:
        value = getattr(import_module("msgflux.nn.extensions"), name)
    elif name == "Parameter":
        value = getattr(import_module("msgflux.nn.parameter"), name)
    elif name in _MODULE_EXPORTS:
        value = getattr(import_module("msgflux.nn.modules"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value
