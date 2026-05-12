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
    "EventStream",
    "EventType",
    "Parameter",
    "StreamEvent",
    "ToolCallMetadata",
    "emit_compaction_post",
    "emit_compaction_pre",
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
    "Tool",
    "ToolLibrary",
    "Transcriber",
]


def __getattr__(name: str):
    if name in {"functional", "modules", "parameter"}:
        value = import_module(f"msgflux.nn.{name}")
    elif name in {
        "EventStream",
        "EventType",
        "StreamEvent",
        "ToolCallMetadata",
        "emit_compaction_post",
        "emit_compaction_pre",
    }:
        value = getattr(import_module("msgflux.runtime"), name)
    elif name == "Parameter":
        value = getattr(import_module("msgflux.nn.parameter"), name)
    elif name in _MODULE_EXPORTS:
        value = getattr(import_module("msgflux.nn.modules"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value
