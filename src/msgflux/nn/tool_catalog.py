"""Helpers for canonical tool catalogs at nn execution boundaries."""

from msgflux.core.dotdict import dotdict
from msgflux.nn.modules.tool_runtime import ToolCatalogView
from msgflux.tools.definitions import ToolCatalog


def adapt_model_tool_catalog(model_execution_params: dotdict) -> dotdict:
    """Adapt the canonical Agent view at the legacy Model boundary."""
    catalog = model_execution_params.get("tool_catalog")
    if isinstance(catalog, ToolCatalogView):
        model_execution_params.tool_catalog = ToolCatalog.from_view(catalog)
    return model_execution_params
