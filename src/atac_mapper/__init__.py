"""Single-cell chromatin accessibility analysis toolkit."""

from importlib.metadata import version

from . import topic_matching

__all__ = ["topic_matching"]
__version__ = version("atac_mapper")
