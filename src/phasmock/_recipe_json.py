"""JSON-specific recipe parsing.

JSON recipes share the same recipe model as YAML recipes. The JSON decoder and
its format-specific validation belong in this module.
"""

from __future__ import annotations

from os import PathLike

from phasmock._recipe_types import ParsedRecipeData


def from_json(path: str | PathLike[str]) -> ParsedRecipeData:
    """Parse a JSON recipe into its typed dictionary form.

    JSON recipe decoding has not been implemented yet.
    """
    raise NotImplementedError
