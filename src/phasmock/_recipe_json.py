"""JSON-specific recipe parsing.

JSON recipes share the same recipe model as YAML recipes. The JSON decoder and
its format-specific validation belong in this module.
"""

from __future__ import annotations

from os import PathLike

from phasmock.recipe import MockRecipe


def from_json(path: str | PathLike[str]) -> MockRecipe:
    """Load a JSON recipe.

    JSON recipe decoding has not been implemented yet.
    """
    raise NotImplementedError
