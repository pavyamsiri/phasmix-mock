"""Regression tests for type-discriminated recipe validation."""

import unittest
from pathlib import Path
from typing import override

import pytest
import strictyaml

from phasmock._recipe_yaml import recipe_schema

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class RecipeSchemaTests(unittest.TestCase):
    """Test the parsing of recipe schemas."""

    contents: str  # pyright: ignore[reportUninitializedInstanceVariable]
    schema: strictyaml.Map  # pyright: ignore[reportUninitializedInstanceVariable]

    @override
    def setUp(self) -> None:
        """Set up the test environment."""
        self.contents = (EXAMPLES / "two-arm.yaml").read_text()
        self.schema = recipe_schema()

    def test_full_example_and_reusable_schema(self) -> None:
        """Test a full recipe and reuse the schema."""
        for _ in range(2):
            data = strictyaml.load(self.contents, self.schema).data  # pyright: ignore[reportAny]
            assert data["background"][0]["parameters"]["x_scale"] == 0.4
            assert isinstance(data["signal"][0]["parameters"]["winding"], int)
            assert len(data["signal"]) == 2  # pyright: ignore[reportAny]

    def test_component_defaults(self) -> None:
        """Test default values for components."""
        contents = self.contents
        for line in (
            "    x_offset: 0.0\n",
            "    y_offset: 0.0\n",
            "    winding: 1\n",
            "    flattening_strength: 0.1\n",
        ):
            contents = contents.replace(line, "")
        data = strictyaml.load(contents, self.schema).data  # pyright: ignore[reportAny]
        assert data["background"][0]["parameters"]["x_offset"] == 0.0
        assert data["signal"][0]["parameters"]["winding"] == 1
        assert data["signal"][0]["parameters"]["flattening_strength"] == 0.1

    def test_either_type_in_either_sequence(self) -> None:
        """Test the supported background and signal component types."""
        contents = self.contents.replace("background:", "TEMP:")
        contents = contents.replace("signal:", "background:").replace("TEMP:", "signal:")
        data = strictyaml.load(contents, self.schema).data  # pyright: ignore[reportAny]
        assert data["background"][0]["type"] == "alinder-v1"
        assert data["signal"][0]["type"] == "gaussian-v1"

    def test_invalid_components(self) -> None:
        """Test invalid components."""
        replacements = (
            ("type: gaussian-v1", "type: unknown"),
            ("type: gaussian-v1", "type: alinder-v1"),
            ("type: alinder-v1", "type: gaussian-v1"),
            ("- type: gaussian-v1\n  parameters:", "- parameters:"),
            ("    variance: 1.0\n", ""),
            ("    rho: 0.2\n", ""),
            ("    variance: 1.0", "    variance: invalid"),
            ("    variance: 1.0", "    variance: 1.0\n    alpha: 0.3"),
            ("    winding: 1", "    winding: 0"),
        )
        for old, new in replacements:
            with (
                self.subTest(old=old, new=new),
                pytest.raises(strictyaml.YAMLValidationError),
            ):
                strictyaml.load(self.contents.replace(old, new), self.schema)  # pyright: ignore[reportUnusedCallResult]
