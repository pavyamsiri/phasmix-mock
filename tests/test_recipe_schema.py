"""Regression tests for type-discriminated recipe validation."""

from pathlib import Path
import unittest

import strictyaml

from phasmock.recipe import recipe_schema


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class RecipeSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contents = (EXAMPLES / "two-arm.yaml").read_text()
        self.schema = recipe_schema()

    def test_full_example_and_reusable_schema(self) -> None:
        for _ in range(2):
            data = strictyaml.load(self.contents, self.schema).data
            self.assertEqual(data["background"][0]["parameters"]["x_scale"], 0.4)
            self.assertIsInstance(data["signal"][0]["parameters"]["winding"], int)
            self.assertEqual(len(data["signal"]), 2)

    def test_component_defaults(self) -> None:
        contents = self.contents
        for line in (
            "    x_offset: 0.0\n",
            "    y_offset: 0.0\n",
            "    winding: 1\n",
            "    flattening_strength: 0.1\n",
        ):
            contents = contents.replace(line, "")
        data = strictyaml.load(contents, self.schema).data
        self.assertEqual(data["background"][0]["parameters"]["x_offset"], 0.0)
        self.assertEqual(data["signal"][0]["parameters"]["winding"], 1)
        self.assertEqual(data["signal"][0]["parameters"]["flattening_strength"], 0.1)

    def test_either_type_in_either_sequence(self) -> None:
        contents = self.contents.replace("background:", "TEMP:")
        contents = contents.replace("signal:", "background:").replace(
            "TEMP:", "signal:"
        )
        data = strictyaml.load(contents, self.schema).data
        self.assertEqual(data["background"][0]["type"], "alinder-v1")
        self.assertEqual(data["signal"][0]["type"], "gaussian-v1")

    def test_invalid_components(self) -> None:
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
            with self.subTest(old=old, new=new):
                with self.assertRaises(strictyaml.YAMLValidationError):
                    strictyaml.load(self.contents.replace(old, new), self.schema)

    def test_short_example_is_incomplete(self) -> None:
        with self.assertRaises(strictyaml.YAMLValidationError):
            strictyaml.load((EXAMPLES / "two-arm_short.yaml").read_text(), self.schema)


if __name__ == "__main__":
    unittest.main()
