from pathlib import Path
import ast
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryQualityTests(unittest.TestCase):
    def test_expected_package_layout_exists(self):
        expected = (
            ROOT / "README.md",
            ROOT / "CONTRIBUTING.md",
            ROOT / "SECURITY.md",
            ROOT / "CHANGELOG.md",
            ROOT / "LICENSE",
            ROOT / "setup.py",
            ROOT / "q8_kernels" / "__init__.py",
            ROOT / "q8_kernels" / "_version.py",
        )
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), f"Missing required file: {path}")

    def test_readme_documents_cuda_requirements(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for phrase in ("CUDA Toolkit 12.4", "PyTorch 2.4", "sm_89"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, readme)

    def test_workflow_is_present(self):
        workflow = ROOT / ".github" / "workflows" / "repository-checks.yml"
        self.assertTrue(workflow.is_file())

    def test_package_metadata_is_importable_without_cuda(self):
        import q8_kernels

        self.assertEqual(q8_kernels.__version__, "0.1.0")

    def test_custom_autograd_backward_arity_matches_forward_inputs(self):
        expected = {
            ("q8_kernels/functional/fast_hadamard.py", "HadamardTransformFn"): 3,
            ("q8_kernels/functional/linear.py", "Q8LinearFunc"): 7,
            ("q8_kernels/functional/linear.py", "FP8LinearFunc"): 7,
        }
        for (relative_path, class_name), expected_arity in expected.items():
            with self.subTest(class_name=class_name):
                module = ast.parse(
                    (ROOT / relative_path).read_text(encoding="utf-8")
                )
                class_node = next(
                    node
                    for node in module.body
                    if isinstance(node, ast.ClassDef) and node.name == class_name
                )
                backward = next(
                    node
                    for node in class_node.body
                    if isinstance(node, ast.FunctionDef) and node.name == "backward"
                )
                return_nodes = [
                    node
                    for node in ast.walk(backward)
                    if isinstance(node, ast.Return)
                    and isinstance(node.value, ast.Tuple)
                ]
                self.assertEqual(len(return_nodes), 1)
                self.assertEqual(len(return_nodes[0].value.elts), expected_arity)


if __name__ == "__main__":
    unittest.main()
