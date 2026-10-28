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

    def test_lora_backward_has_safe_quantized_and_parameter_gradient_paths(self):
        source = (ROOT / "q8_kernels" / "functional" / "linear.py").read_text(
            encoding="utf-8"
        )
        module = ast.parse(source)
        lora_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "Q8LinearLora"
        )
        forward = next(
            node
            for node in lora_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        backward = next(
            node
            for node in lora_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "backward"
        )

        # INT8 inputs must have a defined activation passed to q8_mm(_bias).
        self.assertIn("a_quant = a", source)

        # Both backward branches must have the quantized output available.
        grad_quantize = [
            node
            for node in ast.walk(backward)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "quantize_fp8"
        ]
        self.assertTrue(grad_quantize)
        self.assertIn("grad_output_fp8", source)

        # Context storage must never include the old non-tensor None sentinel.
        save_calls = [
            node
            for node in ast.walk(forward)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "save_for_backward"
        ]
        self.assertTrue(save_calls)
        self.assertFalse(
            any(
                isinstance(argument, ast.Constant) and argument.value is None
                for call in save_calls
                for argument in call.args
            )
        )

        # Parameter gradients are reduced to their parameter shapes across
        # arbitrary leading batch/sequence dimensions.
        self.assertIn("flat_grad_out_lora_b", source)
        self.assertIn("flat_lora_input", source)

    def test_rms_norm_optional_weights_are_not_saved_as_none(self):
        source = (ROOT / "q8_kernels" / "functional" / "rms_norm.py").read_text(
            encoding="utf-8"
        )
        module = ast.parse(source)
        rms_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "RMSNorm8bit"
        )
        forward = next(
            node
            for node in rms_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        backward = next(
            node
            for node in rms_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "backward"
        )

        self.assertIn("ctx.has_weights", source)
        save_calls = [
            node
            for node in ast.walk(forward)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "save_for_backward"
        ]
        self.assertTrue(save_calls)
        self.assertFalse(
            any(
                isinstance(argument, ast.Constant) and argument.value is None
                for call in save_calls
                for argument in call.args
            )
        )
        self.assertIn("weights = None", source)
        self.assertTrue(any(isinstance(node, ast.Return) for node in ast.walk(backward)))

if __name__ == "__main__":
    unittest.main()
