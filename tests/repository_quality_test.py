from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryQualityTests(unittest.TestCase):
    def test_expected_package_layout_exists(self):
        expected = (
            ROOT / "README.md",
            ROOT / "CONTRIBUTING.md",
            ROOT / "SECURITY.md",
            ROOT / "CHANGELOG.md",
            ROOT / "setup.py",
            ROOT / "q8_kernels" / "__init__.py",
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


if __name__ == "__main__":
    unittest.main()
