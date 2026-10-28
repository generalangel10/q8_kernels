# Changelog

This file records notable repository changes. The project has not yet adopted
formal release tags; entries below describe changes on the development branch.

## Unreleased

- Clarified CUDA, GPU architecture, installation, and validation requirements.
- Added contributor and security-reporting guidance.
- Added CPU-safe repository checks and issue templates.
- Added package version metadata and a canonical MIT license file.
- Made package metadata importable on CPU-only machines without eagerly
  loading CUDA-backed subpackages.
- Corrected gradient tuple arity for the Hadamard, Q8 linear, and FP8 linear
  custom autograd functions.
