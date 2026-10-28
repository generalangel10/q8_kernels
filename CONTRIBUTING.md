# Contributing to Q8 Kernels

Thanks for helping improve Q8 Kernels. Contributions are welcome when they
include enough context for someone else to reproduce, review, and maintain the
change.

## Before opening an issue

Search existing issues first. For bugs and performance reports, use the issue
template and include the exact commit, environment, tensor shapes, dtypes, and
traceback. Performance claims should include the baseline, warm-up policy,
number of measured iterations, and hardware.

## Before opening a pull request

1. Keep the change focused and explain the motivation.
2. Run the CPU-safe checks:

   ```bash
   python -m compileall -q q8_kernels
   python -m unittest discover -s tests -v
   ```

3. If CUDA code changes, test on the supported GPU architecture and record the
   CUDA, driver, PyTorch, compiler, and GPU versions.
4. Update the README or relevant documentation when installation, API,
   compatibility, or benchmark behaviour changes.
5. Do not commit model weights, build directories, generated binaries, API
   keys, or other credentials.

## CUDA and benchmark changes

The default CI job intentionally does not compile CUDA extensions because
hosted runners do not provide the required GPU toolchain. CUDA changes need a
separate reproducible local or self-hosted validation record. Avoid presenting
one machine's benchmark as a universal speed guarantee.

## Review expectations

Pull requests should explain compatibility impact, failure modes, and how the
change was tested. Maintainers may request a smaller patch when a change mixes
refactoring, formatting, and behaviour changes.
