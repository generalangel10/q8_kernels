# Q8 Kernels

---

CUDA kernels and PyTorch modules for FP8/INT8 inference workloads.

Q8 Kernels provides CUDA implementations for quantised matrix multiplication,
flash attention, RMSNorm, RoPE, fused activations, fast Hadamard transforms,
and token-wise quantisation. The project is currently optimised for NVIDIA
Ada Lovelace GPUs (`sm_89`).

> **Status:** experimental research software. APIs and supported model
> integrations may change between revisions. The package currently requires a
> CUDA-capable NVIDIA GPU and is not a CPU fallback implementation.
## Features:
-8bit GEMM(with fused gelu and bias) / 2x faster than cuBLAS FP8 and 3.5x faster than torch.mm <br />
-FP8 Flash Attention 2 with Fast Hadamard Transform(also supports cross attention mask) / 2x faster than flash attention 2 <br />
-Mixed Precision Fast Hadamard Transform  <br />
-RMSNorm <br />
-Mixed Precision FMA <br />
-RoPE Layer <br />
-Quantizers <br />

All operations are implemented in CUDA. 
Current version supports Ada architecture (`sm_89`). Ampere support is not
currently guaranteed.

## Requirements

- NVIDIA GPU with Ada Lovelace (`sm_89`) support
- CUDA Toolkit 12.4 or newer
- PyTorch 2.4 or newer with CUDA support
- Python 3.7 or newer
- A working C++ compiler and `ninja`

CPU-only environments can run the repository's metadata and syntax checks, but
cannot build or execute the CUDA extensions.

## Installation

q8_kernels requires CUDA Version >= 12.4 and pytorch >=2.4.
q8_kernels was tested on Windows machine. Dont see problem with building on Linux systems.
Install ninja ```pip install ninja```
Make sure that ninja is installed and that it works correctly (e.g. ninja --version).
Without ninja installation is very slow.

```
git clone https://github.com/KONAKONA666/q8_kernels.git
cd q8_kernels
git submodule update --init --recursive
python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install .
```

It takes ~10–15 minutes to compile and install all modules, depending on the
machine and compiler cache. For an editable development install:

```bash
python -m pip install -e .
```


## Supported models
Speed ups are computed relative to transformers with inference with 16bit and flash attention 2 
|Model name | Speed up                                 |
| -------- | -------- |
| [LTXVideo](https://github.com/KONAKONA666/LTX-Video)| up to 2.5x |

The reported speed-up depends on GPU architecture, tensor shapes, PyTorch,
CUDA, and the comparison baseline. Reproduce benchmarks on your own hardware
before using those numbers for planning.

The `q8_kernels.convert_weights` console entry point is available after
installation for supported weight-conversion workflows.

## Development checks

The repository includes a lightweight CI check that runs without CUDA. Run it
locally with:

```bash
python -m compileall -q q8_kernels
python -m unittest discover -s tests -v
```

These checks validate source syntax and repository metadata only. CUDA kernel
correctness and performance still require a CUDA-enabled machine and should be
reported with the GPU model, driver, CUDA, PyTorch, and Python versions.

## Reporting issues

Please use the issue templates and include a minimal reproduction when
possible. For runtime or performance issues, include:

- GPU model and driver version
- CUDA Toolkit and PyTorch versions
- Python version and operating system
- The exact commit or package version
- Input tensor shapes, dtypes, and a complete traceback

Security-sensitive reports should follow [SECURITY.md](SECURITY.md) instead of
being posted publicly.

## Acknowledgement
Thanks to:
[Flash attention](https://github.com/Dao-AILab/flash-attention/tree/main)

[@66RING](https://github.com/66RING/tiny-flash-attention)

[fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform)

[cutlass](https://github.com/NVIDIA/cutlass)

[@weishengying](https://github.com/weishengying): Check his CUTE exercises and flash attn implementations

## Authors
KONAKONA666
## License

MIT License; see [LICENSE](LICENSE).

The bundled CUTLASS submodule retains its own upstream license and notices.
