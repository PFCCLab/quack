# 🦆 QuACK: A Quirky Assortment of CuTe Kernels 🦆 ❤️ PaddlePaddle

> [!NOTE]
>
> This repo is a fork of the original QuACK project, with modifications to enhance compatibility and integration with PaddlePaddle.
> Currently branch is align with 3d0ab3ec2164749caac8f269f771e66a40efd2de
>
> **Installation**
>
> ```bash
> git clone https://github.com/PFCCLab/quack.git
> cd quack
> pip install .
> ```
>
> **Usage**
>
> ```python
> import paddle
> paddle.enable_compat(scope={"quack", "triton"})  # Enable torch proxy before importing quack
> import quack
> # use quack
> ```

The original README.md content is as follows:

---

Kernels are written in the [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).

## Installation

``` bash
pip install quack-kernels
```

## Requirements

- H100 or B200 GPU
- CUDA toolkit 12.9+
- Python 3.12

## Kernels 🐥

- 🦆 RMSNorm forward + backward
- 🦆 Softmax forward + backward
- 🦆 Cross entropy forward + backward
- 🦆 Layernorm forward
- 🦆 Hopper gemm + epilogue
- 🦆 Blackwell gemm + epilogue

## Usage

```
from quack import rmsnorm, softmax, cross_entropy
```

## Documentations

[2025-07-10] We have a comprehensive
[blogpost](media/2025-07-10-membound-sol.md) on how to get memory-bound kernels
to speed-of-light, right in the comfort of Python thanks to the [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).

## Performance

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  >
</figure>
</div>

See our [blogpost](media/2025-07-10-membound-sol.md) for the details.

## Development

To set up the development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```
