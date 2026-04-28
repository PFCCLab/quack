__version__ = "0.3.7"

import os
import types
import torch

# Paddle compat: shim torch.compiler if missing
if not hasattr(torch, "compiler"):
    _compiler = types.ModuleType("torch.compiler")
    _compiler.is_compiling = lambda: False
    _compiler.disable = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)
    torch.compiler = _compiler
    # Also register via proxy overrides if running under paddle.enable_compat()
    try:
        import paddle.compat.proxy
        paddle.compat.proxy._extend_torch_proxy_overrides(
            {"torch.compiler": paddle.compat.proxy.RawOverriddenAttribute(_compiler)}
        )
    except Exception:
        pass

# Paddle compat: make CustomOpDef callable (same patch as supersonic-moe)
try:
    import paddle
    import inspect
    if not (hasattr(paddle.library.CustomOpDef, "__call__") and inspect.isfunction(paddle.library.CustomOpDef.__call__)):
        def _custom_op_call(self, *args, **kwargs):
            return getattr(getattr(paddle.ops, self._namespace), self._name)(*args, **kwargs)
        paddle.library.CustomOpDef.__call__ = _custom_op_call

    # Paddle compat: PyLayer context doesn't have needs_input_grad.
    # We provide a property that reads _quack_needs_input_grad if set during forward,
    # otherwise defaults to True (conservative).
    _PyLayerCtxCls = paddle.autograd.py_layer.PyLayerContext
    if not hasattr(_PyLayerCtxCls, "needs_input_grad"):
        class _NeedsInputGrad:
            """Returns stop_gradient info if recorded, else True (conservative)."""
            def __init__(self, ctx):
                self._ctx = ctx
            def __getitem__(self, idx):
                flags = getattr(self._ctx, "_quack_needs_input_grad", None)
                if flags is not None:
                    if isinstance(idx, slice):
                        return tuple(flags[i] for i in range(*idx.indices(len(flags))))
                    return flags[idx]
                if isinstance(idx, slice):
                    start, stop, step = idx.indices(64)
                    return tuple(True for _ in range(start, stop, step or 1))
                return True
        _PyLayerCtxCls.needs_input_grad = property(lambda self: _NeedsInputGrad(self))

    # Paddle compat: ctx.saved_tensors (PyTorch property) -> ctx.saved_tensor() (Paddle method)
    if not hasattr(_PyLayerCtxCls, "saved_tensors"):
        _PyLayerCtxCls.saved_tensors = property(lambda self: self.saved_tensor())

    # Paddle compat: ctx.set_materialize_grads (PyTorch) is not in Paddle PyLayer
    if not hasattr(_PyLayerCtxCls, "set_materialize_grads"):
        _PyLayerCtxCls.set_materialize_grads = lambda self, value: None

    # Paddle compat: ctx.mark_non_differentiable (PyTorch) is not in Paddle PyLayer
    if not hasattr(_PyLayerCtxCls, "mark_non_differentiable"):
        _PyLayerCtxCls.mark_non_differentiable = lambda self, *tensors: None
except Exception:
    pass

from quack.rmsnorm import rmsnorm
from quack.softmax import softmax
from quack.cross_entropy import cross_entropy
from quack.rounding import RoundingMode
from quack import copy_utils  # expose as attribute for cutlass DSL AST preprocessor
from quack import activation  # expose as attribute for cutlass DSL AST preprocessor


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    import quack.cute_dsl_ptxas  # noqa: F401

    # Patch to dump ptx and then use system ptxas to compile to cubin
    quack.cute_dsl_ptxas.patch()


__all__ = [
    "rmsnorm",
    "softmax",
    "cross_entropy",
    "RoundingMode",
]
