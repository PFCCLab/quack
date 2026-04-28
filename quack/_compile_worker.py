# Copyright (c) 2025, Tri Dao.
# Persistent subprocess worker for parallel autotuning pre-compilation.
# Receives length-prefixed pickled tasks on stdin, creates FakeTensors
# matching the parent's tensor metadata, and compiles with COMPILE_ONLY=True.
# Stays alive to process multiple configs (amortizes import overhead).

import importlib
import pickle
import struct
import sys

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import quack.cache_utils

quack.cache_utils.COMPILE_ONLY = True

_dtype_map = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}
# Paddle compat: str(tensor.dtype) returns "paddle.bfloat16" etc.
for _k, _v in list(_dtype_map.items()):
    _dtype_map[_k.replace("torch.", "paddle.")] = _v


def _make_fake_tensor(meta):
    shape = meta["shape"]
    stride = meta["stride"]
    dtype = _dtype_map[meta["dtype"]]
    return torch.empty_strided(shape, stride, dtype=dtype, device="cuda")


def _recv(stream):
    """Read a length-prefixed pickled message. Returns None on EOF."""
    try:
        header = stream.read(4)
        if len(header) < 4:
            return None
        length = struct.unpack("<I", header)[0]
        if length == 0:
            return None
        data = stream.read(length)
        return pickle.loads(data)
    except (BrokenPipeError, OSError, pickle.UnpicklingError):
        return None


def _send(stream, msg):
    """Write a length-prefixed pickled message. Returns False on pipe error."""
    try:
        data = pickle.dumps(msg)
        stream.write(struct.pack("<I", len(data)))
        stream.write(data)
        stream.flush()
        return True
    except (BrokenPipeError, OSError):
        return False


def main():
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    # Signal ready
    if not _send(stdout, "READY"):
        return

    fn_cache = {}
    while True:
        payload = _recv(stdin)
        if payload is None:
            break

        try:
            fn_module = payload["fn_module"]
            fn_qualname = payload["fn_qualname"]
            fn_key = (fn_module, fn_qualname)
            if fn_key not in fn_cache:
                mod = importlib.import_module(fn_module)
                obj = mod
                for part in fn_qualname.split("."):
                    obj = getattr(obj, part)
                fn_cache[fn_key] = getattr(obj, "fn", obj)
            fn = fn_cache[fn_key]

            tensor_meta = payload["tensor_meta"]
            kwargs = payload["kwargs"]
            config_kwargs = payload["config_kwargs"]

            with FakeTensorMode():
                fake_args = []
                for meta in tensor_meta:
                    if isinstance(meta, dict) and "shape" in meta:
                        fake_args.append(_make_fake_tensor(meta))
                    else:
                        fake_args.append(meta)
                fn(*fake_args, **kwargs, **config_kwargs)
                if not _send(stdout, "OK"):
                    break
        except Exception as e:
            if not _send(stdout, f"ERR:{e}"):
                break


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    except Exception:
        pass
