# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
"""Persistent shared-library cache for CuTe DSL compiled kernels.

Compiled kernels are exported as object files, linked to .so files, and cached.
On subsequent runs the .so is loaded via tvm_ffi (~1ms) instead of
re-generating IR + re-JIT'ing (~100ms per kernel).

Controls:
  QUACK_CACHE_ENABLED=0       — disable persistent shared-library cache
  QUACK_CACHE_DIR=path        — override default cache directory
  QUACK_CACHE_DEBUG=1         — short structured cache event logs
  QUACK_CACHE_DEBUG_VERBOSE=1 — full key/path dump (noisy)
  QUACK_CACHE_LOCK_TIMEOUT=N  — per-kernel compile lock timeout in seconds
"""

import fcntl
import functools
import hashlib
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from collections import namedtuple
from getpass import getuser
from pathlib import Path

import cutlass
import cutlass.cute as cute
import tvm_ffi

CACHE_ENABLED: bool = os.getenv("QUACK_CACHE_ENABLED", "1") == "1"
CACHE_DIR: str | None = os.getenv("QUACK_CACHE_DIR", None)
COMPILE_ONLY: bool = False
_CACHE_DEBUG: bool = os.getenv(
    "QUACK_CACHE_DEBUG", os.getenv("QUACK_JIT_CACHE_DEBUG", "0")
).lower() in {"1", "true", "yes", "on"}
_CACHE_DEBUG_VERBOSE: bool = os.getenv(
    "QUACK_CACHE_DEBUG_VERBOSE", "0"
).lower() in {"1", "true", "yes", "on"}

import logging
_logger = logging.getLogger(__name__)

# Downstream projects can append directories here to include their sources
# in the cache fingerprint. Must be set before the first jit_cache call.
EXTRA_SOURCE_DIRS: list[Path] = []

EXPORT_FUNC_NAME = "func"
CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop_kernel(*args, **kwargs):
    pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


LOCK_TIMEOUT = _env_int("QUACK_CACHE_LOCK_TIMEOUT", 1800)


def get_cache_path() -> Path:
    if CACHE_DIR is not None:
        cache_dir = Path(CACHE_DIR)
    else:
        cache_dir = Path(tempfile.gettempdir()) / getuser() / "quack_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _hash_source_dir(h, root: Path) -> None:
    """Hash all Python sources under *root* into *h*."""
    for src in sorted(root.rglob("*.py")):
        if not src.is_file():
            continue
        h.update(src.relative_to(root).as_posix().encode())
        content = src.read_bytes()
        h.update(len(content).to_bytes(8, "little"))
        h.update(content)


@functools.lru_cache(maxsize=1)
def _compute_source_fingerprint() -> str:
    """Hash quack + extra source dirs plus runtime ABI stamps into a fingerprint."""
    h = hashlib.sha256()
    h.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    h.update(f"cutlass={cutlass.__version__}".encode())
    h.update(f"tvm_ffi={tvm_ffi.__version__}".encode())
    _hash_source_dir(h, Path(__file__).resolve().parent)
    for extra_dir in EXTRA_SOURCE_DIRS:
        _hash_source_dir(h, Path(extra_dir).resolve())
    return h.hexdigest()


def _key_to_hash(key: tuple) -> str:
    return hashlib.sha256(pickle.dumps(key)).hexdigest()


def _rank_for_log() -> str:
    for name in ("PADDLE_RANK_IN_NODE", "PADDLE_TRAINER_ID", "RANK", "FLAGS_selected_gpus"):
        value = os.getenv(name)
        if value is not None:
            return value
    return "?"


def _short_kernel_signature(fn_name: str, cache_key: tuple) -> str:
    if fn_name == "_compile_gemm" and len(cache_key) >= 26:
        return (
            f"dtype={_short_dtype(cache_key[0])}/{_short_dtype(cache_key[1])}->{_short_dtype(cache_key[2])} "
            f"layout={cache_key[4]}{cache_key[5]}->{cache_key[6]} "
            f"tile={cache_key[8]} cluster={cache_key[9][:2]} "
            f"pp={int(cache_key[10])} pers={int(cache_key[11])} dyn={int(cache_key[12])} "
            f"alpha={cache_key[16]} beta={cache_key[17]} add={int(cache_key[18])} "
            f"varM={int(cache_key[19])} varK={int(cache_key[20])} gather={int(cache_key[21])} "
            f"perm={int(cache_key[22])} sm={cache_key[23][0]}{cache_key[23][1]} round={_short_enum(cache_key[24])} sr={cache_key[25]}"
        )
    return f"argc={len(cache_key)} key={_key_to_hash(cache_key)[:8]}"


def _short_dtype(value) -> str:
    name = getattr(value, "__name__", repr(value))
    return (
        name.replace("BFloat16", "bf16")
        .replace("Float32", "f32")
        .replace("Float16", "f16")
        .replace("Float8E4M3FN", "fp8e4m3")
        .replace("Float8E5M2", "fp8e5m2")
    )


def _short_enum(value) -> str:
    return getattr(value, "name", repr(value))


def _debug_key_summary(cache_key: tuple, max_items: int = 32) -> str:
    items = []
    for idx, value in enumerate(cache_key[:max_items]):
        text = repr(value)
        if len(text) > 96:
            text = text[:93] + "..."
        items.append(f"{idx}:{text}")
    if len(cache_key) > max_items:
        items.append(f"...(+{len(cache_key) - max_items} items)")
    return "; ".join(items)


# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------


class FileLock:
    """Advisory file lock using fcntl.flock with timeout."""

    def __init__(self, lock_path: Path, exclusive: bool, timeout: float = 15):
        self.lock_path = lock_path
        self.exclusive = exclusive
        self.timeout = timeout
        self._fd: int = -1

    def __enter__(self) -> "FileLock":
        flags = os.O_WRONLY | os.O_CREAT if self.exclusive else os.O_RDONLY | os.O_CREAT
        lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        self._fd = os.open(str(self.lock_path), flags)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                fcntl.flock(self._fd, lock_type | fcntl.LOCK_NB)
                return self
            except OSError:
                time.sleep(0.1)
        os.close(self._fd)
        self._fd = -1
        raise RuntimeError(f"Timed out waiting for lock: {self.lock_path}")

    def __exit__(self, *exc) -> None:
        if self._fd >= 0:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = -1


# ---------------------------------------------------------------------------
# Shared-library loader
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _link_inputs() -> tuple[str, str, str]:
    cc = shutil.which("g++") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        raise RuntimeError("No C/C++ compiler found for QuACK shared cache export")
    cute_libs = cute.runtime.find_runtime_libraries(enable_tvm_ffi=False)
    if not cute_libs:
        raise RuntimeError("CuTe runtime library not found for QuACK shared cache export")
    libcute = str(Path(cute_libs[0]))
    libtvm = str(Path(tvm_ffi.__file__).resolve().parent / "lib" / "libtvm_ffi.so")
    if not Path(libtvm).exists():
        raise RuntimeError(f"tvm_ffi runtime library not found: {libtvm}")
    return cc, libcute, libtvm


def _extract_kernel(module, name: str):
    return module[name]


def _try_load_so_file(so_path: Path, qualname: str, sha: str) -> "cute.Kernel | None":
    try:
        return _extract_kernel(
            cute.runtime.load_module(str(so_path), enable_tvm_ffi=True),
            EXPORT_FUNC_NAME,
        )
    except Exception as e:
        if _CACHE_DEBUG:
            _logger.warning(
                "QJIT event=load_failed module=quack kernel=%s kid=%s rank=%s error=%r",
                qualname, sha[:10], _rank_for_log(), e,
            )
        return None


def _export_shared_library(compiled_fn, tmp_o_path: Path, so_path: Path) -> None:
    cc, libcute, libtvm = _link_inputs()
    tmp_so_path = so_path.with_suffix(f".{os.getpid()}.tmp.so")
    try:
        compiled_fn.export_to_c(
            object_file_path=str(tmp_o_path),
            function_name=EXPORT_FUNC_NAME,
        )
        cmd = [
            cc,
            "-shared",
            "-Wl,--no-undefined",
            "-o",
            str(tmp_so_path),
            str(tmp_o_path),
            libcute,
            libtvm,
            f"-Wl,-rpath,{Path(libcute).parent}",
            f"-Wl,-rpath,{Path(libtvm).parent}",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        os.replace(tmp_so_path, so_path)
    finally:
        tmp_o_path.unlink(missing_ok=True)
        tmp_so_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# JIT cache decorator
# ---------------------------------------------------------------------------


def jit_cache(fn):
    """Decorator that caches compiled CuTe DSL kernels in-memory and on disk.

    The decorated function should return a compiled kernel (i.e. call cute.compile).
    The disk cache key is (fn.__qualname__, *args, **sorted_kwargs).

    Concurrency model (per key):
      - Shared lock during disk-cache read (multiple concurrent readers).
      - After a miss, this rank acquires an *exclusive* lock, re-checks whether
        the .so file has since appeared (another rank may have compiled it), and
        only compiles + exports if the file is still absent.  This avoids the
        N-rank cold-start stampede where every rank compiles the same kernel.
    """
    cache = {}
    hits = 0
    misses = 0

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal hits, misses
        cache_key = args + tuple(sorted(kwargs.items())) if kwargs else args

        # 1. In-memory hit
        if cache_key in cache:
            hits += 1
            _logger.debug("JIT %s in-memory hit.", fn.__qualname__)
            return _noop_kernel if COMPILE_ONLY else cache[cache_key]

        # 2. Disk hit (fast path — shared lock, no retry on load failure)
        disk_key = (fn.__qualname__,) + cache_key
        sha = None
        source_fingerprint = None
        so_path = None
        lock_path = None
        if CACHE_ENABLED:
            sha = _key_to_hash(disk_key)
            source_fingerprint = _compute_source_fingerprint()
            cache_dir = get_cache_path() / source_fingerprint
            cache_dir.mkdir(parents=True, exist_ok=True)
            so_path = cache_dir / f"{sha}.so"
            lock_path = cache_dir / f"{sha}.lock"
            if _CACHE_DEBUG_VERBOSE:
                _logger.info(
                    "QJIT_VERBOSE event=lookup module=%s kernel=%s kid=%s fp=%s "
                    "rank=%s exists=%s cache_dir=%s pid=%s key=[%s]",
                    fn.__module__, fn.__qualname__, sha, source_fingerprint,
                    _rank_for_log(), int(so_path.exists()),
                    get_cache_path(), os.getpid(),
                    _debug_key_summary(cache_key),
                )
            # Fast shared-lock read
            try:
                with FileLock(lock_path, exclusive=False, timeout=LOCK_TIMEOUT):
                    if so_path.exists():
                        loaded = _try_load_so_file(so_path, fn.__qualname__, sha)
                        if loaded is not None:
                            if _CACHE_DEBUG:
                                _logger.info(
                                    "QJIT event=hit module=%s kernel=%s kid=%s fp=%s rank=%s sig=\"%s\"",
                                    fn.__module__, fn.__qualname__,
                                    sha[:10], source_fingerprint[:10],
                                    _rank_for_log(),
                                    _short_kernel_signature(fn.__qualname__, cache_key),
                                )
                            else:
                                _logger.info("JIT %s disk-cache hit (%s), loading ...",
                                             fn.__qualname__, sha[:8])
                            cache[cache_key] = loaded
                            hits += 1
                            if not _CACHE_DEBUG:
                                _logger.info("JIT %s disk-cache load done.", fn.__qualname__)
                            return _noop_kernel if COMPILE_ONLY else loaded
            except RuntimeError as e:
                if _CACHE_DEBUG:
                    _logger.info(
                        "QJIT event=lock_timeout module=%s kernel=%s kid=%s rank=%s error=%r",
                        fn.__module__, fn.__qualname__, sha[:10], _rank_for_log(), e,
                    )

        # 3. Cold miss: one rank compiles, the others wait and load the .so.
        if CACHE_ENABLED:
            try:
                with FileLock(lock_path, exclusive=True, timeout=LOCK_TIMEOUT):
                    if so_path.exists():
                        loaded = _try_load_so_file(so_path, fn.__qualname__, sha)
                        if loaded is not None:
                            if _CACHE_DEBUG:
                                _logger.info(
                                    "QJIT event=hit module=%s kernel=%s kid=%s fp=%s rank=%s sig=\"%s\"",
                                    fn.__module__, fn.__qualname__,
                                    sha[:10], source_fingerprint[:10],
                                    _rank_for_log(),
                                    _short_kernel_signature(fn.__qualname__, cache_key),
                                )
                            else:
                                _logger.info("JIT %s disk-cache hit (%s), loading ...",
                                             fn.__qualname__, sha[:8])
                            cache[cache_key] = loaded
                            hits += 1
                            if not _CACHE_DEBUG:
                                _logger.info("JIT %s disk-cache load done.", fn.__qualname__)
                            return _noop_kernel if COMPILE_ONLY else loaded
                        try:
                            os.replace(
                                so_path,
                                so_path.with_suffix(f".{os.getpid()}.bad.so"),
                            )
                        except OSError:
                            pass

                    misses += 1
                    if _CACHE_DEBUG:
                        _logger.info(
                            "QJIT event=miss module=%s kernel=%s kid=%s fp=%s rank=%s sig=\"%s\"",
                            fn.__module__, fn.__qualname__, sha[:10],
                            source_fingerprint[:10], _rank_for_log(),
                            _short_kernel_signature(fn.__qualname__, cache_key),
                        )
                    else:
                        _logger.info("JIT %s compiling (cache miss) ...", fn.__qualname__)

                    compiled_fn = fn(*args, **kwargs)

                    if _CACHE_DEBUG:
                        _logger.info(
                            "QJIT event=compile_done module=%s kernel=%s kid=%s rank=%s",
                            fn.__module__, fn.__qualname__, sha[:10], _rank_for_log(),
                        )
                    else:
                        _logger.info("JIT %s compile done.", fn.__qualname__)

                    cache[cache_key] = compiled_fn
                    try:
                        so_path.parent.mkdir(parents=True, exist_ok=True)
                        if _CACHE_DEBUG:
                            _logger.info(
                                "QJIT event=export_start module=%s kernel=%s kid=%s rank=%s",
                                fn.__module__, fn.__qualname__, sha[:10], _rank_for_log(),
                            )
                        _export_shared_library(
                            compiled_fn,
                            so_path.with_suffix(f".{os.getpid()}.tmp.o"),
                            so_path,
                        )
                        if _CACHE_DEBUG:
                            _logger.info(
                                "QJIT event=export_done module=%s kernel=%s kid=%s rank=%s bytes=%s",
                                fn.__module__, fn.__qualname__, sha[:10],
                                _rank_for_log(),
                                so_path.stat().st_size if so_path.exists() else "missing",
                            )
                    except Exception as e:
                        _logger.warning(
                            "QJIT event=export_failed module=%s kernel=%s kid=%s rank=%s error=%r",
                            fn.__module__, fn.__qualname__, sha[:10], _rank_for_log(), e,
                        )
                    return _noop_kernel if COMPILE_ONLY else compiled_fn
            except RuntimeError as e:
                if not str(e).startswith("Timed out waiting for lock:"):
                    raise
                if _CACHE_DEBUG:
                    _logger.info(
                        "QJIT event=lock_timeout module=%s kernel=%s kid=%s rank=%s error=%r",
                        fn.__module__, fn.__qualname__, sha[:10], _rank_for_log(), e,
                    )

        misses += 1
        if _CACHE_DEBUG and CACHE_ENABLED:
            _logger.info(
                "QJIT event=miss module=%s kernel=%s kid=%s fp=%s rank=%s sig=\"%s\"",
                fn.__module__, fn.__qualname__, sha[:10],
                source_fingerprint[:10], _rank_for_log(),
                _short_kernel_signature(fn.__qualname__, cache_key),
            )
        else:
            _logger.info("JIT %s compiling (cache miss) ...", fn.__qualname__)

        compiled_fn = fn(*args, **kwargs)
        cache[cache_key] = compiled_fn
        if _CACHE_DEBUG and CACHE_ENABLED:
            _logger.info(
                "QJIT event=compile_done module=%s kernel=%s kid=%s rank=%s",
                fn.__module__, fn.__qualname__, sha[:10], _rank_for_log(),
            )
        else:
            _logger.info("JIT %s compile done.", fn.__qualname__)
        return _noop_kernel if COMPILE_ONLY else compiled_fn

    def cache_clear():
        nonlocal hits, misses
        cache.clear()
        hits = 0
        misses = 0

    def cache_info():
        return CacheInfo(hits=hits, misses=misses, maxsize=None, currsize=len(cache))

    wrapper.cache = cache
    wrapper.cache_clear = cache_clear
    wrapper.cache_info = cache_info
    return wrapper
