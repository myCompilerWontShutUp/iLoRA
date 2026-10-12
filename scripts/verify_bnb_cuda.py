"""Check that bitsandbytes loads the CUDA binary matching torch's own CUDA runtime.

Run inside the activated conda env after scripts/setup_bnb_cuda_runtime.sh. Exits non-zero if
torch is not the pinned 2.0.0 CUDA 11.7 build, no GPU is visible, the runtime bitsandbytes picks is
not torch's, or the loaded bitsandbytes binary is not libbitsandbytes_cuda<torch CUDA>.so.
"""
import ctypes
import os
import sys

EXPECTED_TORCH = '2.0.0'
EXPECTED_CUDA = '11.7'

failures = []


def check(ok, message):
    print(f"  [{'OK' if ok else 'FAIL'}] {message}")
    if not ok:
        failures.append(message)


import torch  # noqa: E402

cuda_available = torch.cuda.is_available()
check(torch.__version__.split('+')[0] == EXPECTED_TORCH, f"torch.__version__ = {torch.__version__}")
check(torch.version.cuda == EXPECTED_CUDA, f"torch.version.cuda = {torch.version.cuda}")
check(cuda_available, f"torch.cuda.is_available() = {cuda_available}")
check(cuda_available, f"GPU = {torch.cuda.get_device_name(0) if cuda_available else 'none'}")

tag = (torch.version.cuda or '').replace('.', '')
expected_binary = f"libbitsandbytes_cuda{tag}.so"

conda_prefix = os.environ.get('CONDA_PREFIX', '')
link = os.path.join(conda_prefix, 'lib', 'libcudart.so')
check(bool(conda_prefix) and os.path.exists(link), f"{link} -> {os.path.realpath(link) if os.path.exists(link) else 'missing'}")
if os.path.exists(link):
    version = ctypes.c_int()
    ctypes.CDLL(link).cudaRuntimeGetVersion(ctypes.byref(version))
    runtime = f"{version.value // 1000}.{version.value % 1000 // 10}"
    check(runtime == torch.version.cuda, f"that runtime reports CUDA {runtime}")

try:
    import bitsandbytes
    from bitsandbytes.cextension import COMPILED_WITH_CUDA
    from bitsandbytes.cuda_setup.main import CUDASetup, determine_cuda_runtime_lib_path
except Exception as e:  # report instead of crashing, so the other checks stay visible
    check(False, f"import bitsandbytes: {type(e).__name__}: {e}")
else:
    check(True, f"import bitsandbytes {bitsandbytes.__version__}")
    picked = determine_cuda_runtime_lib_path()
    check(picked is not None and os.path.exists(link) and os.path.samefile(picked, link),
          f"runtime bitsandbytes picks = {picked}")
    setup = CUDASetup.get_instance()
    loaded = str(getattr(setup.lib, '_name', None) or '')  # bitsandbytes passes a Path to LoadLibrary
    check(os.path.basename(loaded) == expected_binary, f"loaded binary = {loaded or 'none'} (expected {expected_binary})")
    check(getattr(setup, 'cuda_version_string', None) == tag,
          f"bitsandbytes detected CUDA version {getattr(setup, 'cuda_version_string', None)}")
    check(bool(COMPILED_WITH_CUDA), f"COMPILED_WITH_CUDA = {COMPILED_WITH_CUDA}")
    try:
        with open('/proc/self/maps') as f:
            mapped = sorted({line.split()[-1] for line in f if 'libbitsandbytes' in line})
        check(mapped == [os.path.realpath(loaded)], f"libbitsandbytes mapped in this process: {mapped}")
    except OSError:
        pass

if failures:
    print(f"verify_bnb_cuda: {len(failures)} check(s) failed", file=sys.stderr)
    sys.exit(1)
print("verify_bnb_cuda: bitsandbytes uses torch's CUDA runtime.")
