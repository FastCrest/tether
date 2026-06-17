"""Create a stub flash_attn package so FluxVLA imports don't fail.

PI05FlowMatching uses PyTorch SDPA, not flash-attn. The imports happen
via transformers' attention dispatch which checks for flash_attn at
import time. This stub satisfies the import without the real CUDA lib.
"""
import os
import sys

base = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "flash_attn")
os.makedirs(base, exist_ok=True)

init = """\
# Stub — PI05FlowMatching uses SDPA, not flash-attn.
def flash_attn_func(*args, **kwargs):
    raise RuntimeError("flash_attn stub: use SDPA instead")

def flash_attn_varlen_func(*args, **kwargs):
    raise RuntimeError("flash_attn stub: use SDPA instead")

flash_attn_supports_top_left_mask = False

# Auto-stub any submodule import
import types as _types
import sys as _sys

class _StubFinder:
    def find_module(self, name, path=None):
        if name.startswith("flash_attn."):
            return self
        return None
    def load_module(self, name):
        if name in _sys.modules:
            return _sys.modules[name]
        mod = _types.ModuleType(name)
        mod.__path__ = []
        mod.__loader__ = self
        _sys.modules[name] = mod
        return mod

_sys.meta_path.insert(0, _StubFinder())
"""
with open(os.path.join(base, "__init__.py"), "w") as f:
    f.write(init)

interface = """\
# Stub — all functions raise if actually called.
def _stub(*args, **kwargs):
    raise RuntimeError("flash_attn stub: use SDPA instead")

flash_attn_func = _stub
flash_attn_varlen_func = _stub
flash_attn_unpadded_qkvpacked_func = _stub
flash_attn_varlen_qkvpacked_func = _stub
flash_attn_qkvpacked_func = _stub
flash_attn_with_kvcache = _stub
"""
with open(os.path.join(base, "flash_attn_interface.py"), "w") as f:
    f.write(interface)

# Create common submodule dirs
for submod in ["layers", "modules", "ops"]:
    subdir = os.path.join(base, submod)
    os.makedirs(subdir, exist_ok=True)
    with open(os.path.join(subdir, "__init__.py"), "w") as f:
        f.write(f"# flash_attn.{submod} stub\n")

with open(os.path.join(base, "layers", "rotary.py"), "w") as f:
    f.write("# stub\nclass RotaryEmbedding: pass\ndef apply_rotary_emb(*a, **k): raise RuntimeError('stub')\n")

with open(os.path.join(base, "bert_padding.py"), "w") as f:
    f.write("# stub\ndef unpad_input(*a, **k): raise RuntimeError('stub')\ndef pad_input(*a, **k): raise RuntimeError('stub')\n")

print(f"Created flash_attn stub at {base}")

# Create fake dist-info so importlib.metadata.distribution("flash-attn") works.
# transformers checks this via PACKAGE_DISTRIBUTION_MAPPING.
dist_info = os.path.join(
    sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages", "flash_attn-2.7.0.dist-info"
)
os.makedirs(dist_info, exist_ok=True)
with open(os.path.join(dist_info, "METADATA"), "w") as f:
    f.write("Metadata-Version: 2.1\nName: flash-attn\nVersion: 2.7.0\n")
with open(os.path.join(dist_info, "RECORD"), "w") as f:
    f.write("")
with open(os.path.join(dist_info, "top_level.txt"), "w") as f:
    f.write("flash_attn\n")
with open(os.path.join(dist_info, "INSTALLER"), "w") as f:
    f.write("pip\n")
print(f"Created flash-attn dist-info at {dist_info}")

# Also stub FluxVLA's CUDA extension modules (not built without setup.py install).
# PI05FlowMatching uses PyTorch SDPA, these are only needed for the Triton inference variant.
fluxvla_ops = "/opt/FluxVLA/fluxvla/ops/cuda"
for ext_name, subdir in [
    ("gemma_rotary_embedding_ext", "gemma_rotary_embedding"),
    ("rotary_pos_embedding_ext", "rotary_pos_embedding"),
    ("matmul_bias_ext", "matmul_bias"),
]:
    stub_path = os.path.join(fluxvla_ops, subdir, f"{ext_name}.py")
    if os.path.isdir(os.path.join(fluxvla_ops, subdir)):
        with open(stub_path, "w") as f:
            f.write(f"# Stub for {ext_name} — CUDA extension not built\n"
                    f"def forward(*a, **k): raise RuntimeError('{ext_name} stub')\n")
        print(f"Created stub: {stub_path}")

