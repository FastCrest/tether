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

with open(os.path.join(base, "bert_padding.py"), "w") as f:
    f.write("# stub\ndef unpad_input(*a, **k): raise RuntimeError('stub')\ndef pad_input(*a, **k): raise RuntimeError('stub')\n")

print(f"Created flash_attn stub at {base}")
