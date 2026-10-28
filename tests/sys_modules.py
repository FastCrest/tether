"""Swap named entries in sys.modules for a test, restoring only those entries.

`unittest.mock.patch.dict("sys.modules", ...)` restores the whole dict on exit, which
also evicts every module first imported inside the block. Re-importing onnx then
re-runs its C extension and protobuf aborts the process on duplicate descriptors.
"""

import contextlib
import sys

_MISSING = object()


@contextlib.contextmanager
def swap_modules(modules):
    previous = {name: sys.modules.get(name, _MISSING) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
