"""CPU checks for tether.accel: import surface, device gate, pi05 adapter contract, FP8 weights."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tether.accel as accel  # noqa: E402
from tether.accel.pi05 import inference as pi05_inference  # noqa: E402
from tether.accel.pi05.inference import AccelPi05Inference, active_views, prompt_ids  # noqa: E402


def test_package_and_adapter_import_without_triton():
    code = (
        "import sys; import tether.accel, tether.accel.pi05.inference, "
        "tether.accel.groot_n17.frontend, tether.accel.kernels, tether.noninferiority; "
        "assert 'triton' not in sys.modules, sorted(m for m in sys.modules if 'triton' in m)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    "cap,target", [((11, 0), "jetson_thor"), ((8, 9), "rtx4090"), ((12, 0), "rtx5090")]
)
def test_supported_devices(monkeypatch, cap, target):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: cap)
    assert accel.require_supported_device() == target


@pytest.mark.parametrize("cap", [(8, 7), (8, 6), (9, 0), (7, 5)])
def test_unsupported_devices_refuse_loudly(monkeypatch, cap):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: cap)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d=None: "Test GPU")
    with pytest.raises(accel.UnsupportedAccelDevice) as err:
        accel.require_supported_device()
    assert ("TensorRT" in str(err.value)) == (cap == (8, 7))


def test_no_cuda_refuses(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(accel.UnsupportedAccelDevice):
        accel.require_supported_device()


def _img():
    return np.zeros((1, 3, 224, 224), np.float32)


def test_active_views_and_prompt_ids():
    m = lambda v: np.array([v])  # noqa: E731
    assert active_views((_img(), _img(), _img()), (m(True), m(False), m(True))) == [0, 2]
    with pytest.raises(ValueError, match="one to three"):
        active_views((_img(),) * 3, (m(False),) * 3)
    with pytest.raises(ValueError, match="batch size 1"):
        active_views((np.zeros((2, 3, 224, 224)),) * 3, (np.array([True, True]),) * 3)
    ids = prompt_ids(np.array([[5, 6, 7, 0]]), np.array([[1, 1, 1, 0]]))
    assert ids.tolist() == [5, 6, 7] and ids.dtype == np.int64
    with pytest.raises(ValueError):
        prompt_ids(np.array([5, 6]), np.array([1, 1]))


class _FakeEngine:
    def __init__(self, views, chunk):
        self.views, self.chunk, self._graphs, self.calls = views, chunk, {}, []

    def set_prompt(self, ids):
        self.ids = list(ids)

    def infer(self, images, noise=None):
        self.calls.append((tuple(images.shape), np.asarray(noise).copy()))
        return torch.from_numpy(np.asarray(noise) * 2.0)


def _adapter(monkeypatch, chunk=50):
    monkeypatch.setattr(torch.cuda, "device", lambda d: __import__("contextlib").nullcontext())
    a = AccelPi05Inference(weights=None, device="cpu", chunk=chunk, target="rtx4090")
    engines = {}
    a._engine = lambda v: engines.setdefault(v, _FakeEngine(v, chunk))
    return a, engines


def _call(a, noise, masks=(True, True, False)):
    return a.predict_action_chunk(
        img_base=_img(),
        img_wrist_l=_img(),
        img_wrist_r=_img(),
        mask_base=np.array([masks[0]]),
        mask_wrist_l=np.array([masks[1]]),
        mask_wrist_r=np.array([masks[2]]),
        lang_tokens=np.array([[1, 2, 3, 0]]),
        lang_masks=np.array([[1, 1, 1, 0]]),
        noise=noise,
        state=np.zeros((1, 8), np.float32),
        episode_id="t0_ep0",
    )


def test_predict_action_chunk_contract(monkeypatch):
    a, engines = _adapter(monkeypatch)
    noise = np.random.default_rng(0).standard_normal((1, 50, 32)).astype(np.float32)
    out = _call(a, noise)
    assert out.shape == (1, 50, 32) and out.dtype == np.float32
    np.testing.assert_array_equal(out, noise * 2.0)  # the shared rollout noise reaches the engine
    eng = engines[2]
    assert eng.calls[0][0] == (2, 3, 224, 224) and eng.ids == [1, 2, 3]
    assert a.get_stats()["calls"] == 1 and a.get_stats()["target"] == "rtx4090"
    _call(a, noise, masks=(True, False, False))
    assert set(engines) == {1, 2}  # one engine per active-camera count


def test_predict_rejects_wrong_noise_and_nonfinite(monkeypatch):
    a, engines = _adapter(monkeypatch)
    with pytest.raises(ValueError, match="noise"):
        _call(a, np.zeros((1, 10, 32), np.float32))
    bad = np.full((1, 50, 32), np.nan, np.float32)
    with pytest.raises(RuntimeError, match="nonfinite"):
        _call(a, bad)


def test_from_checkpoint_rejects_wrong_variant(tmp_path, monkeypatch):
    monkeypatch.setattr(pi05_inference, "require_supported_device", lambda d: "rtx4090")
    (tmp_path / "config.json").write_text('{"paligemma_variant": "gemma_300m"}')
    with pytest.raises(ValueError, match="gemma_2b"):
        AccelPi05Inference.from_checkpoint(tmp_path, device="cpu")


def test_satisfies_rollout_inference_protocol():
    from tether.eval.libero_rollout import InferenceProtocol

    for name in ("reset_cache", "predict_action_chunk", "get_stats"):
        assert hasattr(InferenceProtocol, name) and callable(getattr(AccelPi05Inference, name))


def test_fp8_weight_quantization_is_per_channel_and_bounded():
    from tether.accel.kernels.fp8 import FP8_MAX, quantize_weight

    w = torch.randn(64, 128) * torch.logspace(-3, 2, 64)[:, None]
    q = quantize_weight(w, torch.randn(64))
    assert q.w.dtype == torch.float8_e4m3fn and q.scale.shape == (64,)
    assert q.w.float().abs().amax(1).max() <= FP8_MAX
    rel = (q.w.float() * q.scale[:, None] - w).abs().amax(1) / w.abs().amax(1)
    assert rel.max() < 0.07  # E4M3 has 3 mantissa bits: worst-case relative step 1/16


def test_bucket_rounds_up():
    from tether.accel._common import bucket

    assert [bucket(n, 16) for n in (1, 16, 17, 32)] == [16, 16, 32, 32]
