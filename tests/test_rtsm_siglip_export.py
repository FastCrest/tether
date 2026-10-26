"""Numerical admission and standalone artifact boundary for the RTSM encoder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tether.exporters import rtsm_siglip as exporter


def test_local_checkpoint_requires_an_explicit_transform():
    with pytest.raises(ValueError, match="local checkpoint requires"):
        exporter.export_rtsm_siglip(Path("unused"), checkpoint=Path("weights.pt"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"samples": 0},
        {"samples": True},
        {"samples": 33},
        {"samples": 3.5},
        {"seed": -1},
        {"seed": True},
        {"seed": 2**32},
        {"preprocess": "guess"},
    ],
)
def test_bad_arguments_refuse_before_any_output(tmp_path, kwargs):
    output = tmp_path / "export"
    with pytest.raises(ValueError):
        exporter.export_rtsm_siglip(output, **kwargs)
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path):
    (tmp_path / "sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        exporter.export_rtsm_siglip(tmp_path)
    assert (tmp_path / "sentinel").read_text() == "keep"


def test_python310_refuses_before_optional_imports_and_preserves_argument_errors(
    tmp_path, monkeypatch
):
    with monkeypatch.context() as patch:
        patch.setattr(exporter.sys, "version_info", (3, 10, 16))
        with pytest.raises(ValueError, match="Python 3.11"):
            exporter.export_rtsm_siglip(tmp_path / "export")
        with pytest.raises(ValueError, match="local checkpoint requires"):
            exporter.export_rtsm_siglip(tmp_path / "export", checkpoint=Path("weights.pt"))
    assert not (tmp_path / "export").exists()


def test_linux_aarch64_refuses_before_optional_imports(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter.sys, "platform", "linux")
    monkeypatch.setattr(exporter.platform, "machine", lambda: "aarch64")
    with pytest.raises(ValueError, match="Linux aarch64"):
        exporter.export_rtsm_siglip(tmp_path / "export")
    assert not (tmp_path / "export").exists()


def _unit_embedding():
    result = np.ones((1, 768), dtype=np.float32)
    return result / np.linalg.norm(result)


def test_parity_accepts_unit_embeddings_and_measures_error():
    embedding = _unit_embedding()
    metrics = exporter._parity_metrics(embedding, embedding.copy())
    assert metrics["max_absolute_error"] == 0
    assert metrics["cosine_similarity"] == pytest.approx(1)


@pytest.mark.parametrize(
    "invalid",
    [
        np.ones((1, 767), dtype=np.float32),
        np.zeros((1, 768), dtype=np.float32),
        np.full((1, 768), np.nan, dtype=np.float32),
        np.full((1, 768), np.inf, dtype=np.float32),
        -_unit_embedding(),
    ],
)
def test_parity_refuses_bad_or_changed_embeddings(invalid):
    with pytest.raises(ValueError):
        exporter._parity_metrics(_unit_embedding(), invalid)


@pytest.fixture
def fake_open_clip(monkeypatch):
    torch = pytest.importorskip("torch")
    open_clip = pytest.importorskip("open_clip")
    from torchvision.transforms import Compose, Resize, ToTensor

    class Visual(torch.nn.Module):
        image_size = (224, 224)

        def forward(self, images):
            return images.mean(dim=(2, 3)).repeat(1, 256)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = Visual()

    transform = Compose([Resize((224, 224)), ToTensor()])
    calls = []

    def factory(*args, **kwargs):
        calls.append(kwargs)
        return Model(), None, transform

    monkeypatch.setattr(open_clip, "create_model_and_transforms", factory)
    monkeypatch.setattr(open_clip, "image_transform", lambda *args, **kwargs: transform)
    return calls


def test_safetensors_cache_symlink_keeps_loader_extension(tmp_path, fake_open_clip):
    pytest.importorskip("onnxruntime")
    blob = tmp_path / "extensionless-content-hash"
    blob.write_bytes(b"fixture weights")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.symlink_to(blob)
    exporter.export_rtsm_siglip(
        tmp_path / "encoder",
        checkpoint=checkpoint,
        preprocess="pretrained",
        samples=3,
    )
    assert fake_open_clip[0]["pretrained"] == str(checkpoint)
    assert fake_open_clip[0]["weights_only"] is True


def test_tiny_encoder_artifact_is_hashed_and_explicitly_not_a_policy(tmp_path, fake_open_clip):
    pytest.importorskip("onnxruntime")
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"test factory substitutes a tiny visual model")
    output = tmp_path / "encoder"
    manifest_path = exporter.export_rtsm_siglip(
        output,
        checkpoint=checkpoint,
        preprocess="rtsm-local",
        samples=3,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "VERIFIED_CPU_NUMERICAL_PARITY"
    assert manifest["checkpoint"]["sha256"] == exporter._sha256(checkpoint)
    assert manifest["checkpoint"]["source"] == "local-unverified-provenance"
    assert manifest["pretrained"] is None
    assert manifest["preprocessing"]["mode"] == "rtsm-local"
    assert manifest["preprocessing"]["webli_config_reference"]["applied"] is False
    assert manifest["parity"]["samples"] == 3
    assert len(manifest["parity"]["cases"]) == 3
    assert "tether_config.json" not in manifest["files"]
    assert any("No policy serving contract" in limit for limit in manifest["limits"])
    for relative, identity in manifest["files"].items():
        assert exporter._sha256(output / relative) == identity["sha256"]


def test_failed_parity_never_publishes_an_export(tmp_path, fake_open_clip, monkeypatch):
    pytest.importorskip("onnxruntime")
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    output = tmp_path / "failed"

    def refuse(*args):
        raise ValueError("deliberately changed embedding")

    monkeypatch.setattr(exporter, "_parity_metrics", refuse)
    with pytest.raises(ValueError, match="changed embedding"):
        exporter.export_rtsm_siglip(
            output, checkpoint=checkpoint, preprocess="pretrained", samples=3
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-*"))


def test_checkpoint_mutation_during_export_refuses(tmp_path, fake_open_clip, monkeypatch):
    pytest.importorskip("onnxruntime")
    import open_clip

    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"original checkpoint")
    original_factory = open_clip.create_model_and_transforms

    def mutate(*args, **kwargs):
        result = original_factory(*args, **kwargs)
        checkpoint.write_bytes(b"changed checkpoint")
        return result

    monkeypatch.setattr(open_clip, "create_model_and_transforms", mutate)
    output = tmp_path / "changed"
    with pytest.raises(ValueError, match="Checkpoint changed"):
        exporter.export_rtsm_siglip(
            output, checkpoint=checkpoint, preprocess="pretrained", samples=3
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".changed-*"))


def test_atomic_publication_preserves_concurrently_created_empty_directory(tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "manifest.json").write_text("verified")
    output = tmp_path / "someone-elses-output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        exporter._publish_stage(staged, output)
    assert output.is_dir() and not list(output.iterdir())
    assert (staged / "manifest.json").read_text() == "verified"


def test_applied_preprocessing_records_real_objects_not_webli_defaults():
    pytest.importorskip("torchvision")
    from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize

    transform = Compose(
        [
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop((224, 224)),
            Normalize((0.48, 0.45, 0.40), (0.26, 0.27, 0.28)),
        ]
    )
    actual = exporter._applied_preprocessing(transform)
    assert actual["color_mode"] == "RGB"
    assert actual["operations"][0]["size"] == 224
    assert actual["operations"][0]["interpolation"] == "bicubic"
    assert actual["operations"][1] == {"type": "CenterCrop", "size": [224, 224]}
    assert actual["operations"][2]["mean"] == [0.48, 0.45, 0.40]
    assert actual["operations"][2]["std"] == [0.26, 0.27, 0.28]
