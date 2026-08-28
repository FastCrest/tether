"""Canonical loading and validation for ``tether_config.json``.

Export Config v1 is deliberately strict at trust boundaries.  Legacy files
are normalized only from fields or artifacts that are actually present; this
module never guesses a model family, action width, or denoising step count.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping


CONFIG_FILENAME = "tether_config.json"
SCHEMA_VERSION = 1
EXPORT_KINDS = frozenset(
    {"monolithic_onnx", "decomposed_onnx", "trt_engine", "triton_bundle", "config_only"}
)
ARTIFACT_ROLES = frozenset({"model", "weights", "engine", "tokenizer", "config", "auxiliary"})
DTYPES = frozenset(
    {"float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool", "string"}
)
_SYMBOLIC_DIM = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_KINDS = {
    "monolithic": "monolithic_onnx",
    "decomposed": "decomposed_onnx",
}

PRODUCER_LAYOUTS: dict[str, dict[str, Any]] = {
    "monolithic": {
        "export_kind": "monolithic_onnx",
        "artifacts": (("model.onnx", "model"),),
    },
    "pi05_split": {
        "export_kind": "decomposed_onnx",
        "artifacts": (
            ("vlm_prefix.onnx", "model"),
            ("expert_denoise.onnx", "model"),
        ),
    },
    "expert_stack": {
        "export_kind": "decomposed_onnx",
        "artifacts": (("expert_stack.onnx", "model"),),
    },
    "pi0_prefix": {
        "export_kind": "decomposed_onnx",
        "pipeline": "prefix_optimum + expert_custom",
        "artifacts": (
            ("vision_encoder/model.onnx", "model"),
            ("multi_modal_projector.onnx", "model"),
            ("text_embedder.onnx", "model"),
            ("decoder_prefill/model.onnx", "model"),
            ("expert_stack.onnx", "model"),
        ),
    },
    "dreamzero": {
        "export_kind": "config_only",
        "artifacts": (),
    },
}

SMOLVLA_VLM_MODEL_PATHS = (
    "vision_encoder.onnx",
    "text_embedder.onnx",
    "decoder_prefill.onnx",
)
SMOLVLA_FULL_BUNDLE_MODEL_PATHS = frozenset(
    ("expert_stack.onnx", *SMOLVLA_VLM_MODEL_PATHS)
)


class ExportConfigError(ValueError):
    """The export config is malformed or inconsistent with its artifacts."""


class UnsupportedExportKindError(ExportConfigError):
    """The config is valid but the selected operation cannot use its layout."""


class UnsupportedExportPipelineError(ExportConfigError):
    """The config declares a pipeline that the selected operation cannot execute."""


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExportConfigError(f"{field} must be a positive integer, got {value!r}")
    return value


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExportConfigError(f"{field} must be a non-empty string")
    return value


def _artifact_path(value: Any, field: str) -> str:
    path = _nonempty_string(value, field)
    if "\\" in path:
        raise ExportConfigError(f"{field} must use POSIX separators")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ExportConfigError(f"{field} must be a relative path without '..'")
    if path != pure.as_posix() or path in {".", ""}:
        raise ExportConfigError(f"{field} must be a normalized relative POSIX path")
    return path


def _validate_tensor(tensor: Any, field: str) -> None:
    if not isinstance(tensor, Mapping):
        raise ExportConfigError(f"{field} must be an object")
    _nonempty_string(tensor.get("name"), f"{field}.name")
    dtype = tensor.get("dtype")
    if dtype not in DTYPES:
        raise ExportConfigError(f"{field}.dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    shape = tensor.get("shape")
    if not isinstance(shape, list):
        raise ExportConfigError(f"{field}.shape must be an array")
    for index, dim in enumerate(shape):
        if dim is None:
            continue
        if isinstance(dim, bool):
            raise ExportConfigError(f"{field}.shape[{index}] is not a valid dimension")
        if isinstance(dim, int):
            if dim <= 0:
                raise ExportConfigError(f"{field}.shape[{index}] must be positive")
            continue
        if isinstance(dim, str) and _SYMBOLIC_DIM.fullmatch(dim):
            continue
        raise ExportConfigError(
            f"{field}.shape[{index}] must be a positive integer, symbolic name, or null"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tether_config(
    payload: Mapping[str, Any],
    root: str | Path | None = None,
    *,
    inspect_artifacts: bool = True,
) -> dict[str, Any]:
    """Validate and return a shallow copy of a canonical Export Config v1."""
    if not isinstance(payload, Mapping):
        raise ExportConfigError("tether_config.json must contain a JSON object")
    config = dict(payload)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ExportConfigError(
            f"schema_version must be {SCHEMA_VERSION}, got {config.get('schema_version')!r}"
        )
    _nonempty_string(config.get("model_id"), "model_id")
    _nonempty_string(config.get("model_type"), "model_type")
    _positive_int(config.get("action_dim"), "action_dim")
    _positive_int(config.get("num_denoising_steps"), "num_denoising_steps")
    _positive_int(config.get("opset"), "opset")
    export_kind = config.get("export_kind")
    if export_kind not in EXPORT_KINDS:
        raise ExportConfigError(
            f"export_kind must be one of {sorted(EXPORT_KINDS)}, got {export_kind!r}"
        )

    artifacts = config.get("artifacts")
    if not isinstance(artifacts, list):
        raise ExportConfigError("artifacts must be an array")
    if export_kind != "config_only" and not artifacts:
        raise ExportConfigError(f"artifacts must not be empty for export_kind={export_kind!r}")
    seen_paths: set[str] = set()
    resolved_root = Path(root).resolve() if root is not None else None
    for index, artifact in enumerate(artifacts):
        field = f"artifacts[{index}]"
        if not isinstance(artifact, Mapping):
            raise ExportConfigError(f"{field} must be an object")
        relative = _artifact_path(artifact.get("path"), f"{field}.path")
        if relative in seen_paths:
            raise ExportConfigError(f"{field}.path duplicates {relative!r}")
        seen_paths.add(relative)
        role = artifact.get("role")
        if role not in ARTIFACT_ROLES:
            raise ExportConfigError(
                f"{field}.role must be one of {sorted(ARTIFACT_ROLES)}, got {role!r}"
            )
        expected_digest = artifact.get("sha256")
        if expected_digest is not None and (
            not isinstance(expected_digest, str) or not _SHA256.fullmatch(expected_digest)
        ):
            raise ExportConfigError(f"{field}.sha256 must be lowercase 64-hex")
        if resolved_root is not None and inspect_artifacts:
            resolved = (resolved_root / relative).resolve()
            try:
                resolved.relative_to(resolved_root)
            except ValueError as exc:
                raise ExportConfigError(
                    f"{field}.path resolves outside the export directory"
                ) from exc
            if not resolved.is_file():
                raise ExportConfigError(f"{field}.path is missing: {relative}")
            if expected_digest is not None:
                actual = _sha256_file(resolved)
                if actual != expected_digest:
                    raise ExportConfigError(
                        f"{field}.sha256 mismatch for {relative}: expected {expected_digest}, got {actual}"
                    )

    io_contract = config.get("io_contract")
    if not isinstance(io_contract, Mapping):
        raise ExportConfigError("io_contract must be an object")
    for direction in ("inputs", "outputs"):
        tensors = io_contract.get(direction)
        if not isinstance(tensors, list):
            raise ExportConfigError(f"io_contract.{direction} must be an array")
        names: set[str] = set()
        for index, tensor in enumerate(tensors):
            field = f"io_contract.{direction}[{index}]"
            _validate_tensor(tensor, field)
            name = str(tensor["name"])
            if name in names:
                raise ExportConfigError(f"{field}.name duplicates {name!r}")
            names.add(name)
    return config


def _legacy_artifacts(config: Mapping[str, Any], root: Path) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(candidate: Any, role: str) -> None:
        if not isinstance(candidate, str) or not candidate:
            return
        raw = Path(candidate)
        if raw.is_absolute():
            try:
                candidate = raw.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                return
        try:
            relative = _artifact_path(candidate, "legacy artifact path")
        except ExportConfigError:
            return
        if relative not in seen and (root / relative).is_file():
            artifacts.append({"path": relative, "role": role})
            seen.add(relative)

    for container_name in ("files", "components"):
        container = config.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for key, value in container.items():
            role = "weights" if "weight" in str(key) else "engine" if "trt" in str(key) else "model"
            add(value, role)
    for filename, role in (
        ("model.onnx", "model"),
        ("expert_stack.onnx", "model"),
        ("vlm_prefix.onnx", "model"),
        ("expert_denoise.onnx", "model"),
        ("expert_stack.trt", "engine"),
    ):
        add(filename, role)
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file() or candidate.name == CONFIG_FILENAME:
            continue
        role: str | None = None
        if candidate.suffix == ".onnx":
            role = "model"
        elif candidate.suffix in {".data", ".bin", ".safetensors"}:
            role = "weights"
        elif candidate.suffix in {".trt", ".engine"}:
            role = "engine"
        elif "tokenizer" in candidate.name or candidate.name in {
            "merges.txt",
            "vocab.json",
            "special_tokens_map.json",
        }:
            role = "tokenizer"
        if role is not None:
            add(candidate.resolve().relative_to(root.resolve()).as_posix(), role)
    return artifacts


def _onnx_metadata(
    root: Path, artifacts: list[dict[str, Any]]
) -> tuple[int | None, dict[str, list[dict[str, Any]]] | None]:
    onnx_paths = [root / item["path"] for item in artifacts if str(item["path"]).endswith(".onnx")]
    if not onnx_paths:
        return None, None
    try:
        import onnx  # type: ignore[import-not-found]
    except ImportError:
        return None, None

    opsets: list[int] = []
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    dtype_names = {
        1: "float32",
        2: "uint8",
        3: "int8",
        4: "uint16",
        5: "int16",
        6: "int32",
        7: "int64",
        8: "string",
        9: "bool",
        10: "float16",
        11: "float64",
    }

    def tensor_descriptor(value: Any) -> dict[str, Any]:
        tensor_type = value.type.tensor_type
        dtype = dtype_names.get(int(tensor_type.elem_type))
        if dtype not in DTYPES:
            raise ExportConfigError(f"ONNX tensor {value.name!r} uses unsupported dtype")
        shape: list[int | str | None] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value) if int(dim.dim_value) > 0 else None)
            elif dim.HasField("dim_param") and dim.dim_param:
                shape.append(str(dim.dim_param))
            else:
                shape.append(None)
        return {"name": value.name, "dtype": dtype, "shape": shape}

    for graph_index, path in enumerate(onnx_paths):
        model = onnx.load(str(path), load_external_data=False)
        opsets.extend(int(item.version) for item in model.opset_import if not item.domain)
        prefix = (
            f"{path.relative_to(root).as_posix()}:" if len(onnx_paths) > 1 else ""
        )
        initializers = {item.name for item in model.graph.initializer}
        for value in model.graph.input:
            if value.name not in initializers:
                desc = tensor_descriptor(value)
                desc["name"] = prefix + desc["name"]
                inputs.append(desc)
        for value in model.graph.output:
            desc = tensor_descriptor(value)
            desc["name"] = prefix + desc["name"]
            outputs.append(desc)
    return (max(opsets) if opsets else None), {"inputs": inputs, "outputs": outputs}


def _onnx_external_artifacts(root: Path, model_paths: list[str]) -> list[tuple[str, str]]:
    """Return only external files referenced by the declared ONNX graphs."""
    try:
        import onnx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ExportConfigError("onnx is required to build a canonical ONNX manifest") from exc

    discovered: list[tuple[str, str]] = []
    seen: set[str] = set()
    for model_path in model_paths:
        model = onnx.load(str(root / model_path), load_external_data=False)
        model_parent = PurePosixPath(model_path).parent
        for tensor in model.graph.initializer:
            if tensor.data_location != onnx.TensorProto.EXTERNAL:
                continue
            location = next(
                (entry.value for entry in tensor.external_data if entry.key == "location"),
                None,
            )
            if not location:
                raise ExportConfigError(
                    f"ONNX initializer {tensor.name!r} has no external-data location"
                )
            candidate = (root / model_parent / PurePosixPath(location)).resolve()
            try:
                relative = candidate.relative_to(root.resolve()).as_posix()
            except ValueError as exc:
                raise ExportConfigError(
                    f"external data for {model_path} resolves outside the export directory"
                ) from exc
            relative = _artifact_path(relative, f"external data for {model_path}")
            if relative not in seen:
                discovered.append((relative, "weights"))
                seen.add(relative)
    return discovered


def build_onnx_artifacts(
    output_dir: str | Path,
    model_paths: list[str] | tuple[str, ...],
) -> list[dict[str, str]]:
    """Build an exact manifest for declared ONNX graphs and their external data."""
    root = Path(output_dir)
    normalized = [_artifact_path(path, "declared ONNX artifact") for path in model_paths]
    if len(normalized) != len(set(normalized)):
        raise ExportConfigError("declared ONNX artifact paths must be unique")
    for path in normalized:
        if not (root / path).is_file():
            raise ExportConfigError(f"declared ONNX artifact is missing: {path}")
    artifacts = [{"path": path, "role": "model"} for path in normalized]
    artifacts.extend(
        {"path": path, "role": role}
        for path, role in _onnx_external_artifacts(root, normalized)
    )
    return artifacts


def replace_owned_onnx_artifacts(
    output_dir: str | Path,
    artifacts: list[Mapping[str, Any]],
    *,
    model_paths: list[str] | tuple[str, ...],
    previously_owned_paths: list[str] | tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], list[str]]:
    """Replace one producer component's manifest entries declaratively.

    ``previously_owned_paths`` is persisted by the producer, so rerunning an
    export removes external-data files that a newly written graph no longer
    references. Other producers' entries are preserved verbatim.
    """
    current = build_onnx_artifacts(output_dir, model_paths)
    owned = {
        _artifact_path(path, "previously owned artifact")
        for path in (*previously_owned_paths, *model_paths)
    }
    owned.update(item["path"] for item in current)
    retained = [dict(item) for item in artifacts if item.get("path") not in owned]
    current_paths = [item["path"] for item in current]
    return retained + current, current_paths


def build_producer_config(
    output_dir: str | Path,
    *,
    producer: str,
    model_id: str,
    model_type: str,
    action_dim: int,
    num_denoising_steps: int,
    opset: int,
    optional_artifacts: list[tuple[str, str]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical config from one producer's declared layout.

    Unlike legacy normalization, this function never scans ``output_dir``.
    External weights are included only when a declared ONNX graph references
    them, and optional assets must be passed explicitly by the producer.
    """
    if producer not in PRODUCER_LAYOUTS:
        raise ExportConfigError(f"unknown config producer {producer!r}")
    root = Path(output_dir)
    layout = PRODUCER_LAYOUTS[producer]
    declared = list(layout["artifacts"])
    model_paths = [path for path, role in declared if role == "model" and path.endswith(".onnx")]
    if model_paths:
        onnx_artifacts = build_onnx_artifacts(root, model_paths)
        declared = [(item["path"], item["role"]) for item in onnx_artifacts]
        inspected_opset, io_contract = _onnx_metadata(
            root,
            [{"path": path, "role": role} for path, role in declared if role == "model"],
        )
        if io_contract is None:
            raise ExportConfigError("could not inspect the declared ONNX I/O contract")
        if inspected_opset is not None and inspected_opset != opset:
            raise ExportConfigError(
                f"declared opset {opset} does not match ONNX opset {inspected_opset}"
            )
    else:
        io_contract = {"inputs": [], "outputs": []}

    for artifact in optional_artifacts or []:
        if artifact not in declared:
            declared.append(artifact)

    config: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "model_type": model_type,
        "action_dim": action_dim,
        "num_denoising_steps": num_denoising_steps,
        "opset": opset,
        "export_kind": layout["export_kind"],
        "artifacts": [{"path": path, "role": role} for path, role in declared],
        "io_contract": io_contract,
    }
    if "pipeline" in layout:
        config["pipeline"] = layout["pipeline"]
    if metadata:
        for key, value in metadata.items():
            if key not in config:
                config[key] = value
    return validate_tether_config(config, root=root, inspect_artifacts=True)


def decomposed_layout(config: Mapping[str, Any]) -> str:
    """Classify a canonical decomposed layout without guessing compatibility."""
    if config.get("export_kind") != "decomposed_onnx":
        raise UnsupportedExportKindError(
            f"decomposed layout requires export_kind='decomposed_onnx', "
            f"got {config.get('export_kind')!r}"
        )
    if config.get("pipeline") is not None:
        raise UnsupportedExportPipelineError(
            f"unsupported decomposed pipeline {config.get('pipeline')!r}"
        )
    paths = {
        str(item["path"])
        for item in config.get("artifacts", [])
        if item.get("role") == "model" and str(item.get("path", "")).endswith(".onnx")
    }
    if paths == {"vlm_prefix.onnx", "expert_denoise.onnx"}:
        return "pi05_split"
    if paths == {"expert_stack.onnx"}:
        return "expert_stack"
    if config.get("model_type") == "smolvla" and paths == SMOLVLA_FULL_BUNDLE_MODEL_PATHS:
        return "smolvla_full_bundle"
    raise UnsupportedExportPipelineError(
        f"unsupported decomposed artifact layout: {sorted(paths)}"
    )


def normalize_legacy_tether_config(payload: Mapping[str, Any], root: str | Path) -> dict[str, Any]:
    """Normalize a legacy config using only present metadata/artifact inspection."""
    config = dict(payload)
    root_path = Path(root)
    if config.get("schema_version") == SCHEMA_VERSION:
        return config
    config["schema_version"] = SCHEMA_VERSION
    config["export_kind"] = _LEGACY_KINDS.get(config.get("export_kind"), config.get("export_kind"))
    if (
        not config.get("export_kind")
        and config.get("export_format") == "dreamzero_decomposed"
    ):
        config["export_kind"] = "config_only"
    if not config.get("model_id"):
        for key in ("checkpoint_path", "checkpoint", "source_model"):
            if isinstance(config.get(key), str) and config[key]:
                config["model_id"] = config[key]
                break
    if not config.get("model_type") and isinstance(config.get("model_family"), str):
        config["model_type"] = config["model_family"]
    if not config.get("action_dim"):
        expert = config.get("expert")
        if isinstance(expert, Mapping) and expert.get("action_dim"):
            config["action_dim"] = expert["action_dim"]
    if not config.get("num_denoising_steps"):
        if config.get("num_inference_steps"):
            config["num_denoising_steps"] = config["num_inference_steps"]
        for container_name in ("expert", "decomposed"):
            if config.get("num_denoising_steps"):
                break
            container = config.get(container_name)
            if isinstance(container, Mapping) and container.get("num_denoising_steps"):
                config["num_denoising_steps"] = container["num_denoising_steps"]
                break
    if config.get("opset") is None and config.get("opset_version") is not None:
        config["opset"] = config["opset_version"]
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = (
            []
            if config.get("export_kind") == "config_only"
            else _legacy_artifacts(config, root_path)
        )
        config["artifacts"] = artifacts
    if not config.get("export_kind"):
        model_paths = {
            str(item.get("path"))
            for item in artifacts
            if isinstance(item, Mapping)
            and item.get("role") == "model"
            and str(item.get("path", "")).endswith(".onnx")
        }
        if model_paths == {"model.onnx"}:
            config["export_kind"] = "monolithic_onnx"
        elif model_paths == {"expert_stack.onnx"} or model_paths == {
            "vlm_prefix.onnx",
            "expert_denoise.onnx",
        }:
            config["export_kind"] = "decomposed_onnx"
    if config.get("export_kind") == "config_only" and not isinstance(
        config.get("io_contract"), Mapping
    ):
        config["io_contract"] = {"inputs": [], "outputs": []}
    if config.get("opset") is None or not isinstance(config.get("io_contract"), Mapping):
        inspected_opset, inspected_io = _onnx_metadata(root_path, artifacts)
        if config.get("opset") is None and inspected_opset is not None:
            config["opset"] = inspected_opset
        if not isinstance(config.get("io_contract"), Mapping) and inspected_io is not None:
            config["io_contract"] = inspected_io
    return config


def load_tether_config(
    path: str | Path,
    *,
    inspect_artifacts: bool = True,
) -> dict[str, Any]:
    """Load, legacy-normalize, and validate an export config."""
    requested = Path(path)
    config_path = requested / CONFIG_FILENAME if requested.is_dir() else requested
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {CONFIG_FILENAME}: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExportConfigError(f"Malformed {CONFIG_FILENAME} at {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ExportConfigError(f"{CONFIG_FILENAME} must contain a JSON object")
    normalized = normalize_legacy_tether_config(payload, config_path.parent)
    return validate_tether_config(
        normalized,
        root=config_path.parent,
        inspect_artifacts=inspect_artifacts,
    )


def write_tether_config(output_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    """Validate then atomically write a canonical config."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = validate_tether_config(payload, root=root, inspect_artifacts=True)
    target = root / CONFIG_FILENAME
    fd, temporary_name = tempfile.mkstemp(prefix=f".{CONFIG_FILENAME}.", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def require_supported_export_kind(
    config: Mapping[str, Any], supported: set[str], operation: str
) -> str:
    """Return the canonical kind or raise an explicit unsupported-layout error."""
    kind = str(config.get("export_kind", ""))
    if kind not in supported:
        raise UnsupportedExportKindError(
            f"{operation} does not support export_kind={kind!r}; supported: {sorted(supported)}"
        )
    return kind


def require_supported_pipeline(
    config: Mapping[str, Any], supported: set[str], operation: str
) -> str | None:
    """Reject a declared component pipeline unless the operation supports it.

    A missing pipeline denotes the canonical layout for ``export_kind``.  A
    producer that declares a custom component pipeline must be opted into by
    name so readers never guess that it is compatible from one shared file.
    """
    pipeline = config.get("pipeline")
    if pipeline is None:
        return None
    pipeline_name = _nonempty_string(pipeline, "pipeline")
    if pipeline_name not in supported:
        raise UnsupportedExportPipelineError(
            f"{operation} does not support pipeline={pipeline_name!r}; "
            f"supported: {sorted(supported)}"
        )
    return pipeline_name


__all__ = [
    "ARTIFACT_ROLES",
    "CONFIG_FILENAME",
    "DTYPES",
    "EXPORT_KINDS",
    "PRODUCER_LAYOUTS",
    "SMOLVLA_FULL_BUNDLE_MODEL_PATHS",
    "SMOLVLA_VLM_MODEL_PATHS",
    "ExportConfigError",
    "SCHEMA_VERSION",
    "UnsupportedExportKindError",
    "UnsupportedExportPipelineError",
    "build_producer_config",
    "build_onnx_artifacts",
    "decomposed_layout",
    "load_tether_config",
    "normalize_legacy_tether_config",
    "require_supported_export_kind",
    "require_supported_pipeline",
    "replace_owned_onnx_artifacts",
    "validate_tether_config",
    "write_tether_config",
]
