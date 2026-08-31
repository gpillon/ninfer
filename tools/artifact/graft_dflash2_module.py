"""Append the DFlash2 drafter module to an already-converted nvfp4full artifact.

The v2 recipe is the v1 image plus the module's 66 objects: the module specs are
the tail of ``inventory_nvfp4full.TENSOR_SPECS``, so every object a v1 file
carries keeps its position, its layout and its payload-relative offset, and only
the JSON directory grows. Rebuilding v2 from scratch would mean re-downloading
and re-encoding the base and quantized 27B checkpoints (~70 GB) to reproduce
bytes the local file already holds; this grafts the module onto them instead.

The module itself is encoded by ``convert_nvfp4full.materialize_dflash2_object``,
the same function the full recipe uses, so the two paths cannot drift.

Canonical invocation::

    python3 -m tools.artifact.graft_dflash2_module \
      --artifact models/qwen3_8_27b_nvfp4full.ninfer \
      --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
      --out out/qwen3_8_27b_nvfp4full-v2.ninfer

``--device cpu`` is a supported way to run it without a CUDA torch build: the
module is 34 matrices and the encoder is device-parameterized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Iterator

import torch

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ArtifactWriter,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_8_27b import convert_nvfp4full as convert
from tools.convert.qwen3_8_27b import inventory_nvfp4full as inventory
from tools.convert.qwen3_8_27b import recipe_nvfp4full as recipe


RECIPE_ID = convert.RECIPE_ID
COPY_CHUNK_BYTES = 64 << 20


def _dflash2_names() -> frozenset[str]:
    return frozenset(spec.name for spec in inventory.DFLASH2_TENSOR_SPECS)


def _validate_source_artifact(artifact: Artifact) -> None:
    """The input must be exactly the module-less image of this same recipe."""

    if (
        artifact.identity.model_id != inventory.MODEL_ID
        or artifact.identity.weights_id != inventory.WEIGHTS_ID
    ):
        raise ValueError(
            f"{artifact.path}: identity {artifact.identity.model_id}/"
            f"{artifact.identity.weights_id} is not "
            f"{inventory.MODEL_ID}/{inventory.WEIGHTS_ID}"
        )
    module = _dflash2_names()
    present = [obj.name for obj in artifact.objects]
    if module.intersection(present):
        raise ValueError(f"{artifact.path}: already carries DFlash2 module objects")
    expected = [spec.name for spec in inventory.OBJECT_SPECS if spec.name not in module]
    if present != expected:
        missing = [name for name in expected if name not in set(present)]
        extra = [name for name in present if name not in set(expected)]
        raise ValueError(
            f"{artifact.path}: object list is not the module-less {RECIPE_ID} image "
            f"(missing {missing[:2]}, unexpected {extra[:2]})"
        )
    # Shape/format/layout equality is what makes the payloads transferable
    # verbatim: an object whose planned byte length differs cannot be copied into
    # the v2 slot it is being written to.
    planned = {spec.name: spec for spec in inventory.OBJECT_SPECS}
    for obj in artifact.objects:
        spec = planned[obj.name]
        if getattr(spec, "shape", None) is not None:
            if tuple(obj.shape) != tuple(spec.shape) or obj.format != spec.format:
                raise ValueError(f"{obj.name}: tensor signature differs from the recipe")
            if obj.layout != spec.layout:
                raise ValueError(f"{obj.name}: storage layout differs from the recipe")
        elif obj.encoding != spec.encoding:
            raise ValueError(f"{obj.name}: resource encoding differs from the recipe")


def _copy_chunks(payload: memoryview) -> Iterator[memoryview]:
    for begin in range(0, len(payload), COPY_CHUNK_BYTES):
        yield payload[begin : begin + COPY_CHUNK_BYTES]


def graft(
    artifact_path: str | Path,
    dflash2_dir: str | Path,
    output_path: str | Path,
    device: str | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    source_path = Path(artifact_path)
    dflash2 = Path(dflash2_dir)
    output = Path(output_path)
    if output.resolve() == source_path.resolve():
        raise ValueError("refusing to graft an artifact onto itself; pass a distinct --out")

    dflash2_summary = convert.validate_dflash2_config(
        json.loads((dflash2 / "config.json").read_text(encoding="utf-8"))
    )
    resolved_device = pick_device(device if device is not None else "cuda")
    module = _dflash2_names()
    parents: dict[str, object] = {}

    with Artifact(source_path) as source:
        _validate_source_artifact(source)
        # Resource payloads (tokenizer, chat template, and the rest) are carried over
        # verbatim, and their lengths are what the object plan needs to place every
        # tensor after them -- so the plan is built from the source's own copies.
        resources = {
            spec.name: bytes(source.payload(spec.name))
            for spec in inventory.OBJECT_SPECS
            if isinstance(spec, inventory.ResourceSpec)
        }
        plan = family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)
        with ShardReader.from_file(dflash2 / "model.safetensors") as reader:
            family_recipe.preflight_source_reader(reader, recipe.DFLASH2_RECIPES)
            output.parent.mkdir(parents=True, exist_ok=True)
            with ArtifactWriter(
                output,
                ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
                plan.specs,
            ) as writer:
                total = len(inventory.OBJECT_SPECS)
                for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                    if spec.name in module:
                        payload = convert.materialize_dflash2_object(
                            spec, reader, resolved_device, parents
                        )
                        writer.write(spec.name, payload)
                        del payload
                    elif spec.name in resources:
                        writer.write(spec.name, resources[spec.name])
                    else:
                        writer.write(spec.name, _copy_chunks(source.payload(spec.name)))
                    if index % 128 == 0 or index == total:
                        print(f"[{index}/{total}] objects written", flush=True)

    errors = [entry["relative_frobenius_error"] for entry in parents.values()]
    elapsed = time.perf_counter() - started
    report: dict[str, object] = {
        "recipe_id": RECIPE_ID,
        "grafted_from": {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
        },
        "source": {
            "dflash2": {
                "model_path": str(dflash2),
                "config": dflash2_summary,
            }
        },
        "artifact": {"path": str(output), "bytes": output.stat().st_size},
        "objects": {
            "count": len(inventory.OBJECT_SPECS),
            "grafted": len(module),
        },
        "local_nvfp4": {
            "encoder_profile": recipe.LOCAL_ENCODER_PROFILE,
            "parents": parents,
            "relative_frobenius_error_max": max(errors) if errors else None,
            "relative_frobenius_error_mean": (sum(errors) / len(errors)) if errors else None,
        },
        "elapsed_seconds": elapsed,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--dflash2-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()

    report = graft(
        arguments.artifact, arguments.dflash2_model, arguments.out, arguments.device
    )
    report_path = arguments.out.with_suffix(arguments.out.suffix + ".graft.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    artifact = report["artifact"]
    local = report["local_nvfp4"]
    print(f"wrote {artifact['path']} ({artifact['bytes']} bytes)")
    print(
        "module relative Frobenius error: "
        f"max {local['relative_frobenius_error_max']:.6f} "
        f"mean {local['relative_frobenius_error_mean']:.6f}"
    )
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
