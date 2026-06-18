"""Tests for the MS lesion segmentation tool.

The tool shells out to the LST-AI ``lst`` CLI; tests mock the subprocess (and the
device probe) so they run without LST-AI, a GPU, or model weights — they exercise the
wrapper's argument building, success/failure handling, and result parsing.
"""

import subprocess
from pathlib import Path

import pytest

from medmcp_neuro_ms.tools import _neuro_ms, segmentation
from medmcp_neuro_ms.tools.segmentation import (
    list_ms_lesion_regions,
    segment_ms_lesions,
)

_STATS = "Num_Lesions,Num_Vox,Lesion_Volume\n5,4236,4242.5\n"
_ANNOT_STATS = (
    "Region,Num_Lesions,Num_Vox,Lesion_Volume\n"
    "Periventricular,3,3000,3000.0\n"
    "Juxtacortical,1,500,500.0\n"
    "Subcortical,1,400,400.0\n"
    "Infratentorial,0,0,0.0\n"
)


def _fake_lst_factory():
    """Return a subprocess.run replacement that writes LST-AI's expected outputs."""

    def fake_run(cmd, **kwargs):
        out = Path(cmd[cmd.index("--output") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "space-flair_seg-lst.nii.gz").write_bytes(b"\x1f\x8b\x08seg")
        (out / "lesion_stats.csv").write_text(_STATS)
        if "--segment_only" not in cmd:
            (out / "space-flair_desc-annotated_seg-lst.nii.gz").write_bytes(b"\x1f\x8bann")
            (out / "annotated_lesion_stats.csv").write_text(_ANNOT_STATS)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return fake_run


@pytest.fixture
def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    t1 = tmp_path / "sub-x_T1w.nii.gz"
    flair = tmp_path / "sub-x_FLAIR.nii.gz"
    t1.write_bytes(b"\x1f\x8bt1")
    flair.write_bytes(b"\x1f\x8bfl")
    return t1, flair


@pytest.fixture(autouse=True)
def _patch_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the tool runnable without LST-AI / GPU."""
    monkeypatch.setattr(segmentation, "find_lst", lambda: "lst")
    monkeypatch.setattr(segmentation, "lst_subprocess_env", dict)
    monkeypatch.setattr(_neuro_ms, "gpu_present", lambda: False)


def test_list_ms_lesion_regions() -> None:
    """Lists the four McDonald-criteria regions and carries display rules."""
    result = list_ms_lesion_regions()
    assert result["regions"] == [
        "Periventricular",
        "Juxtacortical",
        "Subcortical",
        "Infratentorial",
    ]
    assert "_render" in result


def test_segment_success_with_annotation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _inputs: tuple[Path, Path]
) -> None:
    """Full run parses lesion load, count, and per-region volumes; outputs are surfaced."""
    monkeypatch.setattr(segmentation.subprocess, "run", _fake_lst_factory())
    t1, flair = _inputs
    out = tmp_path / "out"

    res = segment_ms_lesions(t1, flair, output_dir=out, device="cpu")

    assert res["device"] == "cpu"
    assert res["lesion_count"] == 5
    assert res["total_lesion_volume_mm3"] == 4242.5
    assert res["region_volumes_mm3"]["Periventricular"] == 3000.0
    assert Path(res["lesion_mask_path"]).exists()
    assert Path(res["annotated_mask_path"]).exists()
    # outputs are stem-prefixed under output_dir
    assert Path(res["lesion_mask_path"]).name == "sub-x_FLAIR_seg-lst.nii.gz"


def test_segment_only_skips_annotation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _inputs: tuple[Path, Path]
) -> None:
    """annotate=False passes --segment_only and yields no annotated outputs."""
    monkeypatch.setattr(segmentation.subprocess, "run", _fake_lst_factory())
    t1, flair = _inputs

    res = segment_ms_lesions(t1, flair, output_dir=tmp_path / "out", annotate=False)

    assert res["annotated_mask_path"] is None
    assert res["region_volumes_mm3"] is None
    assert res["lesion_count"] == 5


def test_auto_device_resolves_to_cpu_without_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _inputs: tuple[Path, Path]
) -> None:
    """device='auto' -> 'cpu' when no GPU is visible."""
    monkeypatch.setattr(segmentation.subprocess, "run", _fake_lst_factory())
    monkeypatch.setattr(segmentation, "resolve_device", _neuro_ms.resolve_device)
    t1, flair = _inputs

    res = segment_ms_lesions(t1, flair, output_dir=tmp_path / "out", device="auto")

    assert res["device"] == "cpu"


def test_missing_input_raises(tmp_path: Path, _inputs: tuple[Path, Path]) -> None:
    """A missing input raises FileNotFoundError before invoking LST-AI."""
    t1, _ = _inputs
    with pytest.raises(FileNotFoundError):
        segment_ms_lesions(t1, tmp_path / "absent_FLAIR.nii.gz", output_dir=tmp_path / "o")


def test_non_niigz_input_raises(tmp_path: Path) -> None:
    """A non-.nii.gz input raises ValueError."""
    t1 = tmp_path / "sub-x_T1w.nii"
    flair = tmp_path / "sub-x_FLAIR.nii.gz"
    t1.write_bytes(b"x")
    flair.write_bytes(b"x")
    with pytest.raises(ValueError, match="zipped NIfTI"):
        segment_ms_lesions(t1, flair, output_dir=tmp_path / "o")


def test_failure_when_no_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _inputs: tuple[Path, Path]
) -> None:
    """If LST-AI writes no mask, a RuntimeError is raised."""

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(segmentation.subprocess, "run", fake_run)
    t1, flair = _inputs
    with pytest.raises(RuntimeError, match="LST-AI failed"):
        segment_ms_lesions(t1, flair, output_dir=tmp_path / "out", device="cpu")
