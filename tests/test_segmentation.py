"""Tests for the LST-AI-backed MS lesion segmentation tool.

The tool shells out to the LST-AI ``lst`` CLI; the subprocess (and helpers) are mocked
so these run without LST-AI, a GPU, or model weights — exercising argument building,
the annotate / segment-only paths, result parsing, and error handling.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from medmcp_neuro_ms.tools.segmentation import (
    SegmentResult,
    list_ms_lesion_regions,
    segment_ms_lesions,
)

_FIND_LST = "medmcp_neuro_ms.tools.segmentation.find_lst"
_LST_ENV = "medmcp_neuro_ms.tools.segmentation.lst_subprocess_env"
_SUBPROCESS_RUN = "medmcp_neuro_ms.tools.segmentation.subprocess.run"
_GPU_PRESENT = "medmcp_neuro_ms.tools._neuro_ms.gpu_present"

_STATS = "Num_Lesions,Num_Vox,Lesion_Volume\n5,4236,4242.5\n"
_ANNOT_STATS = (
    "Region,Num_Lesions,Num_Vox,Lesion_Volume\n"
    "Periventricular,3,3000,3000.0\n"
    "Juxtacortical,1,500,500.0\n"
    "Subcortical,1,400,400.0\n"
    "Infratentorial,0,0,0.0\n"
)


def _mock_lst_run(
    cmd: list[str],
    *,
    capture_output: bool,
    text: bool,
    timeout: int,
    env: dict[str, str],
) -> MagicMock:
    """Fake the `lst` subprocess: write LST-AI's expected outputs, return success."""
    out = Path(cmd[cmd.index("--output") + 1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "space-flair_seg-lst.nii.gz").write_bytes(b"\x1f\x8bseg")
    (out / "lesion_stats.csv").write_text(_STATS)
    if "--segment_only" not in cmd:
        (out / "space-flair_desc-annotated_seg-lst.nii.gz").write_bytes(b"\x1f\x8bann")
        (out / "annotated_lesion_stats.csv").write_text(_ANNOT_STATS)
    result = MagicMock()
    result.returncode = 0
    result.stdout = ""
    result.stderr = ""
    return result


def _make_inputs(tmp_path: Path) -> tuple[Path, Path]:
    t1 = tmp_path / "sub-x_T1w.nii.gz"
    flair = tmp_path / "sub-x_FLAIR.nii.gz"
    t1.write_bytes(b"\x1f\x8bt1")
    flair.write_bytes(b"\x1f\x8bfl")
    return t1, flair


def _run(tmp_path: Path, *, device: str = "cpu", annotate: bool = True) -> SegmentResult:
    """Run segment_ms_lesions with the lst subprocess + helpers mocked."""
    t1, flair = _make_inputs(tmp_path)
    with (
        patch(_FIND_LST, return_value="lst"),
        patch(_LST_ENV, return_value={}),
        patch(_SUBPROCESS_RUN, side_effect=_mock_lst_run),
    ):
        return segment_ms_lesions(
            t1, flair, output_dir=tmp_path / "out", device=device, annotate=annotate
        )


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


def test_segment_success_with_annotation(tmp_path: Path) -> None:
    """Full run parses lesion load, count, and per-region volumes; outputs are surfaced."""
    res = _run(tmp_path, device="cpu", annotate=True)

    assert res["device"] == "cpu"
    assert res["lesion_count"] == 5
    assert res["total_lesion_volume_mm3"] == 4242.5
    regions = res["region_volumes_mm3"]
    assert regions is not None
    assert regions["Periventricular"] == 3000.0
    assert Path(res["lesion_mask_path"]).exists()
    assert Path(res["lesion_mask_path"]).name == "sub-x_FLAIR_seg-lst.nii.gz"
    annotated = res["annotated_mask_path"]
    assert annotated is not None
    assert Path(annotated).exists()


def test_segment_only_skips_annotation(tmp_path: Path) -> None:
    """annotate=False passes --segment_only and yields no annotated outputs."""
    res = _run(tmp_path, device="cpu", annotate=False)
    assert res["annotated_mask_path"] is None
    assert res["region_volumes_mm3"] is None
    assert res["lesion_count"] == 5


def test_segment_only_passes_flag(tmp_path: Path) -> None:
    """The --segment_only flag is forwarded to the lst CLI when annotate=False."""
    t1, flair = _make_inputs(tmp_path)
    with (
        patch(_FIND_LST, return_value="lst"),
        patch(_LST_ENV, return_value={}),
        patch(_SUBPROCESS_RUN, side_effect=_mock_lst_run) as mock_run,
    ):
        segment_ms_lesions(t1, flair, output_dir=tmp_path / "out", device="cpu", annotate=False)
    cmd = mock_run.call_args[0][0]
    assert "--segment_only" in cmd
    assert "--stripped" not in cmd


def test_already_stripped_passes_flag(tmp_path: Path) -> None:
    """already_stripped forwards --stripped to the lst CLI."""
    t1, flair = _make_inputs(tmp_path)
    with (
        patch(_FIND_LST, return_value="lst"),
        patch(_LST_ENV, return_value={}),
        patch(_SUBPROCESS_RUN, side_effect=_mock_lst_run) as mock_run,
    ):
        segment_ms_lesions(
            t1, flair, output_dir=tmp_path / "out", device="cpu", already_stripped=True
        )
    assert "--stripped" in mock_run.call_args[0][0]


def test_auto_device_resolves_to_cpu_without_gpu(tmp_path: Path) -> None:
    """device='auto' -> 'cpu' when no GPU is visible."""
    t1, flair = _make_inputs(tmp_path)
    with (
        patch(_FIND_LST, return_value="lst"),
        patch(_LST_ENV, return_value={}),
        patch(_GPU_PRESENT, return_value=False),
        patch(_SUBPROCESS_RUN, side_effect=_mock_lst_run),
    ):
        res = segment_ms_lesions(t1, flair, output_dir=tmp_path / "out", device="auto")
    assert res["device"] == "cpu"


def test_missing_input_raises(tmp_path: Path) -> None:
    """A missing input raises FileNotFoundError before invoking LST-AI."""
    t1, _ = _make_inputs(tmp_path)
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


def test_failure_when_no_output(tmp_path: Path) -> None:
    """If LST-AI writes no mask, a RuntimeError is raised."""

    def _no_output_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
        env: dict[str, str],
    ) -> MagicMock:
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "boom"
        return result

    t1, flair = _make_inputs(tmp_path)
    with (
        patch(_FIND_LST, return_value="lst"),
        patch(_LST_ENV, return_value={}),
        patch(_SUBPROCESS_RUN, side_effect=_no_output_run),
        pytest.raises(RuntimeError, match="LST-AI failed"),
    ):
        segment_ms_lesions(t1, flair, output_dir=tmp_path / "out", device="cpu")
