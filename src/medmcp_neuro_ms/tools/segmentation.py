"""MS lesion segmentation using LST-AI (ONNX deep-learning ensemble).

Runs the LST-AI pipeline on a co-registered **T1w + FLAIR** pair:
registration to MNI (picsl-greedy) → skull stripping (HD-BET v2) → an ONNX UNet3D
ensemble lesion segmentation → optional region annotation by McDonald criteria
(periventricular / juxtacortical / subcortical / infratentorial). Outputs the binary
lesion mask in FLAIR space, the region-annotated map, the total lesion load and count,
and per-region volumes.

GPU-accelerated: HD-BET runs on torch-CUDA and the ONNX ensemble on the CUDA execution
provider when a GPU is available; registration (greedy) is CPU either way.
"""

import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TypedDict

from medmcp_neuro_ms.tools._neuro_ms import (
    find_lst,
    lst_subprocess_env,
    nii_stem,
    resolve_device,
)

# Region labels of the annotated map / annotated_lesion_stats.csv (LST-AI / McDonald).
_LESION_REGIONS: list[str] = [
    "Periventricular",
    "Juxtacortical",
    "Subcortical",
    "Infratentorial",
]

# Fixed output names LST-AI writes into its --output directory.
_LST_SEG = "space-flair_seg-lst.nii.gz"
_LST_ANNOT = "space-flair_desc-annotated_seg-lst.nii.gz"
_LST_STATS = "lesion_stats.csv"
_LST_ANNOT_STATS = "annotated_lesion_stats.csv"


class SegmentResult(TypedDict):
    """MS lesion segmentation result."""

    lesion_mask_path: str
    annotated_mask_path: str | None
    lesion_stats_path: str
    annotated_stats_path: str | None
    total_lesion_volume_mm3: float
    lesion_count: int
    region_volumes_mm3: dict[str, float] | None
    input_t1: str
    input_flair: str
    device: str
    _render: str


class RegionListResult(TypedDict):
    """The lesion regions LST-AI annotates."""

    regions: list[str]
    _render: str


def _parse_total_stats(path: Path) -> tuple[int, float]:
    """Parse lesion_stats.csv -> (lesion_count, total_volume_mm3).

    Columns: Num_Lesions, Num_Vox, Lesion_Volume (header + one data row).
    """
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return 0, 0.0
    col = {name: i for i, name in enumerate(rows[0])}
    data = rows[1]
    return int(float(data[col["Num_Lesions"]])), float(data[col["Lesion_Volume"]])


def _parse_region_stats(path: Path) -> dict[str, float]:
    """Parse annotated_lesion_stats.csv -> {region: volume_mm3}.

    Columns: Region, Num_Lesions, Num_Vox, Lesion_Volume (header + one row per region).
    """
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return {}
    col = {name: i for i, name in enumerate(rows[0])}
    region_i, vol_i = col["Region"], col["Lesion_Volume"]
    return {row[region_i]: float(row[vol_i]) for row in rows[1:] if row}


def segment_ms_lesions(
    t1_path: Path,
    flair_path: Path,
    output_dir: Path | None = None,
    device: str = "auto",
    already_stripped: bool = False,
    annotate: bool = True,
    threads: int = 4,
) -> SegmentResult:
    """Segment multiple-sclerosis lesions from a T1w + FLAIR pair using LST-AI.

    LST-AI registers the images to MNI, skull-strips them (HD-BET), runs an ONNX UNet3D
    ensemble to produce a binary lesion mask in FLAIR space, and (by default) annotates
    each lesion by region per McDonald criteria. Both a T1w and a FLAIR are required.

    Args:
        t1_path: Absolute path to the T1w image (.nii.gz).
        flair_path: Absolute path to the FLAIR image (.nii.gz), co-registered to the T1w.
        output_dir: Directory for outputs. Defaults to flair_path's directory.
        device: 'auto' (default; GPU if present, else CPU), a GPU id ('0', '1', …), or 'cpu'.
        already_stripped: Set True only if BOTH inputs are already skull-stripped
            (skips HD-BET). A mix of stripped/non-stripped is not supported.
        annotate: Annotate lesions by region (default True). False = binary mask only.
        threads: CPU threads for registration. Default 4.

    Returns:
        SegmentResult with output paths, total lesion volume (mm³) and count, and — when
        annotate is True — per-region volumes.

    Raises:
        FileNotFoundError: If an input does not exist.
        ValueError: If an input is not a .nii.gz file.
        RuntimeError: If LST-AI is not installed or segmentation fails.
    """
    for label, p in (("T1w", t1_path), ("FLAIR", flair_path)):
        if not p.exists():
            raise FileNotFoundError(f"{label} input not found: {p}")
        if not p.name.endswith(".nii.gz"):
            raise ValueError(f"{label} must be a zipped NIfTI (.nii.gz): {p}")

    lst = find_lst()
    resolved_device = resolve_device(device)
    out_dir = output_dir if output_dir is not None else flair_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = nii_stem(flair_path)

    # LST-AI writes fixed filenames into --output; run it in a private scratch dir and
    # surface the results under out_dir with the input stem so repeated runs don't clash.
    with tempfile.TemporaryDirectory() as scratch:
        lst_out = Path(scratch) / "out"
        lst_tmp = Path(scratch) / "tmp"
        cmd = [
            lst,
            "--t1",
            str(t1_path),
            "--flair",
            str(flair_path),
            "--output",
            str(lst_out),
            "--temp",
            str(lst_tmp),
            "--device",
            resolved_device,
            "--threads",
            str(threads),
        ]
        if already_stripped:
            cmd.append("--stripped")
        if not annotate:
            cmd.append("--segment_only")

        print(
            f"[medmcp-neuro-ms] segment_ms_lesions: running LST-AI on {flair_path.name} "
            f"(device={resolved_device}, annotate={annotate})...",
            file=sys.stderr,
            flush=True,
        )
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800, env=lst_subprocess_env()
        )
        if proc.stderr:
            sys.stderr.write(proc.stderr)
            sys.stderr.flush()

        seg_src = lst_out / _LST_SEG
        stats_src = lst_out / _LST_STATS
        if not seg_src.exists() or not stats_src.exists():
            raise RuntimeError(
                f"LST-AI failed (exit {proc.returncode}): {proc.stderr.strip()[-2000:]}"
            )

        # Surface outputs under out_dir, stem-prefixed.
        lesion_mask = out_dir / f"{stem}_seg-lst.nii.gz"
        lesion_stats = out_dir / f"{stem}_lesion_stats.csv"
        _copy(seg_src, lesion_mask)
        _copy(stats_src, lesion_stats)
        lesion_count, total_volume = _parse_total_stats(stats_src)

        annotated_mask: Path | None = None
        annotated_stats: Path | None = None
        region_volumes: dict[str, float] | None = None
        if annotate and (lst_out / _LST_ANNOT).exists():
            annotated_mask = out_dir / f"{stem}_desc-annotated_seg-lst.nii.gz"
            _copy(lst_out / _LST_ANNOT, annotated_mask)
            if (lst_out / _LST_ANNOT_STATS).exists():
                annotated_stats = out_dir / f"{stem}_annotated_lesion_stats.csv"
                _copy(lst_out / _LST_ANNOT_STATS, annotated_stats)
                region_volumes = _parse_region_stats(lst_out / _LST_ANNOT_STATS)

    result: SegmentResult = {
        "lesion_mask_path": str(lesion_mask),
        "annotated_mask_path": str(annotated_mask) if annotated_mask else None,
        "lesion_stats_path": str(lesion_stats),
        "annotated_stats_path": str(annotated_stats) if annotated_stats else None,
        "total_lesion_volume_mm3": round(total_volume, 2),
        "lesion_count": lesion_count,
        "region_volumes_mm3": (
            {k: round(v, 2) for k, v in region_volumes.items()} if region_volumes else None
        ),
        "input_t1": str(t1_path),
        "input_flair": str(flair_path),
        "device": resolved_device,
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            "Report the MS lesion segmentation as a compact key-value list:\n"
            "  Lesion mask:   <lesion_mask_path>\n"
            "  Total volume:  <total_lesion_volume_mm3> mm^3\n"
            "  Lesion count:  <lesion_count>\n"
            "  Device:        <device>\n"
            "If region_volumes_mm3 is present, add a short per-region breakdown "
            "(Periventricular / Juxtacortical / Subcortical / Infratentorial) in mm^3, "
            "and mention the annotated map path. Omit internal keys.\n"
            "NEXT ACTION: Give the user the lesion mask path and total lesion load. The "
            "mask is in FLAIR space — offer to overlay it on the FLAIR in the viewer. Note "
            "that LST-AI is a research-only tool, not for clinical use."
        ),
    }
    return result


def list_ms_lesion_regions() -> RegionListResult:
    """List the lesion regions LST-AI annotates (McDonald-criteria locations).

    Returns:
        RegionListResult with the region names as they appear in the annotated stats CSV.
    """
    return {
        "regions": list(_LESION_REGIONS),
        "_render": (
            "DISPLAY RULES — follow exactly:\n"
            "State that LST-AI annotates lesions into four McDonald-criteria regions: "
            "Periventricular, Juxtacortical, Subcortical, Infratentorial.\n"
            "These are the rows of the annotated_lesion_stats CSV (volumes in mm^3)."
        ),
    }


def _copy(src: Path, dst: Path) -> None:
    """Copy src to dst (small helper to keep the flow readable)."""
    shutil.copy(src, dst)
