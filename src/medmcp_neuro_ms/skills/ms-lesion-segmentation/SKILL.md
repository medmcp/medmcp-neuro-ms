---
name: ms-lesion-segmentation
description: Workflow for multiple-sclerosis (MS) white-matter lesion segmentation and lesion-load quantification from a T1w + FLAIR pair using LST-AI
license: Apache-2.0
compatibility: Requires the medmcp-neuro-ms MCP server (console script medmcp-neuro-ms).
---

# MS lesion segmentation & lesion-load workflow

`segment_ms_lesions` runs **LST-AI** — a deep-learning UNet3D ensemble (ONNX) — to
segment MS white-matter lesions and report **total lesion load** (volume mm³), lesion
**count**, and a per-region breakdown. It does its own registration to MNI
(picsl-greedy) and skull stripping (HD-BET), so you only provide the raw scans.

## When to use

- The user wants **MS lesion segmentation**, a **lesion mask**, or **lesion load /
  lesion volume / lesion count** from a brain MRI.
- The user has both a **T1w** and a **FLAIR** of the same session.

## Required inputs

LST-AI needs **both** a T1w **and** a FLAIR, co-registered to the same session. If only
one contrast is available, tell the user LST-AI cannot run (it is a T1+FLAIR method) —
do not substitute another modality.

## Steps

1. **Identify the T1w and the FLAIR.** Match by contrast, not by guessing — the filename
   usually contains `T1w` / `FLAIR`. If you cannot tell which is which, ask.
2. **Do not skull-strip or register first.** LST-AI handles both internally. Only pass
   `already_stripped=True` if the user states **both** images are already skull-stripped.
3. **Run `segment_ms_lesions`** with `device="auto"` (GPU when present, else CPU — tell
   the user if it resolves to CPU, which is slower). Keep `annotate=True` (default) unless
   the user only wants the binary mask.
4. **Report lesion load.** From the result: `total_lesion_volume_mm3`, `lesion_count`,
   and the lesion mask path. If `region_volumes_mm3` is present, give the per-region
   breakdown. Follow the result's `_render` rules.

## Region annotation

With `annotate=True`, lesions are labelled into four McDonald-criteria regions —
**Periventricular, Juxtacortical, Subcortical, Infratentorial** — written to the
annotated map and `*_annotated_lesion_stats.csv`. Call `list_ms_lesion_regions()` for
the exact region names.

## Gotchas

- **Research only.** LST-AI is **not** validated/approved for clinical use. If the user
  describes a clinical decision context, flag this clearly before proceeding.
- **Output is a binary mask in FLAIR space** (`*_seg-lst.nii.gz`) — offer to overlay it
  on the FLAIR in the viewer. The annotated map (`*_desc-annotated_seg-lst.nii.gz`) uses
  labels 1–4 for the four regions.
- **Lesion load comparisons over time** need consistent acquisition; raw mm³ differences
  between scans can reflect protocol/scanner changes, not true disease change — caveat any
  longitudinal comparison.
- **Runtime** — minutes on GPU; substantially longer on CPU. The result reports which
  `device` was used.
- **Errors**: report and stop; do not retry with modified inputs without asking the user.
