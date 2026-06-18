# medmcp-neuro-ms

MS white-matter **lesion segmentation** stack for the [medmcp](https://github.com/medmcp)
ecosystem. It wraps [LST-AI](https://github.com/CompImg/LST-AI) — a deep-learning UNet3D
ensemble — as an **MCP (Model Context Protocol) server** over stdio, so an LLM agent can
segment MS lesions and quantify lesion load from a T1w + FLAIR pair.

> [!WARNING]
> Research software, **not licensed for clinical use**. LST-AI is a research-only tool for
> MS lesion segmentation and has not been validated/approved for clinical usage.

---

## Tool inventory

| Tool name | Description | Inputs | Key outputs |
|---|---|---|---|
| `segment_ms_lesions` | Run LST-AI (registration → HD-BET skull strip → ONNX ensemble → optional region annotation) on a T1w + FLAIR pair | `t1_path`, `flair_path`, `output_dir`, `device="auto"`, `already_stripped=False`, `annotate=True`, `threads=4` | binary lesion mask (FLAIR space), annotated region map, `total_lesion_volume_mm3`, `lesion_count`, `region_volumes_mm3` |
| `list_ms_lesion_regions` | List the McDonald-criteria lesion regions LST-AI annotates | — | `Periventricular`, `Juxtacortical`, `Subcortical`, `Infratentorial` |

- **Annotated vs. binary:** `annotate=True` (default) produces the 4-region map +
  per-region volumes; `annotate=False` (`--segment_only`) yields only the binary mask.
- **Skull stripping:** done internally by HD-BET; set `already_stripped=True` only when
  **both** inputs are already skull-stripped.

### Model / weights provenance

- **LST-AI** — installed from the fork `jqmcginnis/LST-AI` at tag **`v1.3.0`** (a pip-only,
  multi-arch stack: ONNX Runtime inference, `picsl-greedy` registration, HD-BET v2; no
  TensorFlow). Models: a 3-model UNet3D ONNX ensemble + MNI atlas, fetched by LST-AI's
  `download_data` from the fork's v1.3.0 release and **baked into the image**.
- **HD-BET v2** weights — baked at build time.

Both are baked so the stack runs with `--network none` (no runtime downloads).

### Hardware requirements

GPU stack: a CUDA GPU is recommended (HD-BET on torch-CUDA, ONNX ensemble on the CUDA
execution provider). CPU fallback works but is substantially slower. The image builds on
`medmcp-base` (CUDA 12.8 runtime); `onnxruntime-gpu` is pinned to the matching CUDA-12 line.

---

## Architecture

Two environments by design (mirrors the FastSurfer pattern in `medmcp-neuro`):

- **`/opt/lst-venv`** — LST-AI and its heavy CUDA stack (torch cu128, onnxruntime-gpu),
  isolated and invoked as the `lst` CLI subprocess.
- **`/app/.venv`** — the light MCP wrapper (`mcp` + this package) that builds the `lst`
  command, runs it (with the venv bin on `PATH` and the CUDA libs on `LD_LIBRARY_PATH`),
  and parses the lesion mask + stats into the tool result.

This is a GPU stack (`org.medmcp.stack` → `"gpu": true`); the core launches it with
`--device nvidia.com/gpu=all` (CDI).

## Build & run

```bash
# build (needs the shared medmcp-base image: `just docker-base` in the core repo)
DOCKER_BUILDKIT=1 docker build -t medmcp-neuro-ms:dev .

# the core launches stacks; to exercise the tool directly:
docker run --rm --network none --device nvidia.com/gpu=all \
  -v /path/to/data:/data:ro -v /path/to/out:/out \
  --entrypoint /app/.venv/bin/python medmcp-neuro-ms:dev -c "
from pathlib import Path
from medmcp_neuro_ms.tools.segmentation import segment_ms_lesions
print(segment_ms_lesions(Path('/data/T1w.nii.gz'), Path('/data/FLAIR.nii.gz'),
                         output_dir=Path('/out'), device='auto'))"
```

Behind a TLS-intercepting proxy, drop the proxy root CA as `*.crt` into `./certs/`
(gitignored) before building.

## Development

```bash
just setup        # uv sync
just check        # ruff + pyright + pytest
```

## Citation

If you use this stack, please cite LST-AI:

> Wiltgen T, McGinnis J, Schlaeger S, et al. *LST-AI: A deep learning ensemble for accurate
> MS lesion segmentation.* NeuroImage: Clinical, Vol. 42, 2024.
> https://doi.org/10.1016/j.nicl.2024.103611
