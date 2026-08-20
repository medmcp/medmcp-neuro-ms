# medmcp-neuro-ms

Multiple-sclerosis lesion segmentation for the [medmcp](https://github.com/medmcp) ecosystem. Exposes an **MCP (Model Context Protocol) server** over stdio that an LLM agent can invoke to segment MS white-matter lesions and quantify lesion load from a T1w + FLAIR pair. Wraps [LST-AI](https://github.com/CompImg/LST-AI).

<p align="center">
  <a href="https://medmcp.ai"><b>medmcp.ai</b></a> ·
  <a href="https://github.com/medmcp/medmcp">Main repository</a>
</p>

> [!NOTE]
> **This repository is for developers** who build, extend, or run the MS lesion segmentation stack from source. **If you just want to use MedMCP, you don't need this repo** — install the MedMCP app and add this stack through the workspace UI (one-click install). See [medmcp.ai](https://medmcp.ai) or the [main repository](https://github.com/medmcp/medmcp) to get started.

> [!WARNING]
> MedMCP and its ecosystem are research software under active development and are
> **not licensed for clinical use**. LST-AI is a research-only tool for MS lesion
> segmentation; its output is an estimate, not a clinical finding.

---

## Tool inventory

| Tool name | Description | Inputs | Outputs |
|---|---|---|---|
| `segment_ms_lesions` | Segment MS white-matter lesions on a co-registered T1w + FLAIR pair and quantify lesion load | `t1_path: Path`, `flair_path: Path`, `output_dir: Path?`, `device: "auto"\|GPU id\|"cpu"`, `already_stripped: bool`, `annotate: bool`, `threads: int` | Binary lesion mask in FLAIR space, a McDonald-region-annotated map, per-lesion stats CSVs, `total_lesion_volume_mm3`, `lesion_count`, `region_volumes_mm3`, and the resolved device |
| `list_ms_lesion_regions` | List the McDonald-criteria lesion regions LST-AI annotates | — | 4 regions |

## Pipeline

`segment_ms_lesions` runs the full LST-AI v2 pipeline in one call: registration to MNI
space (greedy) → skull stripping (HD-BET) → a three-model UNet3D ensemble (native
PyTorch, `.pt` checkpoints) → optional region annotation derived from a FastSurfer
seg-only aseg. Both a T1w and a FLAIR of the same session are required — LST-AI is a
T1+FLAIR method and no other contrast can substitute.

- **Annotated vs. binary:** `annotate=True` (default) also produces a region-annotated
  map and per-region volumes; `annotate=False` (`--segment_only`) yields only the
  binary mask and skips the FastSurfer pass.
- **Skull stripping** is done internally by HD-BET; set `already_stripped=True` only
  when **both** inputs are already skull-stripped.

The annotated map assigns each lesion to one of the four McDonald-criteria regions —
`Periventricular`, `Juxtacortical`, `Subcortical`, `Infratentorial` — which are the
rows of the annotated stats CSV (volumes in mm³).

## Skill inventory

Skills are SKILL.md files the agent loads on demand to follow multi-step workflows. They are bundled under `src/medmcp_neuro_ms/skills/` and discovered automatically via `server_config()`.

| Skill name | Description |
|---|---|
| `ms-lesion-segmentation` | Workflow for MS lesion segmentation and lesion-load quantification. Covers identifying the T1w and FLAIR by contrast, when LST-AI cannot run (a single contrast is not enough), device selection, and reading total and per-region lesion volumes out of the stats CSVs in mm³. |

---

### Bundled tools

| Tool / weights | Used by | Source | License |
|---|---|---|---|
| LST-AI (`lst-ai`, pinned `2.0.0rc1`) | `segment_ms_lesions` | [upstream](https://github.com/CompImg/LST-AI); `.pt` ensemble + MNI atlas from the `v2.0.0-data` release, baked into the image | [MIT](https://github.com/CompImg/LST-AI/blob/main/LICENSE) |
| greedy (`picsl-greedy`) | registration to MNI | [upstream](https://github.com/pyushkevich/greedy), dependency | Apache-2.0 |
| HD-BET (`brainles_hd_bet`) | skull stripping | [upstream](https://github.com/BrainLesion/HD-BET); all 5 parameter folds baked | AGPL-3.0 |
| FastSurfer `v2.5.4` | region annotation (seg-only FastSurferVINN) | [upstream](https://github.com/Deep-MI/FastSurfer), installed by LST-AI; VINN checkpoints baked | [Apache-2.0](https://github.com/Deep-MI/FastSurfer/blob/dev/LICENSE) |

Everything the pipeline can touch is fetched **and asserted** at build time (and
re-checked offline in CI), so the stack runs with `--network none` and never starts a
job whose weights are missing.

### Citation

Results produced with this stack should cite the underlying work, not this package:

- **LST-AI** — Wiltgen T, McGinnis J, Schlaeger S, et al. LST-AI: A deep learning
  ensemble for accurate MS lesion segmentation. *NeuroImage: Clinical* 42:103611
  (2024). [doi:10.1016/j.nicl.2024.103611](https://doi.org/10.1016/j.nicl.2024.103611)
- **HD-BET** — Isensee F, et al. Automated brain extraction of multisequence MRI using
  artificial neural networks. *Human Brain Mapping* 40(17):4952–4964 (2019).
  [doi:10.1002/hbm.24750](https://doi.org/10.1002/hbm.24750)
- **greedy** — Yushkevich PA, et al. Fast Automatic Segmentation of Hippocampal
  Subfields and Medial Temporal Lobe Subregions in 3 Tesla and 7 Tesla T2-Weighted
  MRI. *Alzheimer's & Dementia* 12:P126–P127 (2016).
  [doi:10.1016/j.jalz.2016.06.205](https://doi.org/10.1016/j.jalz.2016.06.205)
- **FastSurfer** (for `annotate=True`) — cite both, as FastSurfer itself asks:
  Henschel L, et al. FastSurfer — A fast and accurate deep learning based neuroimaging
  pipeline. *NeuroImage* 219:117012 (2020).
  [doi:10.1016/j.neuroimage.2020.117012](https://doi.org/10.1016/j.neuroimage.2020.117012);
  and Henschel L, et al. FastSurferVINN: Building resolution-independence into deep
  learning segmentation methods. *NeuroImage* 251:118933 (2022).
  [doi:10.1016/j.neuroimage.2022.118933](https://doi.org/10.1016/j.neuroimage.2022.118933)

Full third-party attribution belongs in [`NOTICE`](NOTICE).

### Hardware requirements

- `segment_ms_lesions`: CUDA GPU recommended — HD-BET, the UNet3D ensemble, and
  FastSurfer all run on torch-CUDA; registration (greedy) is CPU either way. CPU
  fallback works but is substantially slower; `annotate=True` adds a FastSurfer
  seg-only pass, which is why the stack's tool timeout is 3600 s.
- Disk: the image is large — about 13 GB, most of it the CUDA/PyTorch stack (one
  torch cu128 shared by HD-BET, the ensemble, and FastSurfer) plus the baked weights.
- This is a GPU stack (`org.medmcp.stack` → `"gpu": true`); the core launches it with
  `--device nvidia.com/gpu=all` (CDI). Images are published for **linux/amd64 and
  linux/arm64** (multi-arch manifest).
- **Runs fully offline.** Every model is baked into the image and nothing is
  downloaded at run time.

---

## Development

### Develop in the dev container (recommended)

This repo ships a dev container (`.devcontainer/`) with the full toolchain
(Python 3.12 + uv, `just`, git, Docker CLI). It derives from the shared
`medmcp-base` image, so build that once from the core repo first (`just docker-base`
in a `medmcp` checkout). Then open the repo with the **Dev Container** action in
PyCharm (2024.2+) or **Reopen in Container** in VS Code — `uv sync` runs on first
start. See the core repo's [CONTRIBUTING](https://github.com/medmcp/medmcp/blob/main/CONTRIBUTING.md)
for IDE specifics.

### Local install (alternative)

```bash
just setup     # install uv, sync dev environment, register pre-commit hooks
just check     # lint + format-check + typecheck + tests
just fix       # auto-fix lint and format
```

For local agent use, install the stack into its own uv tool environment:

```bash
uv tool install --editable .
```

The package registers itself via the `[medmcp.stacks]` entry point. The local
agent autodiscovers it on the next session — no manual config needed.

### Container image (deployment)

```bash
just docker-build           # build the stack image (FROM medmcp-base)
```

It is a stdio MCP server. The medmcp **core** launches it on demand via a
`stacks.d/medmcp-neuro-ms.toml` manifest (`docker run -i …`; GPU stacks add
`--device nvidia.com/gpu=all`, CDI), so deployment nodes need no host Python
install. Build both architectures — the core refuses to install a foreign-arch
image rather than failing later with "exec format error". Pin any GPU/CUDA build
in `pyproject.toml` against the fleet driver floor (CUDA 12.8 / driver R570).

Two environments by design (mirrors the FastSurfer pattern in `medmcp-neuro-core`):
**`/opt/lst-venv`** holds LST-AI + FastSurfer and their shared CUDA stack, isolated
and invoked as the `lst` CLI subprocess; **`/app/.venv`** is the light MCP wrapper
(`mcp` + this package) that builds the `lst` command, runs it, and parses the lesion
mask + stats into the tool result.

Behind a TLS-intercepting proxy, drop the proxy root CA as `*.crt` into `./certs/`
(gitignored) before building.

### Staying in sync with the template

Files shared with [medmcp-template](https://github.com/medmcp/medmcp-template) are
listed in `scripts/shared-files.txt`. The **Template drift** workflow reports when
one of them diverges; `./scripts/sync-from-template.sh` pulls them back. A change
that belongs in every stack goes in the template, not here.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: fork, `just setup`, `just check`, open a PR against `main`.

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jqmcginnis"><img src="https://avatars.githubusercontent.com/u/33037028?v=4?s=100" width="100px;" alt="Julian McGinnis"/><br /><sub><b>Julian McGinnis</b></sub></a><br /><a href="https://github.com/medmcp/medmcp-neuro-ms/commits?author=jqmcginnis" title="Code">💻</a> <a href="https://github.com/medmcp/medmcp-neuro-ms/commits?author=jqmcginnis" title="Documentation">📖</a> <a href="#maintenance-jqmcginnis" title="Maintenance">🚧</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://allcontributors.org) specification — contributions of any kind are welcome!

## License

[Apache 2.0](LICENSE). Third-party tools, model weights, and templates bundled by
this stack retain their own licenses and are attributed in [`NOTICE`](NOTICE).
