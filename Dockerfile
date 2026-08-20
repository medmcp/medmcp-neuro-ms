# syntax=docker/dockerfile:1
#
# medmcp-neuro-ms — MS white-matter lesion segmentation stack as a fixed-environment
# MCP stdio server. Wraps LST-AI v2 (native-PyTorch UNet3D ensemble + picsl-greedy
# registration + HD-BET skull strip + FastSurfer-based region annotation). Launched
# by the core via `docker run -i --device nvidia.com/gpu=all` (GPU stack; CDI).
#
# Two environments by design:
#   /app/.venv     — the light MCP wrapper (just `mcp` + this package).
#   /opt/lst-venv  — LST-AI + FastSurfer and their heavy CUDA stack (one torch cu128
#                    covers HD-BET, the UNet3D ensemble AND FastSurferVINN), kept
#                    isolated and invoked as the `lst` subprocess. Mirrors the
#                    FastSurfer pattern in medmcp-neuro-core.
#
# Multi-arch: linux/amd64 and linux/arm64. Everything resolves from PyPI wheels on
# both arches — LST-AI v2 pinned brainles_hd_bet precisely because it is the only
# HD-BET with an aarch64 wheel, and picsl-greedy ships official aarch64 wheels since
# 1.4.0.1 (manylinux_2_28; the ubuntu24.04 base is glibc 2.39).
ARG BASE_IMAGE=medmcp-base:dev

FROM ${BASE_IMAGE} AS runtime

# Stack metadata for one-click install/discovery (read via `docker inspect`). GPU stack.
# tool_timeout_sec 3600: the v2 annotation path runs a FastSurfer seg-only pass, which
# on CPU (the arm64 reality today) can push a run well past the old 1800s budget.
LABEL org.medmcp.stack='{"name": "medmcp-neuro-ms", "gpu": true, "tool_timeout_sec": 3600, "skills_path": "/app/src/medmcp_neuro_ms/skills"}'

# libgomp1: OpenMP runtime for torch/scipy/greedy. No git needed — LST-AI installs
# from PyPI and fetches FastSurfer as a release tarball itself.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Trust extra CA certs at build time behind a TLS-intercepting (MITM) proxy so
# uv/pip/git fetch through it. Drop the proxy root CA as a *.crt into ./certs/
# (gitignored; empty = no-op for CI / non-proxied builds). UV_NATIVE_TLS makes uv use
# the system trust store. Runtime is offline, so no production impact.
COPY certs/ /usr/local/share/ca-certificates/medmcp-extra/
RUN update-ca-certificates
ENV UV_NATIVE_TLS=1

WORKDIR /app

# Python downloaders (HD-BET weights, LST-AI model/atlas bundle, FastSurfer
# checkpoints) use requests/urllib, which trust certifi's bundle — not the system
# store — so point them at the updated bundle to fetch through a MITM proxy.
# Harmless without a proxy CA; runtime is offline.
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# ── LST-AI env (/opt/lst-venv) ──────────────────────────────────────────────────
# Built BEFORE copying src so wrapper-code edits don't re-trigger this ~10GB install.
# LST-AI v2 runs the UNet3D ensemble natively in PyTorch (.pt checkpoints — no ONNX
# Runtime: the ORT CUDA arena transiently grabbed ~40 GB at session init and OOM'd
# under GPU contention; torch's caching allocator stays at a few GB). The same torch
# covers HD-BET (brainles_hd_bet) and FastSurfer. torch is pinned to the CUDA 12.8
# (cu128) build so it runs on any host driver >= R570.
# Installed from PyPI; an exact pre-release pin (==2.0.0rc1) is an explicit
# pre-release request under PEP 440, so uv resolves it without extra flags. Bump to
# the final 2.0.0 once tagged. picsl-greedy resolves from PyPI on both arches.
ARG LST_AI_VERSION=2.0.0rc1
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.12 /opt/lst-venv \
 && uv pip install --python /opt/lst-venv \
        --torch-backend=cu128 --index-strategy unsafe-best-match \
        "lst-ai==${LST_AI_VERSION}"

# ── FastSurfer (region annotation) ──────────────────────────────────────────────
# LST-AI v2 annotates lesions from a FastSurfer aseg (seg-only FastSurferVINN — no
# FreeSurfer license). lst_ai installs and resolves its own pinned FastSurfer
# (v2.5.4 release tarball → /opt/lst-venv/share/lst-ai/FastSurfer-<ref>); the
# seg-path Python deps are ordinary wheels in lst-ai's install_requires, already in
# /opt/lst-venv above, so LST-AI and FastSurfer share the one CUDA stack. A wheel
# install has no setup.py hook, so run the installer module explicitly;
# --checkpoints also pre-fetches the FastSurferVINN (asegdkt) checkpoints the
# seg-only path runs.
RUN /opt/lst-venv/bin/python -m lst_ai.fastsurfer --checkpoints

# ── Bake ALL model weights (the stack runs with --network none) ─────────────────
# Every artifact the pipeline can touch is fetched at build time and ASSERTED, so a
# container never starts work it cannot finish:
#   - LST-AI .pt ensemble + MNI atlas — resolve_data_dir() prefers the lst_ai
#     package directory once the bundle is there, so that is where download_data()
#     unpacks (bundle: the code-decoupled v2.0.0-data release).
#   - HD-BET parameters, one per fold (brainles_hd_bet).
#   - FastSurfer tree + FastSurferVINN (asegdkt) checkpoints — installed above;
#     asserted via the same lookup (find_fastsurfer) the lst CLI uses at run time.
# A silent fetch failure here would ship an image that only appears to work until a
# bind mount stops papering over the gap — hence the asserts, not just the downloads.
RUN /opt/lst-venv/bin/python -c "\
import os, lst_ai; from lst_ai.utils import download_data; \
p = os.path.dirname(lst_ai.__file__); download_data(path=p); \
assert os.listdir(os.path.join(p, 'model')), 'LST-AI model weights missing'; \
assert os.listdir(os.path.join(p, 'atlas')), 'MNI atlas missing'; \
print('LST-AI weights:', sorted(os.listdir(os.path.join(p, 'model'))))" \
 && /opt/lst-venv/bin/python -c "\
import os; from brainles_hd_bet.utils import maybe_download_parameters, get_params_fname; \
[maybe_download_parameters(f) for f in range(5)]; \
missing = [f for f in range(5) if not os.path.isfile(get_params_fname(f))]; \
assert not missing, f'HD-BET params missing for folds: {missing}'; \
print('HD-BET params:', [os.path.basename(get_params_fname(f)) for f in range(5)])" \
 && /opt/lst-venv/bin/python -c "\
import glob; from lst_ai.fastsurfer import find_fastsurfer; \
home = find_fastsurfer(); \
assert home is not None, 'FastSurfer tree missing'; \
ck = sorted(glob.glob(str(home / 'checkpoints' / '*'))); \
assert ck, 'FastSurfer asegdkt checkpoint missing'; \
print('FASTSURFER_HOME:', home); print('FastSurfer checkpoints:', ck)"

# ── MCP wrapper env (/app/.venv) ────────────────────────────────────────────────
# Frozen install from the committed lock (build-time network; runtime offline). Kept
# last so iterating on the wrapper rebuilds only this cheap layer, not LST-AI above.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# No FastSurfer on PATH: lst_ai resolves its own managed tree (find_fastsurfer)
# and deliberately ignores PATH / FASTSURFER_HOME.
ENV PATH=/app/.venv/bin:$PATH \
    LST_AI_VENV=/opt/lst-venv \
    UV_NO_SYNC=1

# stdio MCP server. tini reaps the process and forwards signals; stdio passes through.
ENTRYPOINT ["tini", "--", "medmcp-neuro-ms"]
