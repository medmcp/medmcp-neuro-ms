# syntax=docker/dockerfile:1
#
# medmcp-neuro-ms — MS white-matter lesion segmentation stack as a fixed-environment
# MCP stdio server. Wraps LST-AI (ONNX UNet3D ensemble + picsl-greedy registration +
# HD-BET v2 skull strip). Launched by the core via
# `docker run -i --device nvidia.com/gpu=all` (GPU stack; CDI).
#
# Two environments by design:
#   /app/.venv     — the light MCP wrapper (just `mcp` + this package).
#   /opt/lst-venv  — LST-AI and its heavy CUDA stack (torch cu128 for HD-BET AND the
#                    UNet3D ensemble via onnx2torch), kept isolated and invoked as the
#                    `lst` subprocess. Mirrors the FastSurfer pattern in medmcp-neuro.
ARG BASE_IMAGE=medmcp-base:dev
FROM ${BASE_IMAGE} AS runtime

# Stack metadata for one-click install/discovery (read via `docker inspect`). GPU stack.
LABEL org.medmcp.stack='{"name": "medmcp-neuro-ms", "gpu": true, "tool_timeout_sec": 1800, "skills_path": "/app/src/medmcp_neuro_ms/skills"}'

# git: to pip-install LST-AI from the fork. libgomp1: OpenMP runtime for torch/scipy/greedy.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
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

# Python downloaders (HD-BET weights, LST-AI model/atlas bundle) use requests/urllib,
# which trust certifi's bundle — not the system store — so point them at the updated
# bundle to fetch through a MITM proxy. Harmless without a proxy CA; runtime is offline.
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# ── LST-AI env (/opt/lst-venv) ──────────────────────────────────────────────────
# Built BEFORE copying src so wrapper-code edits don't re-trigger this ~10GB install.
# LST-AI runs its UNet3D ensemble through PyTorch (via onnx2torch), the same torch the
# fork uses for HD-BET — so a single CUDA stack covers brain extraction AND inference.
# torch is pinned to the CUDA 12.8 (cu128) build so it runs on any host driver >= R570;
# onnx + onnx2torch come from the fork's deps. (Dropped onnxruntime-gpu: the ORT CUDA
# arena spiked ~40 GB at init and OOM'd under GPU contention — see the fork's segment.py.)
ARG LST_AI_REF=v1.3.0
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.12 /opt/lst-venv \
 && uv pip install --python /opt/lst-venv \
        --torch-backend=cu128 --index-strategy unsafe-best-match \
        "lst-ai @ git+https://github.com/jqmcginnis/LST-AI@${LST_AI_REF}"

# Bake weights so the stack runs with --network none (no runtime download):
#  - LST-AI model + MNI atlas: download_data() unpacks beside the `lst` script, i.e. the
#    venv bin dir (the CLI resolves model/atlas relative to os.path.dirname(__file__)).
#  - HD-BET v2 parameters.
RUN /opt/lst-venv/bin/python -c "from LST_AI.utils import download_data; download_data('/opt/lst-venv/bin')" \
 && /opt/lst-venv/bin/python -c "from HD_BET.checkpoint_download import maybe_download_parameters; maybe_download_parameters()"

# ── MCP wrapper env (/app/.venv) ────────────────────────────────────────────────
# Frozen install from the committed lock (build-time network; runtime offline). Kept
# last so iterating on the wrapper rebuilds only this cheap layer, not LST-AI above.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH=/app/.venv/bin:$PATH \
    LST_AI_VENV=/opt/lst-venv \
    UV_NO_SYNC=1

# stdio MCP server. tini reaps the process and forwards signals; stdio passes through.
ENTRYPOINT ["tini", "--", "medmcp-neuro-ms"]
