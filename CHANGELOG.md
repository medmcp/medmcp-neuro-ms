# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Template alignment to the ecosystem standard (as in medmcp-totalsegmentator):
  `NOTICE` with per-component licenses and citations (LST-AI MIT, HD-BET AGPL-3.0,
  greedy Apache-2.0, FastSurfer Apache-2.0), `scripts/shared-files.txt` +
  `sync-from-template.sh`, the Template-drift workflow, and all-contributors.

### Changed

- README rewritten to the ecosystem documentation standard: tool/skill inventories,
  pipeline description, bundled-tools table with licenses, citation section with DOIs,
  hardware requirements, and template-sync docs.
- CONTRIBUTING regenerated from the current template; ci.yml re-synced (checkout v7).

- LST-AI now installs from PyPI (`lst-ai==2.0.0rc1`) instead of a git fork; picsl-greedy
  resolves from official PyPI wheels on both amd64 and arm64 (the custom aarch64 wheel
  and its build script are gone), and FastSurfer v2.5.4 is installed by LST-AI's own
  `lst_ai.fastsurfer` module rather than a git clone + requirements filtering.

### Added

- Container scaffold: `Dockerfile` + `.dockerignore` + `.devcontainer` + `just docker-build`; `org.medmcp.stack` label; `rename.sh` also renames the Dockerfile and devcontainer.json; dev-container-first contributor docs.

- Initial template scaffold: pyproject + uv, ruff + pyright strict, pytest, just, pre-commit
- GitHub Actions CI workflow (lint, format-check, pyright, pytest on py3.12 / 3.13)
- Contributor docs: README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY
- Issue and PR templates with medical-context PHI warnings
- Rename helper script for one-shot placeholder replacement
