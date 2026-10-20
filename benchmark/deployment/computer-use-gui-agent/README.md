# Computer-Use × GUIAgent

Benchmark the GUIAgent on OSWorld desktop computing tasks via QEMU VMs.

## Overview

| Dimension | Value |
|-----------|-------|
| Scenario | computer-use |
| Agent | gui-agent |
| Dataset | OSWorld evaluation_examples (pinned commit) |
| Evaluation | Task-specific evaluator (env reward) |
| Hardware | CPU + KVM (`/dev/kvm` required); ~12 GB disk for qcow2 VM |

## Prerequisites

- Docker 24+ and Docker Compose v2
- KVM support on host (`/dev/kvm` must exist — check with `ls /dev/kvm`)
- A multimodal LLM API key (e.g., Kimi k2.6 via DashScope)
- ~24 GB free disk for the Ubuntu qcow2 VM image (downloaded on **first run only**)

> **First-run VM download**: The agent uses OSWorld's `desktop-env` package,
> which downloads an Ubuntu qcow2 VM image (~11 GB) from HuggingFace on first
> use. If `huggingface.co` is unreachable, set `HF_ENDPOINT=https://hf-mirror.com`
> in `.env` — `desktop-env` reads this env var and redirects the download.
> The image is cached at `OS_WORLD_DATA_DIR` (default:
> `agents/gui-agent/.cache/vm`, set in `agents/gui-agent/config/.env`) and
> reused on subsequent runs.
>
> See [docs/china-mirrors.md](../../../docs/china-mirrors.md#huggingface-datasets-models)
> for more HuggingFace mirror options.

## Quick Start

### 1. Prepare data

```bash
cd benchmark/deployment/computer-use-gui-agent
bash scripts/1.prepare_data.sh
```

This clones the OSWorld repo (pinned commit) and generates a manifest of
task JSON files.

### 2. Run

```bash
bash scripts/2.run.sh
```

> The first run will take 20-40 minutes due to the VM image download.
> Subsequent runs reuse the cached image and start in ~30 seconds.

### 3. Observe

- Grafana: http://localhost:3003
- Prometheus: http://localhost:9094
- Logs: `docker logs -f os-world-loadgen`

### 4. Cleanup

```bash
bash scripts/3.cleanup.sh
```

### 5. Analyze

From the repo root:

```bash
python benchmark/analysis/run.py computer-use-gui-agent
```

Report (figures, CSVs, `summary.txt`) goes to `benchmark/analysis/out/computer-use-gui-agent/`.
Raw data is in `benchmark/data/computer-use-gui-agent/`; the previous run is moved to
`benchmark/data/_archive/` on each `2.run.sh` (`AGSBENCH_KEEP_DATA=1` to append).
See [benchmark/analysis](../../analysis/README.md#raw-data-layout) for the data layout.

## Configuration

Key environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OS_WORLD_MODEL` | `kimi-k2.6` | Multimodal LLM for screenshot analysis |
| `OS_WORLD_OBSERVATION_TYPE` | `screenshot` | `a11y_tree` (text-only, any LLM), `screenshot` (needs VLM), or `screenshot_a11y_tree` |
| `OS_WORLD_DATA_DIR` | `/data/agent-sys-bench` | Host path for qcow2 VM images (cached) |
| `OS_WORLD_MAX_STEPS` | `15` | Max agent steps per task |
| `OS_WORLD_REQUEST_NUM` | `1` | Number of tasks to run (paper scale: 358) |
| `HF_ENDPOINT` | *(empty)* | HuggingFace mirror for VM download (e.g., `https://hf-mirror.com`) |

### Observation modes

| Mode | LLM requirement | Description |
|------|----------------|-------------|
| `screenshot` | VLM (Kimi k2.6, GPT-4o) | Agent sees PNG screenshots of the desktop |
| `a11y_tree` | Any LLM (DeepSeek, Qwen) | Agent reads text accessibility tree (no vision needed) |
| `screenshot_a11y_tree` | VLM | Both screenshot and accessibility tree |

For non-VLM models, use `a11y_tree` mode — it works with any OpenAI-compatible
LLM but loses accuracy on visual tasks.

## Architecture

```
os-world-loadgen ──HTTP──▶ os-world-service ──Docker-in-Docker──▶ QEMU VM (Ubuntu)
(agsbench-request:v1)        (gui-agent:v1)         /dev/kvm passthrough
                                 │
                                 ▼
                            LLM API (Kimi k2.6)
```

Each task spins up a full Ubuntu VM via QEMU-in-Docker. The agent observes
the desktop (screenshot or accessibility tree), picks pyautogui actions
via the LLM, and executes them. The task's built-in evaluator scores the result.

The VM image (~11 GB Ubuntu qcow2) is downloaded once from HuggingFace and
cached at `OS_WORLD_DATA_DIR`. Subsequent runs revert to a clean snapshot
per task — no re-download needed.

## Notes

- Uses Python 3.11 (not 3.12) — `desktop-env` has transitive deps incompatible with 3.12
- Image is based on `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (torch pre-installed for easyocr)
- Requires `privileged: true` + `/dev/kvm` device passthrough in Docker Compose

## VM version pinning

The QEMU runtime (`happysixd/osworld-docker`) and the Ubuntu VM disk
(`xlangai/ubuntu_osworld` on HuggingFace) are **version-coupled**: the
qcow2 boots via the OVMF firmware inside the docker image. If the qcow2
is updated (newer shim/UEFI) while the OVMF stays old, the VM hangs at
the bootloader (`BdsDxe: starting Boot0007` / `SeaBIOS: Booting from
Hard Disk`) and never reaches the screenshot endpoint.

To avoid this drift, pin both:

- **Runtime image** — `OS_WORLD_DOCKER_IMAGE` (default
  `happysixd/osworld-docker:latest`, digest
  `sha256:fe8d9a5e5ad6c593d059887ea2c790481b3f32dd42fa961f2441cdbbe2c70cf4`,
  built 2024-10-17). For an immutable pin, set it to
  `happysixd/osworld-docker@sha256:fe8d9a5e5ad6...`.
- **VM disk revision** — `OS_WORLD_VM_REVISION` pins the qcow2 to a
  specific HuggingFace commit (`resolve/<sha>/Ubuntu.qcow2.zip` instead
  of `resolve/main/`). Set this to the commit that was current when the
  runtime image above was built, so image and disk stay matched.

Both overrides are read by the `desktop_env` provider/manager, patched at
image build time by `agents/gui-agent/patches/osworld-provider-patch.py`.
The VM boots with QEMU user-mode networking by default
(`OS_WORLD_VM_NETWORK=user`); set `tap` for the image's bridge + DHCP mode.
See [docs/limitations.md](../../../docs/limitations.md) §5 for the download
caveat, §8 for the boot-compatibility note and §9 for networking.
