# Known Limitations

This document records known architectural limitations and areas for future
improvement in AgentSysBench. These are not bugs — they are design trade-offs or
incomplete features that users should be aware of.

## 1. File-path-based request passing (not fully HTTP-decoupled)

**Affected scenarios**: qa, tool-rich-assistant, office-work

**Description**: The scenario-protocol design goal is that loadgen and agent
communicate purely via HTTP — the loadgen sends a self-contained Request
payload, the agent processes it and returns a Result. However, 3 of 8
scenarios currently pass **file paths** in the Request instead of inline
content:

| Scenario | Field | What the agent reads from the path |
|----------|-------|-----------------------------------|
| qa | `uri` | Document content for chunking + embedding |
| tool-rich-assistant | `task_md_path` | Task .md file (instructions + workspace_files) |
| office-work | `task_md_path` | Same as above |

**Why it works now**: The loadgen and agent service containers share volume
mounts (`.cache/`, `tasks/`), so the agent can access files at the paths
the loadgen sends.

**Impact**: Breaks in multi-node deployment where loadgen and agent are on
different hosts. The agent host doesn't have the files at those paths.

**Fix direction**: Loadgen should read file content at manifest-parse time
and send it inline in the Request (e.g., `task_md_content: str` instead of
`task_md_path: str`). Agent side should accept inline content with fallback
to path-based access for backward compatibility.

**Priority**: Medium — single-node deployment (the default) works correctly.

## 2. webagent Chromium download may fail on restricted networks

**Affected agent**: webagent

**Description**: The webagent Dockerfile runs `playwright install chromium`
which downloads Chromium from `cdn.playwright.dev`. This CDN is blocked in
some regions (e.g., China). The Dockerfile sets
`PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright` as a
mirror, but the mirror may not have all versions.

**Workaround**: Pre-stage the browser binary on a machine with network
access, then `COPY` it into the Docker image. See
[docs/china-mirrors.md](china-mirrors.md) for details.

## 3. `configure-api.sh` LiteLLM proxy build may hang

**Affected script**: `scripts/configure-api.sh`

**Description**: When the LLM provider doesn't support the Responses API
(DashScope, DeepSeek), `configure-api.sh` automatically builds the
`litellm-proxy:local` image. This build uses `pip install 'litellm[proxy]==1.105.1'`
which can be slow or hang on restricted networks if the PIP mirror isn't
configured.

**Workaround**: Run `scripts/setup.sh` first (sets PIP mirror to aliyun),
or build the image manually:
```bash
docker build -t litellm-proxy:local -f - . << 'EOF'
FROM python:3.12
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
RUN pip config set global.index-url ${PIP_INDEX_URL} && pip install 'litellm[proxy]==1.105.1'
EXPOSE 4000
CMD ["litellm", "--port", "4000"]
EOF
```

## 4. HuggingGPT and GUIAgent require specialized hardware

**Affected agents**: hugginggpt, gui-agent

**Description**: All 10 deployment settings under `benchmark/deployment/`
are verified default experiment settings. HuggingGPT and GUIAgent are part
of the default suite but require specialized hardware that the 8 CPU-only
agents do not:

- **HuggingGPT** (`multimodal-hugginggpt`) — multi-GPU host (runs 7 local
  HF models concurrently). Verified on `wei_server` (8× RTX 4090). Use the
  DeepSeek official API (`api.deepseek.com`) for the LLM stages, since the
  GPU host typically cannot reach DashScope. See
  [DEPLOYMENT_NOTES.md](../../DEPLOYMENT_NOTES.md) §1.3/§2.10 and the
  [deployment README](../benchmark/deployment/multimodal-hugginggpt/README.md).
- **GUIAgent** (`computer-use-gui-agent`) — KVM (`/dev/kvm`) for QEMU-in-Docker
  VMs, and a one-time ~11 GB Ubuntu qcow2 download. Verified on a host with
  `/dev/kvm` available. See §5 below for the VM download caveat.

Both agents are fully integrated: `agsbench_` package naming
(`agsbench_hugginggpt`, `agsbench_gui_agent`), scenario-protocol modules,
deployment directories, and unified loadgen / prepare_data are all in place.

## 5. GUIAgent QEMU VM image download (~11 GB)

**Affected agent**: gui-agent (computer-use scenario)

**Description**: The `desktop-env` package downloads an Ubuntu qcow2 VM image
(~11 GB zipped, ~24 GB unzipped) from
`huggingface.co/datasets/xlangai/ubuntu_osworld` on first run. This is a
one-time download — the image is cached at `OS_WORLD_DATA_DIR` (default
`agents/gui-agent/.cache/vm`) and reused on subsequent runs (snapshot revert
per task); the zip is deleted after extraction.

**Workaround**: Set `HF_ENDPOINT=https://hf-mirror.com` in the deployment
`.env` file. `desktop-env` reads this env var and redirects the download to
the mirror. The deployment compose wires this var into the `os-world-service`
container via its `environment:` section, so setting it in the deployment
`.env` (or shell) is sufficient — no need to edit `config/.env`. See the
[deployment README](../benchmark/deployment/computer-use-gui-agent/README.md)
for details.

## 6. SWE-bench evaluation requires HuggingFace access

**Affected scenarios**: swe, terminal-use

**Description**: The `4.evaluate.sh` script in deployment directories runs
the SWE-bench harness, which downloads dataset metadata from HuggingFace.
On restricted networks, this fails. The script passes
`HF_ENDPOINT=https://hf-mirror.com` as a default, but the mirror may not
have all dataset revisions.

**Workaround**: Set `HF_ENDPOINT` to a reachable mirror or pre-download
the dataset.

## 7. No multi-node deployment support in deployment scripts

**Description**: The deployment scripts (`2.run.sh`, `3.cleanup.sh`) assume
single-node deployment (loadgen and agent on the same Docker host).
Multi-node mode (loadgen on one host, agent on another) was supported in
the original codebase via SSH tunnels and `_remote_helpers.sh`, but the
deployment scripts don't expose this.

**Priority**: Low — single-node is the default for benchmark suites.

## 8. GUIAgent VM image/runtime version coupling

**Affected agent**: gui-agent (computer-use scenario)

**Description**: The Ubuntu VM disk (`xlangai/ubuntu_osworld` on
HuggingFace, `Ubuntu.qcow2.zip`) is booted by the OVMF firmware baked
into the `happysixd/osworld-docker` QEMU runtime image. The two are
version-coupled: a newer qcow2 whose shimx64.efi / UEFI boot path
differs from what the runtime's OVMF expects will hang at the
bootloader (`BdsDxe: starting Boot0007` under UEFI, or
`SeaBIOS: Booting from Hard Disk` under legacy) and never expose the
screenshot endpoint — `desktop-env` then times out with
`VM failed to become ready within timeout period`.

At time of writing the runtime image has only a `:latest` tag (digest
`sha256:fe8d9a5e5ad6c593d059887ea2c790481b3f32dd42fa961f2441cdbbe2c70cf4`,
built 2024-10-17), while the qcow2 on HF `main` was last updated
2025-08-18 — i.e. the disk is newer than the runtime, which is the
suspected cause of the boot hang on some hosts.

**Workaround / fix**: pin both sides so they stay matched. The
`desktop_env` docker provider/manager, patched at image build time by
`agents/gui-agent/patches/osworld-provider-patch.py`, read two env vars:

- `OS_WORLD_DOCKER_IMAGE` — the QEMU runtime image ref (default
  `happysixd/osworld-docker:latest`; pin immutably with
  `happysixd/osworld-docker@sha256:fe8d9a5e5ad6...`).
- `OS_WORLD_VM_REVISION` — a HuggingFace commit sha for
  `xlangai/ubuntu_osworld`; the download URL becomes
  `resolve/<sha>/Ubuntu.qcow2.zip` instead of `resolve/main/`.

The image defaults `OS_WORLD_VM_REVISION` to
`1d71c917cb89be34ecb69768c808fe21483dd908` (2024-10-18), the HF commit
current when the runtime image was built, so image and disk don't drift
apart. Set it to `main` to track the latest disk. See the
[deployment README](../benchmark/deployment/computer-use-gui-agent/README.md#vm-version-pinning)
for usage.

## 9. GUIAgent VM networking (tap-mode DHCP)

**Affected agent**: gui-agent (computer-use scenario)

**Description**: `happysixd/osworld-docker` defaults to tap networking: an
in-container bridge with dnsmasq handing the guest a DHCP lease. On some
hosts the DHCP offer never reaches the guest's tap device (ICMP and
forwarded TCP still work), so NetworkManager times out, the guest has no
IP, and `desktop-env` fails with `VM failed to become ready within
timeout period`.

**Workaround / fix**: the provider patch starts the VM container with
QEMU user-mode networking (`NETWORK=user`, ports 5000/9222/8080 forwarded)
unless `OS_WORLD_VM_NETWORK=tap`. Guest egress is NATed through the host's
own IP, so no extra address is needed on the physical network. User mode is
slower than vhost-net tap, which matters little for a screenshot-driven
agent. Tasks that need `huggingface.co` or other sites blocked on the host
network will still fail inside the guest.
