# GUIAgent

OSWorld-based GUI agent that completes desktop computing tasks by observing screenshots and driving keyboard/mouse actions via QEMU VMs.

## Overview

| Dimension | Value |
|-----------|-------|
| Package | `agsbench_gui_agent` |
| Image | `gui-agent:v1` |
| Scenarios served | computer-use |
| Hardware | CPU + KVM (`/dev/kvm` required); ~30 GB disk |

## Build

```bash
cd /path/to/agent-sys-bench
bash scripts/setup.sh
bash scripts/configure-api.sh --llm-preset dashscope --llm-key sk-xxx
cd agents/gui-agent
bash scripts/build_image.sh
```

## Serving Interface

| Scenario | RPC Endpoint | Request Type | Result Type |
|----------|-------------|--------------|-------------|
| computer-use | `agsbench_gui_agent.request.solve_task` | `ComputerTaskRequest` | `ComputerTaskResult` |

## Deployment

Experiments run from `benchmark/deployment/computer-use-gui-agent/`.
See [deployment README](../../benchmark/deployment/computer-use-gui-agent/README.md).

## Configuration

| File | Purpose |
|------|---------|
| `config/.env.example` | LLM endpoint, model, observation type, VM provider |
| `config/os_world-server.yml` | fast-service server config (port 1904) |
| `config/os_world-client.yml` | fast-service client config |

## Notes

- Uses Python 3.11 (not 3.12) — `desktop-env` has transitive deps incompatible with 3.12
- `desktop_env` / `mm_agents` come from OSWorld at a pinned commit (`OSWORLD_COMMIT` in `scripts/build_image.sh`), copied over PyPI `desktop-env==1.0.2`, which supplies the py3.11 dependency set but lacks `mm_agents.kimi`
- The kimi agent runs upstream's thinking-mode prompts; on DashScope set `OS_WORLD_LLM_EXTRA_BODY={"enable_thinking": true}` (as in `.env.example`) or the model drifts out of the expected response format after a few steps
- Requires a **multimodal LLM** (e.g., Kimi k2.6) for screenshot observation
- Uses `a11y_tree` mode by default for non-VLM models (text-only accessibility tree)
- `patches/` directory contains build-time patches for fast-service 3.11 compat and the OSWorld docker provider (VM disk pin, user-mode VM networking, port allocation)
