"""Patch OSWorld's docker provider to reach the QEMU container via the
host gateway instead of localhost.

When OSWorld runs on the host directly, `localhost:<port>` lands on the
QEMU container's published port. When we run OSWorld inside the
os-world-service container, localhost is the service container's
loopback — not the host's. host.docker.internal resolves to the host
gateway (we add that mapping in the compose `extra_hosts`)."""

import importlib.util
import re
import sys
from pathlib import Path

# Resolve the installed package instead of hard-coding site-packages: the
# pytorch base image is conda-based (/opt/conda/lib/python3.11/...).
_DE = Path(importlib.util.find_spec("desktop_env").submodule_search_locations[0])
if not (_DE / "providers/docker/provider.py").exists():
    sys.exit(f"error: desktop_env docker provider not found under {_DE}")

candidates = [
    Path(
        _DE / "providers/docker/provider.py"
    ),
    Path(
        _DE / "providers/docker/manager.py"
    ),
    Path(_DE / "controllers/python.py"),
    Path(_DE / "controllers/setup.py"),
]

patched_any = False
for p in candidates:
    if not p.exists():
        continue
    t = p.read_text()
    # Comprehensive: replace `localhost` adjacent to `:`, in URLs, or
    # quoted. The provider.py returns `f"localhost:{port}:{port}..."`
    # which desktop_env then splits and uses as vm_ip — we need that
    # to become "host.docker.internal".
    new = re.sub(
        r"(https?://|//)localhost(:\d+)?",
        lambda m: m.group(1) + "host.docker.internal" + (m.group(2) or ""),
        t,
    )
    # Chrome's remote debugging rejects non-IP Host headers (security since Chrome 92+).
    # Use the Docker gateway IP directly instead of 'host.docker.internal' hostname.
    import socket as _socket

    try:
        _docker_host_ip = _socket.gethostbyname("host.docker.internal")
    except _socket.gaierror:
        _docker_host_ip = "host.docker.internal"  # fallback
    new = new.replace('"localhost"', f'"{_docker_host_ip}"')
    new = new.replace("'localhost'", f"'{_docker_host_ip}'")
    # Catch f-string and other bare uses like `f"localhost:{port}"`.
    new = re.sub(r"\blocalhost:\{", f"{_docker_host_ip}:" + "{", new)
    new = re.sub(r'f"localhost:', f'f"{_docker_host_ip}:', new)
    new = re.sub(r"f'localhost:", f"f'{_docker_host_ip}:", new)
    if new != t:
        p.write_text(new)
        print(f"patched {p}")
        patched_any = True

if not patched_any:
    print("warning: no localhost references found to patch", file=sys.stderr)


# === Privileged mode for QEMU container ===================================
#
# On ARM64 (GB200 / Grace) hosts, the QEMU container needs privileged
# mode to access /dev/kvm properly and /dev/net/tun for networking.
# Without it, KVM is reported as unavailable and tun fails with
# "Operation not permitted".

provider_priv_path = Path(
    _DE / "providers/docker/provider.py"
)
if provider_priv_path.exists():
    t = provider_priv_path.read_text()
    old_run = """self.container = self.client.containers.run(
                    "happysixd/osworld-docker",
                    environment=self.environment,
                    cap_add=["NET_ADMIN"],
                    devices=devices,"""
    new_run = """self.container = self.client.containers.run(
                    os.environ.get("OS_WORLD_DOCKER_IMAGE", "happysixd/osworld-docker"),
                    environment=self.environment,
                    privileged=True,"""
    if old_run in t:
        t = t.replace(old_run, new_run)
        provider_priv_path.write_text(t)
        print(f"patched QEMU container to use privileged mode in {provider_priv_path}")
    else:
        print(
            f"warning: could not find container.run call to patch for privileged mode",
            file=sys.stderr,
        )


# === VM disk revision pin + zip cleanup ===================================
#
# The qcow2 (xlangai/ubuntu_osworld on HuggingFace) is booted by the OVMF
# firmware inside happysixd/osworld-docker; the two are version-coupled.
# `resolve/main` drifts (re-uploaded 2025-08), so pin the download to the
# revision that matches the runtime image (2024-10-17). Override with
# OS_WORLD_VM_REVISION=<sha> (or `main`). Also delete the ~11 GB zip once
# it is extracted — upstream keeps it, doubling the disk footprint.

_DEFAULT_VM_REVISION = "1d71c917cb89be34ecb69768c808fe21483dd908"  # 2024-10-18
manager_path = Path(
    _DE / "providers/docker/manager.py"
)
if manager_path.exists():
    t = manager_path.read_text()
    old_get = "        with requests.get(URL, headers=headers, stream=True) as response:"
    new_get = (
        "        _rev = os.environ.get('OS_WORLD_VM_REVISION', "
        f"'{_DEFAULT_VM_REVISION}')\n"
        "        URL = URL.replace('resolve/main/', f'resolve/{_rev}/')\n"
        "        logger.info(f'Downloading VM image from {URL}')\n" + old_get
    )
    old_zip = "            zip_ref.extractall(vms_dir)\n"
    new_zip = old_zip + "        os.remove(downloaded_file_path)\n"
    if old_get in t and old_zip in t:
        t = t.replace(old_get, new_get).replace(old_zip, new_zip)
        manager_path.write_text(t)
        print(f"patched VM revision pin + zip cleanup in {manager_path}")
    else:
        print("warning: could not patch VM download in manager.py", file=sys.stderr)


# === ARM64 KVM and environment fixes ======================================
#
# The happysixd/osworld-docker image is x86-only. On ARM64 hosts it runs
# via QEMU user-mode emulation. The image's entry script checks for
# vmx/svm (x86 KVM flags) and exits with code 88 if not found, unless
# KVM=N is set. We also bump CPU_CORES to help compensate for emulation
# overhead.

import platform as _platform

if _platform.machine() in ("aarch64", "arm64"):
    if provider_priv_path.exists():
        t = provider_priv_path.read_text()
        old_env = """self.environment = {"DISK_SIZE": "32G", "RAM_SIZE": "4G", "CPU_CORES": "4"}"""
        new_env = """self.environment = {"DISK_SIZE": "32G", "RAM_SIZE": "8G", "CPU_CORES": "8", "KVM": "N"}"""
        if old_env in t:
            t = t.replace(old_env, new_env)
            provider_priv_path.write_text(t)
            print(
                f"patched QEMU env for ARM64 (KVM=N, more RAM/CPU) in {provider_priv_path}"
            )
        else:
            print(
                f"warning: could not find environment dict to patch for ARM64",
                file=sys.stderr,
            )


# === Port allocation start-value bump ====================================
#
# OSWorld's docker provider hard-codes port-allocation start values:
#   vnc_port      = 8006
#   server_port   = 5000
#   chromium_port = 9222
#   vlc_port      = 8080
#
# On a busy host (lots of unrelated containers + systemd socket-activated
# services) the next-free port the provider picks can collide with a
# port that's bound only on 127.0.0.1 — `psutil.net_connections()` sees
# it, but the provider's own `_get_used_ports` may not (when running
# inside our os-world-service container the host's loopback bindings
# aren't fully visible to the container's network namespace). The
# result: docker run tries to bind 0.0.0.0:8081 and fails with
# "address already in use".
#
# Bump the start values into a higher, less-contested range. Configurable
# via env vars so the operator can shift further if needed.
import os as _os

provider_path = Path(
    _DE / "providers/docker/provider.py"
)
if provider_path.exists():
    bumps = {
        "self._get_available_port(8006)": f"self._get_available_port({_os.environ.get('OS_WORLD_PORT_START_VNC', '18006')})",
        "self._get_available_port(5000)": f"self._get_available_port({_os.environ.get('OS_WORLD_PORT_START_SERVER', '15000')})",
        "self._get_available_port(9222)": f"self._get_available_port({_os.environ.get('OS_WORLD_PORT_START_CHROMIUM', '19222')})",
        "self._get_available_port(8080)": f"self._get_available_port({_os.environ.get('OS_WORLD_PORT_START_VLC', '18080')})",
    }
    t = provider_path.read_text()
    new = t
    for old, replacement in bumps.items():
        new = new.replace(old, replacement)
    if new != t:
        provider_path.write_text(new)
        print(f"patched port-allocation start values in {provider_path}")


# === VM boot timeout bump ================================================
#
# Default _wait_for_vm_ready(timeout=300) is 5 minutes. With 4 VMs
# booting concurrently there can be disk I/O contention on the qcow2
# images. Bump to 600s (10 min) to avoid spurious timeouts.
# Configurable via OS_WORLD_VM_BOOT_TIMEOUT env var.

_boot_timeout = int(_os.environ.get("OS_WORLD_VM_BOOT_TIMEOUT", "600"))
if provider_path.exists():
    t = provider_path.read_text()
    old_wait = "def _wait_for_vm_ready(self, timeout: int = 300):"
    new_wait = f"def _wait_for_vm_ready(self, timeout: int = {_boot_timeout}):"
    if old_wait in t:
        t = t.replace(old_wait, new_wait)
        provider_path.write_text(t)
        print(f"patched VM boot timeout to {_boot_timeout}s in {provider_path}")
    else:
        print(
            "warning: could not find _wait_for_vm_ready to patch timeout",
            file=sys.stderr,
        )


# === QEMU container CPU pinning ==========================================
#
# Pin each QEMU VM to a specific set of physical cores so concurrent VMs
# don't fight over the same cores and don't interfere with other jobs on
# the host. Configured via OS_WORLD_CPUSET_CPUS env var (e.g. "32-63").
# Only applied when the env var is set.

_cpuset = _os.environ.get("OS_WORLD_CPUSET_CPUS", "")
if _cpuset and provider_path.exists():
    t = provider_path.read_text()
    old_detach = "                    detach=True\n                )"
    new_detach = f'                    cpuset_cpus="{_cpuset}",\n                    detach=True\n                )'
    if old_detach in t:
        t = t.replace(old_detach, new_detach)
        provider_path.write_text(t)
        print(f"patched QEMU container cpuset_cpus={_cpuset!r} in {provider_path}")
    else:
        print(
            "warning: could not find detach=True to patch cpuset_cpus", file=sys.stderr
        )


# === Chrome CDP 500 recovery: detect and recover inside the retry loop ===
#
# When a VM snapshot has Chrome in a broken CDP state, /json/version returns
# HTTP 500. The broken state only becomes visible once the VM is reachable —
# probing before the loop catches ConnectionRefused (VM still booting), not 500.
# Fix: set a flag before the loop, detect "500" in the exception message on the
# first affected attempt, kill+restart Chrome inside the VM, then keep retrying.

_setup_path = Path(
    _DE / "controllers/setup.py"
)
if _setup_path.exists():
    _t = _setup_path.read_text()

    # Part A: add _cdp_chrome_restarted flag before the loop
    _old_cdp_flag = (
        '        logger.debug("PLAYWRIGHT ENV: %s", repr(os.environ))\n'
        "        for attempt in range(15):"
    )
    _new_cdp_flag = (
        '        logger.debug("PLAYWRIGHT ENV: %s", repr(os.environ))\n'
        "        _cdp_chrome_restarted = False\n"
        "        for attempt in range(15):"
    )
    if _old_cdp_flag in _t:
        _t = _t.replace(_old_cdp_flag, _new_cdp_flag)
        print(f"patched Chrome CDP 500 flag init in {_setup_path}")
    else:
        print(
            "warning: could not find chrome_open_tabs loop anchor for flag",
            file=sys.stderr,
        )

    # Part B: inside the except block, detect 500 and restart Chrome once
    _old_cdp_inner = (
        "                except Exception as e:\n"
        "                    if attempt < 14:\n"
        '                        logger.error(f"Attempt {attempt + 1}: Failed to connect, retrying. Error: {e}")\n'
        "                        # time.sleep(10)\n"
        "                        continue\n"
    )
    _new_cdp_inner = (
        "                except Exception as e:\n"
        '                    if "500" in str(e) and not _cdp_chrome_restarted:\n'
        '                        logger.warning("[OPEN_TABS_FIX] CDP 500 on attempt %d"\n'
        '                                       " - restarting Chrome in VM", attempt + 1)\n'
        "                        self._execute_setup(\n"
        '                            ["bash", "-c",\n'
        '                             "CHROME_PORT=$(pgrep -ax socat 2>/dev/null"\n'
        "                             \" | grep -o 'localhost:[0-9]*' | cut -d: -f2);\"\n"
        '                             " CHROME_PORT=${CHROME_PORT:-9222};"\n'
        '                             " pkill -x chrome || true; sleep 3;"\n'
        '                             " DISPLAY=:0 nohup /opt/google/chrome/chrome"\n'
        '                             " --remote-debugging-port=$CHROME_PORT"\n'
        '                             " --no-first-run --no-default-browser-check"\n'
        '                             " --disable-features=PrivacySandboxSettings4"\n'
        '                             " > /tmp/chrome-cdp-restart.log 2>&1 &"])\n'
        "                        import time as _cdp_t2; _cdp_t2.sleep(10)\n"
        '                        logger.warning("[OPEN_TABS_FIX] Chrome restarted after 500")\n'
        "                        _cdp_chrome_restarted = True\n"
        "                    if attempt < 14:\n"
        '                        logger.error(f"Attempt {attempt + 1}: Failed to connect, retrying. Error: {e}")\n'
        "                        # time.sleep(10)\n"
        "                        continue\n"
    )
    if _old_cdp_inner in _t:
        _t = _t.replace(_old_cdp_inner, _new_cdp_inner)
        _setup_path.write_text(_t)
        print(f"patched Chrome CDP inner 500 detection in {_setup_path}")
    else:
        print(
            "warning: could not find chrome_open_tabs inner except anchor to patch",
            file=sys.stderr,
        )


# === Chrome CDP connection retry count ====================================
#
# _chrome_open_tabs_setup retries CDP connection 15 times × 5s = 70s max.
# With concurrent VMs under load, Chrome can take longer to become ready,
# causing env_reset to return 500 and silently skip the task (reward=0).
# Increase via OS_WORLD_CDP_RETRIES env var (default 60 = ~5 min).

_setup_path = Path(
    _DE / "controllers/setup.py"
)
_cdp_retries = int(_os.environ.get("OS_WORLD_CDP_RETRIES", "60"))
if _setup_path.exists():
    _t = _setup_path.read_text()
    # Patch both the loop range and the guard condition
    _old_range = "for attempt in range(15):"
    _new_range = f"for attempt in range({_cdp_retries}):"
    _old_guard = "if attempt < 14:"
    _new_guard = f"if attempt < {_cdp_retries - 1}:"
    if _old_range in _t and _old_guard in _t:
        _t = _t.replace(_old_range, _new_range).replace(_old_guard, _new_guard)
        _setup_path.write_text(_t)
        print(
            f"patched Chrome CDP retries to {_cdp_retries} (~{_cdp_retries * 5}s) in {_setup_path}"
        )
    else:
        print("warning: could not find CDP retry pattern to patch", file=sys.stderr)


# === Chrome CDP Host-header fix ==========================================
#
# Chrome 92+ rejects DevTools connections where the HTTP Host header is not
# an IP address or 'localhost'. When setup.py constructs
#   remote_debugging_url = f"http://{host}:{port}"
# with host="host.docker.internal", Chrome returns HTTP 500:
#   "Host header is specified and is not an IP address or localhost."
#
# Fix: resolve host to its IP before building the CDP URL. Affects all
# three _chrome_*_setup methods that use remote_debugging_url.

_setup_path = Path(
    _DE / "controllers/setup.py"
)
if _setup_path.exists():
    _t = _setup_path.read_text()
    _old_cdp_url = '        remote_debugging_url = f"http://{host}:{port}"'
    _new_cdp_url = (
        "        import socket as _cdp_sock\n"
        "        try:\n"
        "            _cdp_host_ip = _cdp_sock.gethostbyname(host)\n"
        "        except Exception:\n"
        "            _cdp_host_ip = host\n"
        '        remote_debugging_url = f"http://{_cdp_host_ip}:{port}"'
    )
    if _old_cdp_url in _t:
        _t = _t.replace(_old_cdp_url, _new_cdp_url)
        _setup_path.write_text(_t)
        count = _t.count(_new_cdp_url.split("\n")[-1])
        print(f"patched Chrome CDP host→IP resolution ({count} sites) in {_setup_path}")
    else:
        print(
            "warning: could not find remote_debugging_url anchor to patch",
            file=sys.stderr,
        )


# === HF_ENDPOINT for task-file downloads ==================================
#
# Task setup ("download") and evaluators (cloud_file gold files) fetch from
# huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache inside
# os-world-service. Upstream honours HF_ENDPOINT only for the VM image, so
# on hosts without huggingface.co access every task fails at setup.
# Rewrite the host to HF_ENDPOINT (if set) for those downloads too.

_HF_HELPER = '''
def _hf_url(url):
    endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    if endpoint and url.startswith("https://huggingface.co/"):
        return endpoint + url[len("https://huggingface.co"):]
    return url
'''
_site = _DE
for _rel, _calls in [
    ("controllers/setup.py", ["requests.get(url, stream=True", "requests.get(db_url, stream=True"]),
    ("evaluators/getters/file.py", ["requests.get(url, stream=True"]),
]:
    _p = _site / _rel
    if not _p.exists():
        continue
    _t = _p.read_text()
    _n = 0
    for _c in _calls:
        _var = _c[len("requests.get("):].split(",")[0]
        _n += _t.count(_c)
        _t = _t.replace(_c, f"requests.get(_hf_url({_var}), stream=True")
    if _n and "\nimport requests\n" in _t:
        if not re.search(r"^import os$", _t, re.M):
            _t = _t.replace("\nimport requests\n", "\nimport os\nimport requests\n", 1)
        _t = _t.replace("\nimport requests\n", "\nimport requests\n" + _HF_HELPER, 1)
        _p.write_text(_t)
        print(f"patched HF_ENDPOINT rewrite ({_n} sites) in {_p}")
    else:
        print(f"warning: could not patch HF_ENDPOINT rewrite in {_p}", file=sys.stderr)


# === VM network mode ======================================================
#
# The image's default tap networking (in-container bridge + dnsmasq DHCP +
# vhost-net) fails on some hosts: dnsmasq's DHCPOFFER never reaches the
# guest's tap, the guest never gets an IP and the provider times out with
# "VM failed to become ready". QEMU user-mode networking (slirp) sidesteps
# the bridge entirely: QEMU's built-in DHCP hands out the address, egress
# is NATed through the host's own IP, and the ports the provider maps are
# forwarded with hostfwd. Lower throughput, but portable.
# OS_WORLD_VM_NETWORK=tap restores the image default.

if provider_priv_path.exists():
    t = provider_priv_path.read_text()
    m = re.search(r"^( *)self\.container = self\.client\.containers\.run\(", t, re.M)
    if m and "OS_WORLD_VM_NETWORK" not in t:
        ind = m.group(1)
        inject = (
            f"{ind}if os.environ.get('OS_WORLD_VM_NETWORK', 'user') == 'user':\n"
            f"{ind}    self.environment.update({{'NETWORK': 'user', 'USER_PORTS': '5000,9222,8080'}})\n"
        )
        t = t[: m.start()] + inject + t[m.start():]
        provider_priv_path.write_text(t)
        print(f"patched VM network mode (OS_WORLD_VM_NETWORK, default user) in {provider_priv_path}")
    else:
        print("warning: could not patch VM network mode in provider.py", file=sys.stderr)
