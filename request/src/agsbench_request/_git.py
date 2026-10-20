"""Pinned git checkout with retries (GitHub access can be flaky)."""

import os
import subprocess
import time


def _git(*args, cwd=None, timeout=600):
    return subprocess.run(["git", *args], cwd=cwd, check=True, timeout=timeout)


def checkout_pinned(repo: str, commit: str, dest: str, retries: int = 8):
    """Ensure `dest` is a checkout of `repo` at `commit`.

    Fetches only the pinned commit (shallow) and retries with backoff,
    so an interrupted or stalled fetch can simply be re-run.
    """
    if os.path.isdir(os.path.join(dest, ".git")):
        head = subprocess.run(["git", "-C", dest, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        if head == commit:
            print(f"{dest} already at {commit[:8]}")
            return
    else:
        os.makedirs(dest, exist_ok=True)
        _git("init", "-q", dest)
        _git("-C", dest, "remote", "add", "origin", repo)

    for attempt in range(1, retries + 1):
        try:
            print(f"Fetching {repo} @ {commit[:8]} (attempt {attempt}/{retries})...")
            _git("-C", dest, "-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=60",
                 "fetch", "--depth", "1", "origin", commit)
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            if attempt == retries:
                raise RuntimeError(f"git fetch failed after {retries} attempts: {repo}") from e
            time.sleep(min(5 * attempt, 30))
    _git("-C", dest, "-c", "advice.detachedHead=false", "checkout", "-q", commit)
