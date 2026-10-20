/**
 * vsandbox — HTTP-based sandbox backend for openclaw benchmarking.
 *
 * Registers "vsandbox" as a custom sandbox backend. Instead of SSH,
 * tool calls are proxied via HTTP to a fast-service RPC server that
 * runs `docker exec` directly.
 *
 * Environment variables (set by run_task.py before launching openclaw):
 *   VSANDBOX_ENDPOINT     — URL of the vsandbox_service (e.g. http://host:1804)
 *   VSANDBOX_SANDBOX_ID   — sandbox container identifier from sandbox_create
 *   VSANDBOX_EXEC_CLIENT  — path to vsandbox_exec_client.py
 *   VSANDBOX_DAEMON_PORT  — TCP port of the local vsandbox_client_daemon (optional)
 *                            When set, runShellCommand() routes through the daemon
 *                            for HTTP keep-alive connection reuse, avoiding ~150ms
 *                            of per-call TCP setup overhead through the SSH tunnel.
 */

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { registerSandboxBackend } from "openclaw/plugin-sdk/sandbox";
import { execFile } from "node:child_process";
import http from "node:http";

/**
 * Spawn vsandbox_exec_client.py and collect stdout/stderr/exitCode.
 * Used by buildExecSpec fallback path.
 */
function spawnClient(args, stdin, signal) {
  const clientPath = process.env.VSANDBOX_EXEC_CLIENT || "vsandbox_exec_client.py";
  return new Promise((resolve, reject) => {
    const child = execFile("python3", [clientPath, ...args], {
      maxBuffer: 50 * 1024 * 1024, // 50 MB
      timeout: 600_000,
      signal,
    }, (error, stdout, stderr) => {
      if (error && error.killed) {
        reject(error);
        return;
      }
      resolve({
        stdout: Buffer.from(stdout, "binary"),
        stderr: Buffer.from(stderr, "binary"),
        code: error ? (error.code ?? 1) : 0,
      });
    });

    // Pipe stdin if provided
    if (stdin != null) {
      const data = typeof stdin === "string" ? Buffer.from(stdin) : stdin;
      child.stdin.write(data);
      child.stdin.end();
    } else {
      child.stdin.end();
    }
  });
}

/**
 * Call the vsandbox_client_daemon via HTTP.
 * The daemon proxies to <REMOTE_HOST> over a persistent keep-alive connection.
 */
function callDaemon(daemonPort, rpcPath, payload, signal) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({ req: payload });
    const opts = {
      hostname: "127.0.0.1",
      port: daemonPort,
      path: `/${rpcPath}`,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body),
      },
      signal,
    };

    const req = http.request(opts, (res) => {
      const chunks = [];
      res.on("data", (chunk) => chunks.push(chunk));
      res.on("end", () => {
        const respBody = Buffer.concat(chunks).toString("utf-8");
        if (res.statusCode >= 400) {
          reject(new Error(`Daemon HTTP ${res.statusCode}: ${respBody.slice(0, 500)}`));
          return;
        }
        try {
          resolve(JSON.parse(respBody));
        } catch (e) {
          reject(new Error(`Daemon response parse error: ${respBody.slice(0, 200)}`));
        }
      });
    });

    req.on("error", (e) => reject(new Error(`Daemon request error: ${e.message}`)));
    req.write(body);
    req.end();
  });
}

export default definePluginEntry({
  id: "vsandbox",
  name: "VSandbox Backend",
  description: "HTTP-based sandbox backend replacing SSH for openclaw benchmarking",

  register(api) {
    registerSandboxBackend("vsandbox", async (params) => {
      const endpoint = process.env.VSANDBOX_ENDPOINT;
      const sandboxId = process.env.VSANDBOX_SANDBOX_ID;
      const clientPath = process.env.VSANDBOX_EXEC_CLIENT || "vsandbox_exec_client.py";
      const daemonPort = process.env.VSANDBOX_DAEMON_PORT
        ? parseInt(process.env.VSANDBOX_DAEMON_PORT, 10)
        : 0;

      console.error(`[vsandbox-plugin] endpoint=${endpoint}, sandboxId=${sandboxId}, daemonPort=${daemonPort}`);

      if (!endpoint) {
        throw new Error("VSANDBOX_ENDPOINT env var is required");
      }
      if (!sandboxId) {
        throw new Error("VSANDBOX_SANDBOX_ID env var is required");
      }

      const workdir = params.cfg?.workspaceRoot || "/workspace";

      return {
        id: "vsandbox",
        runtimeId: sandboxId,
        runtimeLabel: `vsandbox:${sandboxId}`,
        workdir,

        async buildExecSpec({ command, workdir: execWorkdir, env, usePty }) {
          // buildExecSpec must return argv — always uses the client script.
          // (In practice this path is rarely used; runShellCommand dominates.)
          const argv = [
            "python3",
            clientPath,
            "--endpoint", endpoint,
            "--sandbox-id", sandboxId,
            "--command", command,
          ];
          if (execWorkdir) {
            argv.push("--workdir", execWorkdir);
          }
          if (env) {
            for (const [k, v] of Object.entries(env)) {
              argv.push("--env", `${k}=${v}`);
            }
          }

          return {
            argv,
            env: { ...process.env },
            stdinMode: "pipe-open",
          };
        },

        async runShellCommand({ script, args, stdin, allowFailure, signal }) {
          // Fast path: use daemon if available (persistent HTTP connection)
          if (daemonPort > 0) {
            try {
              const payload = {
                sandbox_id: sandboxId,
                script,
                args: args || [],
                stdin_b64: stdin
                  ? Buffer.from(typeof stdin === "string" ? stdin : stdin).toString("base64")
                  : "",
                timeout: 600,
              };

              const result = await callDaemon(
                daemonPort,
                "agsbench_openclaw.vsandbox_service.vsandbox_shell",
                payload,
                signal,
              );

              if (result.error) {
                throw new Error(`vsandbox_shell error: ${result.error}`);
              }

              const stdout = result.stdout_b64
                ? Buffer.from(result.stdout_b64, "base64")
                : Buffer.alloc(0);
              const stderr = result.stderr_b64
                ? Buffer.from(result.stderr_b64, "base64")
                : Buffer.alloc(0);
              const code = result.code ?? 0;

              if (!allowFailure && code !== 0) {
                const stderrStr = stderr.toString("utf-8").slice(0, 500);
                throw new Error(
                  `vsandbox shell command failed (code=${code}): ${stderrStr}`
                );
              }

              return { stdout, stderr, code };
            } catch (err) {
              if (allowFailure) {
                return {
                  stdout: Buffer.alloc(0),
                  stderr: Buffer.from(err.message),
                  code: 1,
                };
              }
              throw err;
            }
          }

          // Fallback: spawn vsandbox_exec_client.py (no daemon)
          const clientArgs = [
            "--run-shell",
            "--endpoint", endpoint,
            "--sandbox-id", sandboxId,
            "--script", script,
          ];
          if (args && args.length > 0) {
            clientArgs.push("--", ...args);
          }

          try {
            const result = await spawnClient(clientArgs, stdin, signal);
            if (!allowFailure && result.code !== 0) {
              const stderrStr = result.stderr.toString("utf-8").slice(0, 500);
              throw new Error(
                `vsandbox shell command failed (code=${result.code}): ${stderrStr}`
              );
            }
            return result;
          } catch (err) {
            if (allowFailure) {
              return {
                stdout: Buffer.alloc(0),
                stderr: Buffer.from(err.message),
                code: 1,
              };
            }
            throw err;
          }
        },
      };
    });
  },
});
