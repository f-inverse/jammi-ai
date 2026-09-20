"""Helpers for testing code that talks to a Jammi engine.

:class:`LiveServer` runs a real ``jammi-server`` subprocess for the duration of
a ``with`` block — for a test suite, a notebook, or a script that needs a
``grpc://`` target and has a server binary to hand (the ``jammi-server`` wheel
puts one on ``PATH``).
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from typing import Mapping

from . import connect


class LiveServer:
    """A real `jammi-server` on kernel-assigned ports — announced, readiness-polled, torn down."""

    # The server's own fixed startup banner, printed between bind and serve.
    BANNER = "jammi-server listening "

    def __init__(
        self,
        artifact_dir: "str | os.PathLike[str]",
        *,
        server_bin: "str | None" = None,
        env: "Mapping[str, str] | None" = None,
    ) -> None:
        # The binary to run; by default the `jammi-server` on PATH.
        self._server_bin = server_bin
        # Deployment knobs for this server, read by its configuration loader
        # exactly as an operator's environment would be (for example
        # `JAMMI_WORKER__ENABLED=false`). Applied over the harness's own keys.
        self._env = dict(env or {})
        # The artifact directory is the caller's: two servers can open the same
        # directory in sequence, and a catalog an embedded session seeded and
        # released can be served as it stands.
        self._artifact_dir = os.fspath(artifact_dir)
        # Set by `__exit__` from the awaited child exit; `None` while running.
        self.returncode = None

    def __enter__(self) -> LiveServer:
        server_bin = self._server_bin or shutil.which("jammi-server")
        if not server_bin:
            raise RuntimeError(
                "jammi-server is not on PATH and no server_bin was given — install the "
                "`jammi-server` wheel or pass the path of a built binary"
            )
        env = dict(os.environ)
        env["JAMMI_ARTIFACT_DIR"] = self._artifact_dir
        # `:0` on BOTH listeners — the child binds, the kernel assigns, nothing
        # here ever holds-and-releases a port number.
        env["JAMMI_SERVER__FLIGHT_LISTEN"] = "127.0.0.1:0"
        env["JAMMI_SERVER__HEALTH_LISTEN"] = "127.0.0.1:0"
        env["JAMMI_SERVER__SERVICES"] = "all"
        env.update(self._env)
        self.proc = subprocess.Popen(
            [server_bin],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        announced: queue.Queue = queue.Queue(maxsize=1)
        self._log: list = []

        def drain() -> None:
            reported = False
            for line in self.proc.stdout:
                self._log.append(line)
                if not reported and line.startswith(self.BANNER):
                    fields = dict(
                        tok.split("=", 1)
                        for tok in line[len(self.BANNER):].split()
                        if "=" in tok
                    )
                    announced.put(int(fields["flight"].rsplit(":", 1)[1]))
                    reported = True  # announced once; keep draining regardless

        self._drain_thread = threading.Thread(target=drain, daemon=True)
        self._drain_thread.start()
        try:
            self.port = announced.get(timeout=30)
        except queue.Empty:
            self.proc.terminate()
            raise RuntimeError(
                "jammi-server never announced its listening ports (did it fail to "
                f"bind?):\n{''.join(self._log)}"
            ) from None
        # Bound is not yet serving — the banner prints before `serve()` — so
        # still poll a real client handshake before handing the server out.
        deadline = time.time() + 30
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"jammi-server exited early:\n{''.join(self._log)}")
            try:
                with connect(self.endpoint) as handshake:
                    handshake.get_server_info()
                return self
            except Exception:
                time.sleep(0.25)
        self.proc.terminate()
        raise RuntimeError("jammi-server did not become ready within 30s")

    @property
    def endpoint(self) -> str:
        """The `grpc://` target of the running server."""
        return f"grpc://127.0.0.1:{self.port}"

    def __exit__(self, *exc) -> None:
        """Stop the child and AWAIT its real exit — the release point itself.

        No `kill` fallback and no sidecar poll of the catalog file: a server
        that will not close on request is a finding, and a successor's
        successful open is what proves the single-process catalog was released.
        """
        self.proc.terminate()
        self.proc.wait(timeout=30)
        self.returncode = self.proc.returncode
