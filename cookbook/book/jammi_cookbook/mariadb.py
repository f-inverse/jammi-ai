"""A MariaDB server a chapter federates as its MySQL source.

pip ships no MySQL-family server, so :func:`server` installs ``mariadb-server``
with the host's package manager when it is missing — apt on Colab and
Debian/Ubuntu, dnf or yum on RHEL/AlmaLinux (the CI image) — which needs root,
as both of those runtimes are. It then starts a private instance on a scratch
data directory and a free loopback port, with one user the chapter connects
as. The instance has no TLS, so its URL says ``sslmode=disabled``.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

USER = "jammi"
PASSWORD = "jammi"
# RHEL-family packages install the daemon under /usr/libexec, off root's PATH.
_DAEMON_PATH = os.pathsep.join([os.environ.get("PATH", ""), "/usr/sbin", "/usr/libexec"])
_STARTUP_SECONDS = 60


def _which(*names: str) -> str | None:
    return next((found for n in names if (found := shutil.which(n, path=_DAEMON_PATH))), None)


@dataclass(frozen=True)
class _Toolchain:
    daemon: str
    initialise: str


def _toolchain() -> _Toolchain | None:
    """The MariaDB daemon and data-directory initialiser, when both are on the
    path. A MySQL install has a ``mysqld`` and no initialiser, so it is not one."""
    daemon = _which("mariadbd", "mysqld")
    initialise = _which("mariadb-install-db", "mysql_install_db")
    return _Toolchain(daemon, initialise) if daemon and initialise else None


def _install() -> _Toolchain:
    """The MariaDB server's toolchain, installed from the package manager first
    if the host has none."""
    if (toolchain := _toolchain()) is None:
        if shutil.which("apt-get"):
            commands = [["apt-get", "update", "-qq"],
                        ["apt-get", "install", "-y", "-qq", "mariadb-server"]]
        elif manager := _which("dnf", "yum"):
            commands = [[manager, "install", "-y", "-q", "mariadb-server"]]
        else:
            raise RuntimeError("no apt-get, dnf or yum to install mariadb-server with")
        env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
        for command in commands:
            subprocess.run(command, check=True, env=env, capture_output=True, text=True)
        toolchain = _toolchain()
    if toolchain is None:
        raise RuntimeError(
            "mariadb-server installed, but its daemon and mariadb-install-db are not on the path"
        )
    return toolchain


@dataclass(frozen=True)
class MariaDB:
    """A running instance: SQL goes in over its socket as root, and a source
    connects over TCP as :data:`USER`."""

    port: int
    socket: Path

    def url(self, database: str) -> str:
        """The ``mysql://`` URL a source registers ``database`` by."""
        return f"mysql://{USER}:{PASSWORD}@127.0.0.1:{self.port}/{database}?sslmode=disabled"

    def execute(self, sql: str) -> None:
        """Run ``sql`` as root."""
        client = _which("mariadb", "mysql")
        subprocess.run([client, f"--socket={self.socket}", "-uroot", "-e", sql],
                       check=True, capture_output=True, text=True)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextlib.contextmanager
def server() -> Iterator[MariaDB]:
    """A fresh MariaDB instance for the ``with`` block, stopped after it."""
    toolchain = _install()
    home = Path(tempfile.mkdtemp(prefix="jammi_mariadb_"))
    data = home / "data"
    subprocess.run([toolchain.initialise, "--no-defaults", f"--datadir={data}", "--user=root"],
                   check=True, capture_output=True, text=True)
    instance = MariaDB(port=_free_port(), socket=home / "mysqld.sock")
    process = subprocess.Popen(
        [toolchain.daemon, "--no-defaults", f"--datadir={data}", "--user=root",
         "--skip-name-resolve", "--bind-address=127.0.0.1", f"--port={instance.port}",
         f"--socket={instance.socket}", f"--pid-file={home / 'mysqld.pid'}",
         f"--log-error={home / 'mysqld.err'}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + _STARTUP_SECONDS
        while True:
            try:
                instance.execute("SELECT 1")
                break
            except subprocess.CalledProcessError:
                if process.poll() is not None or time.monotonic() > deadline:
                    log = (home / "mysqld.err").read_text(errors="replace")[-4000:]
                    raise RuntimeError(f"MariaDB did not start:\n{log}") from None
                time.sleep(0.5)
        # Named for the loopback address, not '%': an initialised data
        # directory may hold an anonymous ''@'localhost', which would outrank
        # a '%' user for a local connection; name resolution is off, so a
        # connection from 127.0.0.1 is matched as exactly that.
        instance.execute(
            f"CREATE USER '{USER}'@'127.0.0.1' IDENTIFIED BY '{PASSWORD}';"
            f"GRANT ALL ON *.* TO '{USER}'@'127.0.0.1';"
        )
        yield instance
    finally:
        process.terminate()
        process.wait(timeout=60)
        shutil.rmtree(home, ignore_errors=True)
