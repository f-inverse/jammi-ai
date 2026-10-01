"""Which hosts the federation chapter's MariaDB helper takes as having MariaDB."""

from __future__ import annotations

from pathlib import Path

from jammi_cookbook import mariadb


def _executable(directory: Path, name: str) -> None:
    path = directory / name
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)


def test_a_mysql_install_is_not_taken_for_mariadb(monkeypatch, tmp_path):
    # MySQL 8 ships a `mysqld` and no data-directory initialiser.
    _executable(tmp_path, "mysqld")
    monkeypatch.setattr(mariadb, "_DAEMON_PATH", str(tmp_path))
    assert mariadb._toolchain() is None


def test_a_mariadb_install_is_its_daemon_and_its_initialiser(monkeypatch, tmp_path):
    _executable(tmp_path, "mariadbd")
    _executable(tmp_path, "mariadb-install-db")
    monkeypatch.setattr(mariadb, "_DAEMON_PATH", str(tmp_path))
    toolchain = mariadb._toolchain()
    assert toolchain is not None
    assert Path(toolchain.daemon).name == "mariadbd"
    assert Path(toolchain.initialise).name == "mariadb-install-db"
