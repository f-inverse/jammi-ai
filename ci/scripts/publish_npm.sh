#!/usr/bin/env bash
# Publishes a packed npm tarball through npm's OIDC trusted publishing with
# provenance. A version already on npm is skipped, so a re-run is idempotent.
# Trusted publishing needs npm >= 11.5.1, so npm is pinned here first.
#
#   bash ci/scripts/publish_npm.sh TARBALL
#
# TARBALL is absolute: npm reads a relative `dir/file.tgz` as a GitHub
# `owner/repo` shorthand.
set -euo pipefail

tarball="${1:?TARBALL}"
case "$tarball" in /*) ;; *) echo "::error::TARBALL must be absolute, got $tarball" >&2; exit 2 ;; esac

npm install -g npm@11.18.0
name="$(tar -xzOf "$tarball" package/package.json | node -p "JSON.parse(require('fs').readFileSync(0, 'utf8')).name")"
version="$(tar -xzOf "$tarball" package/package.json | node -p "JSON.parse(require('fs').readFileSync(0, 'utf8')).version")"
if npm view "$name@$version" version >/dev/null 2>&1; then
  echo "$name $version is already on npm"
else
  npm publish "$tarball" --provenance --access public
fi
