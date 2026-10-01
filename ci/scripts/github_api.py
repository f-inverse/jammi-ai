"""The GitHub REST reads the CI scripts make: one transport, one pagination
rule, one shape check.

Every reader takes `fetch(url, token) -> dict` as a parameter, so a test drives
it against a fake surface with no network. Any transport, HTTP or JSON failure
raises `ApiError`: a caller decides what an unreadable answer means, and it is
never an empty one.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable

API_BASE = "https://api.github.com"
PER_PAGE = 100

FetchFn = Callable[[str, str], dict]


class ApiError(Exception):
    """A GitHub REST read that did not yield the JSON object asked for."""


def default_fetch(url: str, token: str) -> dict:
    """The real GitHub REST call. Never used by a test, which injects its own
    `fetch`."""
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https host
            body = resp.read()
    except (urllib.error.URLError, OSError) as e:
        raise ApiError(f"GET {url} failed: {e}") from e
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as e:
        raise ApiError(f"GET {url} returned invalid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ApiError(f"GET {url} returned a non-object JSON body")
    return parsed


def get(fetch: FetchFn, token: str, url: str) -> dict:
    """One object read; any failure is an `ApiError`."""
    try:
        data = fetch(url, token)
    except ApiError:
        raise
    except Exception as e:  # noqa: BLE001 - any fetch failure is an ApiError
        raise ApiError(f"GET {url} failed: {e}") from e
    if not isinstance(data, dict):
        raise ApiError(f"GET {url} returned a non-object JSON body")
    return data


def paginated(fetch: FetchFn, token: str, url: str, key: str) -> list[dict]:
    """Every item of the list field `key`, page by page while a page is full
    (`PER_PAGE` items): the page-number form of following `Link: rel="next"`,
    which needs no header access from `fetch`."""
    out: list[dict] = []
    page = 1
    while True:
        sep = "&" if "?" in url else "?"
        page_url = f"{url}{sep}per_page={PER_PAGE}&page={page}"
        data = get(fetch, token, page_url)
        if key not in data:
            raise ApiError(f"GET {page_url} returned an unexpected shape (missing {key!r})")
        items = data[key]
        if not isinstance(items, list):
            raise ApiError(f"GET {page_url} field {key!r} is not a list")
        out.extend(items)
        if len(items) < PER_PAGE:
            return out
        page += 1


def list_jobs(fetch: FetchFn, token: str, repo: str, run_id: int, filter_mode: str = "latest") -> list[dict]:
    """The jobs of run `run_id`. `filter=latest` (the default) is the latest
    ATTEMPT of each job, so a `gh run rerun <id> --failed` supersedes a stale
    attempt in place; `filter=all` is every attempt."""
    url = f"{API_BASE}/repos/{repo}/actions/runs/{run_id}/jobs?filter={filter_mode}"
    return paginated(fetch, token, url, "jobs")
