#!/usr/bin/env python3
"""Emit environment diagnostics to help debug CI networking issues."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from highest_volatility.errors import describe_exception


DEFAULT_HOSTS = ["query1.finance.yahoo.com", "example.com"]
DEFAULT_URLS = [
    "https://query1.finance.yahoo.com/v8/finance/chart/AAPL",
    "https://example.com",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture runtime diagnostics")
    parser.add_argument(
        "--host",
        action="append",
        dest="hosts",
        default=[],
        help="Hostname to resolve (may be specified multiple times)",
    )
    parser.add_argument(
        "--url",
        action="append",
        dest="urls",
        default=[],
        help="HTTP(S) URL to probe (may be specified multiple times)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds for HTTP probes (default: 10)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Write a condensed summary to GITHUB_STEP_SUMMARY if available",
    )
    return parser.parse_args()


def resolve_host(host: str) -> dict[str, Any]:
    record: dict[str, Any] = {"host": host}
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception as exc:  # pragma: no cover - depends on environment
        record["error"] = describe_exception(exc)
        return record

    addresses = sorted({info[4][0] for info in infos})
    record["addresses"] = addresses
    return record


async def probe_url(url: str, timeout: float) -> dict[str, Any]:
    record: dict[str, Any] = {"url": url}
    try:
        import aiohttp

        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            try:
                async with session.get(url, allow_redirects=False) as resp:
                    record.update(
                        {
                            "ok": True,
                            "status": resp.status,
                            "reason": resp.reason,
                            "headers": {k: v for k, v in resp.headers.items()},
                        }
                    )
            except aiohttp.ClientResponseError as exc:
                # Provide body-less responses for clarity.
                record["ok"] = False
                record["error"] = describe_exception(exc)
            except Exception as exc:  # pragma: no cover - depends on network
                record["ok"] = False
                record["error"] = describe_exception(exc)
    except Exception as exc:  # pragma: no cover - import/runtime issues
        record["ok"] = False
        record["error"] = describe_exception(exc)
    return record


async def gather_http(urls: list[str], timeout: float) -> list[dict[str, Any]]:
    tasks = [probe_url(url, timeout) for url in urls]
    return await asyncio.gather(*tasks)


def environment_snapshot() -> dict[str, Any]:
    interesting_env = [
        "GITHUB_ACTIONS",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_NUMBER",
        "GITHUB_WORKFLOW",
        "RUNNER_NAME",
        "RUNNER_OS",
        "RUNNER_ARCH",
    ]
    env_payload = {key: os.getenv(key) for key in interesting_env if os.getenv(key) is not None}
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "timestamp": timestamp,
        "python_version": sys.version,
        "platform": platform.platform(),
        "environment": env_payload,
        "cwd": os.getcwd(),
    }


def write_summary(payload: dict[str, Any]) -> None:
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    lines = ["# Diagnostic summary", "", "## DNS"]
    for record in payload.get("dns", []):
        if "error" in record:
            lines.append(f"- {record['host']}: ❌ {record['error']['message']}")
        else:
            addrs = ", ".join(record.get("addresses", [])) or "(none)"
            lines.append(f"- {record['host']}: ✅ {addrs}")
    lines.append("\n## HTTP")
    for record in payload.get("http", []):
        if record.get("ok"):
            lines.append(
                f"- {record['url']}: ✅ {record.get('status')} {record.get('reason', '')}".rstrip()
            )
        else:
            error = record.get("error", {})
            message = error.get("message") or error.get("type", "error")
            lines.append(f"- {record['url']}: ❌ {message}")
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    hosts = args.hosts or DEFAULT_HOSTS
    urls = args.urls or DEFAULT_URLS

    payload: dict[str, Any] = {
        "system": environment_snapshot(),
        "dns": [resolve_host(host) for host in hosts],
    }

    try:
        payload["http"] = asyncio.run(gather_http(urls, args.timeout))
    except KeyboardInterrupt:  # pragma: no cover - CLI convenience
        return 130

    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.summary:
        write_summary(payload)

    # Always succeed: diagnostics should not break the pipeline.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
