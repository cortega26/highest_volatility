from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import closing
from datetime import datetime, timezone

import httpx
import pandas as pd
import pytest
import uvicorn
from playwright.sync_api import Page


def _find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        _, port = sock.getsockname()
    return int(port)


@pytest.fixture(scope="session")
def _apply_app_overrides(tmp_path_factory: pytest.TempPathFactory) -> None:
    from pytest import MonkeyPatch

    from highest_volatility.app import api as api_module

    monkeypatch = MonkeyPatch()

    async def fake_schedule_cache_refresh(*_: object, **__: object) -> None:
        await asyncio.sleep(0)

    def fake_build_universe(limit: int, *, validate: bool = True) -> tuple[list[str], pd.DataFrame]:
        tickers = ["AAPL", "MSFT", "GOOG", "TSLA"]
        fortune = pd.DataFrame(
            {
                "rank": list(range(1, len(tickers) + 1)),
                "company": ["Alpha", "Beta", "Gamma", "Delta"],
                "ticker": tickers,
            }
        )
        return tickers[:limit], fortune.iloc[:limit].copy()

    async def fake_download_price_history(*_: object, **__: object) -> pd.DataFrame:
        index = pd.date_range(datetime.now(tz=timezone.utc) - pd.Timedelta(days=4), periods=5, freq="D")
        return pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=index)

    db_path = tmp_path_factory.mktemp("annotations") / "annotations.db"
    api_module.settings.annotations_db_path = str(db_path)
    for attr in ("annotation_store", "annotation_lock"):
        if hasattr(api_module.app.state, attr):
            delattr(api_module.app.state, attr)

    monkeypatch.setattr(api_module, "schedule_cache_refresh", fake_schedule_cache_refresh, raising=False)
    monkeypatch.setattr(api_module, "build_universe", fake_build_universe, raising=True)
    monkeypatch.setattr(api_module, "download_price_history", fake_download_price_history, raising=False)

    yield

    monkeypatch.undo()


@pytest.fixture(scope="session")
def uvicorn_server(_apply_app_overrides: None) -> str:
    port = _find_free_port()
    config = uvicorn.Config(
        "highest_volatility.app.api:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    with httpx.Client() as client:
        while time.time() < deadline:
            try:
                response = client.get(f"{base_url}/healthz", timeout=1)
            except httpx.HTTPError:
                time.sleep(0.2)
                continue
            if response.status_code in (200, 503):
                break
        else:  # pragma: no cover - defensive timeout guard
            raise RuntimeError("Timed out waiting for uvicorn to start")

    yield base_url

    server.should_exit = True
    thread.join(timeout=10)
    if thread.is_alive():  # pragma: no cover - defensive cleanup
        raise RuntimeError("uvicorn thread failed to exit")


@pytest.mark.e2e
def test_pwa_offline_sync(page: Page, uvicorn_server: str) -> None:
    page.set_default_timeout(10000)
    base_url = uvicorn_server

    page.goto(f"{base_url}/", wait_until="networkidle")
    page.wait_for_selector("#annotations-container .annotation-card")
    page.wait_for_function("navigator.serviceWorker && navigator.serviceWorker.controller")

    page.evaluate("async () => await window.hvDataLayer.syncPendingMutations()")
    page.wait_for_timeout(200)

    page.context.set_offline(True)
    page.wait_for_selector("text=Offline")

    note_area = page.locator('[data-ticker="AAPL"] textarea')
    note_area.fill("client-offline-note")
    page.locator('[data-ticker="AAPL"] button[data-action="save"]').click()
    page.wait_for_timeout(300)

    pending = page.evaluate("async () => (await window.hvDataLayer.getMergedAnnotations()).pending.length")
    assert pending >= 1

    page.click("#refresh-universe")
    page.wait_for_timeout(300)
    assert page.locator('[data-ticker="AAPL"]').is_visible()

    time.sleep(0.3)
    with httpx.Client() as client:
        response = client.put(
            f"{base_url}/annotations/AAPL",
            json={"note": "server-wins", "client_timestamp": datetime.utcnow().isoformat()},
            timeout=5,
        )
        assert response.status_code == 200

    page.context.set_offline(False)
    page.wait_for_selector("text=Online")

    sync_result = page.evaluate("async () => await window.hvDataLayer.syncPendingMutations()")
    assert sync_result["conflicts"], "Expected at least one conflict during sync"

    merged = page.evaluate("async () => await window.hvDataLayer.getMergedAnnotations()")
    assert merged["pending"] == []
    audit_types = {entry["type"] for entry in merged["audit"]}
    assert "conflict" in audit_types

    final_items = {item["ticker"]: item for item in merged["items"]}
    assert final_items["AAPL"]["note"] == "server-wins"
    assert not final_items["AAPL"].get("pending", False)
