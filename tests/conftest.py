import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def api_key() -> str:
    return "test-api-key"


@pytest.fixture(scope="session", autouse=True)
def _configure_api_key(api_key: str) -> None:
    from highest_volatility.app import api as hv_api

    hv_api.settings.api_key = api_key
    hv_api.settings.require_api_key = True


@pytest.fixture
def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}
