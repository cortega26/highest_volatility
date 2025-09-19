"""Tests for the Streamlit entry point bootstrap logic."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


class _DummyContext:
    """Basic context manager that also tolerates arbitrary calls."""

    def __enter__(self) -> "_DummyContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - interface only
        return False

    def __getattr__(self, name: str) -> Callable[..., Any]:
        return lambda *args, **kwargs: None


class _DummySidebar(_DummyContext):
    def header(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - interface only
        return None

    def slider(self, *args: Any, **kwargs: Any) -> Any:
        return kwargs.get("value") or (args[2] if len(args) > 2 else None)

    def number_input(self, *args: Any, **kwargs: Any) -> Any:
        if "value" in kwargs:
            return kwargs["value"]
        return args[2] if len(args) > 2 else None

    def selectbox(self, _label: str, options: list[Any], index: int = 0, **_: Any) -> Any:
        return options[index] if options else None

    def checkbox(self, _label: str, value: bool = False, **_: Any) -> bool:
        return value

    def button(self, *args: Any, **kwargs: Any) -> bool:  # pragma: no cover - deterministic default
        return False


class _DummyStreamlit(ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.sidebar = _DummySidebar()

    def set_page_config(self, *args: Any, **kwargs: Any) -> None:
        return None

    def title(self, *args: Any, **kwargs: Any) -> None:
        return None

    def caption(self, *args: Any, **kwargs: Any) -> None:
        return None

    def header(self, *args: Any, **kwargs: Any) -> None:
        return None

    def info(self, *args: Any, **kwargs: Any) -> None:
        return None

    def warning(self, *args: Any, **kwargs: Any) -> None:
        return None

    def error(self, *args: Any, **kwargs: Any) -> None:
        return None

    def subheader(self, *args: Any, **kwargs: Any) -> None:
        return None

    def dataframe(self, *args: Any, **kwargs: Any) -> None:
        return None

    def download_button(self, *args: Any, **kwargs: Any) -> None:
        return None

    def multiselect(self, *args: Any, **kwargs: Any) -> list[Any]:
        return list(kwargs.get("default", []))

    def slider(self, *args: Any, **kwargs: Any) -> Any:
        return self.sidebar.slider(*args, **kwargs)

    def number_input(self, *args: Any, **kwargs: Any) -> Any:
        return self.sidebar.number_input(*args, **kwargs)

    def selectbox(self, *args: Any, **kwargs: Any) -> Any:
        return self.sidebar.selectbox(*args, **kwargs)

    def checkbox(self, *args: Any, **kwargs: Any) -> bool:
        return self.sidebar.checkbox(*args, **kwargs)

    def button(self, *args: Any, **kwargs: Any) -> bool:
        return self.sidebar.button(*args, **kwargs)

    def markdown(self, *args: Any, **kwargs: Any) -> None:
        return None

    def altair_chart(self, *args: Any, **kwargs: Any) -> None:
        return None

    def tabs(self, labels: list[str]) -> list[_DummyContext]:
        return [_DummyContext() for _ in labels]

    def spinner(self, *args: Any, **kwargs: Any) -> _DummyContext:
        return _DummyContext()

    def cache_data(self, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


class _DummyAltair(ModuleType):
    class Chart:  # pragma: no cover - proxy only
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return None

        def mark_line(self, *args: Any, **kwargs: Any) -> "_DummyAltair.Chart":
            return self

        def encode(self, *args: Any, **kwargs: Any) -> "_DummyAltair.Chart":
            return self

        def properties(self, *args: Any, **kwargs: Any) -> "_DummyAltair.Chart":
            return self

    def __init__(self) -> None:
        super().__init__("altair")

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - attribute proxy
        if name == "Chart":
            return self.Chart
        return lambda *args, **kwargs: None


def test_streamlit_app_bootstrap_inserts_src(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"

    src_real = src_path.resolve()
    filtered_path: list[str] = []
    for entry in sys.path:
        try:
            if Path(entry).resolve() == src_real:
                continue
        except (OSError, RuntimeError):  # pragma: no cover - defensive against virtual entries
            pass
        filtered_path.append(entry)

    monkeypatch.setattr(sys, "path", filtered_path, raising=False)

    monkeypatch.setitem(sys.modules, "streamlit", _DummyStreamlit())
    monkeypatch.setitem(sys.modules, "altair", _DummyAltair())

    cached_modules: dict[str, ModuleType] = {}
    for name in list(sys.modules):
        if name == "highest_volatility" or name.startswith("highest_volatility."):
            cached_modules[name] = sys.modules.pop(name)

    try:
        runpy.run_path(str(repo_root / "src" / "highest_volatility" / "app" / "streamlit_app.py"), run_name="__main__")
    finally:
        for name in list(sys.modules):
            if name == "highest_volatility" or name.startswith("highest_volatility."):
                sys.modules.pop(name)
        sys.modules.update(cached_modules)

    assert sys.path[0] == str(src_path)
