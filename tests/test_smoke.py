"""M0 smoke tests: package imports, CLI help, config defaults, and the registry."""

from __future__ import annotations

from typer.testing import CliRunner

import flypad
from flypad.cli.app import app
from flypad.config import Config, Mode
from flypad.pipeline import Context, register, registered_stages, run


def test_version_is_str() -> None:
    assert isinstance(flypad.__version__, str)


def test_cli_help_runs() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "flypad" in result.output


def test_config_defaults() -> None:
    cfg = Config()
    assert cfg.mode is Mode.corrected
    assert cfg.hardware.n_channels == 96
    assert cfg.acquisition.duration_samples == 360_000


def test_pipeline_registry_runs() -> None:
    @register("demo_stage")
    def _demo(ctx: Context, cfg: Config) -> Context:
        ctx.extra["ran"] = True
        return ctx

    assert "demo_stage" in registered_stages()
    seen: list[str] = []
    out = run(Context(), Config(), order=["demo_stage"], on_progress=seen.append)
    assert out.extra["ran"] is True
    assert seen == ["demo_stage"]


def test_unknown_stage_raises() -> None:
    import pytest

    with pytest.raises(KeyError):
        run(Context(), Config(), order=["nope"])
