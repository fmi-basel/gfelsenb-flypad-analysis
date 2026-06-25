"""flypad CLI — thin entry point over the library (design §11).

Commands are stubs in M0; each milestone wires its command to the library core.
Heavy imports stay inside command bodies so ``flypad --help`` is fast.
"""

from __future__ import annotations

import typer
from rich.console import Console

from flypad import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="flypad — analyse flyPAD capacitance recordings (sips & activity bouts).",
)
console = Console()

_TODO = "[yellow]not yet implemented[/] (M0 scaffold)."


@app.command()
def version() -> None:
    """Print the flypad version."""
    console.print(f"flypad {__version__}")


@app.command()
def run(
    data_dir: str,
    config: str | None = typer.Option(None, "-c", "--config", help="Experiment YAML."),
    mode: str | None = typer.Option(None, "--mode", help="matlab_compat | corrected."),
) -> None:
    """Run the full pipeline on a folder of recordings."""
    console.print(f"[bold]run[/] {data_dir} (config={config}, mode={mode}) — {_TODO}")


@app.command()
def detect(
    data_dir: str,
    config: str | None = typer.Option(None, "-c", "--config", help="Experiment YAML."),
) -> None:
    """Detection only: sips + activity bouts."""
    console.print(f"[bold]detect[/] {data_dir} (config={config}) — {_TODO}")


@app.command()
def validate(
    data_dir: str,
    against: str = typer.Option(..., "--against", help="MATLAB Events .mat ground truth."),
) -> None:
    """Compare Python output against a MATLAB Events .mat (design §9)."""
    console.print(f"[bold]validate[/] {data_dir} vs {against} — {_TODO}")


@app.command()
def stats(results_dir: str) -> None:
    """(Re)compute summary statistics from saved events."""
    console.print(f"[bold]stats[/] {results_dir} — {_TODO}")


@app.command()
def plot(results_dir: str) -> None:
    """Render figures from saved results."""
    console.print(f"[bold]plot[/] {results_dir} — {_TODO}")


@app.command()
def gui() -> None:
    """Launch the desktop GUI (requires the 'gui' extra)."""
    console.print(f"[bold]gui[/] — {_TODO}")


config_app = typer.Typer(no_args_is_help=True, help="Inspect, validate and resolve configuration.")
app.add_typer(config_app, name="config")


@config_app.command("validate")
def config_validate(
    path: str = typer.Argument(..., help="Experiment YAML to validate."),
    preset: str | None = typer.Option(None, "--preset", help="Base preset override."),
) -> None:
    """Resolve preset + YAML and report whether it is valid."""
    from pydantic import ValidationError

    from flypad.config import load_config

    try:
        cfg = load_config(path, preset=preset)
    except (ValidationError, ValueError, TypeError, OSError) as exc:
        console.print(f"[red]invalid[/]: {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]valid[/] — mode={cfg.mode.value}, channels={cfg.hardware.n_channels}")


@config_app.command("schema")
def config_schema(
    out: str | None = typer.Option(None, "-o", "--out", help="Write schema to this path."),
) -> None:
    """Print (or write) the configuration JSON Schema."""
    import json

    from flypad.config import config_json_schema, write_schema

    if out:
        console.print(f"schema written to {write_schema(out)}")
    else:
        typer.echo(json.dumps(config_json_schema(), indent=2))


@config_app.command("show")
def config_show(
    path: str | None = typer.Argument(None, help="Optional experiment YAML."),
    preset: str | None = typer.Option(None, "--preset", help="Base preset."),
    set_: list[str] = typer.Option([], "--set", help="Dotted overrides key=value."),
) -> None:
    """Print the fully-resolved config as YAML."""
    import yaml

    from flypad.config import load_config

    cfg = load_config(path, preset=preset, overrides=set_)
    console.print(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))


def main() -> None:
    """Console-script entry point (``flypad``)."""
    app()


if __name__ == "__main__":
    main()
