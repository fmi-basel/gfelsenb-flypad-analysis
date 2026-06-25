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


def main() -> None:
    """Console-script entry point (``flypad``)."""
    app()


if __name__ == "__main__":
    main()
