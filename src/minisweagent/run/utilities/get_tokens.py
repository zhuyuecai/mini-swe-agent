#!/usr/bin/env python3

"""Sum token usage from SWE-bench experiment trajectories."""

import json
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def total_tokens_in_value(value: Any) -> int:
    if isinstance(value, dict):
        return sum(
            item if key == "total_tokens" and isinstance(item, int) else total_tokens_in_value(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return sum(total_tokens_in_value(item) for item in value)
    return 0


def total_tokens_in_experiment(path: Path) -> int:
    if not path.is_dir():
        msg = f"Experiment path is not a directory: {path}"
        raise ValueError(msg)
    return sum(
        total_tokens_in_value(json.loads(traj.read_text()))
        for child in path.iterdir()
        if child.is_dir()
        for traj in child.glob("*.traj.json")
    )


@app.command()
def main(path: Path = typer.Argument(..., help="Folder containing SWE-bench experiment outcome subfolders.")) -> None:
    """Print the sum of all total_tokens fields in child trajectory files."""
    typer.echo(total_tokens_in_experiment(path))


if __name__ == "__main__":
    app()
