"""Command-line interface."""

from __future__ import annotations

from pathlib import Path

import pyglet
import rich_click as click

from . import __doc__ as app_doc
from .dataset import EmptyDatasetError
from .viewer import Viewer


@click.command(help=app_doc, context_settings={"show_default": True})
@click.argument("dataset_directory", type=Path)
@click.option("-c", "--color", type=str, default="red")
@click.option("-t", "--thickness", type=int, default=5)
def cli(
    dataset_directory: Path,
    *,
    color: str = "red",
    thickness: int = 5,
) -> None:
    """Runs the command-line interface."""
    try:
        _viewer = Viewer(dataset_directory, color=color, thickness=thickness)
    except EmptyDatasetError as e:
        print(e)  # noqa: T201
        return
    pyglet.app.run()
