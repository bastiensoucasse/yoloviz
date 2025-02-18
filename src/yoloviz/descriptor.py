"""YOLO annotated image descriptor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from .annotation import Annotation

if TYPE_CHECKING:
    from pathlib import Path


class Descriptor:
    """YOLO annotated image descriptor."""

    image_file: Path
    """Path to the image file."""

    annotation_file: Path | None
    """Path to the annotation file."""

    def __init__(self, image_file: Path, annotation_file: Path | None = None) -> None:
        """Initializes a YOLO annotated image descriptor.

        Args:
            image_file: Path to the image file.
            annotation_file: Path to the annotation file.

        Raises:
            FileNotFoundError: If the image file or the annotation file is not found.
            ValueError: If the image file and the annotation file don't have the same stem.
        """
        if not image_file.is_file():
            raise FileNotFoundError(image_file)
        if annotation_file:
            if not annotation_file.is_file():
                raise FileNotFoundError(annotation_file)
            if image_file.stem != annotation_file.stem:
                msg = "Image and annotation file stems don't match."
                raise ValueError(msg)

        self.image_file = image_file
        self.annotation_file = annotation_file

    @property
    def name(self) -> str:
        """Image file name."""
        return self.image_file.name

    def load_image(self) -> Image.Image:
        """Returns the loaded image."""
        return Image.open(self.image_file).convert(mode="RGB")

    def load_annotations(self) -> list[Annotation]:
        """Returns the loaded annotations."""
        if not self.annotation_file:
            return []

        text = self.annotation_file.read_text(encoding="utf-8")
        return [Annotation.from_line(line) for line in text.splitlines()]
