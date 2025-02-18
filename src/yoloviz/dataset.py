"""YOLO dataset."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, overload, override

from .descriptor import Descriptor

if TYPE_CHECKING:
    from pathlib import Path

ALLOWED_IMAGE_EXTENSIONS: set[str] = {"png", "jpg", "jpeg", "tif", "tiff"}
"""Allowed image extensions."""


class EmptyDatasetError(Exception):
    """Exception raised when a dataset is empty."""

    def __init__(self, dataset_directory: Path) -> None:
        """Initializes an empty dataset error.

        Args:
            dataset_directory: Path to the dataset directory.
        """
        super().__init__(f"Dataset empty: {dataset_directory}.")


class Dataset(Sequence[Descriptor]):
    """YOLO dataset."""

    images_directory: Path
    """Path to the images directory."""

    annotations_directory: Path | None
    """Path to the annotations directory."""

    image_files: list[Path]
    """List of paths to the image files."""

    def __init__(
        self,
        dataset_directory: Path,
        *,
        images_directory_name: str = "images",
        annotations_directory_name: str = "labels",
    ) -> None:
        """Initializes a YOLO dataset.

        Args:
            dataset_directory: Path to the dataset directory.
            images_directory_name: Name of the images directory.
            annotations_directory_name: Name of the annotations directory.

        Raises:
            FileNotFoundError: If a necessary directory is not found.
            EmptyDatasetError: If the dataset is empty.
        """
        dataset_directory = dataset_directory.resolve()
        if not dataset_directory.is_dir():
            raise FileNotFoundError(dataset_directory)

        images_candidate = dataset_directory / images_directory_name
        annotations_candidate = dataset_directory / annotations_directory_name

        if images_candidate.is_dir():
            self.images_directory = images_candidate
            self.annotations_directory = annotations_candidate if annotations_candidate.is_dir() else None
        elif annotations_candidate.is_dir():
            raise FileNotFoundError(images_candidate)
        else:
            self.images_directory = dataset_directory
            self.annotations_directory = None

        self.image_files = sorted(f for e in ALLOWED_IMAGE_EXTENSIONS for f in self.images_directory.glob(f"*.{e}"))
        if not self.image_files:
            raise EmptyDatasetError(dataset_directory)

    @overload
    def __getitem__(self, key: int) -> Descriptor: ...

    @overload
    def __getitem__(self, key: slice) -> list[Descriptor]: ...

    @override
    def __getitem__(self, key: int | slice) -> Descriptor | list[Descriptor]:
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]

        image_file = self.image_files[key]
        if not self.annotations_directory:
            return Descriptor(image_file)

        annotation_file = self.annotations_directory / image_file.relative_to(self.images_directory).with_suffix(".txt")
        if not annotation_file.is_file():
            return Descriptor(image_file)

        return Descriptor(image_file, annotation_file)

    @override
    def __len__(self) -> int:
        return len(self.image_files)
