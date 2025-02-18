"""YOLO dataset viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from pyglet.image import ImageData
from pyglet.sprite import Sprite
from pyglet.window import Window, key

from .dataset import Dataset
from .renderer import Renderer

if TYPE_CHECKING:
    from pathlib import Path

    from .descriptor import Descriptor


class Viewer(Window):
    """YOLO dataset viewer."""

    dataset: Dataset
    """YOLO dataset."""

    renderer: Renderer
    """Image with annotation bounding boxes renderer."""

    base_scale: float
    """Base scale."""

    zoom_factor: float
    """Zoom factor."""

    current_index: int
    """Current index."""

    sprite: Sprite | None
    """Image sprite."""

    def __init__(self, dataset_directory: Path, *, color: str = "red", thickness: int = 5) -> None:
        """Initializes a YOLO dataset viewer.

        Args:
            dataset_directory: Path to the dataset directory.
            color: Color of the bounding boxes.
            thickness: Thickness of the bounding boxes.
        """
        self.dataset = Dataset(dataset_directory)
        self.renderer = Renderer(color=color, thickness=thickness)
        self.base_scale = 1.0
        self.zoom_factor = 1.0
        self.current_index = 0
        self.sprite = None
        super().__init__()
        self.display_current_image()

    @property
    def current_descriptor(self) -> Descriptor:
        """Current YOLO annotated image descriptor."""
        return self.dataset[self.current_index]

    @property
    def current_caption(self) -> str:
        """Current caption."""
        return f"{self.current_descriptor.name} ({self.current_index + 1}/{len(self.dataset)}) - YOLO Dataset Viewer"

    def reset_zoom(self) -> None:
        """Resets the zoom factor."""
        self.zoom_factor = 1.0

    def display_current_image(self) -> None:
        """Displays the current image."""
        image = self.renderer.render(self.current_descriptor)

        self.set_size(image.width, image.height)
        self.set_caption(self.current_caption)

        pitch = -image.width * len(image.mode)
        image_data = ImageData(image.width, image.height, image.mode, image.tobytes(), pitch=pitch)
        self.sprite = Sprite(image_data)
        self.sprite.scale = self.base_scale * self.zoom_factor

    def display_next_image(self) -> None:
        """Displays the next image."""
        self.current_index = (self.current_index + 1) % len(self.dataset)
        self.reset_zoom()
        self.display_current_image()

    def display_previous_image(self) -> None:
        """Displays the previous image."""
        self.current_index = (self.current_index - 1) % len(self.dataset)
        self.reset_zoom()
        self.display_current_image()

    def zoom_in(self, *, increment: float = 0.1) -> None:
        """Increases the zoom factor."""
        self.zoom_factor += increment
        if self.sprite:
            self.sprite.scale = self.base_scale * self.zoom_factor

    def zoom_out(self, *, increment: float = 0.1) -> None:
        """Decreases the zoom factor (ensuring it doesn't go below a minimum)."""
        self.zoom_factor = max(0.1, self.zoom_factor - increment)
        if self.sprite:
            self.sprite.scale = self.base_scale * self.zoom_factor

    @override
    def on_draw(self) -> None:
        self.clear()
        if self.sprite:
            self.sprite.draw()

    @override
    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol in {key.ESCAPE, key.Q}:
            self.close()
        elif symbol in {key.RIGHT, key.N}:
            self.display_next_image()
        elif symbol in {key.LEFT, key.P}:
            self.display_previous_image()
        elif symbol in {key.PLUS, key.EQUAL, key.NUM_ADD}:
            self.zoom_in()
        elif symbol in {key.MINUS, key.NUM_SUBTRACT}:
            self.zoom_out()
