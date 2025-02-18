"""Image with annotation bounding boxes renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from .annotation import Annotation
    from .descriptor import Descriptor


class Renderer:
    """Image with annotation bounding boxes renderer."""

    color: str
    """Color of the bounding boxes."""

    thickness: int
    """Thickness of the bounding boxes."""

    def __init__(self, *, color: str = "red", thickness: int = 5) -> None:
        """Initializes an image with annotation bounding boxes renderer.

        Args:
            color: Color of the bounding boxes.
            thickness: Thickness of the bounding boxes.
        """
        self.color = color
        self.thickness = thickness

    def render(self, descriptor: Descriptor) -> Image.Image:
        """Render the image with annotation bounding boxes.

        Args:
            descriptor: YOLO annotated image descriptor.

        Returns:
            image: Rendered image.
        """
        image = descriptor.load_image()
        annotations = descriptor.load_annotations()

        draw = ImageDraw.ImageDraw(image)
        for annotation in annotations:
            bounding_box = self.make_bounding_box_from_annotation(annotation, image.size)
            draw.rectangle(bounding_box, outline=self.color, width=self.thickness)

        return image

    @staticmethod
    def make_bounding_box_from_annotation(
        annotation: Annotation,
        image_size: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        """Returns the XYXY bounding box calculated from the given annotation and image size."""
        image_width, image_height = image_size
        x_center = annotation.norm_x_center * image_width
        y_center = annotation.norm_y_center * image_height
        width = annotation.norm_width * image_width
        height = annotation.norm_height * image_height

        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        return x_min, y_min, x_max, y_max
