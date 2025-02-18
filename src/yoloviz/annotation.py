"""YOLO annotation."""

from __future__ import annotations

from typing import NamedTuple


class Annotation(NamedTuple):
    """YOLO annotation."""

    label: int
    """Label identifier of the object (class ID)."""

    norm_x_center: float
    """X-center coordinate normalized between 0 and 1 (relative to the image size)."""

    norm_y_center: float
    """Y-center coordinate normalized between 0 and 1 (relative to the image size)."""

    norm_width: float
    """Width normalized between 0 and 1 (relative to the image size)."""

    norm_height: float
    """Height normalized between 0 and 1 (relative to the image size)."""

    @classmethod
    def from_line(cls, line: str) -> Annotation:
        """Creates a YOLO annotation from a YOLO annotation file line.

        Args:
            line: YOLO annotation file line.

        Returns:
            annotation: Created YOLO annotation.

        Raises:
            ValueError: If the annotation cannot be parsed or validated.
        """
        words = line.split()
        if len(words) != (annotation_value_length := 5):
            msg = f'Annotation parsing error: {annotation_value_length} values are required (got "{line}").'
            raise ValueError(msg)

        try:
            annotation = cls(int(words[0]), float(words[1]), float(words[2]), float(words[3]), float(words[4]))
        except ValueError as e:
            msg = f'Annotation parsing error: Unsupported value detected (got "{line}").'
            raise ValueError(msg) from e

        if annotation.label < 0:
            msg = f"Annotation validation error: Label should be positive or null (got {annotation.label})."
            raise ValueError(msg)
        for attribute in ("x_center", "y_center", "width", "height"):
            value = getattr(annotation, f"norm_{attribute}")
            if value <= 0 or value > 1:
                msg = f"Annotation validation error: {attribute.capitalize()} should be between 0 and 1 (got {value})."
                raise ValueError(msg)
        return annotation
