"""Conservative OpenCV preprocessing with explicit coordinate transforms."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import PreprocessingConfig
from .models import AffineTransform, PageQuality, TransformOperation


def assess_page_quality(image: NDArray[np.uint8], config: PreprocessingConfig) -> PageQuality:
    gray = _to_gray(image)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    skew = estimate_skew_degrees(gray)
    warnings: list[str] = []
    if blur < config.blur_threshold:
        warnings.append("low_sharpness")
    if contrast < config.low_contrast_threshold:
        warnings.append("low_contrast")
    return PageQuality(blur, contrast, skew, tuple(warnings))


def estimate_skew_degrees(gray: NDArray[np.uint8]) -> float | None:
    """Estimate dominant horizontal-line skew, returning None when evidence is weak."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=max(40, gray.shape[1] // 4), maxLineGap=15)
    if lines is None:
        return None
    angles: list[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if abs(angle) <= 15:
            angles.append(angle)
    return float(np.median(angles)) if angles else None


def preprocess_page(
    original: NDArray[np.uint8], config: PreprocessingConfig, quality: PageQuality | None = None,
) -> tuple[NDArray[np.uint8], AffineTransform]:
    """Return a separate image and original-to-current coordinate transform."""
    image = original.copy()
    transform = np.eye(3, dtype=np.float64)
    operations: list[TransformOperation] = []
    if config.deskew:
        angle = (quality or assess_page_quality(image, config)).estimated_skew_degrees
        if angle is not None and abs(angle) >= 0.1:
            height, width = image.shape[:2]
            # OpenCV's affine matrix maps the source image to the rotated image.
            affine = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            image = cv2.warpAffine(image, affine, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
            transform = _to_homogeneous(affine) @ transform
            operations.append(TransformOperation("deskew", {"angle_degrees": angle}))
    if config.denoise:
        image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
        operations.append(TransformOperation("denoise", {"method": "fastNlMeansDenoisingColored"}))
    if config.enhance_contrast:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        operations.append(TransformOperation("contrast_enhancement", {
            "method": "CLAHE", "clip_limit": 2.0, "tile_grid_size": [8, 8],
        }))
    return image, AffineTransform(
        "preprocessing",
        tuple(tuple(float(v) for v in row) for row in transform),
        tuple(operations),
    )


def _to_gray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _to_homogeneous(affine: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.vstack((affine, np.array([0.0, 0.0, 1.0])))
