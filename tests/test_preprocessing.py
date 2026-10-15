import cv2
import numpy as np

from answer_sheet_layout.config import PreprocessingConfig
from answer_sheet_layout.models import BoundingBox
from answer_sheet_layout.preprocessing import assess_page_quality, preprocess_page


def test_default_preprocessing_keeps_pixels_and_identity_transform() -> None:
    image = np.full((100, 120, 3), 255, dtype=np.uint8)
    cv2.line(image, (0, 50), (119, 50), (50, 50, 50), 1)
    config = PreprocessingConfig()
    result, transform = preprocess_page(image, config)
    assert np.array_equal(result, image)
    assert transform.operations == ()
    assert transform.map_box_to_original(BoundingBox(5, 7, 10, 11)) == BoundingBox(5, 7, 10, 11)


def test_quality_flags_low_contrast_image() -> None:
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    quality = assess_page_quality(image, PreprocessingConfig())
    assert "low_contrast" in quality.warnings
