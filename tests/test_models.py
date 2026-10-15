from answer_sheet_layout.models import AffineTransform, BoundingBox, Region, RegionClass


def test_affine_coordinate_round_trip() -> None:
    transform = AffineTransform("translate", ((1.0, 0.0, 10.0), (0.0, 1.0, -5.0), (0.0, 0.0, 1.0)))
    mapped = transform.map_box_to_original(BoundingBox(30, 15, 20, 10))
    assert mapped == BoundingBox(20, 20, 20, 10)


def test_region_validates_confidence() -> None:
    region = Region("r1", "page-1", RegionClass.UNKNOWN, BoundingBox(0, 0, 1, 1), "test", 0.5)
    assert region.confidence == 0.5
