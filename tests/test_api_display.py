"""
Tests for the presentation helpers in api.py (face-box drawing and per-face crops).

They only touch drawing/cropping — no model is loaded and no recognition logic is involved.
"""
import base64

import cv2
import numpy as np

from api import BOX_COLORS, _draw_face_tag, _face_crop
from src.matching import MatchStatus


def _decode(data_url):
    header, b64 = data_url.split(",", 1)
    assert header == "data:image/jpeg;base64"
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)


def test_face_crop_is_a_small_square_jpeg():
    img = np.full((600, 800, 3), 200, np.uint8)
    crop = _decode(_face_crop(img, (300, 200, 500, 400)))
    assert crop.shape[0] == crop.shape[1] == 160


def test_face_crop_clamps_to_the_image_at_the_edges():
    img = np.full((300, 300, 3), 200, np.uint8)
    crop = _decode(_face_crop(img, (-40, -30, 120, 150)))  # face partly outside the image
    assert 0 < crop.shape[0] <= 160 and 0 < crop.shape[1] <= 160


def test_face_crop_outside_the_image_returns_none():
    img = np.full((100, 100, 3), 200, np.uint8)
    assert _face_crop(img, (500, 500, 600, 600)) is None


def test_draw_face_tag_uses_the_status_colour_even_at_the_top_edge():
    img = np.zeros((400, 400, 3), np.uint8)
    color = BOX_COLORS[MatchStatus.CONFIRMED]
    _draw_face_tag(img, (50, 0, 200, 150), 1, color)  # no room above the box -> tag drawn inside it
    assert (img.reshape(-1, 3) == np.array(color, np.uint8)).all(axis=1).any()
