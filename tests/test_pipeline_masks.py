import unittest

import numpy as np
from PIL import Image, ImageDraw

from processing.pipeline import (
    ProcessingConfig,
    build_refined_alpha,
    crop_to_alpha_content,
    validate_garment_mask,
)


def _source_with_garment(size=(320, 320), garment_color=(245, 245, 245), shadow=False):
    image = Image.new("RGBA", size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    if shadow:
        draw.ellipse((76, 246, 244, 272), fill=(210, 210, 210, 255))
    draw.polygon([(86, 86), (126, 58), (194, 58), (234, 86), (212, 154), (198, 252), (122, 252), (108, 154)], fill=garment_color)
    draw.polygon([(86, 86), (38, 158), (76, 176), (116, 112)], fill=garment_color)
    draw.polygon([(234, 86), (282, 158), (244, 176), (204, 112)], fill=garment_color)
    return image


def _alpha_from_shapes(size=(320, 320), include_body=True, include_sleeves=True, include_print=False, neck_opening=False):
    alpha = Image.new("L", size, 0)
    draw = ImageDraw.Draw(alpha)
    if include_body:
        draw.polygon([(86, 86), (126, 58), (194, 58), (234, 86), (212, 154), (198, 252), (122, 252), (108, 154)], fill=255)
    if include_sleeves:
        draw.polygon([(86, 86), (38, 158), (76, 176), (116, 112)], fill=255)
        draw.polygon([(234, 86), (282, 158), (244, 176), (204, 112)], fill=255)
    if include_print:
        draw.rectangle((138, 138, 182, 176), fill=255)
    if neck_opening:
        draw.ellipse((136, 56, 184, 96), fill=0)
    return np.array(alpha, dtype=np.uint8)


def _segmented_from_alpha(source, alpha):
    rgba = np.array(source, dtype=np.uint8)
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


class GarmentMaskTests(unittest.TestCase):
    def setUp(self):
        self.config = ProcessingConfig(alpha_threshold=2, smooth_edges=False)

    def test_white_garment_on_white_background_remains(self):
        source = _source_with_garment(garment_color=(250, 250, 250))
        alpha = _alpha_from_shapes(include_body=True, include_sleeves=True)
        refined, validation = build_refined_alpha(source, _segmented_from_alpha(source, alpha), self.config)
        self.assertFalse(validation.is_suspicious)
        self.assertGreater(np.count_nonzero(refined > 2), 25000)

    def test_colored_sleeves_and_missing_white_torso_are_recovered(self):
        source = _source_with_garment(garment_color=(250, 250, 250))
        draw = ImageDraw.Draw(source)
        draw.polygon([(86, 86), (38, 158), (76, 176), (116, 112)], fill=(200, 20, 20, 255))
        draw.polygon([(234, 86), (282, 158), (244, 176), (204, 112)], fill=(200, 20, 20, 255))
        alpha = _alpha_from_shapes(include_body=False, include_sleeves=True, include_print=True)
        refined, _ = build_refined_alpha(source, _segmented_from_alpha(source, alpha), self.config)
        self.assertGreater(refined[172, 160], 200)
        self.assertGreater(refined[150, 62], 200)
        self.assertGreater(refined[150, 258], 200)

    def test_black_garment_on_white_background_remains(self):
        source = _source_with_garment(garment_color=(25, 25, 25))
        alpha = _alpha_from_shapes(include_body=True, include_sleeves=True)
        refined, _ = build_refined_alpha(source, _segmented_from_alpha(source, alpha), self.config)
        self.assertGreater(np.count_nonzero(refined > 2), 25000)

    def test_gray_sleeves_black_torso_remain_one_product_bbox(self):
        source = _source_with_garment(garment_color=(20, 20, 20))
        alpha = _alpha_from_shapes(include_body=True, include_sleeves=True)
        refined, _ = build_refined_alpha(source, _segmented_from_alpha(source, alpha), self.config)
        cropped = crop_to_alpha_content(_segmented_from_alpha(source, refined), threshold=2)
        self.assertGreater(cropped.size[0], 230)
        self.assertGreater(cropped.size[1], 180)

    def test_printed_graphic_is_preserved_as_foreground(self):
        source = _source_with_garment(garment_color=(245, 245, 245))
        ImageDraw.Draw(source).rectangle((138, 138, 182, 176), fill=(10, 40, 90, 255))
        alpha = _alpha_from_shapes(include_body=True, include_sleeves=True, include_print=True)
        refined, _ = build_refined_alpha(source, _segmented_from_alpha(source, alpha), self.config)
        self.assertGreater(refined[150, 150], 200)

    def test_collar_opening_stays_transparent(self):
        source = _source_with_garment(garment_color=(245, 245, 245))
        alpha = _alpha_from_shapes(include_body=True, include_sleeves=True, neck_opening=True)
        refined, _ = build_refined_alpha(source, _segmented_from_alpha(source, alpha), self.config)
        self.assertEqual(refined[76, 160], 0)
        self.assertGreater(refined[116, 160], 200)

    def test_multiple_disconnected_visual_regions_are_not_dropped(self):
        source = _source_with_garment(garment_color=(245, 245, 245))
        alpha = np.zeros((320, 320), dtype=np.uint8)
        alpha[78:112, 74:116] = 255
        alpha[78:112, 204:246] = 255
        alpha[136:178, 138:182] = 255
        refined, _ = build_refined_alpha(source, _segmented_from_alpha(source, alpha), self.config)
        self.assertGreater(refined[94, 94], 200)
        self.assertGreater(refined[94, 224], 200)
        self.assertGreater(refined[156, 160], 200)

    def test_shadow_does_not_expand_crop_bounds(self):
        source = _source_with_garment(garment_color=(245, 245, 245), shadow=True)
        alpha = _alpha_from_shapes(include_body=True, include_sleeves=True)
        refined, _ = build_refined_alpha(source, _segmented_from_alpha(source, alpha), self.config)
        cropped = crop_to_alpha_content(_segmented_from_alpha(source, refined), threshold=2)
        self.assertLess(cropped.size[1], 230)

    def test_very_light_garment_on_light_background_is_not_removed_by_color(self):
        source = _source_with_garment(garment_color=(253, 253, 253))
        alpha = _alpha_from_shapes(include_body=True, include_sleeves=True)
        refined, _ = build_refined_alpha(source, _segmented_from_alpha(source, alpha), self.config)
        self.assertGreater(refined[172, 160], 200)

    def test_large_internal_opening_stays_transparent(self):
        source = _source_with_garment(garment_color=(245, 245, 245))
        alpha = _alpha_from_shapes(include_body=True, include_sleeves=True)
        alpha[130:190, 146:174] = 0
        refined, _ = build_refined_alpha(source, _segmented_from_alpha(source, alpha), self.config)
        self.assertEqual(refined[160, 160], 0)
        self.assertGreater(refined[200, 160], 200)

    def test_validation_flags_sparse_fragmented_masks(self):
        alpha = np.zeros((320, 320), dtype=bool)
        alpha[80:130, 40:90] = True
        alpha[80:130, 230:280] = True
        validation = validate_garment_mask(alpha, self.config)
        self.assertTrue(validation.is_suspicious)


if __name__ == "__main__":
    unittest.main()
