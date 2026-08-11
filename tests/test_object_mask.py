from __future__ import annotations

import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from src.object_mask import (
    cleanup_object_mask,
    dilate_mask,
    largest_component,
    load_object_mask_for_image,
    load_object_mask_manifest,
    track_and_refine_mask,
)


class ObjectMaskTests(unittest.TestCase):
    def test_largest_component_and_dilation(self) -> None:
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[20:60, 30:70] = 1
        mask[2:5, 2:5] = 1

        largest = largest_component(mask)
        dilated = dilate_mask(largest, 4)

        self.assertEqual(int(largest.sum()), 1600)
        self.assertGreater(int(dilated.sum()), int(largest.sum()))
        self.assertEqual(int(largest[3, 3]), 0)

    def test_cleanup_removes_thin_board_like_protrusion(self) -> None:
        mask = np.zeros((120, 160), dtype=np.uint8)
        cv2.ellipse(mask, (80, 65), (42, 36), 0, 0, 360, 1, -1)
        mask[62:66, 15:60] = 1

        cleaned = cleanup_object_mask(
            mask, morphology_px=7, smoothing_px=9)

        self.assertEqual(int(cleaned[63, 20]), 0)
        self.assertEqual(int(cleaned[65, 80]), 1)

    def test_tracks_translated_textured_object(self) -> None:
        rng = np.random.default_rng(4)
        height, width = 140, 180
        previous = np.full((height, width, 3), 35, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (80, 72), 32, 1, -1)
        texture = rng.integers(70, 230, previous.shape, dtype=np.uint8)
        previous[mask > 0] = texture[mask > 0]
        matrix = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]])
        current = cv2.warpAffine(
            previous, matrix, (width, height),
            borderValue=(35, 35, 35))

        tracked, confidence, inlier_count = track_and_refine_mask(
            previous, current, mask, grabcut_iterations=1)

        moments = cv2.moments(tracked.astype(np.uint8))
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        self.assertGreater(confidence, 0.5)
        self.assertGreaterEqual(inlier_count, 6)
        self.assertAlmostEqual(center_x, 85.0, delta=2.0)
        self.assertAlmostEqual(center_y, 69.0, delta=2.0)

    def test_rejects_mask_from_replaced_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "img_000001.png")
            mask_path = os.path.join(directory, "mask.png")
            manifest_path = os.path.join(directory, "manifest.json")
            image = np.zeros((20, 30, 3), dtype=np.uint8)
            cv2.imwrite(image_path, image)
            cv2.imwrite(mask_path, np.ones((20, 30), dtype=np.uint8) * 255)
            stat = os.stat(image_path)
            payload = {
                "frames": {
                    "img_000001.png": {
                        "mask": "mask.png",
                        "image_size_bytes": stat.st_size,
                        "image_mtime_ns": stat.st_mtime_ns,
                    }
                }
            }
            with open(manifest_path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            manifest = load_object_mask_manifest(manifest_path)
            loaded, _ = load_object_mask_for_image(
                manifest, image_path)
            self.assertEqual(loaded.shape, image.shape[:2])

            image[0, 0] = 255
            cv2.imwrite(image_path, image)
            os.utime(
                image_path,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
            )
            with self.assertRaises(RuntimeError):
                load_object_mask_for_image(manifest, image_path)


if __name__ == "__main__":
    unittest.main()
