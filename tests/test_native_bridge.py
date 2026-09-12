import unittest
import numpy as np

from native.c_bridge import (
    is_cpp_available,
    compute_frame_hash,
    detect_screen_changes,
    wait_for_screen_stabilize,
)


class TestNativeBridge(unittest.TestCase):
    def test_cpp_engine_available(self):
        self.assertTrue(is_cpp_available(), "C++ fast vision engine should be compiled and loaded")

    def test_dhash_computation(self):
        # Create gradient frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :50, :] = 200
        frame[:, 50:, :] = 50

        hash_val = compute_frame_hash(frame)
        self.assertIsInstance(hash_val, int)
        self.assertGreater(hash_val, 0)

        # Identical frame produces identical hash
        frame_copy = frame.copy()
        self.assertEqual(hash_val, compute_frame_hash(frame_copy))

        # Different frame produces different hash
        frame_diff = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_diff[50:, :, :] = 255
        self.assertNotEqual(hash_val, compute_frame_hash(frame_diff))

    def test_detect_screen_changes_identical(self):
        f1 = np.full((120, 160, 3), 100, dtype=np.uint8)
        f2 = f1.copy()

        diff_ratio, roi = detect_screen_changes(f1, f2)
        self.assertEqual(diff_ratio, 0.0)
        self.assertIsNone(roi)

    def test_detect_screen_changes_with_roi(self):
        f1 = np.zeros((200, 300, 3), dtype=np.uint8)
        f2 = f1.copy()

        # Simulate a 50x40 window appearing at x=[60, 110), y=[30, 70)
        f2[30:70, 60:110, :] = 220

        diff_ratio, roi = detect_screen_changes(f1, f2, threshold=20)

        expected_pixels = 50 * 40
        total_pixels = 200 * 300
        expected_ratio = expected_pixels / total_pixels

        self.assertAlmostEqual(diff_ratio, expected_ratio, places=3)
        self.assertIsNotNone(roi)
        min_x, min_y, max_x, max_y = roi
        self.assertEqual(min_x, 60)
        self.assertEqual(min_y, 30)
        self.assertEqual(max_x, 109)
        self.assertEqual(max_y, 69)

    def test_screen_stabilization_polls_and_returns(self):
        # Create frames with distinct spatial patterns so dHash changes until stable
        f1 = np.zeros((60, 80, 3), dtype=np.uint8)
        f1[:, :20] = 255
        f2 = np.zeros((60, 80, 3), dtype=np.uint8)
        f2[:, 20:40] = 255
        f3 = np.zeros((60, 80, 3), dtype=np.uint8)
        f3[:, 40:60] = 255

        frames = [f1, f2, f3, f3, f3]  # f3 repeated
        idx = 0

        def fake_capture():
            nonlocal idx
            f = frames[min(idx, len(frames) - 1)]
            idx += 1
            return f


        stabilized = wait_for_screen_stabilize(
            fake_capture,
            max_wait_sec=0.8,
            poll_interval_sec=0.01,
            hash_stability_count=2,
        )
        self.assertEqual(stabilized.shape, (60, 80, 3))
        self.assertTrue(idx >= 4)


if __name__ == "__main__":
    unittest.main()
