import unittest

from camera_geometry_tests import run


class CameraGeometryTest(unittest.TestCase):
    def test_independent_camera_fixture(self):
        result = run()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(row["passed"] for row in result["checks"]))


if __name__ == "__main__":
    unittest.main()
