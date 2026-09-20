import unittest

from visible_oracle_geometry import self_test


class VisibleOracleGeometryTest(unittest.TestCase):
    def test_cpu_visible_oracle_geometry_fixture(self):
        result = self_test()
        self.assertEqual(result["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
