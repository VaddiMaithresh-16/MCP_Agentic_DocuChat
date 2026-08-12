import unittest

from llamacpp_utils import llamacpp_status


class LlamaCppUtilityTests(unittest.TestCase):
    def test_status_has_a_consistent_shape_when_unreachable(self):
        # Nothing is listening on this port in the test environment, so this
        # exercises the "server not running" path without needing a real
        # llama-server process.
        result = llamacpp_status(base_url="http://127.0.0.1:1", timeout=0.2)
        self.assertIn("running", result)
        self.assertIn("models", result)
        self.assertFalse(result["running"])
        self.assertEqual(result["models"], [])


if __name__ == "__main__":
    unittest.main()
