import unittest

from ollama_utils import detect_gpu


class OllamaUtilityTests(unittest.TestCase):
    def test_gpu_detection_returns_a_consistent_shape(self):
        result = detect_gpu()
        self.assertIn("available", result)
        self.assertIn("kind", result)
        self.assertIn("details", result)


if __name__ == "__main__":
    unittest.main()
