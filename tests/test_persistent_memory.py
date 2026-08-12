import tempfile
import unittest

from persistent_memory import PersistentMemory, ResponseCache


class PersistentMemoryTests(unittest.TestCase):
    def test_memory_turns_and_cache_survive_store_use(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentMemory(f"{directory}/memory.sqlite3")
            memory.add_memory("user-1", "I prefer short explanations.")
            memory.add_turn("user-1", "What is RAG?", "RAG combines retrieval and generation.")
            cache = ResponseCache(memory)
            cache.put("question-key", {"answer": "cached answer"})

            self.assertIn("short explanations", memory.context("user-1"))
            self.assertEqual(memory.recent_turns("user-1")[0]["question"], "What is RAG?")
            self.assertEqual(cache.get("question-key")["answer"], "cached answer")

    def test_personal_fact_is_captured_without_a_memory_name(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentMemory(f"{directory}/memory.sqlite3")
            captured = memory.auto_capture("user-1", "I prefer concise answers.")

            self.assertIsNotNone(captured)
            self.assertIn("concise answers", memory.memories("user-1")[0])


if __name__ == "__main__":
    unittest.main()
